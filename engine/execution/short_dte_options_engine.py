"""ShortDTEOptionsEngine — 1-7 DTE options overlay for Metadron Capital.

============================================================
LAYER:  layer7_execution (options arm)
ROLE:   Turn the daily scan into concrete 1-7 DTE option contracts that are
        priced, risk-gated and sized for L7UnifiedExecutionSurface, using
        Schwab chains as the ONLY market-data source.
============================================================

Pipeline per underlying (all horizons == the contract's DTE, never 30/45 d):

  1. Price history (Schwab)          → log returns, realised vol
  2. Momentum / RSI gate             → RSI(14) level + slope, 20-bar breakout,
                                        RSI divergence, 5/10/21-day momentum,
                                        acceleration   →  direction score ∈[-1,1]
  3. Beta corridor fair value        → BetaCorridor.calculate_target_beta on
                                        SPY drift/vol; corridor position and
                                        target beta give the directional
                                        FAIR-VALUE tilt (no futures hedge)
  4. Monte Carlo of the full scan    → AR(1) paths (MonteCarloBridge model) for
                                        EVERY underlying over exactly `dte`
                                        days: confidence, VaR95, P(up),
                                        and MC Greeks (Δ, Γ by bump-and-
                                        revalue on common paths)
     Gate: confidence ≥ 0.20 and VaR95 ≥ −40 %.
  5. Chain (Schwab, 1-7 DTE only)    → build a SHORT-TENOR vol surface from
                                        the real chain (1/2/3/5/7 d tenors)
  6. Black-Scholes only at chain DTE → Newton-Raphson IV per contract from
                                        mid; full Greeks; fair value from
                                        surface IV → edge (bps) + mispricing
  7. PredictiveOptionsSignal         → VRP / Skew / Vol-MR / Term / PCR fed
                                        INTO the composite, not just printed
  8. Composite                       → alpha × MC.conf × delta_quality ×
                                        edge × mispricing × predictive_adj
                                        must clear DecisionMatrix 0.55
  9. Sizing                          → OptionsSizer (≥200 bps edge, Kelly ×1.5
                                        clipped to 5 % NAV) then vega budget,
                                        then bucket caps (IG 10 / HY 10 /
                                        DIST 5 → total 25 % notional)
 10. Convexity                       → ConvexityHedgeManager put ladder mapped
                                        onto REAL SPY contracts at dte_max
 11. Regime structure                → OptionsStrategyBuilder.select_for_regime
                                        + theta_gamma_optimize decide single
                                        leg vs vertical

Output is a list of ``OptionTradeIntent`` that L7 routes to
``SchwabBroker.place_option_order`` / ``place_option_spread``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .options_engine import (
    BlackScholesModel,
    ConvexityHedgeManager,
    OptionsSizer,
    OptionsStrategyBuilder,
    PredictiveOptionsSignal,
    VolatilitySurface,
)

try:
    from engine.ml.bridges.monte_carlo_bridge import MonteCarloBridge
except Exception:  # pragma: no cover
    MonteCarloBridge = None  # type: ignore[assignment]

try:
    from engine.ml.pattern_recognition import ChartPatternDetector, MomentumSignalEngine
except Exception:  # pragma: no cover
    ChartPatternDetector = None  # type: ignore[assignment]
    MomentumSignalEngine = None  # type: ignore[assignment]

try:
    from engine.portfolio.beta_corridor import BetaCorridor
except Exception:  # pragma: no cover
    BetaCorridor = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Config / dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ShortDTEConfig:
    dte_min: int = 1
    dte_max: int = 7
    risk_free: float = 0.04
    history_days: int = 120
    # momentum / RSI
    rsi_period: int = 14
    breakout_lookback: int = 20
    rsi_breakout_level: float = 60.0
    rsi_breakdown_level: float = 40.0
    rsi_overbought: float = 75.0
    rsi_oversold: float = 25.0
    # Monte Carlo (full-scan)
    mc_paths: int = 3000
    mc_min_confidence: float = 0.20      # on the horizon-normalised confidence (see monte_carlo)
    mc_calibration_days: int = 21        # the full-scan MC confidence is calibrated on a 21-day horizon
    min_contracts: int = 1               # small mandates (Individual $5.7k) trade 1-lots; legacy sizer default was 5
    chain_retry_wait_s: float = 20.0     # pause before the single chain re-pull after a fetch failure
    backfill_max_tries: int = 12         # SP400/SP600 names tried per empty bucket (many lack 1-7 DTE chains)
    backfill_names_per_bucket: int = 3
    max_position_pct_nav: float = 0.10   # Kelly output may fill a whole bucket (10 % NAV = 40 % of options headroom)
    kelly_multiplier: float = 1.5        # aggressive Kelly (architecture: 1.5× Kelly on confirmed mispricing)
    mc_var95_floor: float = -0.40
    mc_seed: int = 42
    # chain filters
    strike_count: int = 24
    max_spread_pct: float = 0.15
    min_open_interest: int = 50
    delta_lo: float = 0.20
    delta_hi: float = 0.60
    delta_target: float = 0.40
    # scoring
    min_composite: float = 0.55
    min_edge_bps: float = 200.0
    # sizing / caps (fractions of NAV)
    bucket_caps: Dict[str, float] = field(default_factory=lambda: {"OPTIONS_IG": 0.10, "OPTIONS_HY": 0.10, "OPTIONS_DISTRESSED": 0.05})
    total_options_cap: float = 0.25
    max_single_option_pct: float = 0.05
    vega_budget_pct_nav: float = 0.005       # $ P&L per 1 vol-point allowed across the book
    # convexity ladder (short-dated → tighter OTM rungs than the 30d default)
    ladder_otm: Tuple[float, float, float] = (0.02, 0.04, 0.07)
    hedge_underlying: str = "SPY"
    market_proxy: str = "SPY"


@dataclass
class MarketContext:
    proxy: str
    spot: float
    rm_annual: float            # annualised drift of the proxy (63d)
    sigma_m: float              # annualised realised vol (20d)
    vix: float                  # $VIX level if quoted, else implied from proxy realised vol
    target_beta: float
    base_beta: float
    corridor_position: str      # BELOW / WITHIN / ABOVE
    direction_bias: float       # [-1, 1] fair-value directional tilt from the corridor
    regime: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
@dataclass
class CTARead:
    """WonderTrader CTA trend core on daily bars: dual-MA (10/30) ⊕ Donchian breakout (20) ⊕
    ROC-momentum z (12/60), regime-weighted consensus (TRENDING/RANGE/STRESS/CRASH) — the
    momentum approach documented in the WonderTrader engine, reused for options direction."""
    regime: str = "RANGE"
    direction: int = 0
    strength: float = 0.0        # [0, 1]
    consensus: float = 0.0       # signed = direction × strength
    dominant: str = ""
    dual_ma: float = 0.0
    breakout: float = 0.0
    momentum: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MomentumRead:
    rsi: float
    rsi_prev: float
    rsi_slope: float
    breakout: str               # "breakout" / "breakdown" / ""
    breakout_conf: float
    divergence: str             # "bullish" / "bearish" / ""
    divergence_conf: float
    mom_5d: float
    mom_10d: float
    mom_21d: float
    acceleration: float
    direction_score: float      # [-1, 1]
    notes: List[str] = field(default_factory=list)
    confirmed: bool = False     # RSI breakout/breakdown regime confirmed by a price pattern or 10d momentum
    cta: Optional[CTARead] = None   # WonderTrader CTA core read (regime-weighted trend consensus)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MCRead:
    dte: int
    confidence: float
    mean_return: float
    var_95: float
    paths_positive_pct: float
    passed: bool
    reason: str
    terminal_prices: Optional[np.ndarray] = None   # for MC Greeks / payoff

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("terminal_prices", None)
        return d


@dataclass
class ContractEval:
    symbol: str
    underlying: str
    put_call: str
    strike: float
    expiry: str
    dte: int
    bid: float
    ask: float
    mid: float
    schwab_iv: float
    bsm_iv: float
    surface_iv: float
    fair_value: float
    edge_bps: float
    mispricing_pct: float       # (surface_iv - bsm_iv) / surface_iv
    bsm_greeks: Dict[str, float]
    mc_greeks: Dict[str, float]
    mc_value: float
    mc_p_itm: float
    delta_quality: float
    composite: float
    alpha: float
    mc_conf: float
    predictive_adj: float
    reject_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OptionTradeIntent:
    ticker: str
    bucket: str
    structure: str               # "SINGLE" | "VERTICAL"
    direction: str               # "BULLISH" | "BEARISH"
    contract_symbol: str
    put_call: str
    strike: float
    expiry: str
    dte: int
    contracts: int
    limit_price: float
    notional: float
    composite: float
    edge_bps: float
    greeks: Dict[str, float]
    legs: List[dict]
    sizing: Dict[str, Any]
    rationale: List[str]
    instrument_type: str = "OPTION"
    signal_type: str = "OPTIONS_OVERLAY"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Short-tenor vol surface built from the REAL chain
# ---------------------------------------------------------------------------
class ShortTenorVolSurface(VolatilitySurface):
    """1/2/3/5/7-day surface fitted to Schwab chain IVs.

    Keeps the ``1M``/``1Y`` aliases (→ shortest / longest short tenor) so the
    existing PredictiveOptionsSignal maths (skew_25d, term_spread) operate on
    the 1-7 DTE term structure rather than on a synthetic 30-365 d grid.
    """

    TENORS = {"1D": 1, "2D": 2, "3D": 3, "5D": 5, "7D": 7}

    def __init__(self, chain: List[Any], vix: float, hist_vol_30d: float, hist_vol_90d: float):
        self._chain = chain
        super().__init__(vix=vix, hist_vol_30d=hist_vol_30d, hist_vol_90d=hist_vol_90d)

    def _term_structure_factors(self) -> Dict[str, float]:
        # Short-end baseline before chain fit: mildly inverted when VRP is high
        vrp = self.vix_decimal - self.hv30
        tilt = 0.04 if vrp > 0.05 else -0.02
        return {label: 1.0 + tilt * (1 - i / max(len(self.TENORS) - 1, 1)) for i, label in enumerate(self.TENORS)}

    def _build(self) -> None:
        super()._build()  # heuristic baseline for every tenor
        by_dte: Dict[int, List[Any]] = {}
        for q in self._chain:
            by_dte.setdefault(int(q.dte), []).append(q)
        dtes = sorted(by_dte)
        for label, days in self.TENORS.items():
            if not dtes:
                break
            nearest = min(dtes, key=lambda d: abs(d - days))
            if abs(nearest - days) > 2:
                continue
            fitted = self._fit_tenor(by_dte[nearest])
            if fitted:
                self.surface[label].update(fitted)
        # aliases used by PredictiveOptionsSignal
        labels = list(self.TENORS)
        self.surface["1M"] = dict(self.surface[labels[0]])   # shortest tenor
        self.surface["1Y"] = dict(self.surface[labels[-1]])  # longest short tenor

    @staticmethod
    def _fit_tenor(quotes: List[Any]) -> Dict[str, float]:
        def med(xs):
            return float(np.median(xs)) if xs else None

        atm = med([q.iv for q in quotes if abs(q.moneyness - 1.0) <= 0.01])
        p25 = med([q.iv for q in quotes if q.put_call == "PUT" and 0.18 <= abs(q.delta) <= 0.32])
        p10 = med([q.iv for q in quotes if q.put_call == "PUT" and 0.05 <= abs(q.delta) <= 0.15])
        c25 = med([q.iv for q in quotes if q.put_call == "CALL" and 0.18 <= abs(q.delta) <= 0.32])
        c10 = med([q.iv for q in quotes if q.put_call == "CALL" and 0.05 <= abs(q.delta) <= 0.15])
        out = {}
        for k, v in (("ATM", atm), ("25d_put", p25), ("10d_put", p10), ("25d_call", c25), ("10d_call", c10)):
            if v is not None and v > 0:
                out[k] = v
        return out

    def term_spread(self) -> float:
        """7D ATM minus 1D ATM (short-end term structure)."""
        return self.get_atm_vol("1Y") - self.get_atm_vol("1M")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ShortDTEOptionsEngine:
    """Produces 1-7 DTE option trade intents from Schwab data."""

    def __init__(self, broker, nav: Optional[float] = None, regime: str = "NORMAL",
                 config: Optional[ShortDTEConfig] = None):
        self.broker = broker
        self.cfg = config or ShortDTEConfig()
        self.regime = regime
        self.nav = float(nav) if nav else float(getattr(getattr(broker, "state", None), "nav", 0.0) or 100_000.0)
        self.bs = BlackScholesModel
        self.sizer = OptionsSizer()
        self.sizer.MIN_EDGE_BPS = self.cfg.min_edge_bps
        self.backfill_candidates: Dict[str, List[str]] = {}   # {"SP400": [...], "SP600": [...]} set by the cycle
        self.last_backfill: Dict[str, List[str]] = {}
        self.sizer.MIN_CONTRACTS = self.cfg.min_contracts
        self.sizer.MAX_CONTRACTS_PCT_NAV = self.cfg.max_position_pct_nav
        self.sizer.KELLY_MULTIPLIER = self.cfg.kelly_multiplier
        self.sizer.MAX_CONTRACTS_PCT_NAV = self.cfg.max_single_option_pct
        self.sizer.MC_STEPS = self.cfg.dte_max            # steps == days for 1-7 DTE
        self._rng = np.random.RandomState(self.cfg.mc_seed)
        self._mc = MonteCarloBridge(seed=self.cfg.mc_seed) if MonteCarloBridge else None
        self._beta = BetaCorridor(nav=self.nav) if BetaCorridor else None
        self.hedge_mgr = ConvexityHedgeManager(nav=self.nav, regime=regime)
        self.hedge_mgr.PUT_LADDER = list(self.cfg.ladder_otm)
        self.last_run: Dict[str, Any] = {}
        self.existing_option_notional: Dict[str, float] = {}   # bucket → $ already deployed

    # ------------------------------------------------------------------
    # 1-3. market context (beta corridor fair value)
    # ------------------------------------------------------------------
    def market_context(self) -> MarketContext:
        cfg = self.cfg
        hist = self.broker.get_price_history(cfg.market_proxy, days=cfg.history_days)
        close = np.asarray(hist.get("close", []), dtype=float)
        spot = float(close[-1]) if close.size else float(self.broker.get_quote(cfg.market_proxy) or 0.0)
        if close.size < 25:
            return MarketContext(cfg.market_proxy, spot, 0.0, 0.15, 15.0, 0.0, 0.0, "UNKNOWN", 0.0, self.regime)
        rets = np.diff(np.log(close))
        rm = float(np.mean(rets[-63:]) * TRADING_DAYS)
        sigma_m = float(np.std(rets[-20:]) * math.sqrt(TRADING_DAYS))
        vix = self._vix_level(sigma_m)
        target_beta, base_beta, pos = 0.0, 0.0, "UNKNOWN"
        if self._beta is not None:
            try:
                st = self._beta.calculate_target_beta(Rm=rm, sigma_m=sigma_m, market_returns=rets)
                target_beta, base_beta, pos = float(st.target_beta), float(st.base_beta), st.corridor_position
            except Exception as e:
                logger.warning("BetaCorridor failed: %s", e)
        # Fair-value directional tilt: where drift sits inside the 7-12 % corridor,
        # scaled by the target beta the corridor wants.
        # Continuous: the tilt grows with the DISTANCE from the corridor edge (a drift 1 % under
        # the 7 % floor is a mild bearish tilt; 10 % under is a full one) — no cliff at the edge.
        if pos == "BELOW":
            bias = -0.15 - 0.85 * min(1.0, (0.07 - rm) / 0.10)
        elif pos == "ABOVE":
            bias = 0.15 + 0.85 * min(1.0, (rm - 0.12) / 0.10)
        elif pos == "WITHIN":
            bias = (rm - 0.095) / 0.025 * 0.6
        else:
            bias = 0.0
        bias = float(np.clip(bias + 0.25 * np.sign(target_beta) * min(1.0, abs(target_beta) / 0.425), -1.0, 1.0))
        regime = self._infer_regime(vix, rm)
        return MarketContext(cfg.market_proxy, spot, rm, sigma_m, vix, target_beta, base_beta, pos, bias, regime)

    def _vix_level(self, sigma_m: float) -> float:
        for sym in ("$VIX", "$VIX.X", "VIX"):
            try:
                q = self.broker.get_quote(sym)
                if q and 5.0 < q < 150.0:
                    return float(q)
            except Exception:
                pass
        return float(sigma_m * 100.0 * 1.10)   # implied ≈ realised + typical VRP

    @staticmethod
    def _infer_regime(vix: float, rm: float) -> str:
        if vix >= 35:
            return "CRASH"
        if vix >= 27:
            return "STRESS"
        if vix >= 21 or rm < 0:
            return "CAUTIOUS"
        if vix <= 14 and rm > 0.12:
            return "RISK_ON"
        return "NORMAL"

    # ------------------------------------------------------------------
    # 2. RSI / momentum read
    # ------------------------------------------------------------------
    def cta_read(self, close: np.ndarray, high: Optional[np.ndarray] = None, low: Optional[np.ndarray] = None) -> CTARead:
        """WonderTrader CTA core on daily bars (see wondertrader_engine: _DualMovingAverage,
        _ChannelBreakout, _MomentumStrategy, _REGIME_WEIGHTS, _detect_regime, _dynamic_stops)."""
        try:
            from engine.execution import wondertrader_engine as wt
        except Exception:
            return CTARead()
        close = np.asarray(close, dtype=float)
        if close.size < 35:
            return CTARead()
        high = np.asarray(high, dtype=float) if high is not None and len(high) == close.size else close
        low = np.asarray(low, dtype=float) if low is not None and len(low) == close.size else close
        regime = wt._detect_regime(close)
        w = wt.WonderTraderEngine._REGIME_WEIGHTS.get(regime, wt.WonderTraderEngine._REGIME_WEIGHTS["RANGE"])
        d_ma, s_ma = wt._DualMovingAverage(10, 30).generate(close)
        d_bo, s_bo = wt._ChannelBreakout(20).generate(high, low, close)
        d_mo, s_mo = wt._MomentumStrategy(12, min(60, max(20, close.size - 13))).generate(close)
        parts = {"dual_ma": w["dual_ma"] * d_ma * s_ma, "breakout": w["breakout"] * d_bo * s_bo, "momentum": w["momentum"] * d_mo * s_mo}
        score = float(sum(parts.values()))
        direction = 0 if abs(score) < 0.05 else (1 if score > 0 else -1)
        dominant = max(parts, key=lambda k: abs(parts[k])) if direction else ""
        sl, tp = (wt._dynamic_stops(close, direction) if direction else (0.0, 0.0))
        return CTARead(regime, direction, min(abs(score), 1.0), float(np.clip(score, -1, 1)), dominant,
                       parts["dual_ma"], parts["breakout"], parts["momentum"], float(sl), float(tp))

    def momentum_read(self, close: np.ndarray, high: Optional[np.ndarray] = None, low: Optional[np.ndarray] = None) -> MomentumRead:
        cfg = self.cfg
        close = np.asarray(close, dtype=float)
        notes: List[str] = []
        if close.size < cfg.rsi_period + 5 or MomentumSignalEngine is None:
            return MomentumRead(50, 50, 0, "", 0, "", 0, 0, 0, 0, 0, 0.0, ["insufficient history"])
        me = MomentumSignalEngine(close)
        rsi_arr = me.rsi(cfg.rsi_period)
        rsi = float(np.nan_to_num(rsi_arr[-1], nan=50.0))
        rsi_prev = float(np.nan_to_num(rsi_arr[-4], nan=rsi)) if rsi_arr.size >= 4 else rsi
        slope = rsi - rsi_prev
        mtf = me.multi_timeframe_momentum()
        accel = me.momentum_acceleration(period=10) if close.size > 21 else 0.0
        div = me.rsi_divergence(cfg.rsi_period, lookback=min(30, close.size - cfg.rsi_period - 1)) or {}
        bo = {}
        if ChartPatternDetector is not None:
            try:
                bo = ChartPatternDetector(close).detect_breakout(lookback=cfg.breakout_lookback) or {}
            except Exception:
                bo = {}
        score = 0.0
        # RSI breakout: crossing up through 60 (or down through 40) with slope
        if rsi >= cfg.rsi_breakout_level and rsi_prev < cfg.rsi_breakout_level:
            score += 0.35; notes.append(f"RSI breakout {rsi_prev:.0f}→{rsi:.0f}")
        elif rsi <= cfg.rsi_breakdown_level and rsi_prev > cfg.rsi_breakdown_level:
            score -= 0.35; notes.append(f"RSI breakdown {rsi_prev:.0f}→{rsi:.0f}")
        elif rsi > cfg.rsi_overbought and slope < 0:
            score -= 0.20; notes.append(f"RSI overbought {rsi:.0f} rolling over (fade)")
        elif rsi < cfg.rsi_oversold and slope > 0:
            score += 0.20; notes.append(f"RSI oversold {rsi:.0f} turning (bounce)")
        elif rsi >= cfg.rsi_breakout_level and slope >= -2:
            score += 0.25; notes.append(f"RSI {rsi:.0f} breakout regime (held > {cfg.rsi_breakout_level:.0f})")
        elif rsi <= cfg.rsi_breakdown_level and slope <= 2:
            score -= 0.25; notes.append(f"RSI {rsi:.0f} breakdown regime (held < {cfg.rsi_breakdown_level:.0f})")
        else:
            score += float(np.clip(slope / 40.0, -0.15, 0.15))
        # price breakout / breakdown
        if bo.get("pattern") == "breakout":
            score += 0.30 * float(bo.get("confidence", 0.5)); notes.append(f"20-bar price breakout > {bo.get('level')}")
        elif bo.get("pattern") == "breakdown":
            score -= 0.30 * float(bo.get("confidence", 0.5)); notes.append(f"20-bar price breakdown < {bo.get('level')}")
        # divergence
        if div.get("divergence") == "bullish":
            score += 0.20 * float(div.get("confidence", 0.5)); notes.append("bullish RSI divergence")
        elif div.get("divergence") == "bearish":
            score -= 0.20 * float(div.get("confidence", 0.5)); notes.append("bearish RSI divergence")
        # short-horizon momentum (matches 1-7 DTE)
        m5, m10, m21 = mtf.get(5, 0.0), mtf.get(10, 0.0), mtf.get(21, 0.0)
        score += float(np.clip(m5 / 3.0, -0.25, 0.25)) + float(np.clip(m10 / 5.0, -0.15, 0.15))
        score += float(np.clip(accel / 5.0, -0.10, 0.10))
        tech_score = float(np.clip(score, -1.0, 1.0))
        # WonderTrader CTA core (regime-weighted dual-MA ⊕ Donchian ⊕ ROC-z): 40 % of the read
        cta = self.cta_read(close, high, low)
        if cta.direction:
            notes.append(f"CTA {cta.regime} {'LONG' if cta.direction > 0 else 'SHORT'} {cta.strength:.2f} via {cta.dominant} "
                         f"(SL {cta.stop_loss:.2f} / TP {cta.take_profit:.2f})")
        score = float(np.clip(0.60 * tech_score + 0.40 * cta.consensus, -1.0, 1.0))
        rsi_note = any(n.startswith("RSI break") or "breakout regime" in n or "breakdown regime" in n for n in notes)
        pattern_agrees = (bo.get("pattern") == ("breakout" if score > 0 else "breakdown") and float(bo.get("confidence", 0)) >= 0.5)
        mom_agrees = (m10 > 1.0 and score > 0) or (m10 < -1.0 and score < 0)
        cta_agrees = cta.direction != 0 and (cta.direction > 0) == (score > 0) and cta.strength >= 0.15
        # confirmed = an RSI regime or CTA trend, corroborated by an independent second read
        votes = sum([rsi_note, pattern_agrees, mom_agrees, cta_agrees])
        confirmed = bool(votes >= 2 and (rsi_note or cta_agrees))
        if confirmed:
            notes.append("momentum pattern CONFIRMED (" + " + ".join(n for n, f in (("RSI regime", rsi_note), ("price pattern", pattern_agrees), ("10d momentum", mom_agrees), ("CTA trend core", cta_agrees)) if f) + ")")
        return MomentumRead(rsi, rsi_prev, slope, bo.get("pattern", ""), float(bo.get("confidence", 0.0)),
                            div.get("divergence", ""), float(div.get("confidence", 0.0)),
                            m5, m10, m21, accel, score, notes, confirmed, cta)

    # ------------------------------------------------------------------
    # 4. Monte Carlo over exactly `dte` days (full-scan model, MC Greeks)
    # ------------------------------------------------------------------
    def monte_carlo(self, returns: np.ndarray, spot: float, dte: int) -> MCRead:
        cfg = self.cfg
        horizon = max(1, int(dte))
        returns = np.asarray(returns, dtype=float)
        if returns.size < 20 or self._mc is None:
            return MCRead(dte, 0.0, 0.0, 0.0, 0.5, False, "insufficient history for MC")
        mu, phi, sigma = self._mc._fit_ar1(returns)
        rng = np.random.RandomState(cfg.mc_seed + horizon)
        paths = self._mc._generate_paths(mu, phi, sigma, float(returns[-1]), cfg.mc_paths, horizon, rng)
        cum = np.sum(paths, axis=1)
        stats = self._mc._compute_statistics(paths)
        # The bridge's confidence (path agreement + |mean return|) is calibrated on the 21-day full
        # scan; over 1-7 days drift/σ shrinks ∝ √h, so normalise back to the calibration horizon.
        conf = float(min(1.0, self._mc._compute_confidence(stats) * math.sqrt(cfg.mc_calibration_days / horizon)))
        terminal = spot * np.exp(cum)
        var_pct = float(np.exp(stats["var_95"]) - 1.0)
        passed, reason = True, "ok"
        if conf < cfg.mc_min_confidence:
            passed, reason = False, f"MC confidence {conf:.2f} < {cfg.mc_min_confidence}"
        elif var_pct < cfg.mc_var95_floor:
            passed, reason = False, f"MC VaR95 {var_pct:.1%} < {cfg.mc_var95_floor:.0%}"
        return MCRead(dte, float(conf), float(stats["mean_return"]), var_pct, float(stats["paths_positive_pct"]), passed, reason, terminal)

    def mc_option_value(self, terminal: np.ndarray, strike: float, is_call: bool, spot: float, dte: int) -> Dict[str, float]:
        """Discounted MC value + MC Greeks (bump-and-revalue on common paths)."""
        T = max(dte, 0.5) / 365.0
        disc = math.exp(-self.cfg.risk_free * T)

        def val(s0: float) -> float:
            st = terminal * (s0 / spot)
            pay = np.maximum(st - strike, 0.0) if is_call else np.maximum(strike - st, 0.0)
            return float(disc * np.mean(pay))

        h = spot * 0.005
        v0, vu, vd = val(spot), val(spot + h), val(spot - h)
        p_itm = float(np.mean(terminal > strike)) if is_call else float(np.mean(terminal < strike))
        return {"value": v0, "delta": (vu - vd) / (2 * h), "gamma": (vu - 2 * v0 + vd) / (h * h), "p_itm": p_itm}

    # ------------------------------------------------------------------
    # 5-8. contract evaluation (BSM only at the chain DTE)
    # ------------------------------------------------------------------
    def evaluate_contract(self, q, spot: float, surface: ShortTenorVolSurface, mc: MCRead,
                          alpha: float, predictive_adj: float) -> ContractEval:
        cfg = self.cfg
        is_call = q.put_call == "CALL"
        T = max(q.dte, 0.5) / 365.0
        mid = q.mid
        bsm_iv = self.bs.implied_vol(mid, spot, q.strike, T, cfg.risk_free, is_call=is_call)
        if not np.isfinite(bsm_iv) or bsm_iv <= 0:
            bsm_iv = q.iv
        surface_iv = surface.interpolate_vol(q.dte, q.moneyness)
        fair = (self.bs.call_price if is_call else self.bs.put_price)(spot, q.strike, T, cfg.risk_free, surface_iv)
        fair = max(fair, 0.01)
        edge_bps = (fair - mid) / mid * 10_000 if mid > 0 else 0.0
        mispricing = (surface_iv - bsm_iv) / surface_iv if surface_iv > 0 else 0.0
        greeks = {
            "delta": self.bs.delta(spot, q.strike, T, cfg.risk_free, bsm_iv, is_call),
            "gamma": self.bs.gamma(spot, q.strike, T, cfg.risk_free, bsm_iv),
            "theta": self.bs.theta(spot, q.strike, T, cfg.risk_free, bsm_iv, is_call),
            "vega": self.bs.vega(spot, q.strike, T, cfg.risk_free, bsm_iv),
            "rho": self.bs.rho(spot, q.strike, T, cfg.risk_free, bsm_iv, is_call),
            "iv": bsm_iv,
            "schwab_delta": q.delta, "schwab_gamma": q.gamma, "schwab_theta": q.theta, "schwab_vega": q.vega,
        }
        mcg = self.mc_option_value(mc.terminal_prices, q.strike, is_call, spot, q.dte) if mc.terminal_prices is not None else {"value": fair, "delta": greeks["delta"], "gamma": greeks["gamma"], "p_itm": 0.5}
        d = abs(greeks["delta"])
        delta_quality = float(np.clip(1.0 - abs(d - cfg.delta_target) / cfg.delta_target, 0.0, 1.0))
        edge_score = float(np.clip(edge_bps / (2 * cfg.min_edge_bps), 0.0, 1.0)) if edge_bps > 0 else 0.0
        mis_score = float(np.clip(0.5 + mispricing * 2.0, 0.0, 1.0))
        # MC agreement: MC value vs market — cheap under our own paths adds to edge
        mc_edge = float(np.clip((mcg["value"] - mid) / mid, -0.5, 0.5)) if mid > 0 else 0.0
        edge_mix = 0.5 * edge_score + 0.5 * float(np.clip(0.5 + mc_edge, 0, 1))
        # composite = alpha × MC.conf × delta_quality × edge × mispricing, normalised as the
        # geometric mean of the five [0,1] factors (so each factor keeps equal weight and the
        # DecisionMatrix 0.55 bar is demanding but attainable), then scaled by the
        # PredictiveOptionsSignal adjustment (VRP / skew / vol-MR / term / PCR).
        product = alpha * mc.confidence * delta_quality * edge_mix * mis_score
        composite = float(np.clip((product ** 0.2) * predictive_adj, 0.0, 1.0)) if product > 0 else 0.0
        reject = ""
        if q.spread_pct > cfg.max_spread_pct:
            reject = f"spread {q.spread_pct:.1%} > {cfg.max_spread_pct:.0%}"
        elif q.open_interest < cfg.min_open_interest:
            reject = f"OI {q.open_interest} < {cfg.min_open_interest}"
        elif not (cfg.delta_lo <= d <= cfg.delta_hi):
            reject = f"|delta| {d:.2f} outside [{cfg.delta_lo},{cfg.delta_hi}]"
        elif edge_bps < cfg.min_edge_bps:
            reject = f"edge {edge_bps:.0f}bps < {cfg.min_edge_bps:.0f}"
        elif composite < cfg.min_composite:
            reject = f"composite {composite:.2f} < {cfg.min_composite}"
        return ContractEval(q.symbol, q.underlying, q.put_call, q.strike, q.expiry, q.dte, q.bid, q.ask, mid,
                            q.iv, float(bsm_iv), float(surface_iv), float(fair), float(edge_bps), float(mispricing),
                            {k: float(v) for k, v in greeks.items()}, {k: float(v) for k, v in mcg.items() if k != "p_itm"},
                            float(mcg["value"]), float(mcg["p_itm"]), delta_quality, composite, alpha, mc.confidence,
                            predictive_adj, reject)

    # ------------------------------------------------------------------
    # 7. predictive options signals → adjustment factor + direction
    # ------------------------------------------------------------------
    @staticmethod
    def predictive_adjustment(signals: List[Any], direction: str) -> Tuple[float, List[str]]:
        adj, notes = 1.0, []
        for s in signals:
            if s.direction == direction:
                adj *= 1.0 + 0.25 * s.strength
            elif s.direction not in ("NEUTRAL", direction):
                adj *= 1.0 - 0.35 * s.strength
            notes.append(f"{s.name}:{s.direction}:{s.strength:.2f}")
        return float(np.clip(adj, 0.4, 1.4)), notes

    # ------------------------------------------------------------------
    # 9. sizing with bucket + vega caps
    # ------------------------------------------------------------------
    def _bucket_budget(self, bucket: str, committed: Dict[str, float], eligible_buckets: Optional[set] = None) -> float:
        """Bucket cap (10/10/5 of NAV) with SPILL-OVER: room in buckets that have NO eligible
        candidate this cycle flows to the buckets that do, never beyond the 25 % total cap."""
        cap = self.cfg.bucket_caps.get(bucket, 0.05) * self.nav
        if eligible_buckets is not None:
            idle = [b for b in self.cfg.bucket_caps if b not in eligible_buckets and b != bucket]
            live = [b for b in self.cfg.bucket_caps if b in eligible_buckets] or [bucket]
            spill = sum(self.cfg.bucket_caps[b] for b in idle) * self.nav / max(1, len(live))
            cap += spill
        used = committed.get(bucket, 0.0) + self.existing_option_notional.get(bucket, 0.0)
        total_used = sum(committed.values()) + sum(self.existing_option_notional.values())
        total_room = self.cfg.total_options_cap * self.nav - total_used
        return max(0.0, min(cap - used, total_room))

    def size(self, ev: ContractEval, spot: float, budget: float, vega_used: float) -> Dict[str, Any]:
        # fair-value vol = chain-fitted surface IV at (dte, moneyness); market IV would make BS == market
        res = self.sizer.size_option(spot=spot, strike=ev.strike, expiry_days=max(ev.dte, 1), vol=ev.surface_iv,
                                     is_call=ev.put_call == "CALL", nav=self.nav, budget_dollars=budget,
                                     risk_free=self.cfg.risk_free, market_price=ev.mid)
        if res.get("rejected"):
            return res
        contracts = int(res["contracts"])
        vega_budget = self.cfg.vega_budget_pct_nav * self.nav
        per_contract_vega = abs(ev.bsm_greeks["vega"]) * 100.0
        room = max(0.0, vega_budget - vega_used)
        if per_contract_vega > 0 and contracts * per_contract_vega > room:
            contracts = int(room // per_contract_vega)
            res["vega_clipped"] = True
        res["contracts"] = contracts
        res["position_vega"] = contracts * per_contract_vega
        if contracts < self.sizer.MIN_CONTRACTS:
            res["rejected"] = True
            res["reject_reason"] = f"vega budget / bucket room allow only {contracts} contracts (< {self.sizer.MIN_CONTRACTS})"
        return res

    # ------------------------------------------------------------------
    # 11. regime structure
    # ------------------------------------------------------------------
    def regime_structure(self, spot: float, surface: ShortTenorVolSurface, dte: int, direction: str) -> Dict[str, Any]:
        builder = OptionsStrategyBuilder(spot, surface, self.cfg.risk_free)
        out: Dict[str, Any] = {"structure": "SINGLE"}
        try:
            profiles = builder.select_for_regime(self.regime)
            out["regime_strategies"] = [p.name for p in profiles]
            tg = builder.theta_gamma_optimize(expiry_days=max(dte, 1), vega_cost=0.5)
            out["theta_gamma"] = tg.get("metrics", {})
        except Exception as e:
            out["error"] = str(e)
        atm_iv = surface.get_atm_vol("1M")
        # rich short-dated vol → prefer a vertical (sell the wing) to cut vega/theta bleed
        if atm_iv > 1.25 * max(surface.hv30, 1e-6) or self.regime in ("STRESS", "CRASH"):
            out["structure"] = "VERTICAL"
            out["structure_reason"] = f"ATM IV {atm_iv:.1%} rich vs HV {surface.hv30:.1%} / regime {self.regime}"
        return out

    def _evaluate(self, ticker: str, bucket: str, ctx: "MarketContext", per_ticker: Dict[str, Any]):
        """Pass 1 for one underlying: history → momentum/CTA/corridor direction → 1-7 DTE chain →
        BSM on that DTE → MC at that horizon → best contract + structure. Returns a pending tuple
        (to be sized in pass 2) or None when the name is rejected (reason recorded in per_ticker)."""
        cfg = self.cfg
        rec: Dict[str, Any] = {"ticker": ticker, "bucket": bucket, "status": "", "reasons": []}
        per_ticker[ticker] = rec
        hist = self.broker.get_price_history(ticker, days=cfg.history_days)
        close = np.asarray(hist.get("close", []), dtype=float)
        if close.size < 30:
            rec["status"] = "SKIP"; rec["reasons"].append("insufficient price history"); return None
        spot = float(self.broker.get_quote(ticker) or close[-1])
        rets = np.diff(np.log(close))
        hv30 = float(np.std(rets[-21:]) * math.sqrt(TRADING_DAYS))
        hv90 = float(np.std(rets[-63:]) * math.sqrt(TRADING_DAYS))
        mom = self.momentum_read(close, hist.get("high"), hist.get("low"))
        rec["momentum"] = mom.to_dict()
        # direction = momentum score blended with the beta-corridor fair value tilt
        w_m = 0.80 if mom.confirmed else 0.65      # confirmed RSI/momentum pattern → corridor is a tilt, not a veto
        dir_score = float(np.clip(w_m * mom.direction_score + (1 - w_m) * ctx.direction_bias, -1, 1))
        rec["direction_score"] = dir_score; rec["momentum_confirmed"] = mom.confirmed
        if abs(dir_score) < 0.15:
            rec["status"] = "NO_TRADE"; rec["reasons"].append(f"direction score {dir_score:+.2f} too weak (|s|<0.15)"); return None
        direction = "BULLISH" if dir_score > 0 else "BEARISH"
        want = "CALL" if direction == "BULLISH" else "PUT"
        alpha = abs(dir_score)

        chain = self.broker.get_option_chain(ticker, cfg.dte_min, cfg.dte_max, strike_count=cfg.strike_count)
        if not chain and getattr(self.broker, "last_chain_error", ""):
            # fetch failed (rate limit / transient) — this is NOT "no chain"; pause and retry once
            time.sleep(cfg.chain_retry_wait_s)
            chain = self.broker.get_option_chain(ticker, cfg.dte_min, cfg.dte_max, strike_count=cfg.strike_count)
            if not chain and getattr(self.broker, "last_chain_error", ""):
                rec["status"] = "ERROR"; rec["reasons"].append(f"chain fetch failed twice: {self.broker.last_chain_error[:80]}"); return None
        if not chain:
            rec["status"] = "SKIP"; rec["reasons"].append("no 1-7 DTE chain"); return None
        surface = ShortTenorVolSurface(chain, vix=ctx.vix, hist_vol_30d=hv30, hist_vol_90d=hv90)
        pcr = self._put_call_ratio(chain)
        signals = PredictiveOptionsSignal(surface).all_signals(pcr=pcr)
        pred_adj, pred_notes = self.predictive_adjustment(signals, direction)
        rec["predictive"] = {"adj": pred_adj, "signals": pred_notes, "pcr": pcr}

        # Monte Carlo per DTE present in the chain — full-scan model at exactly that horizon
        mc_by_dte: Dict[int, MCRead] = {}
        for dte in sorted({q.dte for q in chain}):
            mc_by_dte[dte] = self.monte_carlo(rets, spot, dte)
        rec["monte_carlo"] = {d: m.to_dict() for d, m in mc_by_dte.items()}
        passing_dtes = [d for d, m in mc_by_dte.items() if m.passed]
        if not passing_dtes:
            rec["status"] = "MC_REJECT"; rec["reasons"].extend(m.reason for m in mc_by_dte.values()); return None
        # MC directional agreement — do not buy calls into a distribution skewed down
        mc_dir_ok = all((mc_by_dte[d].paths_positive_pct >= 0.5) == (direction == "BULLISH") for d in passing_dtes)
        if not mc_dir_ok:
            rec["reasons"].append("MC path skew disagrees with momentum direction → alpha haircut 40%")
            alpha *= 0.6

        evals: List[ContractEval] = []
        for q in chain:
            if q.put_call != want or q.dte not in mc_by_dte or not mc_by_dte[q.dte].passed:
                continue
            evals.append(self.evaluate_contract(q, spot, surface, mc_by_dte[q.dte], alpha, pred_adj))
        evals.sort(key=lambda e: e.composite, reverse=True)
        rec["top_contracts"] = [e.to_dict() for e in evals[:5]]
        ok = [e for e in evals if not e.reject_reason]
        if not ok:
            rec["status"] = "NO_EDGE"
            rec["reasons"].extend(sorted({e.reject_reason for e in evals})[:4] or ["no contracts in delta band"])
            return None
        best = ok[0]
        structure = self.regime_structure(spot, surface, best.dte, direction)
        rec["structure"] = structure
        return (ticker, bucket, rec, best, structure, chain, spot, mom, ctx, mc_by_dte, pred_adj, direction)

    # ------------------------------------------------------------------
    # main scan
    # ------------------------------------------------------------------
    def scan(self, universe: List[Tuple[str, str]], nav: Optional[float] = None) -> Dict[str, Any]:
        """universe = [(ticker, bucket), ...]; bucket ∈ OPTIONS_IG / OPTIONS_HY / OPTIONS_DISTRESSED."""
        cfg = self.cfg
        if nav:
            self.nav = float(nav)
            self.hedge_mgr.nav = self.nav
        ctx = self.market_context()
        self.regime = ctx.regime
        self.hedge_mgr.update_regime(ctx.regime)
        intents: List[OptionTradeIntent] = []
        per_ticker: Dict[str, Any] = {}
        committed: Dict[str, float] = {}
        vega_used = 0.0
        pending: List[tuple] = []

        for ticker, bucket in universe:
            t = self._evaluate(ticker, bucket, ctx, per_ticker)
            if t is not None:
                pending.append(t)

        # ---- bucket back-fill (operator rule): when OPTIONS_HY / OPTIONS_DISTRESSED have no
        # eligible name, fill the remainder from the S&P 400 / S&P 600 universe runs (even if the
        # names are not HY / distressed by classification) rather than spilling room into IG.
        eligible = {b for _, b, *_ in pending}
        fills: Dict[str, List[str]] = {}
        for bkt, source in (("OPTIONS_HY", "SP400"), ("OPTIONS_DISTRESSED", "SP600")):
            if bkt in eligible or not self.backfill_candidates:
                continue
            tried = 0
            for t2 in self.backfill_candidates.get(source, []):
                if t2 in per_ticker or tried >= self.cfg.backfill_max_tries:
                    continue
                tried += 1
                t = self._evaluate(t2, bkt, ctx, per_ticker)
                per_ticker[t2]["backfill"] = f"{bkt} ← {source}"
                if t is not None:
                    pending.append(t); fills.setdefault(bkt, []).append(t2); eligible.add(bkt)
                    if len(fills[bkt]) >= self.cfg.backfill_names_per_bucket:
                        break
        self.last_backfill = fills


        # ---- pass 2: size (bucket caps 10/10/5 honoured; empty buckets were back-filled above)
        pending.sort(key=lambda t: -t[3].composite)          # strongest composite gets budget first
        for ticker, bucket, rec, best, structure, chain, spot, mom, ctx, mc_by_dte, pred_adj, direction in pending:
            budget = self._bucket_budget(bucket, committed)
            if budget <= 0:
                rec["status"] = "BUCKET_FULL"; rec["reasons"].append(f"{bucket} cap reached"); continue
            sizing = self.size(best, spot, budget, vega_used)
            rec["sizing"] = {k: v for k, v in sizing.items() if k not in ("paths",)}
            if sizing.get("rejected"):
                rec["status"] = "SIZE_REJECT"; rec["reasons"].append(sizing.get("reject_reason", "sizer rejected")); continue
            contracts = int(sizing["contracts"])
            legs = [{"symbol": best.symbol, "instruction": "BUY_TO_OPEN", "quantity": 1}]
            limit = round(min(best.ask, best.mid + 0.25 * (best.ask - best.bid)), 2)   # pay up to 75 % of spread
            if structure["structure"] == "VERTICAL":
                wing = self._select_wing(chain, best)
                if wing is not None:
                    legs.append({"symbol": wing.symbol, "instruction": "SELL_TO_OPEN", "quantity": 1})
                    limit = round(max(0.05, best.mid - wing.mid), 2)
                    structure["wing"] = {"symbol": wing.symbol, "strike": wing.strike, "mid": wing.mid}
                else:
                    structure["structure"] = "SINGLE"
            notional = contracts * limit * 100.0
            committed[bucket] = committed.get(bucket, 0.0) + notional
            vega_used += float(sizing.get("position_vega", 0.0))
            rationale = mom.notes + [
                f"beta corridor {ctx.corridor_position} (Rm {ctx.rm_annual:+.1%}, target β {ctx.target_beta:+.2f}) → bias {ctx.direction_bias:+.2f}",
                f"MC@{best.dte}d conf {best.mc_conf:.2f} VaR95 {mc_by_dte[best.dte].var_95:+.1%} P(up) {mc_by_dte[best.dte].paths_positive_pct:.0%}",
                f"BSM IV {best.bsm_iv:.1%} vs surface {best.surface_iv:.1%} → edge {best.edge_bps:+.0f}bps",
                f"composite {best.composite:.2f} ≥ {cfg.min_composite}; predictive adj {pred_adj:.2f}",
            ]
            intents.append(OptionTradeIntent(
                ticker=ticker, bucket=bucket, structure=structure["structure"], direction=direction,
                contract_symbol=best.symbol, put_call=best.put_call, strike=best.strike, expiry=best.expiry, dte=best.dte,
                contracts=contracts, limit_price=limit, notional=notional, composite=best.composite, edge_bps=best.edge_bps,
                greeks={**{k: v * contracts * 100 for k, v in best.bsm_greeks.items() if k in ("delta", "gamma", "theta", "vega")},
                        "mc_delta": best.mc_greeks["delta"] * contracts * 100, "mc_gamma": best.mc_greeks["gamma"] * contracts * 100,
                        "iv": best.bsm_iv, "delta_exposure_usd": best.bsm_greeks["delta"] * contracts * 100 * spot},
                legs=legs, sizing={k: v for k, v in sizing.items()}, rationale=rationale,
            ))
            rec["status"] = "INTENT"

        ladder = self.build_protective_ladder(ctx)
        self.last_run = {
            "as_of": date.today().isoformat(), "nav": self.nav, "market": ctx.to_dict(), "regime": self.regime,
            "universe": [t for t, _ in universe], "per_ticker": per_ticker,
            "intents": [i.to_dict() for i in intents], "ladder": ladder,
            "committed_by_bucket": committed, "vega_used": vega_used, "backfill": self.last_backfill,
            "caps": {"bucket": cfg.bucket_caps, "total": cfg.total_options_cap},
        }
        return {"intents": intents, "ladder": ladder, "context": ctx, "report": self.last_run}

    # ------------------------------------------------------------------
    # Extended tenor (8–30 DTE) — PROPOSAL ONLY, never auto-executed.
    # Same pipeline (momentum ⊕ corridor direction → chain → BSM at that DTE → MC at that
    # horizon → composite) but surfaced for the operator's explicit OK.
    # ------------------------------------------------------------------
    def extended_watch(self, universe: List[Tuple[str, str]], dte_min: int = 8, dte_max: int = 30,
                       max_proposals: int = 5, min_composite: Optional[float] = None) -> List[Dict[str, Any]]:
        cfg = self.cfg
        floor = float(min_composite if min_composite is not None else max(cfg.min_composite, 0.60))
        ctx = self.market_context()
        props: List[Dict[str, Any]] = []
        for ticker, bucket in universe:
            try:
                hist = self.broker.get_price_history(ticker, days=cfg.history_days)
                close = np.asarray(hist.get("close", []), dtype=float)
                if close.size < 30:
                    continue
                spot = float(self.broker.get_quote(ticker) or close[-1])
                rets = np.diff(np.log(close))
                hv30 = float(np.std(rets[-21:]) * math.sqrt(TRADING_DAYS)); hv90 = float(np.std(rets[-63:]) * math.sqrt(TRADING_DAYS))
                mom = self.momentum_read(close, hist.get("high"), hist.get("low"))
                w_m = 0.80 if mom.confirmed else 0.65
                dir_score = float(np.clip(w_m * mom.direction_score + (1 - w_m) * ctx.direction_bias, -1, 1))
                if abs(dir_score) < 0.15:
                    continue
                direction = "BULLISH" if dir_score > 0 else "BEARISH"; want = "CALL" if direction == "BULLISH" else "PUT"
                chain = self.broker.get_option_chain(ticker, dte_min, dte_max, strike_count=cfg.strike_count)
                if not chain:
                    continue
                surface = ShortTenorVolSurface(chain, vix=ctx.vix, hist_vol_30d=hv30, hist_vol_90d=hv90)
                pred_adj, pred_notes = self.predictive_adjustment(PredictiveOptionsSignal(surface).all_signals(pcr=self._put_call_ratio(chain)), direction)
                mc_by_dte = {d: self.monte_carlo(rets, spot, d) for d in sorted({q.dte for q in chain})}
                evals = [self.evaluate_contract(q, spot, surface, mc_by_dte[q.dte], abs(dir_score), pred_adj)
                         for q in chain if q.put_call == want and mc_by_dte[q.dte].passed]
                ok = sorted([e for e in evals if not e.reject_reason and e.composite >= floor], key=lambda e: -e.composite)
                if not ok:
                    continue
                best = ok[0]
                structure = self.regime_structure(spot, surface, best.dte, direction)
                wing = self._select_wing(chain, best) if structure.get("structure") == "VERTICAL" else None
                mcr = mc_by_dte[best.dte]
                props.append({
                    "ticker": ticker, "bucket": bucket, "direction": direction, "direction_score": round(dir_score, 2),
                    "structure": structure.get("structure", "SINGLE"), "put_call": best.put_call, "strike": best.strike,
                    "wing_strike": getattr(wing, "strike", None), "expiry": best.expiry, "dte": best.dte,
                    "mid": round(best.mid, 2), "ask": best.ask, "bsm_fair": round(best.fair_value, 2), "edge_bps": round(best.edge_bps),
                    "bsm_iv": round(best.bsm_iv, 4), "surface_iv": round(best.surface_iv, 4), "composite": round(best.composite, 2),
                    "mc_conf": round(mcr.confidence, 2), "mc_p_up": round(mcr.paths_positive_pct, 2), "mc_var95": round(mcr.var_95, 3),
                    "p_itm": round(best.mc_p_itm, 2), "delta": round(best.bsm_greeks["delta"], 2), "gamma": round(best.bsm_greeks["gamma"], 4),
                    "theta": round(best.bsm_greeks["theta"], 3), "vega": round(best.bsm_greeks["vega"], 3),
                    "why": mom.notes[:3] + pred_notes[:2], "status": "PROPOSAL_ONLY — requires operator OK",
                })
            except Exception as e:  # noqa: BLE001
                logger.debug("extended_watch %s: %s", ticker, e)
        props.sort(key=lambda p: -p["composite"])
        self.last_extended = props[:max_proposals]
        return self.last_extended

    # ------------------------------------------------------------------
    def _select_wing(self, chain: List[Any], best: ContractEval):
        same = [q for q in chain if q.put_call == best.put_call and q.dte == best.dte and q.symbol != best.symbol and q.bid > 0]
        if best.put_call == "CALL":
            cands = sorted([q for q in same if q.strike > best.strike], key=lambda q: q.strike)
        else:
            cands = sorted([q for q in same if q.strike < best.strike], key=lambda q: -q.strike)
        for q in cands:
            if abs(q.strike - best.strike) / best.strike >= 0.01 and q.mid < best.mid:
                return q
        return None

    @staticmethod
    def _put_call_ratio(chain: List[Any]) -> float:
        puts = sum(q.open_interest for q in chain if q.put_call == "PUT")
        calls = sum(q.open_interest for q in chain if q.put_call == "CALL")
        return float(puts / calls) if calls > 0 else 0.85

    # ------------------------------------------------------------------
    # 10. convexity ladder on real SPY contracts (dte_max)
    # ------------------------------------------------------------------
    def build_protective_ladder(self, ctx: MarketContext) -> Dict[str, Any]:
        cfg = self.cfg
        chain = self.broker.get_option_chain(cfg.hedge_underlying, cfg.dte_min, cfg.dte_max, strike_count=80, contract_type="PUT")
        puts = [q for q in chain if q.put_call == "PUT"]
        if not puts:
            return {"rungs": [], "reason": "no SPY puts in window"}
        far_dte = max(q.dte for q in puts)
        puts = [q for q in puts if q.dte == far_dte]
        spot = puts[0].underlying_price or ctx.spot
        atm_iv = float(np.median([q.iv for q in puts if abs(q.moneyness - 1) < 0.01]) or ctx.vix / 100)
        theo = self.hedge_mgr.build_put_ladder(spot=spot, vol=atm_iv, risk_free=cfg.risk_free, expiry_days=far_dte)
        # the ConvexityHedgeManager budget is monthly; a 1-7 DTE ladder is rolled weekly,
        # so each cycle may spend  annual_budget × dte/365  (regime-scaled) across the rungs.
        cycle_budget = self.hedge_mgr.hedge_budget() * far_dte / 365.0
        rungs, seen = [], set()
        for hp, otm, w in zip(theo, self.hedge_mgr.PUT_LADDER, self.hedge_mgr.LADDER_WEIGHTS):
            target_k = spot * (1 - otm)
            q = min(puts, key=lambda x: abs(x.strike - target_k))
            if q.symbol in seen:
                continue
            seen.add(q.symbol)
            rung_budget = cycle_budget * w
            cost_1 = max(q.ask, 0.01) * 100.0
            qty = int(rung_budget // cost_1)
            rungs.append({"contract_symbol": q.symbol, "strike": q.strike, "expiry": q.expiry, "dte": q.dte,
                          "otm_pct": otm, "contracts": qty, "limit_price": round(q.ask, 2),
                          "theo_premium": round(hp.entry_premium, 3), "market_mid": round(q.mid, 3),
                          "rung_budget": round(rung_budget, 2), "cost_per_contract": round(cost_1, 2),
                          "affordable": qty >= 1, "delta": q.delta, "instruction": "BUY_TO_OPEN"})
        return {"underlying": cfg.hedge_underlying, "dte": far_dte, "spot": spot, "atm_iv": atm_iv,
                "annual_budget": self.hedge_mgr.hedge_budget(), "cycle_budget": cycle_budget,
                "regime": self.hedge_mgr.regime, "rungs": rungs,
                "placeable": [r for r in rungs if r["affordable"]]}
