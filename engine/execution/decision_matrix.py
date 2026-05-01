"""DecisionMatrix — Multi-gate trade approval engine.

Sits between AlphaOptimizer and ExecutionEngine in the signal pipeline:
    UniverseEngine -> MacroEngine -> MetadronCube -> AlphaOptimizer
        -> **DecisionMatrix** -> ExecutionEngine

Target 95%+ alpha. Four approval gates (binary pass/fail at composite ≥ 0.55).
All gates are cross-asset and asset-agnostic — applied uniformly across the
merged AlphaOptimizer output (equities, options, futures, ETFs):
    1. FUNDAMENTALS     (40%)  Quality/ROIC/FCF + Graham-Dodd grade + earnings + credit
    2. FLOW_HEADLINES   (20%)  ETF flow + news sentiment + sector rotation signals
    3. MACRO_REGIME     (20%)  MetadronCube regime alignment + yield curve + risk state
    4. MOMENTUM         (20%)  RSI, MACD, breakout, mean-reversion, cross-asset momentum

AlphaBetaUnleashed (Dataset 1) — 1-minute cadence beta management:
    Rm_adjusted = Rm_realized + macro.rm_adjustment
    target_beta = corridor_fn(Rm_adjusted) * 4.7 * vol_adj
    MES_hedge_beta = target_beta - sleeve_beta

KellySizer — aggressive Kelly criterion position sizing.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from ..portfolio.beta_corridor import (
    BetaCorridor, BetaState, BetaAction,
    EXECUTION_MULTIPLIER, BETA_MAX, BETA_INV,
    R_LOW, R_HIGH, ALPHA as CORRIDOR_ALPHA, VOL_STANDARD,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DAILY_TARGET = 0.046                  # ~4.6% daily compound
RISK_FREE_RATE = 0.04
TRADING_DAYS = 252
MAX_PORTFOLIO_VAR_PCT = 0.015         # 1.5% daily VaR limit
MAX_SINGLE_POSITION_PCT = 0.20        # 20% max single position
MAX_SECTOR_PCT = 0.35                 # 35% max sector
MAX_LEVERAGE = 3.0                    # Regime-dependent cap
CORRELATION_LIMIT = 0.80              # Reject if corr > 0.80 with existing
MIN_COMPOSITE_SCORE = 0.55            # Minimum composite to approve (raised from 0.45 — more gate scrutiny)
ADV_MIN_RATIO = 0.01                  # Trade size < 1% of ADV
SPREAD_MAX_BPS = 30.0                 # Max acceptable spread

# Gate definitions
GATE_CONFIGS = {
    "FUNDAMENTALS":        {"weight": 0.40, "threshold": 0.45},
    "FLOW_HEADLINES":      {"weight": 0.20, "threshold": 0.35},
    "MACRO_REGIME":        {"weight": 0.20, "threshold": 0.40},
    "MOMENTUM":            {"weight": 0.20, "threshold": 0.35},
}

# MC risk bounds
MC_VAR95_MAX = 0.05           # 5% max VaR95 for portfolio
MC_PROB_PROFIT_MIN = 0.40     # Must have >= 40% profit probability

# Quality tier -> score mapping (from AlphaOptimizer)
QUALITY_TIER_SCORES = {
    "A": 1.00, "B": 0.85, "C": 0.70, "D": 0.55,
    "E": 0.35, "F": 0.20, "G": 0.05,
}

# Regime -> expected alignment modifier
REGIME_ALIGNMENT_MAP = {
    "TRENDING": {"long": 1.0, "short": 0.3},
    "RANGE":    {"long": 0.6, "short": 0.6},
    "STRESS":   {"long": 0.3, "short": 0.8},
    "CRASH":    {"long": 0.1, "short": 1.0},
}


# ---------------------------------------------------------------------------
# DecisionGate
# ---------------------------------------------------------------------------
@dataclass
class DecisionGate:
    """A single gate in the decision matrix."""
    gate_name: str
    score: float = 0.0         # 0-1 normalised score
    weight: float = 0.0        # Gate weight in composite
    pass_threshold: float = 0.5
    passed: bool = False
    details: str = ""

    def evaluate(self) -> bool:
        """Mark pass/fail based on score vs threshold."""
        self.passed = self.score >= self.pass_threshold
        return self.passed

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"Gate({self.gate_name}: {self.score:.3f} "
                f"[thr={self.pass_threshold:.2f}] {status})")


# ---------------------------------------------------------------------------
# KellySizer
# ---------------------------------------------------------------------------
class KellySizer:
    """Kelly criterion position sizing with half-Kelly multiplier.

    Standard Kelly: f* = (p * b - q) / b
        p = win probability
        b = win/loss ratio
        q = 1 - p

    Capped at MAX_SINGLE_POSITION_PCT of NAV.
    """

    def __init__(self, max_position_pct: float = MAX_SINGLE_POSITION_PCT,
                 default_multiplier: float = 1.5):
        self.max_position_pct = max_position_pct
        self.default_multiplier = default_multiplier
        self._sizing_history: List[dict] = []

    @staticmethod
    def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
        """Compute standard Kelly fraction.

        Parameters
        ----------
        win_prob : float
            Probability of winning (0-1).
        win_loss_ratio : float
            Average win / average loss (positive).

        Returns
        -------
        float
            Kelly fraction (can be negative => don't trade).
        """
        if win_loss_ratio <= 0:
            return 0.0
        q = 1.0 - win_prob
        f = (win_prob * win_loss_ratio - q) / win_loss_ratio
        return max(f, 0.0)

    def aggressive_kelly(self, kelly_f: float,
                         multiplier: float = None) -> float:
        """Scale Kelly fraction aggressively for 95%+ alpha pursuit.

        Parameters
        ----------
        kelly_f : float
            Standard Kelly fraction.
        multiplier : float, optional
            Scaling factor (default 1.5x).

        Returns
        -------
        float
            Aggressive Kelly fraction, capped at max_position_pct.
        """
        if multiplier is None:
            multiplier = self.default_multiplier
        aggressive = kelly_f * multiplier
        return min(aggressive, self.max_position_pct)

    def compute_size(self, nav: float, conviction: float,
                     volatility: float, price: float = 1.0) -> dict:
        """Compute position size in shares and dollar amount.

        Parameters
        ----------
        nav : float
            Current net asset value.
        conviction : float
            Conviction score (0-1) used as win_prob proxy.
        volatility : float
            Annualised volatility of the instrument.
        price : float
            Current price per share.

        Returns
        -------
        dict
            keys: kelly_f, aggressive_f, dollar_amount, shares,
                  pct_of_nav, vol_scaled
        """
        # Win probability from conviction; win/loss from inverse vol
        win_prob = np.clip(conviction, 0.01, 0.99)
        vol_daily = volatility / np.sqrt(TRADING_DAYS) if volatility > 0 else 0.01
        # Win/loss ratio: higher for lower vol instruments
        win_loss_ratio = max(1.0 / (vol_daily * 10.0), 0.5)

        kelly_f = self.kelly_fraction(win_prob, win_loss_ratio)
        agg_f = self.aggressive_kelly(kelly_f)

        # Vol-scale: reduce size when vol is high
        vol_adj = VOL_STANDARD / max(volatility, 0.01)
        vol_adj = np.clip(vol_adj, 0.3, 2.0)
        final_f = agg_f * vol_adj

        # Cap at max position
        final_f = min(final_f, self.max_position_pct)

        dollar_amount = nav * final_f
        shares = int(dollar_amount / price) if price > 0 else 0
        dollar_amount = shares * price  # round to whole shares

        result = {
            "kelly_f": round(kelly_f, 6),
            "aggressive_f": round(agg_f, 6),
            "vol_adjustment": round(vol_adj, 4),
            "final_fraction": round(final_f, 6),
            "dollar_amount": round(dollar_amount, 2),
            "shares": shares,
            "pct_of_nav": round(final_f * 100, 2),
            "price": price,
        }
        self._sizing_history.append({
            "timestamp": datetime.now().isoformat(),
            **result,
        })
        return result

    def half_kelly(self, kelly_f: float) -> float:
        """Conservative half-Kelly for uncertain environments."""
        return min(kelly_f * 0.5, self.max_position_pct)

    def get_history(self) -> List[dict]:
        """Return sizing history."""
        return list(self._sizing_history)


# ---------------------------------------------------------------------------
# AlphaBetaUnleashed — 1-minute cadence beta management
# ---------------------------------------------------------------------------
class AlphaBetaUnleashed:
    """1-minute cadence beta management from Dataset 1.

    Core equations:
        Rm_adjusted = Rm_realized + macro.rm_adjustment
        target_beta = corridor_fn(Rm_adjusted) * 4.7 * vol_adj
        MES_hedge_beta = target_beta - sleeve_beta

    Uses BetaCorridor from engine.portfolio.beta_corridor when available,
    falls back to pure-numpy corridor function otherwise.
    """

    def __init__(self, execution_multiplier: float = EXECUTION_MULTIPLIER,
                 beta_max: float = BETA_MAX,
                 beta_inv: float = BETA_INV,
                 vol_standard: float = VOL_STANDARD):
        self.execution_multiplier = execution_multiplier
        self.beta_max = beta_max
        self.beta_inv = beta_inv
        self.vol_standard = vol_standard

        # BetaCorridor integration
        self._corridor = None
        try:
            self._corridor = BetaCorridor()
            logger.info("AlphaBetaUnleashed: BetaCorridor loaded")
        except Exception as exc:
            logger.warning("BetaCorridor init failed: %s", exc)

        # State tracking
        self._last_target_beta: float = 0.0
        self._last_hedge_beta: float = 0.0
        self._beta_history: List[dict] = []
        self._tick_count: int = 0

    # -- Corridor function (pure-numpy fallback) ----------------------------
    @staticmethod
    def _corridor_fn_fallback(rm_adjusted: float) -> float:
        """Piecewise linear corridor mapping Rm_adjusted -> base beta.

        Below R_LOW  => beta_inv (short bias)
        R_LOW-R_HIGH => linear interpolation 0 -> beta_max
        Above R_HIGH => beta_max (full throttle)
        """
        if rm_adjusted < R_LOW:
            # Below corridor: inverse / hedge
            slope = (0.0 - BETA_INV) / (R_LOW - 0.0)
            return max(BETA_INV, BETA_INV + slope * rm_adjusted)
        elif rm_adjusted <= R_HIGH:
            # Within corridor: linear ramp
            frac = (rm_adjusted - R_LOW) / (R_HIGH - R_LOW)
            return frac * BETA_MAX
        else:
            # Above corridor: full throttle
            return BETA_MAX

    def _corridor_fn(self, rm_adjusted: float) -> float:
        """Use BetaCorridor if available, else fallback."""
        if self._corridor is not None:
            try:
                state = self._corridor.compute_target_beta(rm_adjusted)
                if hasattr(state, "target_beta"):
                    return state.target_beta
                return state
            except Exception:
                pass
        return self._corridor_fn_fallback(rm_adjusted)

    def compute_target_beta(self, rm_realized: float,
                            rm_adjustment: float,
                            vol: float) -> float:
        """Compute target portfolio beta.

        Parameters
        ----------
        rm_realized : float
            Realised market return (annualised).
        rm_adjustment : float
            Macro engine adjustment to Rm.
        vol : float
            Current realised volatility.

        Returns
        -------
        float
            Target beta for the portfolio.
        """
        rm_adjusted = rm_realized + rm_adjustment
        base_beta = self._corridor_fn(rm_adjusted)

        # Vol adjustment: scale down when vol exceeds standard
        vol_adj = self.vol_standard / max(vol, 0.01)
        vol_adj = np.clip(vol_adj, 0.25, 2.0)

        # Apply execution multiplier and vol adjustment
        target_beta = base_beta * self.execution_multiplier * vol_adj

        # Clamp to hard limits
        target_beta = np.clip(target_beta, self.beta_inv, self.beta_max)

        self._last_target_beta = target_beta
        self._tick_count += 1

        self._beta_history.append({
            "tick": self._tick_count,
            "timestamp": datetime.now().isoformat(),
            "rm_realized": round(rm_realized, 6),
            "rm_adjustment": round(rm_adjustment, 6),
            "rm_adjusted": round(rm_adjusted, 6),
            "base_beta": round(base_beta, 4),
            "vol": round(vol, 4),
            "vol_adj": round(vol_adj, 4),
            "target_beta": round(target_beta, 4),
        })

        return target_beta

    def compute_hedge_requirement(self, target_beta: float,
                                  sleeve_beta: float) -> dict:
        """Compute MES hedge beta needed to reach target.

        Parameters
        ----------
        target_beta : float
            Desired portfolio beta.
        sleeve_beta : float
            Current aggregate sleeve beta.

        Returns
        -------
        dict
            hedge_beta, direction, magnitude, urgency
        """
        hedge_beta = target_beta - sleeve_beta
        self._last_hedge_beta = hedge_beta

        magnitude = abs(hedge_beta)
        if magnitude < 0.05:
            urgency = "NONE"
            direction = "HOLD"
        elif magnitude < 0.20:
            urgency = "LOW"
            direction = "INCREASE" if hedge_beta > 0 else "DECREASE"
        elif magnitude < 0.50:
            urgency = "MEDIUM"
            direction = "INCREASE" if hedge_beta > 0 else "DECREASE"
        else:
            urgency = "HIGH"
            direction = "INCREASE" if hedge_beta > 0 else "DECREASE"

        return {
            "target_beta": round(target_beta, 4),
            "sleeve_beta": round(sleeve_beta, 4),
            "hedge_beta": round(hedge_beta, 4),
            "direction": direction,
            "magnitude": round(magnitude, 4),
            "urgency": urgency,
            "instrument": "MES" if magnitude > 0.05 else "NONE",
            "timestamp": datetime.now().isoformat(),
        }

    def format_beta_report(self) -> str:
        """Format ASCII beta management report."""
        lines = [
            "",
            "=" * 72,
            "  ALPHA-BETA UNLEASHED  |  1-Minute Cadence Beta Management",
            "=" * 72,
            f"  Execution Multiplier : {self.execution_multiplier:.1f}x",
            f"  Beta Range           : [{self.beta_inv:.3f}, {self.beta_max:.3f}]",
            f"  Vol Standard         : {self.vol_standard:.1%}",
            f"  Corridor Import      : {'BetaCorridor' if self._corridor else 'Fallback'}",
            f"  Ticks Processed      : {self._tick_count:,}",
            "-" * 72,
            f"  Last Target Beta     : {self._last_target_beta:+.4f}",
            f"  Last Hedge Beta      : {self._last_hedge_beta:+.4f}",
        ]

        # Recent history (last 10 ticks)
        if self._beta_history:
            lines.append("-" * 72)
            lines.append("  Recent Beta History (last 10):")
            lines.append(f"  {'Tick':>6}  {'Rm_adj':>8}  {'Base_B':>8}  "
                         f"{'Vol':>6}  {'Target_B':>10}")
            for entry in self._beta_history[-10:]:
                lines.append(
                    f"  {entry['tick']:>6}  "
                    f"{entry['rm_adjusted']:>8.4f}  "
                    f"{entry['base_beta']:>8.4f}  "
                    f"{entry['vol']:>6.2%}  "
                    f"{entry['target_beta']:>+10.4f}"
                )

        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DecisionMatrix — main class
# ---------------------------------------------------------------------------
class DecisionMatrix:
    """Multi-gate trade approval engine.

    Combines all upstream signals (AlphaOptimizer, MetadronCube, MacroEngine,
    ML ensemble) into a unified execution decision.  Each trade must pass
    through six weighted gates.  Composite score determines approval and
    priority ranking.

    Parameters
    ----------
    nav : float
        Current net asset value (default $1,000).
    regime : str
        Current MetadronCube regime (TRENDING/RANGE/STRESS/CRASH).
    max_leverage : float
        Maximum allowed leverage.
    min_composite : float
        Minimum composite score to approve a trade.
    """

    def __init__(self, nav: float = 1_000.0,
                 regime: str = "RANGE",
                 max_leverage: float = MAX_LEVERAGE,
                 min_composite: float = MIN_COMPOSITE_SCORE):
        self.nav = nav
        self.regime = regime.upper()
        self.max_leverage = max_leverage
        self.min_composite = min_composite

        # Sub-components
        self.kelly_sizer = KellySizer()
        self.alpha_beta = AlphaBetaUnleashed()

        # Portfolio state tracking
        self._current_var: float = 0.0
        self._current_leverage: float = 0.0
        self._current_drawdown: float = 0.0
        self._peak_nav: float = nav
        self._sector_exposure: Dict[str, float] = {}
        self._position_tickers: List[str] = []
        self._decision_log: List[dict] = []
        self._approved_count: int = 0
        self._rejected_count: int = 0

    # -- Gate evaluators (4-gate cross-asset, asset-agnostic) ----------------

    def _score_fundamentals(self, proposal: dict) -> DecisionGate:
        """Gate 1 (40%): Fundamentals — quality, ROIC, FCF, credit, earnings.

        Regime context: adjusts quality bar (higher in CRASH, lower in TRENDING).

        MLVoteEnsemble tiers feeding this gate:
            T1 Neural Net, T5 Quality, T7 Distressed, T9 CVR, T10 CreditQuality
        """
        cfg = GATE_CONFIGS["FUNDAMENTALS"]
        gate = DecisionGate(
            gate_name="FUNDAMENTALS",
            weight=cfg["weight"],
            pass_threshold=cfg["threshold"],
        )

        sharpe = proposal.get("sharpe", 0.0)
        quality_tier = proposal.get("quality_tier", "G")
        alpha_signal = proposal.get("alpha_signal", 0.0)
        edge_bps = proposal.get("edge_bps", 0.0)
        credit_quality = proposal.get("credit_quality_score", 0.0)

        sharpe_score = np.clip((sharpe + 1.0) / 4.0, 0.0, 1.0)
        tier_score = QUALITY_TIER_SCORES.get(quality_tier, 0.05)
        alpha_score = np.clip(abs(alpha_signal) / 0.10, 0.0, 1.0)
        edge_score = np.clip(edge_bps / 50.0, 0.0, 1.0)
        credit_score = np.clip(credit_quality, 0.0, 1.0)

        # MLVoteEnsemble tiers: T1 Neural, T5 Quality, T7 Distressed, T9 CVR, T10 Credit
        tier_votes = proposal.get("ml_tier_votes", {})
        t1 = np.clip((tier_votes.get("T1_neural", 0) + 1) / 2.0, 0, 1)
        t5 = np.clip((tier_votes.get("T5_quality", 0) + 1) / 2.0, 0, 1)
        t7 = np.clip((tier_votes.get("T7_distressed", 0) + 1) / 2.0, 0, 1)
        t9 = np.clip((tier_votes.get("T9_cvr", 0) + 1) / 2.0, 0, 1)
        t10 = np.clip((tier_votes.get("T10_credit", 0) + 1) / 2.0, 0, 1)
        ensemble_score = (t1 * 0.20 + t5 * 0.30 + t7 * 0.15 + t9 * 0.15 + t10 * 0.20)

        # Regime quality modifier: stricter in STRESS/CRASH
        regime_mod = {"TRENDING": 1.0, "RANGE": 0.95, "STRESS": 0.80, "CRASH": 0.60}
        quality_mod = regime_mod.get(self.regime, 0.9)

        raw_score = (0.20 * sharpe_score + 0.15 * tier_score +
                     0.15 * alpha_score + 0.10 * credit_score +
                     0.10 * edge_score + 0.30 * ensemble_score)
        gate.score = raw_score * quality_mod

        gate.details = (f"Sharpe={sharpe:.2f} Tier={quality_tier} "
                        f"Alpha={alpha_signal:.4f} Credit={credit_quality:.2f} "
                        f"Ensemble={ensemble_score:.2f} RegimeMod={quality_mod}")
        gate.evaluate()
        return gate

    def _score_flow_headlines(self, proposal: dict) -> DecisionGate:
        """Gate 2 (20%): Flow/Headlines — ETF flow, news sentiment, sector rotation.

        MLVoteEnsemble tiers feeding this gate:
            T6 MiroMomentum (News+AgentSim), T8 EventDriven
        """
        cfg = GATE_CONFIGS["FLOW_HEADLINES"]
        gate = DecisionGate(
            gate_name="FLOW_HEADLINES",
            weight=cfg["weight"],
            pass_threshold=cfg["threshold"],
        )

        # News sentiment from Track B
        news_sentiment = proposal.get("news_sentiment", 0.5)
        etf_flow = proposal.get("etf_flow_score", 0.5)
        sector_rotation = proposal.get("sector_rotation_score", 0.5)

        # MLVoteEnsemble tiers: T6 MiroMomentum, T8 EventDriven
        tier_votes = proposal.get("ml_tier_votes", {})
        t6 = np.clip((tier_votes.get("T6_miro", 0) + 1) / 2.0, 0, 1)
        t8 = np.clip((tier_votes.get("T8_event", 0) + 1) / 2.0, 0, 1)
        ensemble_score = t6 * 0.60 + t8 * 0.40

        gate.score = (0.30 * news_sentiment + 0.20 * etf_flow +
                      0.15 * sector_rotation + 0.35 * ensemble_score)
        gate.details = (f"News={news_sentiment:.2f} ETF_flow={etf_flow:.2f} "
                        f"Sector={sector_rotation:.2f} T6+T8={ensemble_score:.2f}")
        gate.evaluate()
        return gate

    def _score_macro_regime(self, proposal: dict) -> DecisionGate:
        """Gate 3 (20%): Macro/Regime — direction alignment + risk state.

        Regime context: determines if trade direction (long/short) fits the
        current macro environment.

        MLVoteEnsemble tiers feeding this gate:
            T3 Volatility Regime, T4 Monte Carlo
        """
        cfg = GATE_CONFIGS["MACRO_REGIME"]
        gate = DecisionGate(
            gate_name="MACRO_REGIME",
            weight=cfg["weight"],
            pass_threshold=cfg["threshold"],
        )

        side = proposal.get("side", "long").lower()
        regime_scores = REGIME_ALIGNMENT_MAP.get(self.regime, {"long": 0.5, "short": 0.5})
        direction_score = regime_scores.get(side, 0.5)

        position_pct = proposal.get("position_pct", 0.05)
        position_vol = proposal.get("volatility", 0.20)

        # VaR headroom
        daily_vol = position_vol / np.sqrt(TRADING_DAYS)
        inc_var = position_pct * daily_vol * 1.645
        total_var = self._current_var + inc_var
        var_score = np.clip((MAX_PORTFOLIO_VAR_PCT - total_var) / MAX_PORTFOLIO_VAR_PCT, 0.0, 1.0)

        # Drawdown check
        dd_score = 1.0
        if self._current_drawdown > 0.10:
            dd_score = max(0.0, 1.0 - (self._current_drawdown - 0.10) / 0.15)

        # MLVoteEnsemble tiers: T3 VolRegime, T4 MonteCarlo
        tier_votes = proposal.get("ml_tier_votes", {})
        t3 = np.clip((tier_votes.get("T3_vol_regime", 0) + 1) / 2.0, 0, 1)
        t4 = np.clip((tier_votes.get("T4_monte_carlo", 0) + 1) / 2.0, 0, 1)
        ensemble_score = t3 * 0.50 + t4 * 0.50

        gate.score = (0.30 * direction_score + 0.20 * var_score +
                      0.15 * dd_score + 0.35 * ensemble_score)
        gate.details = (f"Regime={self.regime} Dir={side}={direction_score:.2f} "
                        f"VaR={total_var:.4f} DD={self._current_drawdown:.2%} "
                        f"T3+T4={ensemble_score:.2f}")
        gate.evaluate()
        return gate

    def _score_momentum_cross_asset(self, proposal: dict) -> DecisionGate:
        """Gate 4 (20%): Momentum — RSI, MACD, breakout, cross-asset momentum.

        Asset-agnostic: applies to equities, options, futures, ETFs uniformly.

        MLVoteEnsemble tiers feeding this gate:
            T1 Neural Net (price direction), T2 Momentum/Mean-Reversion
        """
        cfg = GATE_CONFIGS["MOMENTUM"]
        gate = DecisionGate(
            gate_name="MOMENTUM",
            weight=cfg["weight"],
            pass_threshold=cfg["threshold"],
        )

        side = proposal.get("side", "long").lower()
        rsi = proposal.get("rsi", 50.0)
        macd_signal = proposal.get("macd_signal", 0.0)
        breakout = proposal.get("breakout", False)
        momentum_5d = proposal.get("momentum_5d", 0.0)

        # RSI score (direction-aware)
        if side == "long":
            if rsi < 30:
                rsi_score = 0.85
            elif 40 <= rsi <= 70:
                rsi_score = 0.90
            elif rsi > 80:
                rsi_score = 0.10
            else:
                rsi_score = 0.50
        else:
            if rsi > 70:
                rsi_score = 0.90
            elif 50 <= rsi <= 70:
                rsi_score = 0.60
            elif rsi < 30:
                rsi_score = 0.10
            else:
                rsi_score = 0.40

        macd_raw = macd_signal if side == "long" else -macd_signal
        macd_score = np.clip((macd_raw + 1.0) / 2.0, 0.0, 1.0)
        breakout_score = 0.90 if breakout else 0.40
        mom = momentum_5d if side == "long" else -momentum_5d
        mom_score = np.clip((mom + 0.05) / 0.10, 0.0, 1.0)

        # MLVoteEnsemble tiers: T1 Neural (direction), T2 Momentum
        tier_votes = proposal.get("ml_tier_votes", {})
        t1 = np.clip((tier_votes.get("T1_neural", 0) + 1) / 2.0, 0, 1)
        t2 = np.clip((tier_votes.get("T2_momentum", 0) + 1) / 2.0, 0, 1)
        ensemble_score = t1 * 0.40 + t2 * 0.60

        gate.score = (0.20 * rsi_score + 0.15 * macd_score +
                      0.15 * breakout_score + 0.15 * mom_score +
                      0.35 * ensemble_score)
        gate.details = (f"RSI={rsi:.1f} MACD={macd_signal:+.3f} "
                        f"Breakout={'Y' if breakout else 'N'} "
                        f"Mom5d={momentum_5d:+.2%} T1+T2={ensemble_score:.2f}")
        gate.evaluate()
        return gate

    # -- Composite score -----------------------------------------------------
    def compute_composite_score(self, gate_results: List[DecisionGate]) -> float:
        """Compute weighted composite score from all gates.

        Parameters
        ----------
        gate_results : list[DecisionGate]
            Results from all six gates.

        Returns
        -------
        float
            Composite score in [0, 1].
        """
        total_weight = sum(g.weight for g in gate_results)
        if total_weight == 0:
            return 0.0
        composite = sum(g.weighted_score for g in gate_results) / total_weight
        return round(np.clip(composite, 0.0, 1.0), 6)

    # -- Main evaluation methods ---------------------------------------------
    def evaluate_trade(self, trade_proposal: dict) -> dict:
        """Evaluate a single trade through all six gates.

        Parameters
        ----------
        trade_proposal : dict
            Must contain at minimum: ticker, side.
            Optional keys feed into specific gates (sharpe, quality_tier,
            alpha_signal, edge_bps, regime_confidence, cube_score, ml_vote,
            agent_consensus, conviction_tier, rsi, macd_signal, breakout,
            adv, spread_bps, volatility, etc.).

        Returns
        -------
        dict
            approved (bool), composite_score, gate_results, ticker, side,
            rejection_reasons, timestamp.
        """
        ticker = trade_proposal.get("ticker", "UNKNOWN")
        side = trade_proposal.get("side", "long")

        # Run all 4 gates (cross-asset, asset-agnostic)
        gates = [
            self._score_fundamentals(trade_proposal),
            self._score_flow_headlines(trade_proposal),
            self._score_macro_regime(trade_proposal),
            self._score_momentum_cross_asset(trade_proposal),
        ]

        composite = self.compute_composite_score(gates)
        all_critical_passed = gates[0].passed  # FUNDAMENTALS (40%) is the critical gate
        approved = composite >= self.min_composite and all_critical_passed

        rejection_reasons = []
        if composite < self.min_composite:
            rejection_reasons.append(
                f"Composite {composite:.3f} < min {self.min_composite:.3f}")
        for g in gates:
            if not g.passed:
                rejection_reasons.append(
                    f"{g.gate_name}: {g.score:.3f} < {g.pass_threshold:.3f}")

        if approved:
            self._approved_count += 1
        else:
            self._rejected_count += 1

        result = {
            "ticker": ticker,
            "side": side,
            "approved": approved,
            "composite_score": composite,
            "gate_results": gates,
            "gate_scores": {g.gate_name: round(g.score, 4) for g in gates},
            "rejection_reasons": rejection_reasons,
            "timestamp": datetime.now().isoformat(),
        }

        self._decision_log.append({
            "ticker": ticker,
            "side": side,
            "approved": approved,
            "composite": composite,
            "timestamp": result["timestamp"],
        })

        return result

    def evaluate_batch(self, proposals: List[dict]) -> List[dict]:
        """Evaluate multiple trade proposals and rank by composite score.

        Parameters
        ----------
        proposals : list[dict]
            List of trade proposals (same format as evaluate_trade).

        Returns
        -------
        list[dict]
            Evaluated trades sorted by composite score (descending).
            Each dict includes all evaluate_trade fields plus 'rank'.
        """
        results = []
        for proposal in proposals:
            result = self.evaluate_trade(proposal)
            results.append(result)

        # Sort by composite score descending
        results.sort(key=lambda r: r["composite_score"], reverse=True)

        # Add rank
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    def get_position_sizing(self, approved_trade: dict,
                            nav: float = None) -> dict:
        """Compute Kelly-criterion position sizing for an approved trade.

        Parameters
        ----------
        approved_trade : dict
            Trade that passed evaluate_trade (must have composite_score).
        nav : float, optional
            Override NAV (defaults to self.nav).

        Returns
        -------
        dict
            Kelly sizing output + composite context.
        """
        if nav is None:
            nav = self.nav

        conviction = approved_trade.get("composite_score", 0.5)
        volatility = approved_trade.get("volatility", 0.20)
        price = approved_trade.get("price", 1.0)

        sizing = self.kelly_sizer.compute_size(
            nav=nav,
            conviction=conviction,
            volatility=volatility,
            price=price,
        )

        # Apply conviction tier multiplier
        conviction_tier = approved_trade.get("conviction_tier", "NONE")
        tier_multipliers = {
            "MAXIMUM": 2.0, "AGGRESSIVE": 1.5,
            "CONTROLLED": 1.2, "NONE": 1.0,
        }
        tier_mult = tier_multipliers.get(conviction_tier, 1.0)

        adjusted_shares = int(sizing["shares"] * tier_mult)
        adjusted_dollars = adjusted_shares * price

        # Ensure we don't exceed max position
        max_dollars = nav * MAX_SINGLE_POSITION_PCT
        if adjusted_dollars > max_dollars:
            adjusted_dollars = max_dollars
            adjusted_shares = int(adjusted_dollars / price) if price > 0 else 0

        sizing["adjusted_shares"] = adjusted_shares
        sizing["adjusted_dollars"] = round(adjusted_dollars, 2)
        sizing["conviction_tier"] = conviction_tier
        sizing["tier_multiplier"] = tier_mult
        sizing["ticker"] = approved_trade.get("ticker", "UNKNOWN")
        sizing["side"] = approved_trade.get("side", "long")

        return sizing

    def apply_portfolio_constraints(self, trades: List[dict],
                                    current_portfolio: dict) -> List[dict]:
        """Apply portfolio-level constraints to approved trades.

        Constraints:
            - Sector concentration: max 35% per GICS sector
            - Single position: max 20% of NAV
            - Correlation: reject if corr > 0.80 with existing position
            - Max total positions: leverage-dependent

        Parameters
        ----------
        trades : list[dict]
            Approved trades with sizing info.
        current_portfolio : dict
            Keys: positions (list of dicts with ticker, sector, weight),
                  sector_weights (dict), total_weight.

        Returns
        -------
        list[dict]
            Filtered list of trades that pass constraints. Each trade has
            'constraint_status' field added.
        """
        existing_sectors = current_portfolio.get("sector_weights", {})
        existing_tickers = set(
            p.get("ticker", "") for p in current_portfolio.get("positions", [])
        )
        existing_weights = current_portfolio.get("total_weight", 0.0)
        correlation_matrix = current_portfolio.get("correlation_matrix", {})

        approved = []
        running_sector_weights = dict(existing_sectors)
        running_total_weight = existing_weights

        for trade in trades:
            ticker = trade.get("ticker", "UNKNOWN")
            sector = trade.get("sector", "Unknown")
            trade_weight = trade.get("adjusted_dollars", 0.0) / max(self.nav, 1.0)
            constraints_passed = True
            constraint_notes = []

            # 1. Sector concentration
            sector_weight = running_sector_weights.get(sector, 0.0) + trade_weight
            if sector_weight > MAX_SECTOR_PCT:
                constraints_passed = False
                constraint_notes.append(
                    f"Sector {sector} would be {sector_weight:.1%} > {MAX_SECTOR_PCT:.0%}"
                )

            # 2. Single position limit
            if trade_weight > MAX_SINGLE_POSITION_PCT:
                constraints_passed = False
                constraint_notes.append(
                    f"Position {trade_weight:.1%} > {MAX_SINGLE_POSITION_PCT:.0%}"
                )

            # 3. Correlation check
            if ticker in correlation_matrix:
                for existing_ticker in existing_tickers:
                    corr_key = (ticker, existing_ticker)
                    alt_key = (existing_ticker, ticker)
                    corr_val = correlation_matrix.get(
                        corr_key, correlation_matrix.get(alt_key, 0.0))
                    if abs(corr_val) > CORRELATION_LIMIT:
                        constraints_passed = False
                        constraint_notes.append(
                            f"Corr({ticker},{existing_ticker})={corr_val:.2f} > "
                            f"{CORRELATION_LIMIT:.2f}"
                        )
                        break

            # 4. Total leverage check
            if running_total_weight + trade_weight > self.max_leverage:
                constraints_passed = False
                constraint_notes.append(
                    f"Leverage would be {running_total_weight + trade_weight:.2f}x > "
                    f"{self.max_leverage:.1f}x"
                )

            # 5. Duplicate check
            if ticker in existing_tickers:
                constraint_notes.append(f"Already holding {ticker} — will add to position")

            trade["constraint_status"] = {
                "passed": constraints_passed,
                "notes": constraint_notes,
            }

            if constraints_passed:
                running_sector_weights[sector] = (
                    running_sector_weights.get(sector, 0.0) + trade_weight
                )
                running_total_weight += trade_weight
                existing_tickers.add(ticker)
                approved.append(trade)

        return approved

    # -- Portfolio state updates ---------------------------------------------
    def update_portfolio_state(self, nav: float, leverage: float,
                               drawdown: float, var_pct: float,
                               sector_exposure: dict = None):
        """Update internal portfolio state for risk gate evaluation.

        Parameters
        ----------
        nav : float
            Current net asset value.
        leverage : float
            Current gross leverage.
        drawdown : float
            Current drawdown from peak (positive number).
        var_pct : float
            Current portfolio VaR as pct of NAV.
        sector_exposure : dict, optional
            Sector -> weight mapping.
        """
        self.nav = nav
        self._current_leverage = leverage
        self._current_drawdown = drawdown
        self._current_var = var_pct
        if nav > self._peak_nav:
            self._peak_nav = nav
        if sector_exposure is not None:
            self._sector_exposure = dict(sector_exposure)

    def update_regime(self, regime: str):
        """Update the current regime from MetadronCube."""
        self.regime = regime.upper()

    # -- Reporting -----------------------------------------------------------
    def format_decision_report(self, results: List[dict]) -> str:
        """Format a comprehensive ASCII decision report.

        Parameters
        ----------
        results : list[dict]
            List of evaluate_trade outputs.

        Returns
        -------
        str
            Multi-section ASCII report.
        """
        approved = [r for r in results if r.get("approved", False)]
        rejected = [r for r in results if not r.get("approved", False)]

        lines = [
            "",
            "=" * 78,
            "  DECISION MATRIX  |  Multi-Gate Trade Approval Report",
            "=" * 78,
            f"  Regime        : {self.regime}",
            f"  NAV           : ${self.nav:,.2f}",
            f"  Daily Target  : {DAILY_TARGET:.1%}",
            f"  Target        : 95%+ alpha, {DAILY_TARGET:.1%} daily compound",
            f"  Min Composite : {self.min_composite:.2f}",
            f"  Current VaR   : {self._current_var:.4f}",
            f"  Leverage      : {self._current_leverage:.2f}x / {self.max_leverage:.1f}x",
            f"  Drawdown      : {self._current_drawdown:.2%}",
            "-" * 78,
            f"  RESULTS: {len(approved)} APPROVED  |  {len(rejected)} REJECTED  |  "
            f"{len(results)} TOTAL",
            "-" * 78,
        ]

        # Approved trades
        if approved:
            lines.append("")
            lines.append("  APPROVED TRADES (by composite score):")
            lines.append(f"  {'#':>3}  {'Ticker':<8} {'Side':<6} {'Score':>7}  "
                         f"{'Alpha':>7} {'Regime':>7} {'Risk':>7} "
                         f"{'Conv':>7} {'Mom':>7} {'Liq':>7}")
            lines.append("  " + "-" * 74)
            for i, r in enumerate(approved, 1):
                gs = r.get("gate_scores", {})
                lines.append(
                    f"  {i:>3}  {r['ticker']:<8} {r['side']:<6} "
                    f"{r['composite_score']:>7.3f}  "
                    f"{gs.get('ALPHA_QUALITY', 0):>7.3f} "
                    f"{gs.get('REGIME_ALIGNMENT', 0):>7.3f} "
                    f"{gs.get('RISK_BUDGET', 0):>7.3f} "
                    f"{gs.get('CONVICTION_SCORE', 0):>7.3f} "
                    f"{gs.get('MOMENTUM_CONFIRM', 0):>7.3f} "
                    f"{gs.get('LIQUIDITY_CHECK', 0):>7.3f}"
                )

        # Rejected trades
        if rejected:
            lines.append("")
            lines.append("  REJECTED TRADES:")
            lines.append(f"  {'#':>3}  {'Ticker':<8} {'Side':<6} {'Score':>7}  "
                         f"{'Reason'}")
            lines.append("  " + "-" * 74)
            for i, r in enumerate(rejected, 1):
                reasons = r.get("rejection_reasons", [])
                reason_str = "; ".join(reasons[:3]) if reasons else "Unknown"
                lines.append(
                    f"  {i:>3}  {r['ticker']:<8} {r['side']:<6} "
                    f"{r['composite_score']:>7.3f}  {reason_str}"
                )

        # Gate summary statistics
        lines.append("")
        lines.append("  GATE PASS RATES:")
        lines.append("  " + "-" * 40)
        gate_names = list(GATE_CONFIGS.keys())
        for gn in gate_names:
            passed_count = sum(
                1 for r in results
                for g in r.get("gate_results", [])
                if g.gate_name == gn and g.passed
            )
            total = len(results)
            rate = passed_count / total if total > 0 else 0.0
            bar_len = int(rate * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            lines.append(f"  {gn:<20} [{bar}] {rate:>6.1%} "
                         f"({passed_count}/{total})")

        # Session stats
        lines.append("")
        lines.append("  SESSION STATISTICS:")
        lines.append("  " + "-" * 40)
        total_decisions = self._approved_count + self._rejected_count
        approval_rate = (self._approved_count / total_decisions * 100
                         if total_decisions > 0 else 0.0)
        lines.append(f"  Total Decisions     : {total_decisions}")
        lines.append(f"  Approved            : {self._approved_count}")
        lines.append(f"  Rejected            : {self._rejected_count}")
        lines.append(f"  Approval Rate       : {approval_rate:.1f}%")
        lines.append(f"  Peak NAV            : ${self._peak_nav:,.2f}")

        lines.append("=" * 78)
        return "\n".join(lines)

    def get_decision_log(self) -> List[dict]:
        """Return the full decision log."""
        return list(self._decision_log)

    def get_stats(self) -> dict:
        """Return session-level statistics."""
        total = self._approved_count + self._rejected_count
        return {
            "total_decisions": total,
            "approved": self._approved_count,
            "rejected": self._rejected_count,
            "approval_rate": (self._approved_count / total if total > 0 else 0.0),
            "nav": self.nav,
            "peak_nav": self._peak_nav,
            "regime": self.regime,
            "leverage": self._current_leverage,
            "drawdown": self._current_drawdown,
            "var_pct": self._current_var,
        }

    def reset(self):
        """Reset decision matrix state for a new session."""
        self._current_var = 0.0
        self._current_leverage = 0.0
        self._current_drawdown = 0.0
        self._peak_nav = self.nav
        self._sector_exposure = {}
        self._position_tickers = []
        self._decision_log = []
        self._approved_count = 0
        self._rejected_count = 0


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------
def build_trade_proposal(ticker: str, side: str = "long",
                         sharpe: float = 0.0,
                         quality_tier: str = "D",
                         alpha_signal: float = 0.0,
                         edge_bps: float = 0.0,
                         regime_confidence: float = 0.5,
                         cube_score: float = 0.5,
                         ml_vote: int = 0,
                         agent_consensus: float = 0.5,
                         conviction_tier: str = "NONE",
                         rsi: float = 50.0,
                         macd_signal: float = 0.0,
                         breakout: bool = False,
                         momentum_5d: float = 0.0,
                         momentum_20d: float = 0.0,
                         volatility: float = 0.20,
                         adv: float = 1_000_000.0,
                         spread_bps: float = 5.0,
                         market_cap: float = 1e9,
                         price: float = 100.0,
                         sector: str = "Unknown",
                         sector_bot_score: float = 0.5,
                         **kwargs) -> dict:
    """Convenience builder for a trade proposal dict.

    All parameters have sensible defaults; override as needed.

    Returns
    -------
    dict
        Ready for DecisionMatrix.evaluate_trade().
    """
    proposal = {
        "ticker": ticker,
        "side": side,
        "sharpe": sharpe,
        "quality_tier": quality_tier,
        "alpha_signal": alpha_signal,
        "edge_bps": edge_bps,
        "regime_confidence": regime_confidence,
        "cube_score": cube_score,
        "ml_vote": ml_vote,
        "agent_consensus": agent_consensus,
        "conviction_tier": conviction_tier,
        "rsi": rsi,
        "macd_signal": macd_signal,
        "breakout": breakout,
        "momentum_5d": momentum_5d,
        "momentum_20d": momentum_20d,
        "volatility": volatility,
        "adv": adv,
        "spread_bps": spread_bps,
        "market_cap": market_cap,
        "price": price,
        "sector": sector,
        "sector_bot_score": sector_bot_score,
    }
    proposal.update(kwargs)
    return proposal


def quick_evaluate(ticker: str, side: str = "long",
                   nav: float = 1_000.0,
                   regime: str = "RANGE",
                   **kwargs) -> dict:
    """One-liner: build proposal, evaluate, return result."""
    matrix = DecisionMatrix(nav=nav, regime=regime)
    proposal = build_trade_proposal(ticker=ticker, side=side, **kwargs)
    return matrix.evaluate_trade(proposal)


def batch_evaluate(proposals: List[dict],
                   nav: float = 1_000.0,
                   regime: str = "RANGE") -> Tuple[List[dict], str]:
    """Evaluate a batch and return (results, report)."""
    matrix = DecisionMatrix(nav=nav, regime=regime)
    results = matrix.evaluate_batch(proposals)
    report = matrix.format_decision_report(results)
    return results, report


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
def _self_test():
    """Run a quick self-test of all components."""
    print("\n" + "=" * 60)
    print("  DecisionMatrix — Self-Test")
    print("=" * 60)

    # 1. Kelly sizer
    ks = KellySizer()
    kf = ks.kelly_fraction(0.60, 2.0)
    akf = ks.aggressive_kelly(kf)
    sz = ks.compute_size(nav=1_000, conviction=0.75, volatility=0.20, price=150.0)
    print(f"\n  Kelly fraction (p=0.60, b=2.0): {kf:.4f}")
    print(f"  Aggressive Kelly (1.5x):        {akf:.4f}")
    print(f"  Size for $1k NAV, 75% conv:     {sz['shares']} shares "
          f"(${sz['dollar_amount']:,.2f})")

    # 2. AlphaBetaUnleashed
    abu = AlphaBetaUnleashed()
    tb = abu.compute_target_beta(rm_realized=0.09, rm_adjustment=0.01, vol=0.15)
    hedge = abu.compute_hedge_requirement(target_beta=tb, sleeve_beta=0.5)
    print(f"\n  Target beta (Rm=9%, adj=1%, vol=15%): {tb:+.4f}")
    print(f"  Hedge requirement (sleeve=0.5):       {hedge['hedge_beta']:+.4f} "
          f"({hedge['urgency']})")

    # 3. DecisionMatrix — single trade
    dm = DecisionMatrix(nav=1_000, regime="TRENDING")
    strong_trade = build_trade_proposal(
        ticker="NVDA", side="long",
        sharpe=2.5, quality_tier="A", alpha_signal=0.05, edge_bps=35,
        regime_confidence=0.85, cube_score=0.80,
        ml_vote=4, agent_consensus=0.90, conviction_tier="AGGRESSIVE",
        rsi=55, macd_signal=0.5, breakout=True,
        momentum_5d=0.03, volatility=0.30,
        adv=5_000_000, spread_bps=3.0, market_cap=2e12, price=800.0,
        sector="Technology",
    )
    result = dm.evaluate_trade(strong_trade)
    print(f"\n  NVDA evaluation:")
    print(f"    Approved:  {result['approved']}")
    print(f"    Composite: {result['composite_score']:.4f}")
    for g in result["gate_results"]:
        status = "PASS" if g.passed else "FAIL"
        print(f"    {g.gate_name:<20} {g.score:.3f}  [{status}]")

    # 4. Weak trade (should be rejected)
    weak_trade = build_trade_proposal(
        ticker="BADCO", side="long",
        sharpe=-0.5, quality_tier="F", alpha_signal=0.001, edge_bps=2,
        regime_confidence=0.20, cube_score=0.15,
        ml_vote=-3, agent_consensus=0.20, conviction_tier="NONE",
        rsi=82, macd_signal=-0.5,
        momentum_5d=-0.04, volatility=0.50,
        adv=50_000, spread_bps=50.0, market_cap=5e7, price=5.0,
        sector="Unknown",
    )
    result_weak = dm.evaluate_trade(weak_trade)
    print(f"\n  BADCO evaluation:")
    print(f"    Approved:  {result_weak['approved']}")
    print(f"    Composite: {result_weak['composite_score']:.4f}")
    if result_weak["rejection_reasons"]:
        for reason in result_weak["rejection_reasons"][:3]:
            print(f"    Rejected: {reason}")

    # 5. Batch evaluation with report
    batch = [strong_trade, weak_trade, build_trade_proposal(
        ticker="MSFT", side="long",
        sharpe=1.8, quality_tier="B", alpha_signal=0.03, edge_bps=20,
        regime_confidence=0.70, cube_score=0.65,
        ml_vote=2, agent_consensus=0.75, conviction_tier="CONTROLLED",
        rsi=48, macd_signal=0.2,
        momentum_5d=0.015, volatility=0.22,
        adv=3_000_000, spread_bps=2.0, market_cap=3e12, price=400.0,
        sector="Technology",
    )]
    results, report = batch_evaluate(batch, nav=1_000, regime="TRENDING")
    print(report)

    # 6. Position sizing
    for r in results:
        if r["approved"]:
            sizing = dm.get_position_sizing(r, nav=1_000)
            print(f"\n  Sizing for {sizing['ticker']}: "
                  f"{sizing['adjusted_shares']} shares "
                  f"(${sizing['adjusted_dollars']:,.2f})")

    # 7. Beta report
    print(abu.format_beta_report())

    print("\n  Self-test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
