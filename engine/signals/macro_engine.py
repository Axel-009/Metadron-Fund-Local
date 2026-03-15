"""MacroEngine — Regime classification & GMTF money velocity analysis.

Incorporates Dataset 3: Global Monetary Tension Framework (GMTF).
Regimes: BULL / BEAR / TRANSITION / STRESS
Money velocity: Fisher equation V = GDP/M2, sigmoid triggers, carry-to-volatility.
Ranks sector universe by macro regime.
Data: yfinance (market proxies) + FRB/FRED (when API key available).
"""

import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

from ..data.yahoo_data import get_adj_close, get_macro_data, get_returns


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------
class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    TRANSITION = "TRANSITION"
    STRESS = "STRESS"
    CRASH = "CRASH"


class CubeRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGE = "RANGE"
    STRESS = "STRESS"
    CRASH = "CRASH"


# ---------------------------------------------------------------------------
# SDR basket for GMTF
# ---------------------------------------------------------------------------
SDR_WEIGHTS = pd.Series({
    "USD": 0.4338, "EUR": 0.2931, "CNY": 0.1228,
    "JPY": 0.0759, "GBP": 0.0744,
})

# Sigmoid sensitivity for non-linear triggers
SIGMOID_SENSITIVITY = 15.0
ROLLING_WINDOW = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MacroSnapshot:
    """Current macro state."""
    regime: MarketRegime = MarketRegime.TRANSITION
    vix: float = 20.0
    spy_return_1m: float = 0.0
    spy_return_3m: float = 0.0
    yield_10y: float = 4.0
    yield_2y: float = 4.5
    yield_spread: float = -0.5
    credit_spread: float = 3.0
    gold_momentum: float = 0.0
    sector_rankings: dict = field(default_factory=dict)
    gmtf_score: float = 0.0
    money_velocity_signal: float = 0.0
    cube_regime: CubeRegime = CubeRegime.RANGE


@dataclass
class GMTFOutput:
    """Global Monetary Tension Framework output."""
    gmtf_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    gmtf_smoothed: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    gamma_liquidity: float = 1.0
    gamma_fx: float = 1.0
    gamma_wage: float = 1.0
    gamma_reserve: float = 1.0
    rv_signals: Optional[pd.DataFrame] = None
    ctv_ratios: Optional[pd.DataFrame] = None
    ctv_gate: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Core GMTF functions (from Dataset 3)
# ---------------------------------------------------------------------------
def sigmoid_trigger(x, threshold: float, sensitivity: float = SIGMOID_SENSITIVITY):
    """Smooth non-linear transition (0 to 1)."""
    return 1 / (1 + np.exp(-sensitivity * (x - threshold)))


def compute_gammas(
    m2: pd.DataFrame,
    gdp_g: pd.DataFrame,
    unemp: pd.DataFrame,
    fx: pd.DataFrame,
    rates: pd.DataFrame,
) -> tuple:
    """Compute non-linear gamma multipliers for GMTF.

    Returns (gamma_liquidity, gamma_fx, gamma_wage, gamma_reserve).
    """
    # Gamma_L: Liquidity overload (M2 growth > 2x GDP growth)
    g_liquidity = 1 + sigmoid_trigger(m2 - 2 * gdp_g, 0.01) * 1.5

    # Gamma_FX: Fed shock (USD yield spike contagion)
    usd_delta = rates["USD"].diff().fillna(0) if "USD" in rates.columns else pd.Series(0, index=rates.index)
    g_fx_factor = (1 - sigmoid_trigger(usd_delta, 0.0025) * 0.5).values
    if g_fx_factor.ndim == 1:
        g_fx_factor = g_fx_factor[:, None]

    # Gamma_W: Wage-price spiral (labor tightness)
    g_wage = 1 + sigmoid_trigger(0.042 - unemp, 0) * 2.0

    # Gamma_S: Reserve decay (USD share threshold at 57%)
    usd_share = fx["USD"] / fx.sum(axis=1) if "USD" in fx.columns else pd.Series(0.6, index=fx.index)
    g_reserve = 1 + sigmoid_trigger(0.57 - usd_share, 0) * 1.5

    return g_liquidity, g_fx_factor, g_wage, g_reserve


def compute_gmtf(
    m2: pd.DataFrame,
    gdp: pd.DataFrame,
    unemp: pd.DataFrame,
    fx: pd.DataFrame,
    rates: pd.DataFrame,
) -> GMTFOutput:
    """Full GMTF calculation pipeline.

    Returns GMTFOutput with aggregated tension series and gamma multipliers.
    """
    currencies = SDR_WEIGHTS.index.tolist()
    dates = m2.index

    # GDP growth proxy
    gdp_growth = gdp.pct_change().fillna(0.0001)

    # Base impact: theta
    base_theta = (m2 / gdp) * (1 + unemp) * gdp_growth

    # Non-linear triggers
    gl, gfx, gw, gs = compute_gammas(m2, gdp_growth, unemp, fx, rates)

    # Apply multipliers
    triggered_theta = base_theta * gl * gw
    # Apply FX gamma (broadcasting)
    if hasattr(gfx, "shape") and gfx.ndim == 2:
        n_cols = min(gfx.shape[1], triggered_theta.shape[1])
        for i in range(n_cols):
            triggered_theta.iloc[:, i] *= gfx[:len(triggered_theta), 0]

    # Apply reserve decay to USD
    if "USD" in triggered_theta.columns:
        triggered_theta["USD"] *= gs

    # Aggregate with SDR weights
    available = [c for c in currencies if c in triggered_theta.columns]
    weights = SDR_WEIGHTS[available]
    weights = weights / weights.sum()
    gmtf_series = triggered_theta[available].dot(weights)
    gmtf_smoothed = gmtf_series.rolling(window=ROLLING_WINDOW).mean()

    return GMTFOutput(
        gmtf_series=gmtf_series,
        gmtf_smoothed=gmtf_smoothed,
        gamma_liquidity=float(gl.mean().mean()) if hasattr(gl, "mean") else 1.0,
        gamma_fx=float(np.mean(gfx)),
        gamma_wage=float(gw.mean().mean()) if hasattr(gw, "mean") else 1.0,
        gamma_reserve=float(gs.mean()) if hasattr(gs, "mean") else 1.0,
    )


# ---------------------------------------------------------------------------
# Carry-to-Volatility (CtV) & stop-loss
# ---------------------------------------------------------------------------
def compute_ctv_signals(
    yield_data: pd.DataFrame,
    fx_data: pd.DataFrame,
    window: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute carry-to-volatility ratios and entry gates.

    Returns (ctv_ratios, ctv_gate).
    """
    currencies = ["EUR", "JPY", "GBP", "CNY"]
    ctv_ratios = pd.DataFrame(index=yield_data.index)

    for c in currencies:
        y_col = f"yield3m_{c}" if f"yield3m_{c}" in yield_data.columns else None
        y_usd = "yield3m_USD" if "yield3m_USD" in yield_data.columns else None
        fx_col = f"fx_{c}" if f"fx_{c}" in fx_data.columns else None

        if y_col and y_usd and fx_col:
            carry = (yield_data[y_col] - yield_data[y_usd]) / 100
            vol = fx_data[fx_col].pct_change().rolling(window).std() * np.sqrt(252)
            vol = vol.replace(0, np.nan)
            ctv_ratios[c] = carry / vol
        else:
            ctv_ratios[c] = 0.0

    ctv_ratios = ctv_ratios.fillna(0)

    # Stop-loss: trigger if CtV drops > 2 std dev in 1 day
    ctv_rolling_std = ctv_ratios.rolling(window).std().fillna(1)
    stop_signal = (ctv_ratios.diff() < -2 * ctv_rolling_std).astype(int)

    # Gate: CtV > 0.5 to enter, no stop-loss active
    gate = ((ctv_ratios > 0.5) & (stop_signal == 0)).astype(int)

    return ctv_ratios, gate


# ---------------------------------------------------------------------------
# MacroEngine class
# ---------------------------------------------------------------------------
class MacroEngine:
    """Macro regime classification and sector ranking engine."""

    def __init__(self):
        self._snapshot: Optional[MacroSnapshot] = None

    def analyze(self, lookback_days: int = 252) -> MacroSnapshot:
        """Run full macro analysis using Yahoo Finance data."""
        snapshot = MacroSnapshot()

        try:
            start = pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30)
            start_str = start.strftime("%Y-%m-%d")

            # Fetch core data
            macro = get_macro_data(start=start_str)
            if macro.empty:
                self._snapshot = snapshot
                return snapshot

            # VIX
            if "VIX" in macro.columns:
                snapshot.vix = float(macro["VIX"].dropna().iloc[-1])

            # SPY returns
            if "S&P 500" in macro.columns:
                spy = macro["S&P 500"].dropna()
                if len(spy) > 21:
                    snapshot.spy_return_1m = float(spy.iloc[-1] / spy.iloc[-22] - 1)
                if len(spy) > 63:
                    snapshot.spy_return_3m = float(spy.iloc[-1] / spy.iloc[-64] - 1)

            # Yields
            if "10Y Yield" in macro.columns:
                snapshot.yield_10y = float(macro["10Y Yield"].dropna().iloc[-1])
            if "5Y Yield" in macro.columns:
                snapshot.yield_2y = float(macro["5Y Yield"].dropna().iloc[-1])
            snapshot.yield_spread = snapshot.yield_10y - snapshot.yield_2y

            # Credit spread (HY - IG proxy)
            if "HY Corporate" in macro.columns and "IG Corporate" in macro.columns:
                hy_ret = macro["HY Corporate"].pct_change().rolling(20).std() * np.sqrt(252)
                ig_ret = macro["IG Corporate"].pct_change().rolling(20).std() * np.sqrt(252)
                spread = (hy_ret - ig_ret).dropna()
                if len(spread) > 0:
                    snapshot.credit_spread = float(spread.iloc[-1]) * 100

            # Gold momentum
            if "Gold" in macro.columns:
                gold = macro["Gold"].dropna()
                if len(gold) > 63:
                    snapshot.gold_momentum = float(gold.iloc[-1] / gold.iloc[-64] - 1)

            # Classify regime
            snapshot.regime = self._classify_regime(snapshot)
            snapshot.cube_regime = self._classify_cube_regime(snapshot)

            # Rank sectors
            snapshot.sector_rankings = self._rank_sectors(snapshot, start_str)

        except Exception as e:
            snapshot.regime = MarketRegime.TRANSITION

        self._snapshot = snapshot
        return snapshot

    def get_snapshot(self) -> MacroSnapshot:
        if self._snapshot is None:
            return self.analyze()
        return self._snapshot

    # --- Regime classification -----------------------------------------------

    def _classify_regime(self, snap: MacroSnapshot) -> MarketRegime:
        """Rule-based regime classification."""
        score = 0

        # VIX component
        if snap.vix > 35:
            score -= 3
        elif snap.vix > 25:
            score -= 1
        elif snap.vix < 15:
            score += 2
        else:
            score += 1

        # SPY momentum
        if snap.spy_return_3m > 0.10:
            score += 3
        elif snap.spy_return_3m > 0.03:
            score += 1
        elif snap.spy_return_3m < -0.10:
            score -= 3
        elif snap.spy_return_3m < -0.03:
            score -= 1

        # Yield curve
        if snap.yield_spread > 0.5:
            score += 1
        elif snap.yield_spread < -0.5:
            score -= 1

        # Credit
        if snap.credit_spread > 5:
            score -= 2

        if score >= 4:
            return MarketRegime.BULL
        elif score >= 1:
            return MarketRegime.TRANSITION
        elif score >= -2:
            return MarketRegime.BEAR
        else:
            return MarketRegime.STRESS

    def _classify_cube_regime(self, snap: MacroSnapshot) -> CubeRegime:
        """Map market regime to cube regime for MetadronCube."""
        if snap.vix > 40:
            return CubeRegime.CRASH
        elif snap.regime == MarketRegime.STRESS:
            return CubeRegime.STRESS
        elif snap.regime == MarketRegime.BULL:
            return CubeRegime.TRENDING
        else:
            return CubeRegime.RANGE

    # --- Sector ranking ------------------------------------------------------

    def _rank_sectors(self, snap: MacroSnapshot, start_str: str) -> dict[str, float]:
        """Rank sectors by macro favourability."""
        from ..data.universe_engine import SECTOR_ETFS
        etfs = list(SECTOR_ETFS.values())
        inv = {v: k for k, v in SECTOR_ETFS.items()}

        try:
            returns = get_returns(etfs, start=start_str)
            if returns.empty:
                return {}
        except Exception:
            return {}

        rankings = {}
        for etf in returns.columns:
            sector = inv.get(etf, etf)
            r = returns[etf].dropna()
            if len(r) < 21:
                continue

            # Composite score: momentum + risk-adjusted
            mom_1m = float(r.iloc[-21:].sum())
            mom_3m = float(r.iloc[-63:].sum()) if len(r) >= 63 else mom_1m
            vol = float(r.std() * np.sqrt(252))
            sharpe = (mom_3m * 4) / vol if vol > 0 else 0

            # Regime adjustment
            if snap.regime == MarketRegime.STRESS:
                # Favour defensives
                if sector in ["Utilities", "Consumer Staples", "Health Care"]:
                    sharpe *= 1.3
                elif sector in ["Consumer Discretionary", "Information Technology"]:
                    sharpe *= 0.7
            elif snap.regime == MarketRegime.BULL:
                if sector in ["Information Technology", "Consumer Discretionary", "Communication Services"]:
                    sharpe *= 1.2

            rankings[sector] = round(sharpe, 4)

        # Sort descending
        return dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
