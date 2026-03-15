"""MetadronCube — C(t) = f(L_t, R_t, F_t).

Multi-layer tensor sitting between MacroEngine and AlphaOptimizer.
    Layer 0  FedPlumbingLayer    → SOFR, HY spreads, M2V proxy
    Layer 1  LiquidityTensor     → reserves/TGA/ON-RRP/repo/credit → L(t) in [-1,+1]
    Layer 2  ReserveFlowKernel   → impulse: ΔReserves → ΔEquity/Credit
    Risk     RiskStateModel       → VIX + realized vol + credit spread → R(t) in [0,1]
    Flow     CapitalFlowModel     → sector momentum, leader/laggard → F(t)
    Layer 4  RegimeEngine         → TRENDING / RANGE / STRESS / CRASH
    Gate-Z   GateZAllocator       → 5-sleeve capital allocation
    RiskGovernor: beta/VaR/leverage/gamma corridor [7%–12%]

Regime leverage:
    TRENDING  2.5x  β≤0.65
    RANGE     2.0x  β≤0.30
    STRESS    1.5x  β≤0.10
    CRASH     0.8x  β≤-0.20
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from .macro_engine import MacroSnapshot, CubeRegime, MarketRegime


# ---------------------------------------------------------------------------
# Regime parameters
# ---------------------------------------------------------------------------
REGIME_PARAMS = {
    CubeRegime.TRENDING: {"max_leverage": 2.5, "beta_cap": 0.65, "equity_pct": 0.50},
    CubeRegime.RANGE:    {"max_leverage": 2.0, "beta_cap": 0.30, "equity_pct": 0.35},
    CubeRegime.STRESS:   {"max_leverage": 1.5, "beta_cap": 0.10, "equity_pct": 0.15},
    CubeRegime.CRASH:    {"max_leverage": 0.8, "beta_cap": -0.20, "equity_pct": 0.05},
}

# Beta corridor bounds (7%–12% return target)
R_LOW = 0.07
R_HIGH = 0.12
BETA_MAX = 2.0
BETA_INV = -0.136


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class LiquidityState:
    """L(t) — aggregate liquidity tensor in [-1, +1]."""
    value: float = 0.0
    sofr_signal: float = 0.0
    credit_impulse: float = 0.0
    m2_velocity: float = 0.0
    hy_spread_z: float = 0.0


@dataclass
class RiskState:
    """R(t) — aggregate risk in [0, 1]. 0=calm, 1=extreme stress."""
    value: float = 0.3
    vix_component: float = 0.0
    realized_vol: float = 0.0
    credit_spread_component: float = 0.0


@dataclass
class FlowState:
    """F(t) — capital flow model."""
    value: float = 0.0
    sector_momentum: dict = field(default_factory=dict)
    leader_sectors: list = field(default_factory=list)
    laggard_sectors: list = field(default_factory=list)


@dataclass
class SleeveAllocation:
    """Gate-Z 5-sleeve capital allocation."""
    p1_directional_equity: float = 0.35   # Directional equities
    p2_factor_rotation: float = 0.20      # Factor rotation
    p3_commodities_macro: float = 0.15    # Commodities/Macro
    p4_options_convexity: float = 0.15    # Options convexity
    p5_hedges_volatility: float = 0.15    # Hedges/Volatility

    def as_dict(self) -> dict:
        return {
            "P1_Directional_Equity": self.p1_directional_equity,
            "P2_Factor_Rotation": self.p2_factor_rotation,
            "P3_Commodities_Macro": self.p3_commodities_macro,
            "P4_Options_Convexity": self.p4_options_convexity,
            "P5_Hedges_Volatility": self.p5_hedges_volatility,
        }

    def total(self) -> float:
        return sum(self.as_dict().values())


@dataclass
class CubeOutput:
    """Full MetadronCube output."""
    regime: CubeRegime = CubeRegime.RANGE
    liquidity: LiquidityState = field(default_factory=LiquidityState)
    risk: RiskState = field(default_factory=RiskState)
    flow: FlowState = field(default_factory=FlowState)
    sleeves: SleeveAllocation = field(default_factory=SleeveAllocation)
    max_leverage: float = 2.0
    beta_cap: float = 0.30
    target_beta: float = 0.0
    risk_budget_pct: float = 0.10  # 7–12% corridor


# ---------------------------------------------------------------------------
# MetadronCube
# ---------------------------------------------------------------------------
class MetadronCube:
    """C(t) = f(L_t, R_t, F_t) — multi-layer allocation tensor."""

    def __init__(self):
        self._last_output: Optional[CubeOutput] = None

    def compute(self, macro: MacroSnapshot) -> CubeOutput:
        """Compute full cube state from macro snapshot."""
        output = CubeOutput()
        output.regime = macro.cube_regime

        # Get regime parameters
        params = REGIME_PARAMS.get(output.regime, REGIME_PARAMS[CubeRegime.RANGE])
        output.max_leverage = params["max_leverage"]
        output.beta_cap = params["beta_cap"]

        # Layer 1: Liquidity tensor
        output.liquidity = self._compute_liquidity(macro)

        # Risk state
        output.risk = self._compute_risk(macro)

        # Flow state
        output.flow = self._compute_flow(macro)

        # Gate-Z: 5-sleeve allocation
        output.sleeves = self._compute_sleeves(output)

        # Target beta from corridor
        output.target_beta = self._compute_target_beta(
            Rm=macro.spy_return_3m * 4,  # annualise
            sigma_m=macro.vix / 100,
        )
        output.target_beta = max(BETA_INV, min(output.beta_cap, output.target_beta))

        # Risk budget
        output.risk_budget_pct = self._compute_risk_budget(output)

        self._last_output = output
        return output

    def get_last(self) -> Optional[CubeOutput]:
        return self._last_output

    # --- Layer computations --------------------------------------------------

    def _compute_liquidity(self, macro: MacroSnapshot) -> LiquidityState:
        """L(t) in [-1, +1]."""
        ls = LiquidityState()

        # SOFR signal (proxy from short-term yield)
        ls.sofr_signal = np.clip((5.0 - macro.yield_2y) / 3.0, -1, 1)

        # Credit impulse (inverse of credit spread)
        ls.credit_impulse = np.clip(1.0 - macro.credit_spread / 5.0, -1, 1)

        # M2 velocity proxy (yield curve steepness)
        ls.m2_velocity = np.clip(macro.yield_spread / 2.0, -1, 1)

        # HY spread z-score
        ls.hy_spread_z = np.clip(-macro.credit_spread / 3.0, -1, 1)

        # Aggregate
        ls.value = np.clip(
            0.3 * ls.sofr_signal + 0.3 * ls.credit_impulse
            + 0.2 * ls.m2_velocity + 0.2 * ls.hy_spread_z,
            -1, 1,
        )
        return ls

    def _compute_risk(self, macro: MacroSnapshot) -> RiskState:
        """R(t) in [0, 1]. 0=calm, 1=extreme."""
        rs = RiskState()
        # VIX component
        rs.vix_component = np.clip(macro.vix / 60.0, 0, 1)
        # Realized vol proxy
        rs.realized_vol = np.clip(macro.vix / 40.0, 0, 1)
        # Credit spread
        rs.credit_spread_component = np.clip(macro.credit_spread / 8.0, 0, 1)
        # Aggregate
        rs.value = np.clip(
            0.4 * rs.vix_component + 0.3 * rs.realized_vol
            + 0.3 * rs.credit_spread_component,
            0, 1,
        )
        return rs

    def _compute_flow(self, macro: MacroSnapshot) -> FlowState:
        """Capital flow model from sector rankings."""
        fs = FlowState()
        fs.sector_momentum = macro.sector_rankings
        sectors = list(macro.sector_rankings.keys())
        if sectors:
            n = max(1, len(sectors) // 3)
            fs.leader_sectors = sectors[:n]
            fs.laggard_sectors = sectors[-n:]
        vals = list(macro.sector_rankings.values())
        fs.value = float(np.mean(vals)) if vals else 0.0
        return fs

    def _compute_sleeves(self, cube: CubeOutput) -> SleeveAllocation:
        """Gate-Z 5-sleeve allocation based on regime and risk."""
        sa = SleeveAllocation()
        regime = cube.regime
        risk = cube.risk.value

        if regime == CubeRegime.TRENDING:
            sa.p1_directional_equity = 0.50 - risk * 0.15
            sa.p2_factor_rotation = 0.20
            sa.p3_commodities_macro = 0.10
            sa.p4_options_convexity = 0.10
            sa.p5_hedges_volatility = 0.10 + risk * 0.05
        elif regime == CubeRegime.RANGE:
            sa.p1_directional_equity = 0.30 - risk * 0.10
            sa.p2_factor_rotation = 0.25
            sa.p3_commodities_macro = 0.15
            sa.p4_options_convexity = 0.15
            sa.p5_hedges_volatility = 0.15 + risk * 0.05
        elif regime == CubeRegime.STRESS:
            sa.p1_directional_equity = 0.15
            sa.p2_factor_rotation = 0.15
            sa.p3_commodities_macro = 0.20
            sa.p4_options_convexity = 0.20
            sa.p5_hedges_volatility = 0.30
        else:  # CRASH
            sa.p1_directional_equity = 0.05
            sa.p2_factor_rotation = 0.05
            sa.p3_commodities_macro = 0.20
            sa.p4_options_convexity = 0.25
            sa.p5_hedges_volatility = 0.45

        # Normalise to 1.0
        total = sa.total()
        if total > 0 and abs(total - 1.0) > 0.01:
            factor = 1.0 / total
            sa.p1_directional_equity *= factor
            sa.p2_factor_rotation *= factor
            sa.p3_commodities_macro *= factor
            sa.p4_options_convexity *= factor
            sa.p5_hedges_volatility *= factor

        return sa

    def _compute_target_beta(self, Rm: float, sigma_m: float) -> float:
        """Beta corridor from Dataset 1.

        Linear interpolation in [R_LOW, R_HIGH] with vol-normalization.
        """
        # Base linear interpolation
        if Rm <= R_LOW:
            base_beta = -0.029
        elif Rm >= R_HIGH:
            base_beta = 0.425
        else:
            slope = (0.425 - (-0.029)) / (R_HIGH - R_LOW)
            base_beta = -0.029 + slope * (Rm - R_LOW)

        # Vol normalisation (thesis standard: 15% vol)
        EXECUTION_MULTIPLIER = 4.7
        vol_adj = 0.15 / max(sigma_m, 0.05)
        target = base_beta * EXECUTION_MULTIPLIER * vol_adj

        return max(BETA_INV, min(BETA_MAX, target))

    def _compute_risk_budget(self, cube: CubeOutput) -> float:
        """Risk budget as % of NAV — targets 7–12% corridor."""
        base = (R_LOW + R_HIGH) / 2  # 9.5%
        # Adjust by regime
        if cube.regime == CubeRegime.TRENDING:
            return min(R_HIGH, base * 1.2)
        elif cube.regime == CubeRegime.CRASH:
            return max(R_LOW * 0.5, base * 0.4)
        elif cube.regime == CubeRegime.STRESS:
            return max(R_LOW, base * 0.7)
        return base
