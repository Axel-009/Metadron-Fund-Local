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

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import deque

from .macro_engine import MacroSnapshot, CubeRegime, MarketRegime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime parameters
# ---------------------------------------------------------------------------
REGIME_PARAMS = {
    CubeRegime.TRENDING: {"max_leverage": 2.5, "beta_cap": 0.65, "equity_pct": 0.50, "hedge_pct": 0.05},
    CubeRegime.RANGE:    {"max_leverage": 2.0, "beta_cap": 0.30, "equity_pct": 0.35, "hedge_pct": 0.15},
    CubeRegime.STRESS:   {"max_leverage": 1.5, "beta_cap": 0.10, "equity_pct": 0.15, "hedge_pct": 0.30},
    CubeRegime.CRASH:    {"max_leverage": 0.8, "beta_cap": -0.20, "equity_pct": 0.05, "hedge_pct": 0.45},
}

# Beta corridor bounds (7%–12% return target)
R_LOW = 0.07
R_HIGH = 0.12
BETA_MAX = 2.0
BETA_INV = -0.136
EXECUTION_MULTIPLIER = 4.7
VOL_STANDARD = 0.15


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
    fed_funds_impact: float = 0.0
    reverse_repo_signal: float = 0.0
    tga_balance_signal: float = 0.0
    reserve_flow: float = 0.0
    net_liquidity_score: float = 0.0


@dataclass
class RiskState:
    """R(t) — aggregate risk in [0, 1]. 0=calm, 1=extreme stress."""
    value: float = 0.3
    vix_component: float = 0.0
    realized_vol: float = 0.0
    credit_spread_component: float = 0.0
    vix_term_structure: float = 0.0
    correlation_stress: float = 0.0
    tail_risk: float = 0.0
    implied_vs_realized: float = 0.0


@dataclass
class FlowState:
    """F(t) — capital flow model."""
    value: float = 0.0
    sector_momentum: dict = field(default_factory=dict)
    leader_sectors: list = field(default_factory=list)
    laggard_sectors: list = field(default_factory=list)
    rotation_velocity: float = 0.0
    breadth: float = 0.0
    persistence: float = 0.0
    mean_reversion_signal: float = 0.0


@dataclass
class SleeveAllocation:
    """Gate-Z 5-sleeve capital allocation."""
    p1_directional_equity: float = 0.35
    p2_factor_rotation: float = 0.20
    p3_commodities_macro: float = 0.15
    p4_options_convexity: float = 0.15
    p5_hedges_volatility: float = 0.15

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

    def normalize(self):
        t = self.total()
        if t > 0 and abs(t - 1.0) > 0.01:
            f = 1.0 / t
            self.p1_directional_equity *= f
            self.p2_factor_rotation *= f
            self.p3_commodities_macro *= f
            self.p4_options_convexity *= f
            self.p5_hedges_volatility *= f


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
    risk_budget_pct: float = 0.10
    cube_tensor: Optional[np.ndarray] = None
    timestamp: str = ""
    regime_confidence: float = 0.0
    transition_probability: float = 0.0


@dataclass
class RegimeTransition:
    """Regime transition record."""
    from_regime: CubeRegime = CubeRegime.RANGE
    to_regime: CubeRegime = CubeRegime.RANGE
    timestamp: str = ""
    confidence: float = 0.0
    trigger: str = ""


# ---------------------------------------------------------------------------
# FedPlumbingLayer
# ---------------------------------------------------------------------------
class FedPlumbingLayer:
    """Layer 0: Fed plumbing — SOFR, reserves, TGA, ON-RRP.

    Models the Fed's balance sheet mechanics and their impact on liquidity.
    """

    def __init__(self):
        self._sofr_target: float = 5.25
        self._tga_balance_bn: float = 700.0
        self._rrp_balance_bn: float = 500.0
        self._reserve_balance_bn: float = 3200.0
        self._fed_funds_rate: float = 5.25

    def compute(self, macro: MacroSnapshot) -> dict:
        """Compute Fed plumbing signals."""
        # SOFR proxy from short-term yield
        sofr_proxy = macro.yield_2y
        sofr_spread = sofr_proxy - self._sofr_target
        sofr_signal = np.clip(sofr_spread / 1.0, -1, 1)

        # TGA balance impact (higher TGA = tighter liquidity)
        tga_signal = np.clip((800 - self._tga_balance_bn) / 400, -1, 1)

        # RRP impact (higher RRP = money parked, less in market)
        rrp_signal = np.clip((1000 - self._rrp_balance_bn) / 500, -1, 1)

        # Reserve adequacy
        reserve_signal = np.clip((self._reserve_balance_bn - 3000) / 500, -1, 1)

        # Fed funds effective rate impact
        ff_impact = np.clip((5.0 - self._fed_funds_rate) / 2.0, -1, 1)

        # Net plumbing score
        net = 0.25 * sofr_signal + 0.20 * tga_signal + 0.20 * rrp_signal + 0.20 * reserve_signal + 0.15 * ff_impact

        return {
            "sofr_signal": float(sofr_signal),
            "tga_signal": float(tga_signal),
            "rrp_signal": float(rrp_signal),
            "reserve_signal": float(reserve_signal),
            "ff_impact": float(ff_impact),
            "net_plumbing": float(np.clip(net, -1, 1)),
        }

    def update_balances(self, tga: float = None, rrp: float = None, reserves: float = None, ff_rate: float = None):
        if tga is not None: self._tga_balance_bn = tga
        if rrp is not None: self._rrp_balance_bn = rrp
        if reserves is not None: self._reserve_balance_bn = reserves
        if ff_rate is not None: self._fed_funds_rate = ff_rate


# ---------------------------------------------------------------------------
# LiquidityTensor
# ---------------------------------------------------------------------------
class LiquidityTensor:
    """Layer 1: Multi-dimensional liquidity state.

    Combines SOFR, credit impulse, M2 velocity, HY spreads, and Fed plumbing
    into a single L(t) value in [-1, +1].
    """

    WEIGHTS = {
        "sofr": 0.20,
        "credit": 0.25,
        "m2v": 0.15,
        "hy_spread": 0.15,
        "fed_plumbing": 0.25,
    }

    def compute(self, macro: MacroSnapshot, fed_plumbing: dict) -> LiquidityState:
        ls = LiquidityState()

        # SOFR signal
        ls.sofr_signal = np.clip((5.0 - macro.yield_2y) / 3.0, -1, 1)

        # Credit impulse
        ls.credit_impulse = np.clip(1.0 - macro.credit_spread / 5.0, -1, 1)

        # M2 velocity proxy
        ls.m2_velocity = np.clip(macro.yield_spread / 2.0, -1, 1)

        # HY spread z-score
        ls.hy_spread_z = np.clip(-macro.credit_spread / 3.0, -1, 1)

        # Fed plumbing
        ls.fed_funds_impact = fed_plumbing.get("ff_impact", 0)
        ls.reverse_repo_signal = fed_plumbing.get("rrp_signal", 0)
        ls.tga_balance_signal = fed_plumbing.get("tga_signal", 0)
        ls.reserve_flow = fed_plumbing.get("reserve_signal", 0)

        net_plumbing = fed_plumbing.get("net_plumbing", 0)

        # Weighted aggregate
        ls.value = float(np.clip(
            self.WEIGHTS["sofr"] * ls.sofr_signal +
            self.WEIGHTS["credit"] * ls.credit_impulse +
            self.WEIGHTS["m2v"] * ls.m2_velocity +
            self.WEIGHTS["hy_spread"] * ls.hy_spread_z +
            self.WEIGHTS["fed_plumbing"] * net_plumbing,
            -1, 1,
        ))

        # Net liquidity score 0-100
        ls.net_liquidity_score = (ls.value + 1) * 50

        return ls


# ---------------------------------------------------------------------------
# ReserveFlowKernel
# ---------------------------------------------------------------------------
class ReserveFlowKernel:
    """Layer 2: Reserve flow impact computation.

    Models how changes in bank reserves flow through to equities and credit.
    ΔReserves → ΔEquity/Credit with time lags.
    """

    def __init__(self, lag_days: int = 5, decay: float = 0.85):
        self.lag_days = lag_days
        self.decay = decay
        self._reserve_history: deque = deque(maxlen=60)

    def compute_impulse(self, current_reserves: float) -> float:
        """Compute reserve flow impulse."""
        self._reserve_history.append(current_reserves)
        if len(self._reserve_history) < 2:
            return 0.0

        history = list(self._reserve_history)
        deltas = [history[i] - history[i - 1] for i in range(1, len(history))]

        # Weighted impulse with decay
        impulse = 0.0
        weight = 1.0
        for d in reversed(deltas[-self.lag_days:]):
            impulse += d * weight
            weight *= self.decay

        return float(np.clip(impulse / 100, -1, 1))

    def compute_equity_impact(self, impulse: float) -> float:
        """Map reserve impulse to equity market impact."""
        return float(np.clip(impulse * 0.7, -1, 1))

    def compute_credit_impact(self, impulse: float) -> float:
        """Map reserve impulse to credit spread impact."""
        return float(np.clip(-impulse * 0.5, -1, 1))


# ---------------------------------------------------------------------------
# RiskStateModel (Enhanced)
# ---------------------------------------------------------------------------
class RiskStateModel:
    """Enhanced R(t) in [0, 1]. Multi-factor risk computation."""

    WEIGHTS = {
        "vix": 0.30,
        "realized_vol": 0.20,
        "credit": 0.20,
        "correlation": 0.15,
        "tail_risk": 0.15,
    }

    def compute(self, macro: MacroSnapshot) -> RiskState:
        rs = RiskState()

        # VIX component (0-1, scales with VIX/60)
        rs.vix_component = float(np.clip(macro.vix / 60.0, 0, 1))

        # Realized vol proxy
        rs.realized_vol = float(np.clip(macro.vix / 40.0, 0, 1))

        # Credit spread component
        rs.credit_spread_component = float(np.clip(macro.credit_spread / 8.0, 0, 1))

        # VIX term structure (flat/inverted = higher risk)
        rs.vix_term_structure = float(np.clip(macro.vix / 50.0, 0, 1))

        # Implied vs realized (higher IV/RV = elevated risk expectations)
        rs.implied_vs_realized = float(np.clip((macro.vix / 100) / max(rs.realized_vol, 0.01) - 1, -0.5, 0.5) + 0.5)

        # Correlation stress (high VIX → correlations spike)
        rs.correlation_stress = float(np.clip((macro.vix - 20) / 30, 0, 1))

        # Tail risk (extreme VIX levels)
        rs.tail_risk = float(np.clip((macro.vix - 30) / 20, 0, 1))

        # Weighted aggregate
        rs.value = float(np.clip(
            self.WEIGHTS["vix"] * rs.vix_component +
            self.WEIGHTS["realized_vol"] * rs.realized_vol +
            self.WEIGHTS["credit"] * rs.credit_spread_component +
            self.WEIGHTS["correlation"] * rs.correlation_stress +
            self.WEIGHTS["tail_risk"] * rs.tail_risk,
            0, 1,
        ))

        return rs


# ---------------------------------------------------------------------------
# CapitalFlowModel (Enhanced)
# ---------------------------------------------------------------------------
class CapitalFlowModel:
    """Enhanced F(t) — capital flow with persistence and mean-reversion."""

    def __init__(self, persistence_decay: float = 0.90, reversion_speed: float = 0.1):
        self.persistence_decay = persistence_decay
        self.reversion_speed = reversion_speed
        self._prev_flow: float = 0.0
        self._flow_history: deque = deque(maxlen=252)

    def compute(self, macro: MacroSnapshot) -> FlowState:
        fs = FlowState()
        fs.sector_momentum = macro.sector_rankings
        sectors = list(macro.sector_rankings.keys())

        if sectors:
            n = max(1, len(sectors) // 3)
            fs.leader_sectors = sectors[:n]
            fs.laggard_sectors = sectors[-n:]

        vals = list(macro.sector_rankings.values())
        raw_flow = float(np.mean(vals)) if vals else 0.0

        # Persistence (momentum)
        persistent_flow = self.persistence_decay * self._prev_flow + (1 - self.persistence_decay) * raw_flow
        fs.persistence = self.persistence_decay

        # Mean reversion component
        if len(self._flow_history) > 20:
            long_avg = float(np.mean(list(self._flow_history)))
            mean_rev = self.reversion_speed * (long_avg - persistent_flow)
            fs.mean_reversion_signal = mean_rev
            persistent_flow += mean_rev

        fs.value = persistent_flow
        self._prev_flow = persistent_flow
        self._flow_history.append(raw_flow)

        # Rotation velocity
        if len(self._flow_history) >= 5:
            recent = list(self._flow_history)[-5:]
            fs.rotation_velocity = float(np.std(recent))

        # Breadth (dispersion of sector returns)
        if vals:
            fs.breadth = float(np.std(vals))

        return fs


# ---------------------------------------------------------------------------
# RegimeEngine
# ---------------------------------------------------------------------------
class RegimeEngine:
    """Formal regime determination with transition probabilities."""

    # Markov transition matrix (from → to)
    TRANSITION_PROBS = {
        CubeRegime.TRENDING: {CubeRegime.TRENDING: 0.85, CubeRegime.RANGE: 0.10, CubeRegime.STRESS: 0.04, CubeRegime.CRASH: 0.01},
        CubeRegime.RANGE: {CubeRegime.TRENDING: 0.15, CubeRegime.RANGE: 0.70, CubeRegime.STRESS: 0.12, CubeRegime.CRASH: 0.03},
        CubeRegime.STRESS: {CubeRegime.TRENDING: 0.05, CubeRegime.RANGE: 0.20, CubeRegime.STRESS: 0.55, CubeRegime.CRASH: 0.20},
        CubeRegime.CRASH: {CubeRegime.TRENDING: 0.02, CubeRegime.RANGE: 0.08, CubeRegime.STRESS: 0.40, CubeRegime.CRASH: 0.50},
    }

    def __init__(self):
        self._current_regime = CubeRegime.RANGE
        self._regime_history: list[RegimeTransition] = []
        self._regime_days: int = 0

    def determine(self, macro_regime: CubeRegime, risk: float, liquidity: float, flow: float) -> tuple[CubeRegime, float]:
        """Determine regime with confidence scoring.

        Returns (regime, confidence).
        """
        # Score each potential regime
        scores = {}

        for regime in CubeRegime:
            score = 0.0

            # Macro regime prior carries 50% weight — upstream classification is authoritative
            macro_match = 1.0 if macro_regime == regime else 0.0

            if regime == CubeRegime.TRENDING:
                score += (1 - risk) * 0.15
                score += max(liquidity, 0) * 0.15
                score += max(flow, 0) * 0.20
                score += macro_match * 0.50

            elif regime == CubeRegime.RANGE:
                score += (1 - abs(risk - 0.3)) * 0.15
                score += (1 - abs(liquidity)) * 0.15
                score += (1 - abs(flow)) * 0.20
                score += macro_match * 0.50

            elif regime == CubeRegime.STRESS:
                score += risk * 0.15
                score += max(-liquidity, 0) * 0.15
                score += max(-flow, 0) * 0.20
                score += macro_match * 0.50

            elif regime == CubeRegime.CRASH:
                score += max(risk - 0.7, 0) * 0.20
                score += max(-liquidity - 0.5, 0) * 0.15
                score += macro_match * 0.65

            # Apply transition probability prior (light touch — macro prior is dominant)
            trans_prob = self.TRANSITION_PROBS.get(self._current_regime, {}).get(regime, 0.1)
            score *= (0.9 + 0.1 * trans_prob)

            scores[regime] = max(score, 0.01)

        # Select highest scoring regime
        best_regime = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best_regime] / total if total > 0 else 0.5

        # Record transition
        if best_regime != self._current_regime:
            self._regime_history.append(RegimeTransition(
                from_regime=self._current_regime,
                to_regime=best_regime,
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
            ))
            self._regime_days = 0
        else:
            self._regime_days += 1

        self._current_regime = best_regime
        return best_regime, confidence

    def get_transition_probability(self, target: CubeRegime) -> float:
        return self.TRANSITION_PROBS.get(self._current_regime, {}).get(target, 0.0)


# ---------------------------------------------------------------------------
# GateZAllocator (Enhanced)
# ---------------------------------------------------------------------------
class GateZAllocator:
    """Gate-Z 5-sleeve allocation with risk-adjusted weights."""

    # Base allocations per regime
    BASE_ALLOCATIONS = {
        CubeRegime.TRENDING: (0.50, 0.20, 0.10, 0.10, 0.10),
        CubeRegime.RANGE:    (0.30, 0.25, 0.15, 0.15, 0.15),
        CubeRegime.STRESS:   (0.15, 0.15, 0.20, 0.20, 0.30),
        CubeRegime.CRASH:    (0.05, 0.05, 0.20, 0.25, 0.45),
    }

    # Rebalancing bands (don't rebalance unless drift exceeds these)
    REBALANCE_BAND = 0.03

    def __init__(self):
        self._prev_allocation: Optional[SleeveAllocation] = None

    def allocate(self, regime: CubeRegime, risk: float, liquidity: float = 0.0) -> SleeveAllocation:
        base = self.BASE_ALLOCATIONS.get(regime, self.BASE_ALLOCATIONS[CubeRegime.RANGE])

        sa = SleeveAllocation(
            p1_directional_equity=base[0],
            p2_factor_rotation=base[1],
            p3_commodities_macro=base[2],
            p4_options_convexity=base[3],
            p5_hedges_volatility=base[4],
        )

        # Risk adjustment: shift from equity to hedges
        risk_shift = risk * 0.15
        sa.p1_directional_equity = max(0.02, sa.p1_directional_equity - risk_shift)
        sa.p5_hedges_volatility = min(0.60, sa.p5_hedges_volatility + risk_shift * 0.6)
        sa.p4_options_convexity = min(0.35, sa.p4_options_convexity + risk_shift * 0.4)

        # Liquidity adjustment: positive liquidity favours equities
        if liquidity > 0.3:
            liq_boost = (liquidity - 0.3) * 0.1
            sa.p1_directional_equity = min(0.60, sa.p1_directional_equity + liq_boost)
            sa.p5_hedges_volatility = max(0.05, sa.p5_hedges_volatility - liq_boost)

        # Normalize to 1.0
        sa.normalize()

        # Apply rebalancing bands
        if self._prev_allocation is not None:
            sa = self._apply_bands(sa, self._prev_allocation)

        self._prev_allocation = sa
        return sa

    def _apply_bands(self, new: SleeveAllocation, prev: SleeveAllocation) -> SleeveAllocation:
        """Only adjust if change exceeds rebalancing band."""
        fields = ['p1_directional_equity', 'p2_factor_rotation', 'p3_commodities_macro',
                   'p4_options_convexity', 'p5_hedges_volatility']
        for f in fields:
            new_val = getattr(new, f)
            prev_val = getattr(prev, f)
            if abs(new_val - prev_val) < self.REBALANCE_BAND:
                setattr(new, f, prev_val)
        new.normalize()
        return new


# ---------------------------------------------------------------------------
# RiskGovernor
# ---------------------------------------------------------------------------
class RiskGovernor:
    """Position limit enforcement, VaR budgeting, leverage monitoring."""

    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.25,
        max_leverage: float = 2.5,
        max_drawdown: float = 0.15,
        var_limit_pct: float = 0.02,
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown
        self.var_limit_pct = var_limit_pct

    def check_position_limit(self, position_pct: float) -> tuple[bool, str]:
        if position_pct > self.max_position_pct:
            return False, f"Position {position_pct:.1%} exceeds limit {self.max_position_pct:.1%}"
        return True, "OK"

    def check_sector_limit(self, sector_pct: float) -> tuple[bool, str]:
        if sector_pct > self.max_sector_pct:
            return False, f"Sector {sector_pct:.1%} exceeds limit {self.max_sector_pct:.1%}"
        return True, "OK"

    def check_leverage(self, current_leverage: float, regime: CubeRegime) -> tuple[bool, str]:
        regime_limit = REGIME_PARAMS.get(regime, {}).get("max_leverage", self.max_leverage)
        if current_leverage > regime_limit:
            return False, f"Leverage {current_leverage:.2f} exceeds regime limit {regime_limit:.2f}"
        return True, "OK"

    def check_drawdown(self, current_drawdown: float) -> tuple[bool, str]:
        if abs(current_drawdown) > self.max_drawdown:
            return False, f"Drawdown {current_drawdown:.1%} exceeds limit {self.max_drawdown:.1%}"
        return True, "OK"

    def compute_risk_budget(self, regime: CubeRegime, current_risk: float) -> float:
        """Compute available risk budget as fraction of total."""
        base = (R_LOW + R_HIGH) / 2
        regime_mult = {
            CubeRegime.TRENDING: 1.2,
            CubeRegime.RANGE: 1.0,
            CubeRegime.STRESS: 0.7,
            CubeRegime.CRASH: 0.4,
        }.get(regime, 1.0)
        budget = base * regime_mult
        used = current_risk * budget
        return max(0, budget - used)

    def full_check(self, position_pct: float = 0, sector_pct: float = 0,
                   leverage: float = 0, drawdown: float = 0, regime: CubeRegime = CubeRegime.RANGE) -> dict:
        checks = {}
        checks["position"] = self.check_position_limit(position_pct)
        checks["sector"] = self.check_sector_limit(sector_pct)
        checks["leverage"] = self.check_leverage(leverage, regime)
        checks["drawdown"] = self.check_drawdown(drawdown)
        checks["all_pass"] = all(v[0] for v in checks.values())
        return checks


# ---------------------------------------------------------------------------
# CubeLearningLoop
# ---------------------------------------------------------------------------
class CubeLearningLoop:
    """Adaptive parameter adjustment based on forecast accuracy."""

    def __init__(self, learning_rate: float = 0.01, memory: int = 100):
        self.learning_rate = learning_rate
        self._predictions: deque = deque(maxlen=memory)
        self._actuals: deque = deque(maxlen=memory)
        self._regime_accuracy: dict[str, list[bool]] = {}
        self._weight_adjustments: dict[str, float] = {}

    def record(self, predicted_regime: CubeRegime, predicted_beta: float,
               actual_return: float, actual_vol: float):
        self._predictions.append({
            "regime": predicted_regime.value,
            "beta": predicted_beta,
            "timestamp": datetime.now().isoformat(),
        })
        self._actuals.append({
            "return": actual_return,
            "vol": actual_vol,
        })

    def compute_accuracy(self) -> dict:
        if len(self._predictions) < 10:
            return {"accuracy": 0, "sample_size": len(self._predictions)}

        correct = 0
        total = len(self._predictions)
        for pred, actual in zip(self._predictions, self._actuals):
            pred_direction = 1 if pred["beta"] > 0 else -1
            actual_direction = 1 if actual["return"] > 0 else -1
            if pred_direction == actual_direction:
                correct += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "sample_size": total,
            "correct": correct,
        }

    def suggest_adjustments(self) -> dict:
        accuracy = self.compute_accuracy()
        adjustments = {}

        if accuracy["accuracy"] < 0.45:
            adjustments["reduce_leverage"] = True
            adjustments["increase_hedge"] = 0.05
            adjustments["note"] = "Below 45% accuracy — reduce risk"
        elif accuracy["accuracy"] > 0.60:
            adjustments["increase_leverage"] = True
            adjustments["decrease_hedge"] = 0.03
            adjustments["note"] = "Above 60% accuracy — increase conviction"
        else:
            adjustments["note"] = "Accuracy in acceptable range"

        return adjustments


# ---------------------------------------------------------------------------
# CubeHistory
# ---------------------------------------------------------------------------
class CubeHistory:
    """Track historical cube outputs."""

    def __init__(self, max_size: int = 1000):
        self._outputs: deque[CubeOutput] = deque(maxlen=max_size)

    def record(self, output: CubeOutput):
        output.timestamp = datetime.now().isoformat()
        self._outputs.append(output)

    def get_recent(self, n: int = 10) -> list[CubeOutput]:
        return list(self._outputs)[-n:]

    def get_regime_distribution(self) -> dict[str, int]:
        dist = {}
        for o in self._outputs:
            r = o.regime.value
            dist[r] = dist.get(r, 0) + 1
        return dist

    def get_avg_beta(self, window: int = 20) -> float:
        recent = list(self._outputs)[-window:]
        if not recent:
            return 0
        return float(np.mean([o.target_beta for o in recent]))

    @property
    def size(self) -> int:
        return len(self._outputs)


# ---------------------------------------------------------------------------
# StressScenarioEngine
# ---------------------------------------------------------------------------
class StressScenarioEngine:
    """Run stress tests across multiple market scenarios."""

    SCENARIOS = {
        "2008_GFC": MacroSnapshot(vix=80, spy_return_3m=-0.40, credit_spread=10, yield_spread=-2.0, cube_regime=CubeRegime.CRASH),
        "2020_COVID": MacroSnapshot(vix=65, spy_return_3m=-0.30, credit_spread=8, yield_spread=-0.5, cube_regime=CubeRegime.CRASH),
        "2022_HIKE": MacroSnapshot(vix=35, spy_return_3m=-0.15, credit_spread=5, yield_spread=-0.8, cube_regime=CubeRegime.STRESS),
        "BULL_RUN": MacroSnapshot(vix=12, spy_return_3m=0.15, credit_spread=2, yield_spread=1.5, cube_regime=CubeRegime.TRENDING),
        "RANGE_BOUND": MacroSnapshot(vix=18, spy_return_3m=0.02, credit_spread=3, yield_spread=0.2, cube_regime=CubeRegime.RANGE),
        "VIX_SPIKE": MacroSnapshot(vix=45, spy_return_3m=-0.10, credit_spread=6, yield_spread=-1.0, cube_regime=CubeRegime.CRASH),
    }

    def run_all(self, cube: "MetadronCube") -> dict:
        results = {}
        for name, scenario in self.SCENARIOS.items():
            output = cube.compute(scenario)
            results[name] = {
                "regime": output.regime.value,
                "target_beta": round(output.target_beta, 4),
                "max_leverage": output.max_leverage,
                "equity_allocation": round(output.sleeves.p1_directional_equity, 3),
                "hedge_allocation": round(output.sleeves.p5_hedges_volatility, 3),
                "risk_score": round(output.risk.value, 3),
            }
        return results


# ---------------------------------------------------------------------------
# MetadronCube
# ---------------------------------------------------------------------------
class MetadronCube:
    """C(t) = f(L_t, R_t, F_t) — multi-layer allocation tensor.

    Computes the cube state from macro inputs and determines:
    - Regime classification
    - 5-sleeve capital allocation
    - Target beta
    - Risk budget
    """

    def __init__(self):
        self._last_output: Optional[CubeOutput] = None
        self._fed_plumbing = FedPlumbingLayer()
        self._liquidity_tensor = LiquidityTensor()
        self._reserve_kernel = ReserveFlowKernel()
        self._risk_model = RiskStateModel()
        self._flow_model = CapitalFlowModel()
        self._regime_engine = RegimeEngine()
        self._gate_z = GateZAllocator()
        self._risk_governor = RiskGovernor()
        self._learning = CubeLearningLoop()
        self._history = CubeHistory()
        self._stress = StressScenarioEngine()

    def compute(self, macro: MacroSnapshot) -> CubeOutput:
        """Compute full cube state from macro snapshot."""
        output = CubeOutput()

        # Layer 0: Fed plumbing
        fed_data = self._fed_plumbing.compute(macro)

        # Layer 1: Liquidity tensor
        output.liquidity = self._liquidity_tensor.compute(macro, fed_data)

        # Layer 2: Reserve flow
        reserve_impulse = self._reserve_kernel.compute_impulse(3200)

        # Risk state
        output.risk = self._risk_model.compute(macro)

        # Flow state
        output.flow = self._flow_model.compute(macro)

        # Regime determination
        regime, confidence = self._regime_engine.determine(
            macro_regime=macro.cube_regime,
            risk=output.risk.value,
            liquidity=output.liquidity.value,
            flow=output.flow.value,
        )
        output.regime = regime
        output.regime_confidence = confidence

        # Get regime parameters
        params = REGIME_PARAMS.get(output.regime, REGIME_PARAMS[CubeRegime.RANGE])
        output.max_leverage = params["max_leverage"]
        output.beta_cap = params["beta_cap"]

        # Gate-Z: 5-sleeve allocation
        output.sleeves = self._gate_z.allocate(output.regime, output.risk.value, output.liquidity.value)

        # Target beta from corridor
        output.target_beta = self._compute_target_beta(
            Rm=macro.spy_return_3m * 4,
            sigma_m=macro.vix / 100,
        )
        output.target_beta = max(BETA_INV, min(output.beta_cap, output.target_beta))

        # Risk budget
        output.risk_budget_pct = self._compute_risk_budget(output)

        # Cube tensor
        output.cube_tensor = np.array([output.liquidity.value, output.risk.value, output.flow.value])

        # Record history
        self._history.record(output)
        self._last_output = output

        return output

    def get_last(self) -> Optional[CubeOutput]:
        return self._last_output

    def get_history(self) -> CubeHistory:
        return self._history

    def run_stress_tests(self) -> dict:
        return self._stress.run_all(self)

    def get_learning_stats(self) -> dict:
        return self._learning.compute_accuracy()

    def get_risk_governor(self) -> RiskGovernor:
        return self._risk_governor

    # --- Layer computations --------------------------------------------------

    def _compute_liquidity(self, macro: MacroSnapshot) -> LiquidityState:
        """L(t) in [-1, +1]. Legacy method for backwards compatibility."""
        fed_data = self._fed_plumbing.compute(macro)
        return self._liquidity_tensor.compute(macro, fed_data)

    def _compute_risk(self, macro: MacroSnapshot) -> RiskState:
        """R(t) in [0, 1]. 0=calm, 1=extreme."""
        return self._risk_model.compute(macro)

    def _compute_flow(self, macro: MacroSnapshot) -> FlowState:
        """Capital flow model from sector rankings."""
        return self._flow_model.compute(macro)

    def _compute_sleeves(self, cube: CubeOutput) -> SleeveAllocation:
        """Gate-Z 5-sleeve allocation based on regime and risk."""
        return self._gate_z.allocate(cube.regime, cube.risk.value, cube.liquidity.value)

    def _compute_target_beta(self, Rm: float, sigma_m: float) -> float:
        """Beta corridor from Dataset 1.

        Linear interpolation in [R_LOW, R_HIGH] with vol-normalization.
        """
        if Rm <= R_LOW:
            base_beta = -0.029
        elif Rm >= R_HIGH:
            base_beta = 0.425
        else:
            slope = (0.425 - (-0.029)) / (R_HIGH - R_LOW)
            base_beta = -0.029 + slope * (Rm - R_LOW)

        vol_adj = VOL_STANDARD / max(sigma_m, 0.05)
        target = base_beta * EXECUTION_MULTIPLIER * vol_adj

        return max(BETA_INV, min(BETA_MAX, target))

    def _compute_risk_budget(self, cube: CubeOutput) -> float:
        """Risk budget as % of NAV — targets 7–12% corridor."""
        base = (R_LOW + R_HIGH) / 2
        if cube.regime == CubeRegime.TRENDING:
            return min(R_HIGH, base * 1.2)
        elif cube.regime == CubeRegime.CRASH:
            return max(R_LOW * 0.5, base * 0.4)
        elif cube.regime == CubeRegime.STRESS:
            return max(R_LOW, base * 0.7)
        return base

    def summary(self) -> str:
        """ASCII cube summary."""
        if not self._last_output:
            return "MetadronCube: No data yet"

        o = self._last_output
        lines = [
            "=" * 60,
            "METADRON CUBE STATE",
            "=" * 60,
            f"  Regime:      {o.regime.value} (confidence: {o.regime_confidence:.1%})",
            f"  L(t):        {o.liquidity.value:>+.3f}  (liquidity)",
            f"  R(t):        {o.risk.value:>.3f}   (risk)",
            f"  F(t):        {o.flow.value:>+.3f}  (flow)",
            f"  Target β:    {o.target_beta:>.4f}",
            f"  Max Leverage: {o.max_leverage:.1f}x",
            f"  Beta Cap:    {o.beta_cap:.2f}",
            "",
            "  SLEEVES:",
            f"    P1 Equity:     {o.sleeves.p1_directional_equity:.1%}",
            f"    P2 Factor:     {o.sleeves.p2_factor_rotation:.1%}",
            f"    P3 Commodity:  {o.sleeves.p3_commodities_macro:.1%}",
            f"    P4 Convexity:  {o.sleeves.p4_options_convexity:.1%}",
            f"    P5 Hedges:     {o.sleeves.p5_hedges_volatility:.1%}",
            f"    TOTAL:         {o.sleeves.total():.1%}",
            "",
            f"  History: {self._history.size} records",
            f"  Avg Beta (20d): {self._history.get_avg_beta():.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)
