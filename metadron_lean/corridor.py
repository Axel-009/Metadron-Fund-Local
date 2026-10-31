"""Smart-beta corridor — canonical implementation.

Definition (see CLAUDE.md "Smart Beta — corridor derivation"):

    t        = clip((Rm − R_LOW) / (R_HIGH − R_LOW), 0, 1)
    β(Rm)    = BETA_LOW + SPAN · (3t² − 2t³)                smooth S-curve
    γ(Rm)    = dβ/dRm = SPAN · 6t(1 − t) / (R_HIGH − R_LOW) flattened bell (corridor gamma)
    leverage = β · EXECUTION_MULTIPLIER · vol_scale(σ_m)

Endpoints come from the Merton rule β = (Rm − h)/(λσ²) evaluated at the corridor edges:
λσ² = 0.1101, h = 7.32% (λ ≈ 4.89 at σ = 15%).

Pure Python + math only, so the same file runs in QuantConnect, locally and in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- corridor constants (single source of truth) -----------------------------
R_LOW = 0.07                     # S&P 500 historical earnings-return floor
R_HIGH = 0.12                    # ... ceiling
BETA_LOW = -0.029                # β at R_LOW
BETA_HIGH = 0.425                # β at R_HIGH
SPAN = BETA_HIGH - BETA_LOW      # 0.454
EXECUTION_MULTIPLIER = 4.7       # β → leverage
SIGMA_STD = 0.15                 # thesis standard volatility
THROTTLE = 0.05                  # rebalance only if |Δleverage| exceeds this

# Merton parameters implied by the endpoints
LAMBDA_SIGMA2 = (R_HIGH - R_LOW) / SPAN               # 0.1101
HURDLE = R_LOW - BETA_LOW * LAMBDA_SIGMA2             # 0.0732


def risk_aversion(sigma: float = SIGMA_STD) -> float:
    """λ implied by the corridor for a given volatility (only λσ² is identified)."""
    return LAMBDA_SIGMA2 / (sigma * sigma)


def _t(rm: float) -> float:
    return min(max((rm - R_LOW) / (R_HIGH - R_LOW), 0.0), 1.0)


def corridor_beta(rm: float) -> float:
    """Smooth corridor beta β(Rm): −0.029 at ≤7%, 0.198 at 9.5%, 0.425 at ≥12%."""
    t = _t(rm)
    return BETA_LOW + SPAN * (3 * t * t - 2 * t * t * t)


def corridor_gamma(rm: float) -> float:
    """Corridor gamma γ = dβ/dRm per unit of Rm (peak 13.62 at 9.5%; ×0.01 for 'per 1%')."""
    if rm <= R_LOW or rm >= R_HIGH:
        return 0.0
    t = _t(rm)
    return SPAN * 6 * t * (1 - t) / (R_HIGH - R_LOW)


def vol_scale(sigma_market: float, mode: str = "linear",
              lo: float = 0.25, hi: float = 2.0) -> float:
    """Volatility scaling of leverage. 'linear' = σ_std/σ_m (legacy); 'squared' = Merton-consistent."""
    ratio = SIGMA_STD / max(sigma_market, 0.01)
    scale = ratio * ratio if mode == "squared" else ratio
    return min(max(scale, lo), hi)


@dataclass
class CorridorState:
    rm: float
    beta: float
    gamma: float
    vol_scale: float
    leverage_raw: float
    cap: float
    leverage: float
    position: str          # BELOW / WITHIN / ABOVE

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def target_leverage(rm: float, sigma_market: float, cap: float = 2.0,
                    floor: float = BETA_LOW * EXECUTION_MULTIPLIER,
                    vol_mode: str = "linear") -> CorridorState:
    """Full corridor evaluation: β, γ and capped leverage for the current Rm and volatility.

    cap   — the binding upper limit (regime cap / burst cap / hard gross-leverage cap, whichever
            is lowest — chosen by the caller).
    floor — the hedge floor (default −0.136 = β_L × 4.7).
    """
    b = corridor_beta(rm)
    g = corridor_gamma(rm)
    vs = vol_scale(sigma_market, vol_mode)
    raw = b * EXECUTION_MULTIPLIER * vs
    lev = min(max(raw, floor), cap)
    pos = "BELOW" if rm < R_LOW else ("ABOVE" if rm > R_HIGH else "WITHIN")
    return CorridorState(rm, b, g, vs, raw, cap, lev, pos)


def should_rebalance(current_leverage: float, target: float, throttle: float = THROTTLE) -> bool:
    return abs(target - current_leverage) > throttle


def annualised_drift(closes: list[float], periods_per_year: int = 252) -> float:
    """Geometric annualised return of a price series (realised Rm estimate)."""
    if len(closes) < 2 or closes[0] <= 0:
        return 0.0
    n = len(closes) - 1
    return (closes[-1] / closes[0]) ** (periods_per_year / n) - 1.0


def realised_vol(closes: list[float], periods_per_year: int = 252) -> float:
    """Annualised realised volatility from close prices."""
    rets = [math.log(b / a) for a, b in zip(closes[:-1], closes[1:]) if a > 0 and b > 0]
    if len(rets) < 2:
        return SIGMA_STD
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var * periods_per_year)
