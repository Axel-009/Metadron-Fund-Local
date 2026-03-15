"""Beta Corridor Engine — Dataset 1 integration.

Manages portfolio beta within the 7%–12% return corridor.
Thesis: Alpha extracted through IG/Fallen Angel names or RV mispricing.
Beta is the controlled exposure variable, solved from S&P 500 historical 7-12% earnings.

Key parameters:
    ALPHA = 2% secular alpha headstart (selection alpha)
    R_LOW, R_HIGH = 7%, 12% — the "Gamma Corridor"
    BETA_MAX = 2.0 full throttle cap
    BETA_INV = -0.136 strategic hedge floor
    EXECUTION_MULTIPLIER = 4.7 thesis scaling factor
    MIN_TRADE_THRESHOLD = 0.05 gamma throttle (friction control)

Uses Yahoo Finance for market data (paper broker mode).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from ..data.yahoo_data import get_market_stats, get_adj_close


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALPHA = 0.02                  # 2% secular alpha headstart
R_LOW = 0.07                  # Gamma corridor lower bound
R_HIGH = 0.12                 # Gamma corridor upper bound
BETA_MAX = 2.0                # Full throttle cap
BETA_INV = -0.136             # Strategic hedge floor
EXECUTION_MULTIPLIER = 4.7    # Thesis scaling factor
MIN_TRADE_THRESHOLD = 0.05    # Gamma throttle
VOL_STANDARD = 0.15           # Thesis standard vol (15%)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BetaState:
    """Current beta corridor state."""
    current_beta: float = 0.0
    target_beta: float = 0.0
    Rm: float = 0.0
    sigma_m: float = 0.15
    vol_adjustment: float = 1.0
    base_beta: float = 0.0
    regime_beta_cap: float = BETA_MAX
    corridor_position: str = "NEUTRAL"  # BELOW / WITHIN / ABOVE
    last_price: float = 0.0
    timestamp: str = ""


@dataclass
class BetaAction:
    """Rebalance action from the beta engine."""
    action: str = "HOLD"      # BUY / SELL / HOLD
    quantity: int = 0
    instrument: str = "SPY"   # Paper broker uses SPY as proxy
    target_beta: float = 0.0
    current_beta: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Beta Corridor Engine
# ---------------------------------------------------------------------------
class BetaCorridor:
    """Manages portfolio beta within the 7–12% return corridor.

    Paper broker mode: uses SPY as the beta instrument proxy.
    """

    def __init__(
        self,
        nav: float = 1_000_000,
        alpha: float = ALPHA,
        r_low: float = R_LOW,
        r_high: float = R_HIGH,
        beta_max: float = BETA_MAX,
        beta_inv: float = BETA_INV,
        execution_multiplier: float = EXECUTION_MULTIPLIER,
    ):
        self.nav = nav
        self.alpha = alpha
        self.r_low = r_low
        self.r_high = r_high
        self.beta_max = beta_max
        self.beta_inv = beta_inv
        self.execution_multiplier = execution_multiplier
        self.current_beta = 0.0
        self._history: list[BetaState] = []

    def update_market_stats(self) -> dict:
        """Refresh market drift and vol from Yahoo Finance."""
        stats = get_market_stats(benchmark="^GSPC", lookback_years=1)
        return stats

    def calculate_target_beta(
        self,
        Rm: float,
        sigma_m: float,
        regime_beta_cap: Optional[float] = None,
    ) -> BetaState:
        """Convert market drift into target exposure.

        Includes vol-adjustment to maintain drift/diffusion ratio.
        The Smooth Beta Curve (Section VI of thesis).
        """
        state = BetaState()
        state.Rm = Rm
        state.sigma_m = sigma_m
        state.last_price = 0.0
        state.timestamp = datetime.now().isoformat()

        # 1. Base linear interpolation
        if Rm <= self.r_low:
            base_beta = -0.029
        elif Rm >= self.r_high:
            base_beta = 0.425
        else:
            slope = (0.425 - (-0.029)) / (self.r_high - self.r_low)
            base_beta = -0.029 + slope * (Rm - self.r_low)
        state.base_beta = base_beta

        # 2. Vol normalization (thesis standard: 15% vol)
        vol_adj = VOL_STANDARD / max(sigma_m, 0.05)
        state.vol_adjustment = vol_adj

        target = base_beta * self.execution_multiplier * vol_adj

        # 3. Apply caps
        effective_cap = regime_beta_cap if regime_beta_cap is not None else self.beta_max
        state.regime_beta_cap = effective_cap
        target = max(self.beta_inv, min(effective_cap, target))

        state.target_beta = target
        state.current_beta = self.current_beta

        # Corridor position
        if Rm < self.r_low:
            state.corridor_position = "BELOW"
        elif Rm > self.r_high:
            state.corridor_position = "ABOVE"
        else:
            state.corridor_position = "WITHIN"

        self._history.append(state)
        return state

    def compute_rebalance(
        self,
        state: BetaState,
        instrument_price: float = 500.0,
        contract_multiplier: float = 1.0,
    ) -> BetaAction:
        """Translate target beta into paper broker action.

        Uses SPY as proxy instrument for paper trading.
        """
        action = BetaAction()
        action.target_beta = state.target_beta
        action.current_beta = self.current_beta

        # Target notional
        target_notional = state.target_beta * self.nav
        current_notional = self.current_beta * self.nav
        delta_notional = target_notional - current_notional

        # Quantity in shares
        contract_value = instrument_price * contract_multiplier
        if contract_value <= 0:
            action.action = "HOLD"
            return action

        qty_diff = int(delta_notional / contract_value)

        # Gamma throttle: only rebalance if shift > threshold
        if abs(state.target_beta - self.current_beta) > MIN_TRADE_THRESHOLD and qty_diff != 0:
            if qty_diff > 0:
                action.action = "BUY"
                action.quantity = abs(qty_diff)
            else:
                action.action = "SELL"
                action.quantity = abs(qty_diff)
            action.instrument = "SPY"
            action.reason = (
                f"Beta shift {self.current_beta:.3f} → {state.target_beta:.3f} "
                f"| Rm={state.Rm:.2%} σ={state.sigma_m:.2%} "
                f"| corridor={state.corridor_position}"
            )
            self.current_beta = state.target_beta
        else:
            action.action = "HOLD"
            action.reason = f"Delta {abs(state.target_beta - self.current_beta):.3f} < threshold {MIN_TRADE_THRESHOLD}"

        return action

    def run_cycle(self, regime_beta_cap: Optional[float] = None) -> tuple[BetaState, BetaAction]:
        """Full cycle: fetch data → compute beta → generate action."""
        stats = self.update_market_stats()
        state = self.calculate_target_beta(
            Rm=stats["Rm"],
            sigma_m=stats["sigma_m"],
            regime_beta_cap=regime_beta_cap,
        )
        state.last_price = stats["last_price"]
        action = self.compute_rebalance(state, instrument_price=stats["last_price"])
        return state, action

    def get_history(self) -> list[BetaState]:
        return list(self._history)

    def update_nav(self, new_nav: float):
        """Update NAV for dynamic sizing."""
        self.nav = new_nav
