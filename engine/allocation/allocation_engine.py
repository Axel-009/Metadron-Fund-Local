"""Metadron Capital — Allocation Engine.

Implements portfolio allocation rules, position sizing, bucket management,
beta corridor gating, and kill switch enforcement.

All rules apply identically in backtest mode (backtest=True flag).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger("metadron.allocation")

# ─── Prometheus instrumentation ──────────────────────────────────
_prom_alloc = None

def _get_prom_alloc():
    global _prom_alloc
    if _prom_alloc is not None:
        return _prom_alloc
    try:
        from engine.bridges.prometheus_metrics import get_metrics
        _prom_alloc = get_metrics()
    except Exception:
        _prom_alloc = {}
    return _prom_alloc


class BucketType(str, Enum):
    IG_EQUITY = "IG_EQUITY"
    HY_DISTRESSED = "HY_DISTRESSED"
    DIV_CASHFLOW_ETF = "DIV_CASHFLOW_ETF"
    FI_MACRO = "FI_MACRO"
    EVENT_DRIVEN_CVR = "EVENT_DRIVEN_CVR"
    OPTIONS_IG = "OPTIONS_IG"
    OPTIONS_HY = "OPTIONS_HY"
    OPTIONS_DISTRESSED = "OPTIONS_DISTRESSED"
    MONEY_MARKET = "MONEY_MARKET"
    MARGIN = "MARGIN"


class BetaCorridorLevel(str, Enum):
    HIGH = "HIGH"
    NEUTRAL = "NEUTRAL"
    LOW = "LOW"


class CyclePhase(str, Enum):
    SCANNING = "SCANNING"
    AGGREGATING = "AGGREGATING"
    EXECUTING = "EXECUTING"
    RISK_CHECK = "RISK_CHECK"
    COOLDOWN = "COOLDOWN"


class InstrumentType(str, Enum):
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    FUTURE = "FUTURE"
    ETF = "ETF"
    FIXED_INCOME = "FIXED_INCOME"
    DERIVATIVE = "DERIVATIVE"


@dataclass
class AllocationRules:
    max_drawdown_kill_switch: float = 0.20
    single_name_ig_pct: float = 0.30
    single_name_hy_distressed_pct: float = 0.20
    div_cashflow_etf_pct: float = 0.15
    fi_macro_pct: float = 0.05
    event_driven_cvr_pct: float = 0.05
    options_notional_pct: float = 0.25
    options_ig_pct: float = 0.10
    options_hy_pct: float = 0.10
    options_distressed_pct: float = 0.05
    margin_real_capital_range: tuple = (0.05, 0.15)
    money_market_pct: float = 0.05
    drip_rule: bool = True
    alpha_primary_goal: bool = True
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "max_drawdown_kill_switch": self.max_drawdown_kill_switch,
            "single_name_ig_pct": self.single_name_ig_pct,
            "single_name_hy_distressed_pct": self.single_name_hy_distressed_pct,
            "div_cashflow_etf_pct": self.div_cashflow_etf_pct,
            "fi_macro_pct": self.fi_macro_pct,
            "event_driven_cvr_pct": self.event_driven_cvr_pct,
            "options_notional_pct": self.options_notional_pct,
            "options_ig_pct": self.options_ig_pct,
            "options_hy_pct": self.options_hy_pct,
            "options_distressed_pct": self.options_distressed_pct,
            "margin_real_capital_range_low": self.margin_real_capital_range[0],
            "margin_real_capital_range_high": self.margin_real_capital_range[1],
            "money_market_pct": self.money_market_pct,
            "drip_rule": self.drip_rule,
            "alpha_primary_goal": self.alpha_primary_goal,
            "timestamp": self.timestamp,
        }


@dataclass
class ScanSignal:
    ticker: str
    signal_type: str
    instrument_type: str
    confidence: float
    alpha_score: float
    regime_context: str
    bucket: str = ""
    universe: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "signal_type": self.signal_type,
            "instrument_type": self.instrument_type,
            "confidence": self.confidence,
            "alpha_score": self.alpha_score,
            "regime_context": self.regime_context,
            "bucket": self.bucket,
            "universe": self.universe,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class PositionAllocation:
    ticker: str
    bucket: str
    instrument_type: str
    position_size: float
    dollar_amount: float
    confidence: float
    alpha_score: float
    signal_type: str
    regime_context: str
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "bucket": self.bucket,
            "instrument_type": self.instrument_type,
            "position_size": self.position_size,
            "dollar_amount": round(self.dollar_amount, 2),
            "confidence": self.confidence,
            "alpha_score": self.alpha_score,
            "signal_type": self.signal_type,
            "regime_context": self.regime_context,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class AllocationSlate:
    positions: list = field(default_factory=list)
    total_ig_equity: float = 0.0
    total_hy_distressed: float = 0.0
    total_etf: float = 0.0
    total_fi_macro: float = 0.0
    total_event_cvr: float = 0.0
    total_options_notional: float = 0.0
    total_margin_real: float = 0.0
    total_money_market: float = 0.0
    kill_switch_triggered: bool = False
    beta_corridor: str = "NEUTRAL"
    leverage_multiplier: float = 1.0
    cycle_number: int = 0
    phase: str = "SCANNING"
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "positions": [p.to_dict() if hasattr(p, "to_dict") else p for p in self.positions],
            "bucket_utilization": {
                "ig_equity": round(self.total_ig_equity, 4),
                "hy_distressed": round(self.total_hy_distressed, 4),
                "div_cashflow_etf": round(self.total_etf, 4),
                "fi_macro": round(self.total_fi_macro, 4),
                "event_driven_cvr": round(self.total_event_cvr, 4),
                "options_notional": round(self.total_options_notional, 4),
                "margin_real_capital": round(self.total_margin_real, 4),
                "money_market": round(self.total_money_market, 4),
            },
            "kill_switch_triggered": self.kill_switch_triggered,
            "beta_corridor": self.beta_corridor,
            "leverage_multiplier": self.leverage_multiplier,
            "cycle_number": self.cycle_number,
            "phase": self.phase,
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
        }


# ─── Kill Switch Monitor ──────────────────────────────────────────

class KillSwitchMonitor:
    """Monitors drawdown and enforces the kill switch when breached.

    The kill switch is a one-way latch: once triggered it does NOT
    auto-reset.  An operator must call request_reset() followed by
    confirm_reset() (or the legacy reset() alias) to clear it.
    """

    def __init__(self, threshold: float = 0.20):
        self.threshold = threshold
        self._triggered: bool = False
        self._triggered_at: Optional[str] = None
        self.pending_reset: bool = False
        self._reset_requested_timestamp: Optional[str] = None

    # ── state queries ─────────────────────────────────────────────

    @property
    def triggered(self) -> bool:
        return self._triggered

    def is_triggered(self) -> bool:
        return self._triggered

    # ── trigger ───────────────────────────────────────────────────

    def check(self, drawdown: float) -> bool:
        """Evaluate drawdown; latch kill switch if threshold breached."""
        if drawdown >= self.threshold and not self._triggered:
            self._triggered = True
            self._triggered_at = datetime.now(timezone.utc).isoformat()
            self.pending_reset = True  # awaits manual operator action
            logger.critical(
                "KILL SWITCH TRIGGERED: drawdown=%.4f >= threshold=%.4f at %s",
                drawdown,
                self.threshold,
                self._triggered_at,
            )
        return self._triggered

    # ── operator reset flow ───────────────────────────────────────

    def request_reset(self) -> None:
        """Signal operator intent to reset.

        Sets pending_reset=True and records the request timestamp.
        Does NOT clear the triggered latch — confirm_reset() must
        be called to complete the reset.
        """
        self.pending_reset = True
        self._reset_requested_timestamp = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "Kill switch reset REQUESTED by operator at %s (still triggered until confirmed)",
            self._reset_requested_timestamp,
        )

    def confirm_reset(self) -> None:
        """Complete the operator reset cycle.

        Clears both triggered and pending_reset flags and logs the event.
        """
        prev_triggered_at = self._triggered_at
        self._triggered = False
        self._triggered_at = None
        self.pending_reset = False
        confirmed_at = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "Kill switch RESET CONFIRMED by operator at %s (was triggered at %s)",
            confirmed_at,
            prev_triggered_at,
        )
        # Emit Prometheus metric
        try:
            metrics = _get_prom_alloc()
            if "kill_switch_reset_total" in metrics:
                metrics["kill_switch_reset_total"].inc()
            if "kill_switch_pending_reset" in metrics:
                metrics["kill_switch_pending_reset"].set(0)
        except Exception:
            pass

    def reset(self) -> None:
        """Backward-compatible alias for confirm_reset()."""
        self.confirm_reset()

    # ── status ────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "triggered": self._triggered,
            "triggered_at": self._triggered_at,
            "threshold": self.threshold,
            "pending_reset": self.pending_reset,
            "reset_requested_at": self._reset_requested_timestamp,
        }


# ─── Allocation Engine ────────────────────────────────────────────

class AllocationEngine:
    """Core allocation engine — applies rules, sizes positions, enforces limits."""

    def __init__(
        self,
        rules: Optional[AllocationRules] = None,
        backtest: bool = False,
        cycle_number: int = 0,
    ):
        self.rules = rules or AllocationRules()
        self.backtest = backtest
        self.cycle_number = cycle_number
        self.kill_switch = KillSwitchMonitor(
            threshold=self.rules.max_drawdown_kill_switch
        )
        self._beta_corridor: BetaCorridorLevel = BetaCorridorLevel.NEUTRAL
        self._leverage_multiplier: float = 1.0

        # Per-bucket utilization accumulators (fraction of total capital)
        self._utilization: dict[str, float] = {
            BucketType.IG_EQUITY: 0.0,
            BucketType.HY_DISTRESSED: 0.0,
            BucketType.DIV_CASHFLOW_ETF: 0.0,
            BucketType.FI_MACRO: 0.0,
            BucketType.EVENT_DRIVEN_CVR: 0.0,
            BucketType.OPTIONS_IG: 0.0,
            BucketType.OPTIONS_HY: 0.0,
            BucketType.OPTIONS_DISTRESSED: 0.0,
            BucketType.MONEY_MARKET: 0.0,
            BucketType.MARGIN: 0.0,
        }

    # ── bucket utilization ────────────────────────────────────────

    def reset_cycle_utilization(self) -> None:
        """Zero all bucket utilization accumulators.

        Called automatically at the start of each apply_rules() call
        to prevent cross-cycle accumulation (bucket accumulation bug fix).
        """
        for key in self._utilization:
            self._utilization[key] = 0.0
        logger.debug("Allocation bucket utilization reset for new cycle.")
        try:
            metrics = _get_prom_alloc()
            if "allocation_cycle_reset_total" in metrics:
                metrics["allocation_cycle_reset_total"].inc()
        except Exception:
            pass

    # ── beta corridor ─────────────────────────────────────────────

    def set_beta_corridor(self, level: BetaCorridorLevel) -> None:
        self._beta_corridor = level
        if level == BetaCorridorLevel.HIGH:
            self._leverage_multiplier = 0.5
        elif level == BetaCorridorLevel.LOW:
            self._leverage_multiplier = 1.25
        else:
            self._leverage_multiplier = 1.0
        logger.info(
            "Beta corridor set to %s → leverage_multiplier=%.2f",
            level,
            self._leverage_multiplier,
        )

    # ── core rules application ────────────────────────────────────

    def apply_rules(
        self,
        signals: list,
        total_capital: float,
        drawdown: float = 0.0,
        beta_corridor: Optional[BetaCorridorLevel] = None,
    ) -> "AllocationSlate":
        """Apply allocation rules to a list of ScanSignals.

        Resets bucket utilization at the start of each call to prevent
        cross-cycle accumulation (bucket accumulation bug fix).
        """
        # ── FIX: reset utilization at start of every cycle ────────
        self.reset_cycle_utilization()

        if beta_corridor is not None:
            self.set_beta_corridor(beta_corridor)

        self.cycle_number += 1
        slate = AllocationSlate(
            cycle_number=self.cycle_number,
            beta_corridor=self._beta_corridor.value,
            leverage_multiplier=self._leverage_multiplier,
            phase=CyclePhase.RISK_CHECK.value,
        )

        # Kill switch check
        if self.kill_switch.check(drawdown):
            return self.validate_against_kill_switch(slate, total_capital)

        slate.phase = CyclePhase.AGGREGATING.value

        for sig in signals:
            alloc = self._size_position(sig, total_capital)
            if alloc is None:
                continue
            if not self._fits_in_bucket(alloc):
                logger.debug("Position rejected: bucket cap reached for %s", sig.ticker)
                continue
            self._apply_utilization(alloc, total_capital)
            slate.positions.append(alloc)

        # Populate slate totals from utilization
        slate.total_ig_equity = self._utilization[BucketType.IG_EQUITY]
        slate.total_hy_distressed = self._utilization[BucketType.HY_DISTRESSED]
        slate.total_etf = self._utilization[BucketType.DIV_CASHFLOW_ETF]
        slate.total_fi_macro = self._utilization[BucketType.FI_MACRO]
        slate.total_event_cvr = self._utilization[BucketType.EVENT_DRIVEN_CVR]
        slate.total_options_notional = (
            self._utilization[BucketType.OPTIONS_IG]
            + self._utilization[BucketType.OPTIONS_HY]
            + self._utilization[BucketType.OPTIONS_DISTRESSED]
        )
        slate.total_margin_real = self._utilization[BucketType.MARGIN]
        slate.total_money_market = self._utilization[BucketType.MONEY_MARKET]
        slate.phase = CyclePhase.EXECUTING.value
        slate.timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "apply_rules cycle=%d: %d positions accepted from %d signals",
            self.cycle_number,
            len(slate.positions),
            len(signals),
        )
        return slate

    # ── position sizing ───────────────────────────────────────────

    def _size_position(
        self, sig: ScanSignal, total_capital: float
    ) -> Optional[PositionAllocation]:
        """Compute position size and dollar amount for a signal."""
        bucket = sig.bucket or self._infer_bucket(sig)
        cap = self._bucket_cap(bucket)
        if cap is None or cap <= 0:
            return None

        base_size = sig.confidence * cap * self._leverage_multiplier
        base_size = min(base_size, cap)  # never exceed bucket cap
        dollar_amount = base_size * total_capital

        return PositionAllocation(
            ticker=sig.ticker,
            bucket=bucket,
            instrument_type=sig.instrument_type,
            position_size=round(base_size, 6),
            dollar_amount=dollar_amount,
            confidence=sig.confidence,
            alpha_score=sig.alpha_score,
            signal_type=sig.signal_type,
            regime_context=sig.regime_context,
        )

    def _infer_bucket(self, sig: ScanSignal) -> str:
        itype = sig.instrument_type.upper() if sig.instrument_type else ""
        if itype == InstrumentType.OPTION:
            return BucketType.OPTIONS_IG
        if itype == InstrumentType.ETF:
            return BucketType.DIV_CASHFLOW_ETF
        if itype == InstrumentType.FIXED_INCOME:
            return BucketType.FI_MACRO
        regime = (sig.regime_context or "").upper()
        if "DISTRESSED" in regime or "HY" in regime:
            return BucketType.HY_DISTRESSED
        return BucketType.IG_EQUITY

    def _bucket_cap(self, bucket: str) -> Optional[float]:
        caps = {
            BucketType.IG_EQUITY: self.rules.single_name_ig_pct,
            BucketType.HY_DISTRESSED: self.rules.single_name_hy_distressed_pct,
            BucketType.DIV_CASHFLOW_ETF: self.rules.div_cashflow_etf_pct,
            BucketType.FI_MACRO: self.rules.fi_macro_pct,
            BucketType.EVENT_DRIVEN_CVR: self.rules.event_driven_cvr_pct,
            BucketType.OPTIONS_IG: self.rules.options_ig_pct,
            BucketType.OPTIONS_HY: self.rules.options_hy_pct,
            BucketType.OPTIONS_DISTRESSED: self.rules.options_distressed_pct,
            BucketType.MONEY_MARKET: self.rules.money_market_pct,
            BucketType.MARGIN: self.rules.margin_real_capital_range[1],
        }
        return caps.get(bucket)

    # ── bucket fits check ─────────────────────────────────────────

    def _fits_in_bucket(self, alloc: PositionAllocation) -> bool:
        cap = self._bucket_cap(alloc.bucket)
        if cap is None:
            return False
        current = self._utilization.get(alloc.bucket, 0.0)
        return (current + alloc.position_size) <= cap

    def _apply_utilization(
        self, alloc: PositionAllocation, total_capital: float
    ) -> None:
        self._utilization[alloc.bucket] = (
            self._utilization.get(alloc.bucket, 0.0) + alloc.position_size
        )

    # ── kill switch enforcement ───────────────────────────────────

    def validate_against_kill_switch(
        self, slate: AllocationSlate, total_capital: float
    ) -> AllocationSlate:
        """Build a halt slate when kill switch is active.

        Emits Prometheus counters/gauges for observability.
        """
        halt_slate = AllocationSlate(
            positions=[],
            kill_switch_triggered=True,
            beta_corridor=self._beta_corridor.value,
            leverage_multiplier=0.0,
            cycle_number=self.cycle_number,
            phase=CyclePhase.COOLDOWN.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.critical(
            "KILL SWITCH ACTIVE — all allocations halted (cycle=%d)",
            self.cycle_number,
        )

        # ── Prometheus instrumentation ────────────────────────────
        try:
            metrics = _get_prom_alloc()
            if "kill_switch_triggered_total" in metrics:
                metrics["kill_switch_triggered_total"].inc()
            # New gauge: 1 = triggered/awaiting reset, 0 = clear
            if "kill_switch_pending_reset" in metrics:
                metrics["kill_switch_pending_reset"].set(
                    1 if self.kill_switch.pending_reset else 0
                )
        except Exception:
            pass

        return halt_slate

    # ── introspection ─────────────────────────────────────────────

    def get_utilization(self) -> dict:
        return {k: round(v, 6) for k, v in self._utilization.items()}

    def get_rules(self) -> dict:
        return self.rules.to_dict()

    def get_kill_switch_status(self) -> dict:
        return self.kill_switch.status()
