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


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AllocationRules:
    """Portfolio allocation bucket constraints from the Metadron Capital spec."""
    max_drawdown_kill_switch: float = 0.20
    single_name_ig_pct: float = 0.30
    single_name_hy_distressed_pct: float = 0.20
    div_cashflow_etf_pct: float = 0.15
    fi_macro_pct: float = 0.05
    event_driven_cvr_pct: float = 0.05
    options_notional_pct: float = 0.25  # IG 10%, HY 10%, Distressed 5%
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
    """A signal discovered during a universe scan."""
    ticker: str
    signal_type: str  # BUY, SELL, RV_LONG, etc.
    instrument_type: str  # EQUITY, OPTION, FUTURE, ETF, FIXED_INCOME
    confidence: float  # 0.0 - 1.0
    alpha_score: float  # Expected alpha
    regime_context: str  # BULL, BEAR, TRANSITION, etc.
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
    """A sized position allocation."""
    ticker: str
    bucket: str
    instrument_type: str
    position_size: float  # fraction of NAV
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
    """Complete allocation output from the engine."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Kill Switch Monitor
# ═══════════════════════════════════════════════════════════════════════════

class KillSwitchMonitor:
    """Tracks portfolio drawdown and triggers kill switch at 20% from HWM.

    When triggered, all new orders are halted until manual reset.
    Logs all kill switch events with timestamp.
    """

    def __init__(self, max_drawdown: float = 0.20):
        self.max_drawdown = max_drawdown
        self.high_water_mark: float = 0.0
        self.triggered: bool = False
        self.events: list[dict] = []
        self._trigger_timestamp: Optional[str] = None

    def check(self, current_nav: float, hwm: Optional[float] = None) -> bool:
        """Check if kill switch should trigger.

        Returns True if drawdown exceeds max_drawdown threshold.
        """
        if hwm is not None:
            self.high_water_mark = max(self.high_water_mark, hwm)
        self.high_water_mark = max(self.high_water_mark, current_nav)

        if self.high_water_mark <= 0:
            return self.triggered

        drawdown = (self.high_water_mark - current_nav) / self.high_water_mark

        if drawdown >= self.max_drawdown and not self.triggered:
            self.triggered = True
            self._trigger_timestamp = datetime.now(timezone.utc).isoformat()
            event = {
                "event": "KILL_SWITCH_TRIGGERED",
                "drawdown": round(drawdown, 4),
                "current_nav": round(current_nav, 2),
                "high_water_mark": round(self.high_water_mark, 2),
                "timestamp": self._trigger_timestamp,
            }
            self.events.append(event)
            logger.critical("KILL SWITCH TRIGGERED — drawdown %.2f%% from HWM $%.2f",
                            drawdown * 100, self.high_water_mark)

        return self.triggered

    def reset(self):
        """Manual reset after operator review."""
        if self.triggered:
            event = {
                "event": "KILL_SWITCH_RESET",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "was_triggered_at": self._trigger_timestamp,
            }
            self.events.append(event)
            logger.info("Kill switch manually reset.")
        self.triggered = False
        self._trigger_timestamp = None

    def current_drawdown(self, current_nav: float) -> float:
        """Return current drawdown as a fraction."""
        if self.high_water_mark <= 0:
            return 0.0
        return max(0.0, (self.high_water_mark - current_nav) / self.high_water_mark)

    def status(self, current_nav: float = 0.0) -> dict:
        return {
            "triggered": self.triggered,
            "high_water_mark": round(self.high_water_mark, 2),
            "current_drawdown": round(self.current_drawdown(current_nav), 4),
            "max_drawdown_threshold": self.max_drawdown,
            "trigger_timestamp": self._trigger_timestamp,
            "total_events": len(self.events),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Beta Corridor Engine (Allocation-level)
# ═══════════════════════════════════════════════════════════════════════════

class BetaCorridorEngine:
    """Computes portfolio beta vs SPY and outputs corridor level with leverage multiplier.

    Corridor levels:
        HIGH  → leverage multiplier 0.5x (reduce exposure)
        NEUTRAL → leverage multiplier 1.0x (normal)
        LOW   → leverage multiplier 1.5x (increase exposure)

    The multiplier gates margin bucket deployment.
    """

    # Thresholds for corridor classification
    HIGH_BETA_THRESHOLD = 1.3
    LOW_BETA_THRESHOLD = 0.7

    MULTIPLIERS = {
        BetaCorridorLevel.HIGH: 0.5,
        BetaCorridorLevel.NEUTRAL: 1.0,
        BetaCorridorLevel.LOW: 1.5,
    }

    def __init__(self):
        self.current_beta: float = 1.0
        self.corridor: BetaCorridorLevel = BetaCorridorLevel.NEUTRAL
        self.leverage_multiplier: float = 1.0
        self._history: list[dict] = []

    def compute(self, portfolio_returns: Optional[np.ndarray] = None,
                spy_returns: Optional[np.ndarray] = None,
                beta_override: Optional[float] = None,
                backtest: bool = False) -> dict:
        """Compute beta corridor from portfolio vs SPY returns.

        Args:
            portfolio_returns: Array of portfolio daily returns.
            spy_returns: Array of SPY daily returns.
            beta_override: Direct beta value (skip computation).
            backtest: If True, uses historical data context.

        Returns:
            Dict with corridor, multiplier, beta.
        """
        if beta_override is not None:
            beta = beta_override
        elif portfolio_returns is not None and spy_returns is not None:
            min_len = min(len(portfolio_returns), len(spy_returns))
            if min_len < 5:
                beta = 1.0
            else:
                pr = portfolio_returns[-min_len:]
                sr = spy_returns[-min_len:]
                cov = np.cov(pr, sr)
                var_spy = cov[1, 1]
                beta = cov[0, 1] / var_spy if var_spy > 1e-10 else 1.0
        else:
            beta = self.current_beta

        self.current_beta = round(float(beta), 4)

        if beta >= self.HIGH_BETA_THRESHOLD:
            self.corridor = BetaCorridorLevel.HIGH
        elif beta <= self.LOW_BETA_THRESHOLD:
            self.corridor = BetaCorridorLevel.LOW
        else:
            self.corridor = BetaCorridorLevel.NEUTRAL

        self.leverage_multiplier = self.MULTIPLIERS[self.corridor]

        record = {
            "beta": self.current_beta,
            "corridor": self.corridor.value,
            "leverage_multiplier": self.leverage_multiplier,
            "backtest": backtest,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._history.append(record)

        return record

    def status(self) -> dict:
        return {
            "beta": self.current_beta,
            "corridor": self.corridor.value,
            "leverage_multiplier": self.leverage_multiplier,
            "history_length": len(self._history),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Allocation Engine
# ═══════════════════════════════════════════════════════════════════════════

# ETF tickers for DRIP rule enforcement
DRIP_ETFS = {"TLTW", "QQQ", "SPY", "IWM", "HYG", "LQD", "TLT", "PDBC", "GLD",
             "USO", "XLE", "XLB", "XLI", "XLY", "XLP", "XLV", "XLF", "XLK",
             "XLC", "XLU", "XLRE", "IEF", "SHY", "EMB", "BKLN"}

# IG vs HY classification heuristic — tickers known as HY/distressed
# In production this would come from credit rating APIs
HY_DISTRESSED_SIGNALS = {"DISTRESS_FALLEN_ANGEL", "DISTRESS_RECOVERY", "FALLEN_ANGEL_BUY"}
EVENT_CVR_SIGNALS = {"CVR_BUY", "CVR_SELL", "EVENT_MERGER_ARB", "EVENT_CATALYST",
                     "PEAD_LONG", "PEAD_SHORT"}
FI_SIGNAL_TYPES = {"FI_BUY", "FI_SELL", "MACRO_RV", "CREDIT_SIGNAL"}


class AllocationEngine:
    """Core allocation engine implementing Metadron Capital allocation rules.

    Classifies opportunities into buckets, sizes positions within constraints,
    aggregates across universe runs, and validates against kill switch.

    All methods respect the backtest flag for historical simulation.
    """

    def __init__(self, nav: float = 1_000_000, rules: Optional[AllocationRules] = None,
                 backtest: bool = False):
        self.nav = nav
        self.rules = rules or AllocationRules()
        self.backtest = backtest
        self.kill_switch = KillSwitchMonitor(max_drawdown=self.rules.max_drawdown_kill_switch)
        self.beta_engine = BetaCorridorEngine()
        self.drip_log: list[dict] = []
        self.rule_change_log: list[dict] = []

        # Current bucket utilization (fraction of NAV)
        self._utilization: dict[str, float] = {
            BucketType.IG_EQUITY.value: 0.0,
            BucketType.HY_DISTRESSED.value: 0.0,
            BucketType.DIV_CASHFLOW_ETF.value: 0.0,
            BucketType.FI_MACRO.value: 0.0,
            BucketType.EVENT_DRIVEN_CVR.value: 0.0,
            BucketType.OPTIONS_IG.value: 0.0,
            BucketType.OPTIONS_HY.value: 0.0,
            BucketType.OPTIONS_DISTRESSED.value: 0.0,
            BucketType.MONEY_MARKET.value: 0.0,
            BucketType.MARGIN.value: 0.0,
        }

    def classify_opportunity(self, signal: ScanSignal) -> str:
        """Classify a scanner signal into the appropriate allocation bucket.

        Classification logic:
        1. Check instrument type first (options, FI, ETF, futures)
        2. Then check signal type for event/CVR/distressed
        3. Default: IG equity for standard equities
        """
        itype = signal.instrument_type.upper()
        stype = signal.signal_type.upper()

        # Options classification
        if itype == InstrumentType.OPTION.value:
            if stype in HY_DISTRESSED_SIGNALS:
                return BucketType.OPTIONS_DISTRESSED.value
            if stype in EVENT_CVR_SIGNALS or stype in FI_SIGNAL_TYPES:
                return BucketType.OPTIONS_HY.value
            return BucketType.OPTIONS_IG.value

        # Futures / derivatives → margin bucket
        if itype in (InstrumentType.FUTURE.value, InstrumentType.DERIVATIVE.value):
            return BucketType.MARGIN.value

        # Fixed income
        if itype == InstrumentType.FIXED_INCOME.value:
            if stype in EVENT_CVR_SIGNALS:
                return BucketType.EVENT_DRIVEN_CVR.value
            return BucketType.FI_MACRO.value

        # ETFs
        if itype == InstrumentType.ETF.value:
            if signal.ticker in DRIP_ETFS:
                return BucketType.DIV_CASHFLOW_ETF.value
            return BucketType.DIV_CASHFLOW_ETF.value

        # Equity classification by signal type
        if stype in HY_DISTRESSED_SIGNALS:
            return BucketType.HY_DISTRESSED.value
        if stype in EVENT_CVR_SIGNALS:
            return BucketType.EVENT_DRIVEN_CVR.value
        if stype in FI_SIGNAL_TYPES:
            return BucketType.FI_MACRO.value

        # Default: IG equity
        return BucketType.IG_EQUITY.value

    def _bucket_limit(self, bucket: str) -> float:
        """Return the allocation limit for a bucket as fraction of NAV."""
        limits = {
            BucketType.IG_EQUITY.value: self.rules.single_name_ig_pct,
            BucketType.HY_DISTRESSED.value: self.rules.single_name_hy_distressed_pct,
            BucketType.DIV_CASHFLOW_ETF.value: self.rules.div_cashflow_etf_pct,
            BucketType.FI_MACRO.value: self.rules.fi_macro_pct,
            BucketType.EVENT_DRIVEN_CVR.value: self.rules.event_driven_cvr_pct,
            BucketType.OPTIONS_IG.value: self.rules.options_ig_pct,
            BucketType.OPTIONS_HY.value: self.rules.options_hy_pct,
            BucketType.OPTIONS_DISTRESSED.value: self.rules.options_distressed_pct,
            BucketType.MONEY_MARKET.value: self.rules.money_market_pct,
            BucketType.MARGIN.value: self.rules.margin_real_capital_range[1],
        }
        return limits.get(bucket, 0.05)

    def size_position(self, signal: ScanSignal, bucket: str,
                      current_utilization: Optional[float] = None,
                      beta_multiplier: float = 1.0) -> float:
        """Size a position within bucket constraints.

        Returns position size as fraction of NAV.
        Respects bucket limits, confidence scaling, and beta corridor multiplier.
        """
        limit = self._bucket_limit(bucket)
        used = current_utilization if current_utilization is not None else self._utilization.get(bucket, 0.0)
        remaining = max(0.0, limit - used)

        if remaining <= 0:
            return 0.0

        # Base size: scale by confidence and alpha score
        base_size = min(0.05, remaining)  # max 5% per position
        confidence_scale = max(0.2, min(1.0, signal.confidence))
        alpha_scale = max(0.3, min(1.5, 1.0 + signal.alpha_score))

        size = base_size * confidence_scale * alpha_scale

        # Apply beta corridor multiplier for margin-related buckets
        if bucket in (BucketType.OPTIONS_IG.value, BucketType.OPTIONS_HY.value,
                       BucketType.OPTIONS_DISTRESSED.value, BucketType.MARGIN.value):
            size *= beta_multiplier

        # Clamp to remaining capacity
        size = min(size, remaining)

        return round(size, 6)

    def apply_rules(self, opportunity_list: list[ScanSignal],
                    backtest: bool = False) -> AllocationSlate:
        """Full allocation pipeline: classify, size, and build allocation slate.

        Args:
            opportunity_list: List of scan signals to allocate.
            backtest: If True, applies rules in backtest context.

        Returns:
            AllocationSlate with sized positions.
        """
        use_backtest = backtest or self.backtest
        slate = AllocationSlate()
        slate.beta_corridor = self.beta_engine.corridor.value
        slate.leverage_multiplier = self.beta_engine.leverage_multiplier

        # Sort by alpha_score descending for priority allocation
        sorted_opps = sorted(opportunity_list, key=lambda s: s.alpha_score, reverse=True)

        for signal in sorted_opps:
            bucket = self.classify_opportunity(signal)
            signal.bucket = bucket

            size = self.size_position(
                signal, bucket,
                current_utilization=self._utilization.get(bucket, 0.0),
                beta_multiplier=self.beta_engine.leverage_multiplier,
            )

            if size <= 0:
                continue

            allocation = PositionAllocation(
                ticker=signal.ticker,
                bucket=bucket,
                instrument_type=signal.instrument_type,
                position_size=size,
                dollar_amount=size * self.nav,
                confidence=signal.confidence,
                alpha_score=signal.alpha_score,
                signal_type=signal.signal_type,
                regime_context=signal.regime_context,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            slate.positions.append(allocation)
            self._utilization[bucket] = self._utilization.get(bucket, 0.0) + size

            # Enforce DRIP rule: log ETF distributions for reinvestment
            if bucket == BucketType.DIV_CASHFLOW_ETF.value and self.rules.drip_rule:
                self.drip_log.append({
                    "ticker": signal.ticker,
                    "action": "REINVEST_DISTRIBUTION",
                    "size": size,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "backtest": use_backtest,
                })

        # Update slate totals
        for bucket_type in BucketType:
            val = self._utilization.get(bucket_type.value, 0.0)
            if bucket_type == BucketType.IG_EQUITY:
                slate.total_ig_equity = val
            elif bucket_type == BucketType.HY_DISTRESSED:
                slate.total_hy_distressed = val
            elif bucket_type == BucketType.DIV_CASHFLOW_ETF:
                slate.total_etf = val
            elif bucket_type == BucketType.FI_MACRO:
                slate.total_fi_macro = val
            elif bucket_type == BucketType.EVENT_DRIVEN_CVR:
                slate.total_event_cvr = val
            elif bucket_type in (BucketType.OPTIONS_IG, BucketType.OPTIONS_HY, BucketType.OPTIONS_DISTRESSED):
                slate.total_options_notional += val
            elif bucket_type == BucketType.MARGIN:
                slate.total_margin_real = val
            elif bucket_type == BucketType.MONEY_MARKET:
                slate.total_money_market = val

        slate.timestamp = datetime.now(timezone.utc).isoformat()
        return slate

    def aggregate_runs(self, *runs: AllocationSlate) -> AllocationSlate:
        """Aggregate and de-duplicate across multiple universe run slates.

        De-duplicates by ticker (keeps highest alpha_score).
        Re-ranks by alpha potential + regime fit.
        """
        seen: dict[str, PositionAllocation] = {}

        for run in runs:
            for pos in run.positions:
                p = pos if isinstance(pos, PositionAllocation) else pos
                key = f"{p.ticker}:{p.bucket}"
                existing = seen.get(key)
                if existing is None or p.alpha_score > existing.alpha_score:
                    seen[key] = p

        # Re-rank by alpha score
        ranked = sorted(seen.values(), key=lambda p: p.alpha_score, reverse=True)

        final = AllocationSlate()
        final.positions = ranked
        final.beta_corridor = self.beta_engine.corridor.value
        final.leverage_multiplier = self.beta_engine.leverage_multiplier

        # Recalculate totals
        for p in ranked:
            bucket = p.bucket
            if bucket == BucketType.IG_EQUITY.value:
                final.total_ig_equity += p.position_size
            elif bucket == BucketType.HY_DISTRESSED.value:
                final.total_hy_distressed += p.position_size
            elif bucket == BucketType.DIV_CASHFLOW_ETF.value:
                final.total_etf += p.position_size
            elif bucket == BucketType.FI_MACRO.value:
                final.total_fi_macro += p.position_size
            elif bucket == BucketType.EVENT_DRIVEN_CVR.value:
                final.total_event_cvr += p.position_size
            elif bucket in (BucketType.OPTIONS_IG.value, BucketType.OPTIONS_HY.value, BucketType.OPTIONS_DISTRESSED.value):
                final.total_options_notional += p.position_size
            elif bucket == BucketType.MARGIN.value:
                final.total_margin_real += p.position_size
            elif bucket == BucketType.MONEY_MARKET.value:
                final.total_money_market += p.position_size

        final.timestamp = datetime.now(timezone.utc).isoformat()
        return final

    def validate_against_kill_switch(self, slate: AllocationSlate,
                                     current_nav: float) -> AllocationSlate:
        """Validate slate against kill switch. Returns slate or HALT slate."""
        triggered = self.kill_switch.check(current_nav)

        if triggered:
            halt_slate = AllocationSlate()
            halt_slate.kill_switch_triggered = True
            halt_slate.positions = []
            halt_slate.beta_corridor = slate.beta_corridor
            halt_slate.leverage_multiplier = 0.0
            halt_slate.phase = "HALTED"
            halt_slate.timestamp = datetime.now(timezone.utc).isoformat()
            logger.critical("Allocation HALTED — kill switch active.")
            return halt_slate

        slate.kill_switch_triggered = False
        return slate

    def update_rules(self, new_rules: dict) -> AllocationRules:
        """Update allocation rules from operator instructions."""
        for key, value in new_rules.items():
            if hasattr(self.rules, key):
                setattr(self.rules, key, value)

        self.rules.timestamp = datetime.now(timezone.utc).isoformat()
        self.rule_change_log.append({
            "changes": new_rules,
            "timestamp": self.rules.timestamp,
        })
        logger.info("Allocation rules updated: %s", new_rules)
        return self.rules

    def get_status(self, current_nav: float = 0.0) -> dict:
        """Full engine status snapshot."""
        return {
            "rules": self.rules.to_dict(),
            "bucket_utilization": dict(self._utilization),
            "kill_switch": self.kill_switch.status(current_nav),
            "beta_corridor": self.beta_engine.status(),
            "drip_events": len(self.drip_log),
            "rule_changes": len(self.rule_change_log),
            "nav": self.nav,
            "backtest": self.backtest,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
