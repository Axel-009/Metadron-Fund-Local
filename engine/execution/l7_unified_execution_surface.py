"""L7 Unified Execution Surface — Fused continuous execution arm.

Unifies WonderTrader (micro-price + CTA + routing), ExchangeCore (order matching),
PaperBroker/TradierBroker (bookkeeping), and OptionsEngine (derivatives) into one
continuous execution arm that routes ALL tradeable products through Tradier.

Pipeline position:
    All 29 signal types → L7UnifiedExecutionSurface
        ├── Equity orders  → WonderTrader micro-price → ExchangeCore matching → TradierBroker
        ├── Options orders → OptionsEngine Greeks → vol-adjusted routing → TradierBroker
        └── Futures orders → Beta corridor hedge → TradierBroker
    Paper log maintained in parallel for ML learning / backtesting.

Design rules (per CLAUDE.md):
    - try/except on ALL external imports — system runs degraded, never broken
    - Pure-numpy fallbacks — no crashes if optional packages missing
    - Tradier is PRIMARY execution broker for all products
    - Paper log is ALWAYS maintained for learning loop
    - Fixed income / FX / liquidity are for research only — never executed here
"""

from __future__ import annotations

import time
import uuid
import json
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

# Internal imports — all guarded
try:
    from .paper_broker import (
        PaperBroker, OrderSide, OrderType, OrderStatus,
        SignalType, Order, Position, PortfolioState,
    )
except ImportError:
    PaperBroker = None  # type: ignore[assignment,misc]

try:
    from .tradier_broker import TradierBroker
except ImportError:
    TradierBroker = None  # type: ignore[assignment,misc]

try:
    from .wondertrader_engine import WonderTraderEngine, MicroPriceResult, CTASignal
except ImportError:
    WonderTraderEngine = None  # type: ignore[assignment,misc]
    MicroPriceResult = None  # type: ignore[assignment,misc]

try:
    from .exchange_core_engine import ExchangeCoreEngine, OrderAction, EngineOrderType
except ImportError:
    ExchangeCoreEngine = None  # type: ignore[assignment,misc]

try:
    from .options_engine import OptionsEngine
except ImportError:
    OptionsEngine = None  # type: ignore[assignment,misc]

try:
    from .quant_strategy_executor import QuantStrategyExecutor
except ImportError:
    QuantStrategyExecutor = None  # type: ignore[assignment,misc]

try:
    from ..portfolio.beta_corridor import BetaCorridor, BetaAction
except ImportError:
    BetaCorridor = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProductType(str, Enum):
    """Tradeable product types routed through L7."""
    EQUITY = "EQUITY"
    OPTION = "OPTION"
    FUTURE = "FUTURE"


class RoutingStrategy(str, Enum):
    """Order routing algorithm."""
    SMART = "SMART"       # Adaptive: TWAP for large, immediate for small
    TWAP = "TWAP"         # Time-weighted average price
    VWAP = "VWAP"         # Volume-weighted average price
    IMMEDIATE = "IMMEDIATE"  # Direct market order


class ExecutionUrgency(str, Enum):
    """How urgently the order should fill."""
    LOW = "LOW"           # Patient — TWAP over 30 min
    MEDIUM = "MEDIUM"     # Standard — TWAP over 5 min
    HIGH = "HIGH"         # Aggressive — immediate fill
    CRITICAL = "CRITICAL" # Kill-switch — market order NOW


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class L7Order:
    """Unified order flowing through the L7 execution surface."""
    order_id: str = ""
    ticker: str = ""
    product_type: ProductType = ProductType.EQUITY
    side: str = "BUY"          # BUY, SELL, SHORT, COVER
    quantity: int = 0
    limit_price: Optional[float] = None
    signal_type: str = "HOLD"
    routing: RoutingStrategy = RoutingStrategy.SMART
    urgency: ExecutionUrgency = ExecutionUrgency.MEDIUM
    # Options-specific
    option_type: str = ""      # CALL / PUT
    strike: float = 0.0
    expiry: str = ""           # YYYY-MM-DD
    # Futures-specific
    contract: str = ""         # ES, NQ, VX, ZN, etc.
    # Execution results
    fill_price: float = 0.0
    fill_quantity: int = 0
    slippage_bps: float = 0.0
    transaction_cost: float = 0.0
    micro_price: float = 0.0
    cta_signal_strength: float = 0.0
    status: str = "PENDING"    # PENDING, ROUTED, FILLED, PARTIAL, REJECTED
    reason: str = ""
    created_at: str = ""
    filled_at: str = ""
    # TCA fields
    arrival_price: float = 0.0
    implementation_shortfall: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost_bps: float = 0.0

    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())[:12]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id, "ticker": self.ticker,
            "product_type": self.product_type, "side": self.side,
            "quantity": self.quantity, "limit_price": self.limit_price,
            "signal_type": self.signal_type, "routing": self.routing,
            "fill_price": self.fill_price, "fill_quantity": self.fill_quantity,
            "slippage_bps": self.slippage_bps, "transaction_cost": self.transaction_cost,
            "micro_price": self.micro_price, "status": self.status,
            "reason": self.reason, "created_at": self.created_at,
            "filled_at": self.filled_at, "arrival_price": self.arrival_price,
            "implementation_shortfall": self.implementation_shortfall,
            "market_impact_bps": self.market_impact_bps,
            "timing_cost_bps": self.timing_cost_bps,
            "option_type": self.option_type, "strike": self.strike,
            "expiry": self.expiry, "contract": self.contract,
        }


@dataclass
class TCASnapshot:
    """Transaction Cost Analysis snapshot for a single execution."""
    order_id: str = ""
    ticker: str = ""
    product_type: str = "EQUITY"
    side: str = "BUY"
    quantity: int = 0
    arrival_price: float = 0.0
    fill_price: float = 0.0
    # Decomposition
    spread_cost_bps: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost_bps: float = 0.0
    commission_bps: float = 0.0
    total_cost_bps: float = 0.0
    # Implementation shortfall
    implementation_shortfall_usd: float = 0.0
    # Benchmark
    vwap_price: float = 0.0
    vwap_slippage_bps: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class TCAAggregate:
    """Aggregate TCA metrics for dashboard display."""
    total_trades: int = 0
    total_volume_usd: float = 0.0
    avg_spread_cost_bps: float = 0.0
    avg_market_impact_bps: float = 0.0
    avg_timing_cost_bps: float = 0.0
    avg_commission_bps: float = 0.0
    avg_total_cost_bps: float = 0.0
    total_implementation_shortfall_usd: float = 0.0
    # Per product type
    equity_avg_cost_bps: float = 0.0
    option_avg_cost_bps: float = 0.0
    future_avg_cost_bps: float = 0.0
    # Trend (last 20 trades vs prior 20)
    cost_trend: str = "STABLE"  # IMPROVING, STABLE, DEGRADING
    # Best/worst
    best_execution_ticker: str = ""
    worst_execution_ticker: str = ""


# ---------------------------------------------------------------------------
# Transaction Cost Analyzer
# ---------------------------------------------------------------------------

class TransactionCostAnalyzer:
    """Real-time TCA engine — decomposes execution costs per trade.

    Cost components:
        1. Spread cost: half-spread at time of order
        2. Market impact: price movement caused by our order
        3. Timing cost: adverse price movement between decision and fill
        4. Commission: Tradier commission schedule
    """

    # Tradier commission schedule (per contract/share)
    EQUITY_COMMISSION_PER_SHARE = 0.0       # Tradier: $0 equity commissions
    OPTION_COMMISSION_PER_CONTRACT = 0.35   # Tradier: $0.35/contract
    FUTURE_COMMISSION_PER_CONTRACT = 1.50   # Estimated

    # Market impact model: sqrt(qty / ADV) * volatility * impact_coeff
    IMPACT_COEFFICIENT = 0.10

    def __init__(self, max_history: int = 5000):
        self._history: deque[TCASnapshot] = deque(maxlen=max_history)
        self._by_product: Dict[str, List[TCASnapshot]] = {
            "EQUITY": [], "OPTION": [], "FUTURE": [],
        }

    def analyze(
        self,
        order: L7Order,
        arrival_price: float,
        fill_price: float,
        adv_shares: float = 100_000,
        spread_bps: float = 3.0,
        daily_vol: float = 0.02,
        vwap_price: Optional[float] = None,
    ) -> TCASnapshot:
        """Decompose execution costs for a filled order."""
        qty = order.fill_quantity or order.quantity
        notional = abs(qty * fill_price)

        # 1. Spread cost: half the bid-ask spread
        spread_cost_bps = spread_bps / 2.0

        # 2. Market impact: sqrt model
        participation = qty / max(adv_shares, 1)
        impact_bps = self.IMPACT_COEFFICIENT * (participation ** 0.5) * daily_vol * 10_000

        # 3. Timing cost: price drift from arrival to fill
        if arrival_price > 0:
            if order.side in ("BUY", "COVER"):
                timing_bps = (fill_price - arrival_price) / arrival_price * 10_000
            else:
                timing_bps = (arrival_price - fill_price) / arrival_price * 10_000
        else:
            timing_bps = 0.0

        # 4. Commission
        if order.product_type == ProductType.OPTION:
            commission_usd = abs(qty) * self.OPTION_COMMISSION_PER_CONTRACT
        elif order.product_type == ProductType.FUTURE:
            commission_usd = abs(qty) * self.FUTURE_COMMISSION_PER_CONTRACT
        else:
            commission_usd = abs(qty) * self.EQUITY_COMMISSION_PER_SHARE
        commission_bps = (commission_usd / max(notional, 1)) * 10_000

        total_bps = spread_cost_bps + impact_bps + max(timing_bps, 0) + commission_bps

        # Implementation shortfall
        if arrival_price > 0 and order.side in ("BUY", "COVER"):
            is_usd = (fill_price - arrival_price) * qty
        elif arrival_price > 0:
            is_usd = (arrival_price - fill_price) * qty
        else:
            is_usd = 0.0

        # VWAP slippage
        vwap_slip = 0.0
        if vwap_price and vwap_price > 0:
            if order.side in ("BUY", "COVER"):
                vwap_slip = (fill_price - vwap_price) / vwap_price * 10_000
            else:
                vwap_slip = (vwap_price - fill_price) / vwap_price * 10_000

        snap = TCASnapshot(
            order_id=order.order_id,
            ticker=order.ticker,
            product_type=order.product_type,
            side=order.side,
            quantity=qty,
            arrival_price=arrival_price,
            fill_price=fill_price,
            spread_cost_bps=round(spread_cost_bps, 2),
            market_impact_bps=round(impact_bps, 2),
            timing_cost_bps=round(timing_bps, 2),
            commission_bps=round(commission_bps, 2),
            total_cost_bps=round(total_bps, 2),
            implementation_shortfall_usd=round(is_usd, 2),
            vwap_price=vwap_price or 0.0,
            vwap_slippage_bps=round(vwap_slip, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._history.append(snap)
        pt = order.product_type if isinstance(order.product_type, str) else order.product_type.value
        if pt in self._by_product:
            self._by_product[pt].append(snap)

        return snap

    def get_aggregate(self, last_n: int = 0) -> TCAAggregate:
        """Compute aggregate TCA metrics. last_n=0 means all history."""
        trades = list(self._history)
        if last_n > 0:
            trades = trades[-last_n:]
        if not trades:
            return TCAAggregate()

        agg = TCAAggregate(total_trades=len(trades))
        agg.total_volume_usd = sum(abs(t.quantity * t.fill_price) for t in trades)
        agg.avg_spread_cost_bps = _mean([t.spread_cost_bps for t in trades])
        agg.avg_market_impact_bps = _mean([t.market_impact_bps for t in trades])
        agg.avg_timing_cost_bps = _mean([t.timing_cost_bps for t in trades])
        agg.avg_commission_bps = _mean([t.commission_bps for t in trades])
        agg.avg_total_cost_bps = _mean([t.total_cost_bps for t in trades])
        agg.total_implementation_shortfall_usd = sum(
            t.implementation_shortfall_usd for t in trades
        )

        # Per product
        for pt in ("EQUITY", "OPTION", "FUTURE"):
            pt_trades = [t for t in trades if t.product_type == pt]
            if pt_trades:
                avg = _mean([t.total_cost_bps for t in pt_trades])
                setattr(agg, f"{pt.lower()}_avg_cost_bps", avg)

        # Trend detection
        if len(trades) >= 40:
            recent = _mean([t.total_cost_bps for t in trades[-20:]])
            prior = _mean([t.total_cost_bps for t in trades[-40:-20]])
            if recent < prior * 0.90:
                agg.cost_trend = "IMPROVING"
            elif recent > prior * 1.10:
                agg.cost_trend = "DEGRADING"

        # Best/worst
        if trades:
            best = min(trades, key=lambda t: t.total_cost_bps)
            worst = max(trades, key=lambda t: t.total_cost_bps)
            agg.best_execution_ticker = best.ticker
            agg.worst_execution_ticker = worst.ticker

        return agg

    @property
    def history(self) -> List[TCASnapshot]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Multi-Product Router
# ---------------------------------------------------------------------------

# Research-only instruments — NEVER executed
RESEARCH_ONLY_PREFIXES = frozenset({
    "DX",   # Dollar index
    "6E", "6J", "6B", "6A", "6C", "6S",  # FX futures
    "ZN", "ZB", "ZF", "ZT",  # Treasury futures (used for beta corridor calc only)
    "TLT", "IEF", "SHY", "BND", "AGG",  # Bond ETFs
    "LQD", "VCIT", "VCSH", "HYG", "JNK",  # Credit
    "MBB", "VMBS",  # MBS
})

# Futures that ARE tradeable via Tradier (equity index + VIX)
TRADEABLE_FUTURES = frozenset({"ES", "NQ", "YM", "RTY", "VX"})


class MultiProductRouter:
    """Routes orders by product type through the appropriate execution path.

    All products route to TradierBroker as primary execution broker.
    Paper log is always maintained in parallel.

    Routing paths:
        EQUITY:  → WonderTrader micro-price → ExchangeCore matching → Tradier
        OPTION:  → OptionsEngine Greeks check → vol-adjusted limit → Tradier
        FUTURE:  → Beta corridor validation → Tradier
    """

    def __init__(self):
        self._route_counts: Dict[str, int] = {
            "EQUITY": 0, "OPTION": 0, "FUTURE": 0, "REJECTED": 0,
        }

    def classify(self, order: L7Order) -> ProductType:
        """Auto-classify product type if not explicitly set."""
        ticker = order.ticker.upper()

        # Options: has strike + expiry
        if order.option_type and order.strike > 0:
            return ProductType.OPTION

        # Futures: known contracts
        if order.contract or ticker in TRADEABLE_FUTURES:
            return ProductType.FUTURE

        # Default: equity
        return ProductType.EQUITY

    def is_research_only(self, ticker: str) -> bool:
        """Check if instrument is research-only (FI, FX, credit)."""
        upper = ticker.upper()
        return upper in RESEARCH_ONLY_PREFIXES or any(
            upper.startswith(p) for p in RESEARCH_ONLY_PREFIXES
        )

    def determine_routing(self, order: L7Order) -> RoutingStrategy:
        """Select routing algo based on order size and urgency."""
        if order.urgency == ExecutionUrgency.CRITICAL:
            return RoutingStrategy.IMMEDIATE
        if order.urgency == ExecutionUrgency.HIGH:
            return RoutingStrategy.IMMEDIATE

        # Large orders get TWAP/VWAP
        notional = order.quantity * (order.limit_price or order.arrival_price or 100)
        if notional > 50_000:
            return RoutingStrategy.TWAP
        if notional > 10_000:
            return RoutingStrategy.SMART

        return RoutingStrategy.IMMEDIATE

    def determine_urgency(
        self,
        signal_type: str,
        cta_strength: float = 0.0,
        kill_switch: bool = False,
    ) -> ExecutionUrgency:
        """Infer urgency from signal context."""
        if kill_switch:
            return ExecutionUrgency.CRITICAL

        # High urgency signals
        high_urgency = {
            "MICRO_PRICE_BUY", "MICRO_PRICE_SELL",
            "EVENT_MERGER_ARB", "DISTRESS_FALLEN_ANGEL",
        }
        if signal_type in high_urgency or cta_strength > 0.8:
            return ExecutionUrgency.HIGH

        # Low urgency
        low_urgency = {"HOLD", "QUALITY_BUY", "QUALITY_SELL"}
        if signal_type in low_urgency:
            return ExecutionUrgency.LOW

        return ExecutionUrgency.MEDIUM

    def record_route(self, product_type: str):
        key = product_type if product_type in self._route_counts else "REJECTED"
        self._route_counts[key] = self._route_counts.get(key, 0) + 1

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._route_counts)


# ---------------------------------------------------------------------------
# Slippage & Transaction Cost Model
# ---------------------------------------------------------------------------

class SlippageModel:
    """Estimates and applies slippage to fill prices.

    Components:
        1. Bid-ask spread (calibrated per product type)
        2. Market impact (square-root model)
        3. Timing delay cost
    """

    # Default half-spreads in bps by product
    DEFAULT_HALF_SPREAD = {
        ProductType.EQUITY: 1.5,
        ProductType.OPTION: 15.0,   # Options have wider spreads
        ProductType.FUTURE: 0.5,    # Index futures are tight
    }

    # Impact coefficients by product
    IMPACT_COEFF = {
        ProductType.EQUITY: 0.10,
        ProductType.OPTION: 0.20,
        ProductType.FUTURE: 0.05,
    }

    def estimate_slippage_bps(
        self,
        order: L7Order,
        adv_shares: float = 100_000,
        daily_vol: float = 0.02,
    ) -> float:
        """Pre-trade slippage estimate in bps."""
        pt = order.product_type if isinstance(order.product_type, ProductType) else ProductType.EQUITY
        half_spread = self.DEFAULT_HALF_SPREAD.get(pt, 1.5)
        impact_coeff = self.IMPACT_COEFF.get(pt, 0.10)

        participation = order.quantity / max(adv_shares, 1)
        impact = impact_coeff * (participation ** 0.5) * daily_vol * 10_000

        return half_spread + impact

    def apply_slippage(
        self,
        price: float,
        side: str,
        slippage_bps: float,
    ) -> float:
        """Apply slippage to mid price to get realistic fill price."""
        slip_frac = slippage_bps / 10_000
        if side in ("BUY", "COVER"):
            return price * (1 + slip_frac)
        else:
            return price * (1 - slip_frac)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
