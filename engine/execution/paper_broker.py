"""PaperBroker — Simulated execution engine using Yahoo Finance data.

Not connected to any live broker. Provides:
    - Order placement (market, limit)
    - Position tracking
    - P&L calculation
    - NAV computation
    - Trade history / audit trail
    - Fill simulation with micro-price model

Designed to be swapped for a live broker API later.
"""

import uuid
import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd

from ..data.yahoo_data import get_adj_close


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class SignalType(str, Enum):
    """15 signal types."""
    MICRO_PRICE_BUY = "MICRO_PRICE_BUY"
    MICRO_PRICE_SELL = "MICRO_PRICE_SELL"
    RV_LONG = "RV_LONG"
    RV_SHORT = "RV_SHORT"
    FALLEN_ANGEL_BUY = "FALLEN_ANGEL_BUY"
    ML_AGENT_BUY = "ML_AGENT_BUY"
    ML_AGENT_SELL = "ML_AGENT_SELL"
    DRL_AGENT_BUY = "DRL_AGENT_BUY"
    DRL_AGENT_SELL = "DRL_AGENT_SELL"
    TFT_BUY = "TFT_BUY"
    TFT_SELL = "TFT_SELL"
    MC_BUY = "MC_BUY"
    MC_SELL = "MC_SELL"
    QUALITY_BUY = "QUALITY_BUY"
    QUALITY_SELL = "QUALITY_SELL"
    HOLD = "HOLD"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Order:
    id: str = ""
    ticker: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    limit_price: Optional[float] = None
    fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    signal_type: SignalType = SignalType.HOLD
    timestamp: str = ""
    fill_timestamp: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Enum) else v for k, v in asdict(self).items()}


@dataclass
class Position:
    ticker: str = ""
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    sector: str = ""

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.avg_cost


@dataclass
class PortfolioState:
    cash: float = 1_000_000.0
    positions: dict = field(default_factory=dict)  # ticker → Position
    nav: float = 1_000_000.0
    total_pnl: float = 0.0
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    beta: float = 0.0


# ---------------------------------------------------------------------------
# Paper Broker
# ---------------------------------------------------------------------------
class PaperBroker:
    """Simulated broker using Yahoo Finance prices.

    Drop-in replacement for live broker — same interface.
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        log_dir: Optional[Path] = None,
        slippage_bps: float = 2.0,
    ):
        self.state = PortfolioState(cash=initial_cash, nav=initial_cash)
        self.slippage_bps = slippage_bps
        self.log_dir = log_dir or Path("logs/paper_broker")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._orders: list[Order] = []
        self._trade_log: list[dict] = []

    # --- Order execution -----------------------------------------------------

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        signal_type: SignalType = SignalType.HOLD,
        limit_price: Optional[float] = None,
        reason: str = "",
    ) -> Order:
        """Place and immediately fill a market order using latest Yahoo price."""
        order = Order(
            id=str(uuid.uuid4())[:8],
            ticker=ticker,
            side=side,
            order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
            quantity=quantity,
            limit_price=limit_price,
            signal_type=signal_type,
            timestamp=datetime.now().isoformat(),
            reason=reason,
        )

        # Get current price
        price = self._get_current_price(ticker)
        if price <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = f"No price data for {ticker}"
            self._orders.append(order)
            return order

        # Apply slippage
        slip = price * self.slippage_bps / 10_000
        if side in (OrderSide.BUY, OrderSide.COVER):
            fill_price = price + slip
        else:
            fill_price = price - slip

        # Limit check
        if limit_price:
            if side in (OrderSide.BUY, OrderSide.COVER) and fill_price > limit_price:
                order.status = OrderStatus.CANCELLED
                order.reason = f"Limit {limit_price} < fill {fill_price}"
                self._orders.append(order)
                return order
            if side in (OrderSide.SELL, OrderSide.SHORT) and fill_price < limit_price:
                order.status = OrderStatus.CANCELLED
                order.reason = f"Limit {limit_price} > fill {fill_price}"
                self._orders.append(order)
                return order

        # Execute
        order.fill_price = fill_price
        order.fill_timestamp = datetime.now().isoformat()
        order.status = OrderStatus.FILLED

        self._update_position(order)
        self._orders.append(order)
        self._log_trade(order)

        return order

    def _update_position(self, order: Order):
        """Update positions and cash after fill."""
        ticker = order.ticker
        qty = order.quantity
        price = order.fill_price

        pos = self.state.positions.get(ticker, Position(ticker=ticker))

        if order.side == OrderSide.BUY:
            cost = qty * price
            if cost > self.state.cash:
                qty = int(self.state.cash / price)
                cost = qty * price
            if qty <= 0:
                return
            # Update avg cost
            total_qty = pos.quantity + qty
            if total_qty > 0:
                pos.avg_cost = (pos.avg_cost * pos.quantity + cost) / total_qty
            pos.quantity = total_qty
            self.state.cash -= cost

        elif order.side == OrderSide.SELL:
            sell_qty = min(qty, pos.quantity)
            if sell_qty <= 0:
                return
            proceeds = sell_qty * price
            pnl = (price - pos.avg_cost) * sell_qty
            pos.realized_pnl += pnl
            pos.quantity -= sell_qty
            self.state.cash += proceeds
            self.state.total_pnl += pnl
            if pnl > 0:
                self.state.win_count += 1
            else:
                self.state.loss_count += 1

        elif order.side == OrderSide.SHORT:
            pos.quantity -= qty
            pos.avg_cost = price
            self.state.cash += qty * price

        elif order.side == OrderSide.COVER:
            cover_qty = min(qty, abs(pos.quantity))
            cost = cover_qty * price
            pnl = (pos.avg_cost - price) * cover_qty
            pos.quantity += cover_qty
            pos.realized_pnl += pnl
            self.state.cash -= cost
            self.state.total_pnl += pnl
            if pnl > 0:
                self.state.win_count += 1
            else:
                self.state.loss_count += 1

        pos.current_price = price
        pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

        if pos.quantity != 0:
            self.state.positions[ticker] = pos
        elif ticker in self.state.positions:
            del self.state.positions[ticker]

        self.state.total_trades += 1

    # --- Portfolio state -----------------------------------------------------

    def refresh_prices(self):
        """Update all position prices from Yahoo."""
        tickers = list(self.state.positions.keys())
        if not tickers:
            return
        for ticker in tickers:
            price = self._get_current_price(ticker)
            if price > 0 and ticker in self.state.positions:
                pos = self.state.positions[ticker]
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

    def compute_nav(self) -> float:
        """Compute current NAV."""
        self.refresh_prices()
        positions_value = sum(
            p.quantity * p.current_price
            for p in self.state.positions.values()
        )
        self.state.nav = self.state.cash + positions_value
        return self.state.nav

    def compute_exposures(self) -> dict:
        """Compute gross and net exposure."""
        nav = self.compute_nav()
        if nav <= 0:
            return {"gross": 0, "net": 0, "long": 0, "short": 0}
        long_val = sum(p.market_value for p in self.state.positions.values() if p.quantity > 0)
        short_val = sum(abs(p.market_value) for p in self.state.positions.values() if p.quantity < 0)
        self.state.gross_exposure = (long_val + short_val) / nav
        self.state.net_exposure = (long_val - short_val) / nav
        return {
            "gross": self.state.gross_exposure,
            "net": self.state.net_exposure,
            "long": long_val / nav,
            "short": short_val / nav,
        }

    def get_position(self, ticker: str) -> Optional[Position]:
        return self.state.positions.get(ticker)

    def get_all_positions(self) -> dict[str, Position]:
        return dict(self.state.positions)

    def get_portfolio_summary(self) -> dict:
        """Full portfolio summary."""
        nav = self.compute_nav()
        exp = self.compute_exposures()
        win_rate = (
            self.state.win_count / (self.state.win_count + self.state.loss_count)
            if (self.state.win_count + self.state.loss_count) > 0 else 0
        )
        return {
            "nav": nav,
            "cash": self.state.cash,
            "total_pnl": self.state.total_pnl,
            "total_trades": self.state.total_trades,
            "win_rate": win_rate,
            "positions": len(self.state.positions),
            "gross_exposure": exp["gross"],
            "net_exposure": exp["net"],
        }

    # --- Price fetching ------------------------------------------------------

    def _get_current_price(self, ticker: str) -> float:
        """Get latest price from Yahoo Finance."""
        try:
            prices = get_adj_close(ticker, start=(
                pd.Timestamp.now() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
            )
            if prices.empty:
                return 0.0
            return float(prices.iloc[-1].iloc[0] if isinstance(prices, pd.DataFrame) else prices.iloc[-1])
        except Exception:
            return 0.0

    # --- Logging -------------------------------------------------------------

    def _log_trade(self, order: Order):
        entry = order.to_dict()
        self._trade_log.append(entry)
        log_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_trade_history(self) -> list[dict]:
        return list(self._trade_log)
