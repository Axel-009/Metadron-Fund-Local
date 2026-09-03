"""broker_types — Canonical execution data types shared by the Schwab broker,
the L7 Unified Execution Surface and the engine API.

Formerly the header of ``paper_broker.py``. The PaperBroker simulation class was
retired when Metadron moved to Schwab as the sole execution + data broker; the
value types (Order, Position, PortfolioState, RiskLimiter, PerformanceTracker,
DailyTargetManager, LiveDashboardState, MicroPriceModel) are unchanged so every
downstream import keeps working via ``engine.execution.broker_types``.

The L7 trade log (logs/l7_execution/trade_log/*.jsonl) is the only ledger used
for reconciliation, learning and back-testing — there is no paper book.
"""

import uuid
import json
import time
import csv
import io
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd

try:
    from ..data.openbb_data import get_adj_close
except Exception:  # pragma: no cover
    get_adj_close = None
from ..utils.money import D, money, to_float, safe_div


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
    DRY_RUN = "DRY_RUN"      # fully risk-checked + logged, deliberately NOT sent to Schwab


class SignalType(str, Enum):
    """Signal types for trade classification."""
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
    SOCIAL_BULLISH = "SOCIAL_BULLISH"
    SOCIAL_BEARISH = "SOCIAL_BEARISH"
    SOCIAL_MOMENTUM = "SOCIAL_MOMENTUM"
    SOCIAL_REVERSAL = "SOCIAL_REVERSAL"
    # Distressed asset signals
    DISTRESS_FALLEN_ANGEL = "DISTRESS_FALLEN_ANGEL"
    DISTRESS_RECOVERY = "DISTRESS_RECOVERY"
    DISTRESS_AVOID = "DISTRESS_AVOID"
    # CVR signals
    CVR_BUY = "CVR_BUY"
    CVR_SELL = "CVR_SELL"
    # Event-driven signals
    EVENT_MERGER_ARB = "EVENT_MERGER_ARB"
    EVENT_PEAD_LONG = "EVENT_PEAD_LONG"
    EVENT_PEAD_SHORT = "EVENT_PEAD_SHORT"
    EVENT_CATALYST = "EVENT_CATALYST"
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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

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
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # ── Decimal-precise financial properties ──────────────────
    @property
    def market_value(self) -> float:
        return to_float(money(D(self.quantity) * D(self.current_price)))

    @property
    def cost_basis(self) -> float:
        return to_float(money(D(abs(self.quantity)) * D(self.avg_cost)))

    def settle_buy(self, qty: int, price: float) -> None:
        """Settle a buy fill with Decimal-precise cost averaging."""
        d_cost = D(self.avg_cost) * D(self.quantity) + D(price) * D(qty)
        new_qty = self.quantity + qty
        self.avg_cost = to_float(safe_div(d_cost, new_qty)) if new_qty != 0 else 0.0
        self.quantity = new_qty

    def settle_sell(self, qty: int, price: float) -> float:
        """Settle a sell fill. Returns realized P&L with Decimal precision."""
        d_pnl = D(price - self.avg_cost) * D(qty)
        self.quantity -= qty
        realized = to_float(money(d_pnl))
        self.realized_pnl = to_float(money(D(self.realized_pnl) + d_pnl))
        return realized


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
# MicroPriceModel — Realistic fill simulation
# ---------------------------------------------------------------------------
class MicroPriceModel:
    """Simulates realistic order fill prices using bid/ask spread modelling,
    market impact estimation, and time-of-day slippage adjustments.

    All parameters are calibrated to US equity markets and driven entirely
    by OpenBB data — no live order-book feed required.
    """

    # Approximate average daily volumes for liquidity tiers (shares)
    _LIQUIDITY_TIERS = {
        "mega":  50_000_000,   # AAPL, MSFT, TSLA
        "large": 10_000_000,   # Mid-large S&P 500
        "mid":    2_000_000,   # S&P 400
        "small":    500_000,   # S&P 600
    }

    # Typical half-spread in bps by liquidity tier
    _HALF_SPREAD_BPS = {
        "mega":  0.5,
        "large": 1.5,
        "mid":   3.0,
        "small": 6.0,
    }

    # Time-of-day slippage multiplier (hour of day ET → multiplier)
    # Higher at open (9) and close (15-16), lower midday
    _TOD_MULTIPLIER = {
        9:  1.80,   # Opening auction volatility
        10: 1.30,
        11: 1.00,
        12: 0.90,
        13: 0.90,
        14: 1.00,
        15: 1.40,   # Approaching close
        16: 1.60,   # Closing auction
    }

    def __init__(self, spread_multiplier: float = 1.0):
        self.spread_multiplier = spread_multiplier
        self._adv_cache: dict[str, float] = {}

    def classify_liquidity(self, adv: float) -> str:
        """Return liquidity tier string based on average daily volume."""
        if adv >= self._LIQUIDITY_TIERS["mega"]:
            return "mega"
        elif adv >= self._LIQUIDITY_TIERS["large"]:
            return "large"
        elif adv >= self._LIQUIDITY_TIERS["mid"]:
            return "mid"
        return "small"

    def estimate_adv(self, ticker: str) -> float:
        """Estimate average daily volume from recent OpenBB data."""
        if ticker in self._adv_cache:
            return self._adv_cache[ticker]
        try:
            from ..data.openbb_data import get_prices
            from datetime import timedelta
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            hist = get_prices(ticker, start=start, end=end)
            if hist is not None and not hist.empty:
                # Extract volume from DataFrame (may be MultiIndex or flat)
                if isinstance(hist.columns, pd.MultiIndex):
                    vol = hist["Volume"] if "Volume" in hist.columns.get_level_values(0) else None
                else:
                    vol = hist["Volume"] if "Volume" in hist.columns else None
                if vol is not None and len(vol) > 0:
                    adv = float(vol.mean())
                else:
                    adv = float(self._LIQUIDITY_TIERS["mid"])
            else:
                adv = float(self._LIQUIDITY_TIERS["mid"])
        except Exception:
            adv = float(self._LIQUIDITY_TIERS["mid"])
        self._adv_cache[ticker] = adv
        return adv

    def half_spread(self, ticker: str) -> float:
        """Return estimated half-spread in bps for *ticker*."""
        adv = self.estimate_adv(ticker)
        tier = self.classify_liquidity(adv)
        return self._HALF_SPREAD_BPS[tier] * self.spread_multiplier

    def impact_cost_bps(self, ticker: str, order_size: int) -> float:
        """Market-impact cost using square-root model.

        impact = sqrt(order_size / ADV) * spread_multiplier * base_spread
        Returns cost in basis points.
        """
        adv = self.estimate_adv(ticker)
        if adv <= 0:
            return 0.0
        participation = order_size / adv
        tier = self.classify_liquidity(adv)
        base_spread = self._HALF_SPREAD_BPS[tier]
        impact = np.sqrt(participation) * base_spread * self.spread_multiplier
        return float(impact)

    def time_of_day_multiplier(self) -> float:
        """Return slippage multiplier based on current hour (ET approximation)."""
        try:
            from datetime import timezone, timedelta
            utc_now = datetime.now(timezone.utc)
            et_hour = (utc_now - timedelta(hours=5)).hour  # rough EST
        except Exception:
            et_hour = 12
        return self._TOD_MULTIPLIER.get(et_hour, 1.0)

    def fill_probability(self, ticker: str, order_size: int,
                         order_type: OrderType = OrderType.MARKET) -> float:
        """Estimate probability that the order fills.

        Market orders always fill (1.0).  Limit orders get a heuristic
        probability based on size relative to ADV.
        """
        if order_type == OrderType.MARKET:
            return 1.0
        adv = self.estimate_adv(ticker)
        if adv <= 0:
            return 0.5
        participation = order_size / adv
        # Larger participation → lower limit fill probability
        prob = max(0.1, 1.0 - np.sqrt(participation) * 2.0)
        return float(min(prob, 0.99))

    def compute_fill_price(self, mid_price: float, ticker: str,
                           order_size: int, side: OrderSide) -> float:
        """Return a realistic simulated fill price.

        Components:
            1. Half-spread cost (always paid)
            2. Market-impact cost (sqrt model)
            3. Time-of-day multiplier
        """
        spread_bps = self.half_spread(ticker)
        impact_bps = self.impact_cost_bps(ticker, order_size)
        tod = self.time_of_day_multiplier()

        total_bps = (spread_bps + impact_bps) * tod
        cost_frac = total_bps / 10_000.0

        if side in (OrderSide.BUY, OrderSide.COVER):
            return mid_price * (1.0 + cost_frac)
        else:
            return mid_price * (1.0 - cost_frac)


# ---------------------------------------------------------------------------
# RiskLimiter — Pre-trade risk checks
# ---------------------------------------------------------------------------
class RiskLimiter:
    """Enforces portfolio-level risk limits before order execution.

    All thresholds are expressed as fractions of NAV unless noted.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_sector_pct: float = 0.30,
        max_single_name_pct: float = 0.05,
        daily_loss_limit_pct: float = 0.03,
        max_gross_exposure: float = 2.5,
        max_net_exposure: float = 1.0,
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_single_name_pct = max_single_name_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure

    def check_position_size(self, order_value: float, nav: float) -> tuple[bool, str]:
        """Check if order value exceeds max position size as % of NAV."""
        if nav <= 0:
            return False, "NAV is zero or negative"
        pct = abs(order_value) / nav
        if pct > self.max_position_pct:
            return False, (f"Position size {pct:.1%} exceeds max "
                           f"{self.max_position_pct:.1%} of NAV")
        return True, ""

    def check_single_name_exposure(self, ticker: str, new_value: float,
                                   positions: dict, nav: float) -> tuple[bool, str]:
        """Check single-name concentration after proposed trade."""
        if nav <= 0:
            return False, "NAV is zero or negative"
        existing = 0.0
        if ticker in positions:
            existing = abs(positions[ticker].market_value)
        total = existing + abs(new_value)
        pct = total / nav
        if pct > self.max_single_name_pct:
            return False, (f"Single-name exposure for {ticker} would be "
                           f"{pct:.1%}, exceeds {self.max_single_name_pct:.1%}")
        return True, ""

    def check_sector_concentration(self, sector: str, new_value: float,
                                   positions: dict, nav: float) -> tuple[bool, str]:
        """Check sector concentration after proposed trade."""
        if nav <= 0 or not sector:
            return True, ""  # Skip if sector unknown
        sector_total = sum(
            abs(p.market_value) for p in positions.values()
            if p.sector == sector
        )
        sector_total += abs(new_value)
        pct = sector_total / nav
        if pct > self.max_sector_pct:
            return False, (f"Sector {sector} concentration {pct:.1%} exceeds "
                           f"max {self.max_sector_pct:.1%}")
        return True, ""

    def check_daily_loss(self, daily_pnl: float, nav: float) -> tuple[bool, str]:
        """Check if daily loss limit has been breached."""
        if nav <= 0:
            return False, "NAV is zero or negative"
        loss_pct = -daily_pnl / nav if daily_pnl < 0 else 0.0
        if loss_pct > self.daily_loss_limit_pct:
            return False, (f"Daily loss {loss_pct:.1%} exceeds limit "
                           f"{self.daily_loss_limit_pct:.1%}")
        return True, ""

    def check_gross_exposure(self, gross_exposure: float) -> tuple[bool, str]:
        """Check gross exposure limit."""
        if gross_exposure > self.max_gross_exposure:
            return False, (f"Gross exposure {gross_exposure:.2f}x exceeds "
                           f"max {self.max_gross_exposure:.2f}x")
        return True, ""

    def check_net_exposure(self, net_exposure: float) -> tuple[bool, str]:
        """Check net exposure limit."""
        if abs(net_exposure) > self.max_net_exposure:
            return False, (f"Net exposure {net_exposure:.2f}x exceeds "
                           f"max {self.max_net_exposure:.2f}x")
        return True, ""

    def run_all_checks(self, order_value: float, ticker: str, sector: str,
                       positions: dict, nav: float, daily_pnl: float,
                       gross_exposure: float, net_exposure: float
                       ) -> tuple[bool, list[str]]:
        """Run every pre-trade risk check. Return (passed, list_of_reasons)."""
        failures: list[str] = []

        ok, msg = self.check_position_size(order_value, nav)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_single_name_exposure(ticker, order_value,
                                                  positions, nav)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_sector_concentration(sector, order_value,
                                                  positions, nav)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_daily_loss(daily_pnl, nav)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_gross_exposure(gross_exposure)
        if not ok:
            failures.append(msg)

        ok, msg = self.check_net_exposure(net_exposure)
        if not ok:
            failures.append(msg)

        return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# PerformanceTracker — Broker-level performance analytics
# ---------------------------------------------------------------------------
class PerformanceTracker:
    """Tracks and computes performance metrics for a paper broker session.

    Maintains a daily P&L series and derives drawdown, Sharpe ratio, win-rate
    by signal type, average win/loss ratio, and trade-frequency statistics.
    """

    def __init__(self, initial_nav: float = 1_000_000.0, risk_free_rate: float = 0.05):
        self.initial_nav = initial_nav
        self.risk_free_rate = risk_free_rate  # annualised
        self._daily_navs: list[tuple[str, float]] = []  # (date_str, nav)
        self._daily_pnls: list[tuple[str, float]] = []  # (date_str, pnl)
        self._trade_records: list[dict] = []
        self._high_water_mark: float = initial_nav

    # --- Recording -----------------------------------------------------------

    def record_nav(self, nav: float, date_str: Optional[str] = None):
        """Record an end-of-day NAV snapshot."""
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        self._daily_navs.append((date_str, nav))
        # Compute daily P&L from previous NAV
        if len(self._daily_navs) >= 2:
            prev_nav = self._daily_navs[-2][1]
            pnl = nav - prev_nav
        else:
            pnl = nav - self.initial_nav
        self._daily_pnls.append((date_str, pnl))
        # Update high water mark
        if nav > self._high_water_mark:
            self._high_water_mark = nav

    def record_trade(self, order_dict: dict):
        """Record a completed trade for signal-type analysis."""
        self._trade_records.append(order_dict)

    # --- Daily P&L -----------------------------------------------------------

    def get_daily_pnl_series(self) -> pd.DataFrame:
        """Return a DataFrame with columns ['date', 'pnl']."""
        if not self._daily_pnls:
            return pd.DataFrame(columns=["date", "pnl"])
        df = pd.DataFrame(self._daily_pnls, columns=["date", "pnl"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_daily_returns(self) -> np.ndarray:
        """Return array of daily simple returns from NAV series."""
        if len(self._daily_navs) < 2:
            return np.array([])
        navs = np.array([n for _, n in self._daily_navs], dtype=np.float64)
        returns = np.diff(navs) / navs[:-1]
        return returns

    # --- Drawdown ------------------------------------------------------------

    def get_drawdown(self) -> dict:
        """Return current drawdown and max drawdown as fractions."""
        if not self._daily_navs:
            return {"current_drawdown": 0.0, "max_drawdown": 0.0,
                    "high_water_mark": self.initial_nav}

        navs = np.array([n for _, n in self._daily_navs], dtype=np.float64)
        running_max = np.maximum.accumulate(navs)
        drawdowns = (running_max - navs) / np.where(running_max > 0, running_max, 1.0)

        current_dd = float(drawdowns[-1])
        max_dd = float(np.max(drawdowns))

        return {
            "current_drawdown": current_dd,
            "max_drawdown": max_dd,
            "high_water_mark": float(running_max[-1]),
        }

    # --- Sharpe ratio --------------------------------------------------------

    def sharpe_ratio(self, rolling_window: Optional[int] = None) -> float:
        """Compute annualised Sharpe ratio.

        If *rolling_window* is given, compute over last N days only.
        Uses 252 trading days for annualisation.
        """
        returns = self.get_daily_returns()
        if len(returns) < 2:
            return 0.0
        if rolling_window and rolling_window < len(returns):
            returns = returns[-rolling_window:]

        daily_rf = self.risk_free_rate / 252.0
        excess = returns - daily_rf
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        if std_excess == 0:
            return 0.0
        return float(mean_excess / std_excess * np.sqrt(252))

    def rolling_sharpe(self, window: int = 60) -> list[tuple[str, float]]:
        """Return list of (date, sharpe) for rolling window."""
        returns = self.get_daily_returns()
        dates = [d for d, _ in self._daily_navs]
        result: list[tuple[str, float]] = []
        daily_rf = self.risk_free_rate / 252.0
        for i in range(window, len(returns) + 1):
            chunk = returns[i - window:i]
            excess = chunk - daily_rf
            m = np.mean(excess)
            s = np.std(excess, ddof=1)
            sr = float(m / s * np.sqrt(252)) if s > 0 else 0.0
            # dates are 1-indexed relative to returns (returns[0] goes with dates[1])
            result.append((dates[i], sr))
        return result

    # --- Win rate by signal type ---------------------------------------------

    def win_rate_by_signal(self) -> dict[str, dict]:
        """Return win rate and trade count grouped by signal_type."""
        buckets: dict[str, dict] = {}
        for trade in self._trade_records:
            sig = trade.get("signal_type", "HOLD")
            if sig not in buckets:
                buckets[sig] = {"wins": 0, "losses": 0, "total": 0,
                                "total_pnl": 0.0}
            pnl = trade.get("realized_pnl", 0.0)
            buckets[sig]["total"] += 1
            buckets[sig]["total_pnl"] += pnl
            if pnl > 0:
                buckets[sig]["wins"] += 1
            elif pnl < 0:
                buckets[sig]["losses"] += 1

        for sig, b in buckets.items():
            b["win_rate"] = b["wins"] / b["total"] if b["total"] > 0 else 0.0
        return buckets

    # --- Average win / loss ratio --------------------------------------------

    def avg_win_loss_ratio(self) -> float:
        """Return average winning trade P&L / average losing trade P&L (abs).

        Returns 0.0 if no losing trades.
        """
        wins = [t["realized_pnl"] for t in self._trade_records
                if t.get("realized_pnl", 0) > 0]
        losses = [t["realized_pnl"] for t in self._trade_records
                  if t.get("realized_pnl", 0) < 0]
        if not losses:
            return float("inf") if wins else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses))
        if avg_loss == 0:
            return 0.0
        return float(avg_win / avg_loss)

    # --- Trade frequency -----------------------------------------------------

    def trade_frequency(self) -> dict:
        """Return trade-frequency statistics."""
        if not self._trade_records:
            return {"total_trades": 0, "trades_per_day": 0.0, "days_traded": 0}
        dates = set()
        for t in self._trade_records:
            ts = t.get("fill_timestamp", t.get("timestamp", ""))
            if ts:
                dates.add(ts[:10])
        n_days = max(len(dates), 1)
        total = len(self._trade_records)
        return {
            "total_trades": total,
            "trades_per_day": total / n_days,
            "days_traded": n_days,
        }

    # --- Aggregate metrics ---------------------------------------------------

    def get_all_metrics(self) -> dict:
        """Return a dictionary with all performance metrics."""
        dd = self.get_drawdown()
        wl = self.avg_win_loss_ratio()
        freq = self.trade_frequency()
        return {
            "sharpe_total": self.sharpe_ratio(),
            "sharpe_60d": self.sharpe_ratio(rolling_window=60),
            "current_drawdown": dd["current_drawdown"],
            "max_drawdown": dd["max_drawdown"],
            "high_water_mark": dd["high_water_mark"],
            "avg_win_loss_ratio": wl,
            "win_rate_by_signal": self.win_rate_by_signal(),
            "trade_frequency": freq,
        }


# ---------------------------------------------------------------------------
# Risk Profile — Daily Target Dial-Down
# ---------------------------------------------------------------------------
class RiskProfile(str, Enum):
    """Risk profile tiers for daily target management."""
    AGGRESSIVE = "AGGRESSIVE"    # Pre-target: full leverage, max scanning
    MODERATE = "MODERATE"        # Target hit (5%): reduce leverage 50%
    DEFENSIVE = "DEFENSIVE"      # Target + buffer (6%+): protect gains


class DailyTargetManager:
    """Manages 5% daily compound return target with risk dial-down.

    Once the daily target is hit, the system shifts from alpha-seeking
    to capital-preservation mode. Execution multipliers, position sizing,
    and stop widths all adjust based on progress toward target.

    Target: 5% daily compound (minimum)
    Buffer: 6%+ triggers full defensive mode
    """

    DAILY_TARGET_PCT = 0.05      # 5% daily compound return
    BUFFER_TARGET_PCT = 0.06     # 6% triggers full defensive
    FLOOR_TARGET_PCT = 0.03      # Below 3% = max aggressive

    # Leverage multipliers by risk profile
    _LEVERAGE_MULT = {
        RiskProfile.AGGRESSIVE: 1.0,   # Full leverage
        RiskProfile.MODERATE: 0.50,    # Half leverage
        RiskProfile.DEFENSIVE: 0.20,   # Minimal leverage
    }

    # Position size multipliers
    _POSITION_MULT = {
        RiskProfile.AGGRESSIVE: 1.0,
        RiskProfile.MODERATE: 0.60,
        RiskProfile.DEFENSIVE: 0.25,
    }

    # Stop-loss width multipliers (wider = more protective)
    _STOP_WIDTH_MULT = {
        RiskProfile.AGGRESSIVE: 1.0,
        RiskProfile.MODERATE: 1.5,
        RiskProfile.DEFENSIVE: 2.5,
    }

    def __init__(self, initial_nav: float = 1_000.0):
        self._initial_nav_today = initial_nav
        self._current_profile = RiskProfile.AGGRESSIVE
        self._target_hit_time: Optional[str] = None
        self._profile_history: list[tuple[str, str]] = []

    def reset_day(self, current_nav: float):
        """Reset for new trading day."""
        self._initial_nav_today = current_nav
        self._current_profile = RiskProfile.AGGRESSIVE
        self._target_hit_time = None

    @property
    def profile(self) -> RiskProfile:
        return self._current_profile

    @property
    def daily_return_pct(self) -> float:
        """Current day return as decimal (0.05 = 5%)."""
        if self._initial_nav_today <= 0:
            return 0.0
        return 0.0  # Will be set by update()

    def update(self, current_nav: float) -> RiskProfile:
        """Update risk profile based on current NAV vs start-of-day NAV.

        Returns the current risk profile after update.
        """
        if self._initial_nav_today <= 0:
            return self._current_profile

        daily_return = (current_nav - self._initial_nav_today) / self._initial_nav_today
        now_str = datetime.now().isoformat()

        old_profile = self._current_profile

        if daily_return >= self.BUFFER_TARGET_PCT:
            self._current_profile = RiskProfile.DEFENSIVE
        elif daily_return >= self.DAILY_TARGET_PCT:
            self._current_profile = RiskProfile.MODERATE
            if self._target_hit_time is None:
                self._target_hit_time = now_str
        else:
            self._current_profile = RiskProfile.AGGRESSIVE

        if old_profile != self._current_profile:
            self._profile_history.append((now_str, self._current_profile.value))

        return self._current_profile

    def get_leverage_multiplier(self) -> float:
        """Return leverage multiplier for current risk profile."""
        return self._LEVERAGE_MULT[self._current_profile]

    def get_position_multiplier(self) -> float:
        """Return position size multiplier for current risk profile."""
        return self._POSITION_MULT[self._current_profile]

    def get_stop_width_multiplier(self) -> float:
        """Return stop-loss width multiplier for current risk profile."""
        return self._STOP_WIDTH_MULT[self._current_profile]

    def allow_new_positions(self) -> bool:
        """Whether new positions are allowed under current profile."""
        return self._current_profile != RiskProfile.DEFENSIVE

    def get_state(self) -> dict:
        """Return full state for dashboard emission."""
        return {
            "risk_profile": self._current_profile.value,
            "initial_nav_today": self._initial_nav_today,
            "target_pct": self.DAILY_TARGET_PCT,
            "target_hit_time": self._target_hit_time,
            "leverage_mult": self.get_leverage_multiplier(),
            "position_mult": self.get_position_multiplier(),
            "stop_width_mult": self.get_stop_width_multiplier(),
            "allow_new_positions": self.allow_new_positions(),
            "profile_history": self._profile_history,
        }


# ---------------------------------------------------------------------------
# Live Dashboard State Emitter
# ---------------------------------------------------------------------------
class LiveDashboardState:
    """Collects and emits state for the live observation dashboard.

    When connected to internet and OpenBB API is running, this state
    can be consumed by engine/monitoring/live_dashboard.py for real-time
    terminal visualization or by the FastAPI backend for web dashboard.
    """

    def __init__(self):
        self._last_snapshot: dict = {}
        self._snapshot_history: list[dict] = []
        self._callbacks: list = []

    def register_callback(self, callback):
        """Register a callback for live state updates."""
        self._callbacks.append(callback)

    def emit(self, broker_state: dict, target_state: dict,
             pipeline_state: Optional[dict] = None):
        """Emit a dashboard snapshot.

        Args:
            broker_state: Portfolio summary from PaperBroker
            target_state: Daily target state from DailyTargetManager
            pipeline_state: Optional signal pipeline state
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": broker_state,
            "daily_target": target_state,
            "pipeline": pipeline_state or {},
        }
        self._last_snapshot = snapshot
        self._snapshot_history.append(snapshot)

        # Keep only last 1000 snapshots in memory
        if len(self._snapshot_history) > 1000:
            self._snapshot_history = self._snapshot_history[-500:]

        # Fire callbacks
        for cb in self._callbacks:
            try:
                cb(snapshot)
            except Exception:
                pass

    def get_latest(self) -> dict:
        return self._last_snapshot

    def get_history(self, n: int = 100) -> list[dict]:
        return self._snapshot_history[-n:]
