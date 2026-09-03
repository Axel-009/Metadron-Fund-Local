"""
SchwabAccountRouter — one execution surface over three Schwab accounts.

Presents the *same* broker interface as ``SchwabBroker`` (so L7 / ExecutionEngine
/ LiveLoop are account-agnostic) while enforcing the per-account mandates:

    ROTH        25% options (1-7 DTE) / 75% equities+ETFs
    LLC         equities + ETFs only
    INDIVIDUAL  options only

Routing rules
-------------
* Every BUY / *_TO_OPEN order is routed to the highest-priority account whose
  mandate permits the product class AND that has mandate headroom
  (options notional ≤ options_pct × account NAV; equity MV ≤ equities_pct × NAV).
* Every SELL / *_TO_CLOSE order goes to the account that holds the position.
* The 20% drawdown rule is checked per account AND for the whole portfolio
  BEFORE any add. A scope in ``ROTATE_OR_CLOSE`` never receives a new add;
  the router returns a REJECTED order carrying the rotate/close directive.
* Sleeve percentages from ``AllocationRules`` are scaled onto each account via
  ``scale_sleeves_for_mandate`` and exposed through ``account_sleeve_caps()`` so
  the allocation engine / options engine can honour them per account.

Every account uses the same ``SchwabAuth``; data calls (quotes, chains,
history) are served by the primary (first) account — Schwab market data is
account-independent.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .account_mandates import (
    AccountMandate, DrawdownGuard, DrawdownStatus, ETF_HINTS, load_mandates, scale_sleeves_for_mandate,
)
from .broker_types import Order, OrderSide, OrderStatus, Position, PortfolioState, SignalType
from .schwab_broker import SchwabAuth, SchwabBroker

logger = logging.getLogger(__name__)


def _is_option_symbol(sym: str) -> bool:
    return bool(sym) and (" " in sym.strip() or len(sym) > 12)


class SchwabAccountRouter:
    """Mandate-aware fan-out over several ``SchwabBroker`` instances."""

    paper = False
    MAX_ORDER_NOTIONAL_PCT = SchwabBroker.MAX_ORDER_NOTIONAL_PCT

    def __init__(
        self,
        mandates: Optional[Dict[str, AccountMandate]] = None,
        auth: Optional[SchwabAuth] = None,
        live_orders: Optional[bool] = None,
        log_dir: Optional[str] = None,
        connect: bool = True,
        initial_cash: float = 100_000.0,
        allocation_rules: Any = None,
        drawdown_guard: Optional[DrawdownGuard] = None,
    ):
        self.mandates: Dict[str, AccountMandate] = mandates or load_mandates()
        if not self.mandates:
            raise ValueError("SchwabAccountRouter needs at least one mapped mandate (SCHWAB_ACCOUNT_ROTH/LLC/INDIVIDUAL)")
        self.auth = auth or SchwabAuth()
        self.rules = allocation_rules
        self.guard = drawdown_guard or DrawdownGuard()
        self.brokers: Dict[str, SchwabBroker] = {}
        per_acct_cash = initial_cash / max(1, len(self.mandates))
        for label, m in self.mandates.items():
            self.brokers[label] = SchwabBroker(
                initial_cash=per_acct_cash,
                log_dir=os.path.join(log_dir or "logs/schwab_broker", label.lower()),
                account_number=m.account_last4, live_orders=live_orders, auth=self.auth, connect=connect,
            )
        self.state = PortfolioState(cash=initial_cash, nav=initial_cash)
        self._orders: List[Order] = []
        self._routing_log: List[dict] = []
        self._initial_nav: Optional[float] = None
        self._refresh_aggregate()

    # ------------------------------------------------------------------
    # aggregate book
    # ------------------------------------------------------------------
    @property
    def primary(self) -> SchwabBroker:
        for label in ("ROTH", "LLC", "INDIVIDUAL"):
            if label in self.brokers:
                return self.brokers[label]
        return next(iter(self.brokers.values()))

    @property
    def live_orders(self) -> bool:
        return any(b.live_orders for b in self.brokers.values())

    @property
    def is_connected(self) -> bool:
        return any(b.is_connected for b in self.brokers.values())

    @property
    def connected(self) -> bool:
        return self.is_connected

    @property
    def account_display(self) -> str:
        return " | ".join(f"{l}:{b.account_display or '****' + m.account_last4}" for (l, m), b in zip(self.mandates.items(), self.brokers.values()))

    def _refresh_aggregate(self):
        cash = nav = pnl = 0.0
        merged: Dict[str, Position] = {}
        for label, b in self.brokers.items():
            cash += b.state.cash
            nav += b.state.nav
            pnl += b.state.total_pnl
            for sym, p in b.state.positions.items():
                if sym in merged:
                    q = merged[sym].quantity + p.quantity
                    if q:
                        merged[sym].avg_cost = (merged[sym].avg_cost * merged[sym].quantity + p.avg_cost * p.quantity) / q
                    merged[sym].quantity = q
                    merged[sym].unrealized_pnl += p.unrealized_pnl
                    merged[sym].realized_pnl += p.realized_pnl
                else:
                    merged[sym] = Position(ticker=p.ticker, quantity=p.quantity, avg_cost=p.avg_cost,
                                           current_price=p.current_price, unrealized_pnl=p.unrealized_pnl,
                                           realized_pnl=p.realized_pnl, sector=p.sector)
        self.state.cash, self.state.nav, self.state.total_pnl, self.state.positions = cash, nav, pnl, merged
        long_v = sum(p.market_value for p in merged.values() if p.quantity > 0)
        short_v = sum(abs(p.market_value) for p in merged.values() if p.quantity < 0)
        self.state.gross_exposure = (long_v + short_v) / nav if nav else 0.0
        self.state.net_exposure = (long_v - short_v) / nav if nav else 0.0
        if self._initial_nav is None and nav > 0 and self.is_connected:
            self._initial_nav = nav
        # seed / update drawdown peaks
        for label, b in self.brokers.items():
            self.guard.check(label, b.state.nav)
        self.guard.check("PORTFOLIO", nav)

    def sync_account(self, force: bool = False) -> dict:
        out = {"accounts": {}}
        for label, b in self.brokers.items():
            out["accounts"][label] = b.sync_account(force=force)
        self._refresh_aggregate()
        out.update({"nav": self.state.nav, "cash": self.state.cash, "connected": self.is_connected})
        return out

    def sync_positions(self) -> Dict[str, Position]:
        for b in self.brokers.values():
            b.sync_positions()
        self._refresh_aggregate()
        return dict(self.state.positions)

    def heartbeat(self) -> dict:
        res = {label: b.heartbeat() for label, b in self.brokers.items()}
        self._refresh_aggregate()
        return {"accounts": res, "nav": self.state.nav, "daily_pnl": self.get_daily_pnl(),
                "drawdown": {k: v.to_dict() for k, v in self.guard.latest().items()}}

    # ------------------------------------------------------------------
    # mandate accounting
    # ------------------------------------------------------------------
    def account_snapshot(self, label: str) -> dict:
        b = self.brokers[label]
        m = self.mandates[label]
        nav = b.state.nav or 0.0
        opt_notional = sum(abs(float(o.get("market_value", 0.0) or 0.0)) for o in b.get_option_positions().values())
        eq_mv = sum(abs(p.market_value) for p in b.state.positions.values() if p.sector != "OPTIONS")
        dd = self.guard.check(label, nav)
        return {
            "label": label, "account": b.account_display or "****" + m.account_last4, "connected": b.is_connected,
            "nav": nav, "cash": b.state.cash,
            "options_notional": opt_notional, "options_cap": m.options_pct * nav,
            "options_headroom": max(0.0, m.options_pct * nav - opt_notional),
            "equities_mv": eq_mv, "equities_cap": m.equities_pct * nav,
            "equities_headroom": max(0.0, m.equities_pct * nav - eq_mv),
            "drawdown": dd.to_dict(), "mandate": m.to_dict(),
        }

    def portfolio_snapshot(self) -> dict:
        self._refresh_aggregate()
        return {
            "as_of": datetime.now().isoformat(), "nav": self.state.nav, "cash": self.state.cash,
            "connected": self.is_connected, "live_orders": self.live_orders,
            "drawdown": self.guard.check("PORTFOLIO", self.state.nav).to_dict(),
            "accounts": {l: self.account_snapshot(l) for l in self.brokers},
        }

    def account_sleeve_caps(self, label: str) -> Dict[str, float]:
        """Sleeve caps (fraction of account NAV) for one account, per AllocationRules × mandate."""
        if self.rules is None:
            try:
                from ..allocation.allocation_engine import AllocationRules  # noqa: WPS433
                self.rules = AllocationRules()
            except Exception:  # noqa: BLE001
                return {}
        return scale_sleeves_for_mandate(self.rules, self.mandates[label])

    def options_budget(self) -> Dict[str, float]:
        """Dollar headroom for new 1-7 DTE options per account (drawdown-scaled)."""
        out = {}
        for label in self.brokers:
            s = self.account_snapshot(label)
            if self.mandates[label].allow_options:
                out[label] = round(s["options_headroom"] * s["drawdown"]["add_scale"], 2)
        return out

    # ── fill order (operator rule) ──────────────────────────────────────────
    #   1. INDIVIDUAL  — 100 % options: run the 1–7 DTE engine and fill it first
    #   2. LLC         — equities + ETFs from the same universe runs
    #   3. ROTH        — composite of both (25 % options / 75 % equities)
    # `prefer(label)` pins the destination for a phase; mandate permission and
    # headroom are still enforced, other accounts remain as fallbacks in priority order.
    _preferred: Optional[str] = None

    def prefer(self, label: Optional[str]) -> None:
        self._preferred = label.upper() if label else None

    def _pick_account(self, product: str, ticker: str, notional: float) -> Tuple[Optional[str], str]:
        pdd = self.guard.check("PORTFOLIO", self.state.nav)
        if not pdd.adds_allowed:
            return None, pdd.directive
        key = "priority_options" if product == "OPTION" else "priority_equities"
        cands = sorted((m for m in self.mandates.values() if m.permits(product, ticker)),
                       key=lambda m: (0 if m.label == self._preferred else 1, getattr(m, key)))
        reasons = []
        for m in cands:
            snap = self.account_snapshot(m.label)
            dd = snap["drawdown"]
            if not dd["adds_allowed"]:
                reasons.append(dd["directive"])
                continue
            headroom = snap["options_headroom"] if product == "OPTION" else snap["equities_headroom"]
            headroom *= dd["add_scale"]
            if not self.brokers[m.label].is_connected and snap["nav"] <= 0:
                headroom = float("inf")  # offline dry-run: mandate permission only
            if notional <= headroom or headroom == float("inf"):
                return m.label, "ok"
            reasons.append(f"{m.label}: mandate headroom ${headroom:,.0f} < ${notional:,.0f}")
        if not cands:
            reasons.append(f"no account mandate permits {product} {ticker}")
        return None, "; ".join(reasons)

    def _holder_of(self, symbol: str) -> Optional[str]:
        for label, b in self.brokers.items():
            p = b.state.positions.get(symbol) or b.state.positions.get(symbol.upper())
            if p and p.quantity != 0:
                return label
            if symbol in b.get_option_positions():
                return label
        return None

    def _reject(self, ticker: str, side: OrderSide, quantity: int, signal_type: SignalType, reason: str, product: str) -> Order:
        o = Order(id=f"RTR-{len(self._orders) + 1:06d}", ticker=ticker, side=side, quantity=quantity,
                  status=OrderStatus.REJECTED, signal_type=signal_type, timestamp=datetime.now().isoformat(),
                  reason=f"[router] {reason}")
        self._orders.append(o)
        self._routing_log.append({"ts": o.timestamp, "ticker": ticker, "product": product, "account": None,
                                  "status": "REJECTED", "reason": reason})
        logger.warning("[SchwabAccountRouter] REJECT %s %s: %s", ticker, product, reason)
        return o

    def _log_route(self, label: str, ticker: str, product: str, order: Order):
        self._orders.append(order)
        self._routing_log.append({"ts": order.timestamp, "ticker": ticker, "product": product, "account": label,
                                  "mandate": self.mandates[label].label, "status": str(order.status), "order_id": order.id})

    # ------------------------------------------------------------------
    # order surface (mirrors SchwabBroker)
    # ------------------------------------------------------------------
    def place_order(self, ticker: str, side: OrderSide, quantity: int, signal_type: SignalType = SignalType.HOLD,
                    limit_price: Optional[float] = None, reason: str = "", sector: str = "EQUITY") -> Order:
        side = OrderSide(side) if not isinstance(side, OrderSide) else side
        product = "ETF" if ticker.upper() in ETF_HINTS else "EQUITY"
        if side == OrderSide.SELL:
            label = self._holder_of(ticker)
            if label is None:
                return self._reject(ticker, side, quantity, signal_type, "no account holds this position (shorting disabled)", product)
        else:
            px = limit_price or self.primary._get_current_price(ticker) or 0.0
            label, why = self._pick_account(product, ticker, abs(quantity) * px)
            if label is None:
                return self._reject(ticker, side, quantity, signal_type, why, product)
        o = self.brokers[label].place_order(ticker, side, quantity, signal_type, limit_price, reason, sector)
        self._log_route(label, ticker, product, o)
        self._refresh_aggregate()
        return o

    def place_twap_order(self, ticker: str, side: OrderSide, quantity: int, duration_minutes: int = 30, **kw) -> Order:
        label = self._holder_of(ticker) if side == OrderSide.SELL else None
        if label is None:
            px = self.primary._get_current_price(ticker) or 0.0
            label, why = self._pick_account("ETF" if ticker.upper() in ETF_HINTS else "EQUITY", ticker, abs(quantity) * px)
            if label is None:
                return self._reject(ticker, side, quantity, kw.get("signal_type", SignalType.HOLD), why, "EQUITY")
        o = self.brokers[label].place_twap_order(ticker, side, quantity, duration_minutes, **kw)
        self._log_route(label, ticker, "EQUITY", o)
        return o

    def place_vwap_order(self, ticker: str, side: OrderSide, quantity: int, duration_minutes: int = 60, **kw) -> Order:
        label = self._holder_of(ticker) if side == OrderSide.SELL else None
        if label is None:
            px = self.primary._get_current_price(ticker) or 0.0
            label, why = self._pick_account("ETF" if ticker.upper() in ETF_HINTS else "EQUITY", ticker, abs(quantity) * px)
            if label is None:
                return self._reject(ticker, side, quantity, kw.get("signal_type", SignalType.HOLD), why, "EQUITY")
        o = self.brokers[label].place_vwap_order(ticker, side, quantity, duration_minutes, **kw)
        self._log_route(label, ticker, "EQUITY", o)
        return o

    def place_option_order(self, option_symbol: str, instruction: str, quantity: int, limit_price: float,
                           underlying: str = "", signal_type: SignalType = SignalType.HOLD, reason: str = "") -> Order:
        ins = instruction.upper()
        side = OrderSide.SELL if "SELL" in ins else OrderSide.BUY
        if "CLOSE" in ins:
            label = self._holder_of(option_symbol)
            if label is None:
                return self._reject(option_symbol, side, quantity, signal_type, "no account holds this contract", "OPTION")
        else:
            label, why = self._pick_account("OPTION", underlying or option_symbol.split()[0], abs(quantity) * limit_price * 100)
            if label is None:
                return self._reject(option_symbol, side, quantity, signal_type, why, "OPTION")
        o = self.brokers[label].place_option_order(option_symbol, instruction, quantity, limit_price, underlying, signal_type, reason)
        self._log_route(label, option_symbol, "OPTION", o)
        self._refresh_aggregate()
        return o

    def place_option_spread(self, legs: List[dict], net_price: float, quantity: int, underlying: str = "",
                            signal_type: SignalType = SignalType.HOLD, reason: str = "", strategy: str = "VERTICAL",
                            is_debit: bool = True) -> Order:
        opening = any("OPEN" in str(l.get("instruction", "")).upper() for l in legs)
        if opening:
            label, why = self._pick_account("OPTION", underlying, abs(quantity) * abs(net_price) * 100)
            if label is None:
                return self._reject(underlying or "SPREAD", OrderSide.BUY, quantity, signal_type, why, "OPTION")
        else:
            label = next((self._holder_of(l.get("symbol", "")) for l in legs if self._holder_of(l.get("symbol", ""))), None)
            if label is None:
                return self._reject(underlying or "SPREAD", OrderSide.SELL, quantity, signal_type, "no account holds these legs", "OPTION")
        o = self.brokers[label].place_option_spread(legs, net_price, quantity, underlying, signal_type, reason, strategy, is_debit)
        self._log_route(label, underlying or "SPREAD", "OPTION", o)
        self._refresh_aggregate()
        return o

    def cancel_order(self, schwab_order_id: str) -> bool:
        return any(b.cancel_order(schwab_order_id) for b in self.brokers.values())

    def get_open_orders(self) -> List[dict]:
        out = []
        for label, b in self.brokers.items():
            out += [{**o, "account": label} for o in b.get_open_orders()]
        return out

    # ------------------------------------------------------------------
    # market data (account-independent → primary)
    # ------------------------------------------------------------------
    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, dict]:
        return self.primary.get_quotes(tickers)

    def get_quote(self, ticker: str) -> Optional[float]:
        return self.primary.get_quote(ticker)

    def _get_current_price(self, ticker: str) -> float:
        return self.primary._get_current_price(ticker)

    def get_micro_price(self, ticker: str) -> Optional[float]:
        return self.primary.get_micro_price(ticker)

    def get_price_history(self, ticker: str, days: int = 120, frequency: str = "daily") -> dict:
        return self.primary.get_price_history(ticker, days, frequency)

    def get_returns(self, ticker: str, days: int = 120) -> np.ndarray:
        return self.primary.get_returns(ticker, days)

    def get_option_chain(self, *a, **kw):
        return self.primary.get_option_chain(*a, **kw)

    @property
    def last_chain_error(self) -> str:
        return getattr(self.primary, "last_chain_error", "")

    def refresh_prices(self):
        for b in self.brokers.values():
            b.refresh_prices()
        self._refresh_aggregate()

    # ------------------------------------------------------------------
    # BrokerProtocol surface (aggregates)
    # ------------------------------------------------------------------
    def get_orders(self) -> List[Order]:
        return list(self._orders)

    def get_trade_history(self, last_n: int = 0) -> List[dict]:
        rows = []
        for label, b in self.brokers.items():
            rows += [{**t, "account": label} for t in b.get_trade_history()]
        rows.sort(key=lambda r: r.get("timestamp", ""))
        return rows[-last_n:] if last_n else rows

    def get_all_positions(self) -> Dict[str, Position]:
        return dict(self.state.positions)

    def get_positions(self) -> Dict[str, Position]:
        return self.get_all_positions()

    def get_position(self, ticker: str) -> Optional[Position]:
        return self.state.positions.get(ticker.upper())

    def get_option_positions(self) -> Dict[str, dict]:
        out = {}
        for label, b in self.brokers.items():
            for sym, o in b.get_option_positions().items():
                out[sym if sym not in out else f"{sym}@{label}"] = {**o, "account": label}
        return out

    def compute_nav(self) -> float:
        self.sync_account()
        return self.state.nav

    def get_nav(self) -> float:
        return self.compute_nav()

    def compute_exposures(self) -> dict:
        self._refresh_aggregate()
        opt_long = sum(p.market_value for p in self.state.positions.values() if p.sector == "OPTIONS" and p.quantity > 0)
        return {"gross": self.state.gross_exposure, "net": self.state.net_exposure,
                "options_long_value": opt_long, "nav": self.state.nav}

    def get_drawdown(self) -> dict:
        st = self.guard.check("PORTFOLIO", self.state.nav)
        return {"current_drawdown": st.drawdown, "peak_nav": st.peak_nav, "level": st.level,
                "accounts": {l: self.guard.check(l, b.state.nav).to_dict() for l, b in self.brokers.items()}}

    def get_daily_pnl(self) -> float:
        return sum(b.get_daily_pnl() for b in self.brokers.values())

    def get_equity_pnl(self) -> float:
        return sum(p.unrealized_pnl + p.realized_pnl for p in self.state.positions.values() if p.sector != "OPTIONS")

    def get_options_pnl(self) -> float:
        return sum(p.unrealized_pnl + p.realized_pnl for p in self.state.positions.values() if p.sector == "OPTIONS")

    def get_sector_pnl(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p in self.state.positions.values():
            out[p.sector or "EQUITY"] = out.get(p.sector or "EQUITY", 0.0) + p.unrealized_pnl + p.realized_pnl
        return out

    def get_risk_profile(self):
        return self.primary.get_risk_profile()

    def get_leverage_multiplier(self) -> float:
        return min(b.get_leverage_multiplier() for b in self.brokers.values())

    def get_daily_target_state(self) -> dict:
        return {l: b.get_daily_target_state() for l, b in self.brokers.items()}

    def reset_daily_target(self):
        for b in self.brokers.values():
            b.reset_daily_target()

    def get_performance_metrics(self) -> dict:
        return {l: b.get_performance_metrics() for l, b in self.brokers.items()}

    def get_dashboard_snapshot(self) -> dict:
        return {"portfolio": self.portfolio_snapshot(), "accounts": {l: b.get_dashboard_snapshot() for l, b in self.brokers.items()}}

    def emit_dashboard_state(self, pipeline_state: Optional[dict] = None) -> dict:
        return {l: b.emit_dashboard_state(pipeline_state) for l, b in self.brokers.items()}

    def reconcile(self) -> dict:
        return {l: b.reconcile() for l, b in self.brokers.items()}

    def get_portfolio_summary(self) -> dict:
        return self.portfolio_snapshot()

    def get_status(self) -> dict:
        return {"connected": self.is_connected, "live_orders": self.live_orders, "accounts": self.account_display,
                "nav": self.state.nav, "routing_log_size": len(self._routing_log)}

    def get_routing_log(self, last_n: int = 0) -> List[dict]:
        return self._routing_log[-last_n:] if last_n else list(self._routing_log)

    def __repr__(self) -> str:
        return f"SchwabAccountRouter({self.account_display}, nav=${self.state.nav:,.0f}, live={self.live_orders})"


def build_schwab_broker(connect: bool = True, account_number: Optional[str] = None,
                        initial_cash: float = 100_000.0, live_orders: Optional[bool] = None,
                        log_dir: Optional[str] = None, allocation_rules: Any = None):
    """Factory: multi-account router when mandates are configured, else a single SchwabBroker."""
    mandates = load_mandates()
    if mandates and account_number is None:
        return SchwabAccountRouter(mandates=mandates, live_orders=live_orders, log_dir=log_dir,
                                  connect=connect, initial_cash=initial_cash, allocation_rules=allocation_rules)
    return SchwabBroker(initial_cash=initial_cash, log_dir=log_dir, account_number=account_number,
                        live_orders=live_orders, connect=connect)
