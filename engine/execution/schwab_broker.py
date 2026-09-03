"""SchwabBroker — sole execution + market-data broker for Metadron Capital.

============================================================
LAYER:  layer7_execution
ROLE:   Charles Schwab Trader API (accounts/orders) + Market Data API
        (quotes, option chains, price history) behind the BrokerProtocol
        surface that ExecutionEngine, L7UnifiedExecutionSurface and the
        engine API consume.
============================================================

Everything the platform needs from the outside world at execution time goes
through this one object:

    * order placement      — equities (BUY/SELL/SELL_SHORT/BUY_TO_COVER) and
                             single-leg or multi-leg options (BUY_TO_OPEN ...)
    * risk limits          — RiskLimiter (position 10 % NAV, sector 30 %,
                             daily loss 3 %, gross 250 %, net 150 %) enforced
                             *again* at the broker edge, after L7's 10 gates
    * position sync        — Schwab positions/balances are ground truth;
                             the local book is rebuilt from them on every
                             heartbeat
    * quote cache          — TTL cache (QUOTE_CACHE_TTL from heartbeat_config)
                             for quotes, chains and price history so the
                             1-min L7 heartbeat never hammers the API
    * option chain         — normalised 1-7 DTE chains with Schwab Greeks/IV
                             (consumed by ShortDTEOptionsEngine)

Authentication (three modes, auto-detected):

    SCHWAB_AUTH_MODE=oauth   SCHWAB_APP_KEY + SCHWAB_APP_SECRET + a token file
                             (SCHWAB_TOKEN_PATH, default ~/.metadron/schwab_token.json)
                             holding {access_token, refresh_token, expires_at}.
                             The 30-minute access token is refreshed
                             automatically; the 7-day refresh token must be
                             re-minted by the OAuth login flow
                             (see ``SchwabAuth.build_authorize_url``).
    SCHWAB_AUTH_MODE=token   SCHWAB_ACCESS_TOKEN static bearer (30 min life).
    SCHWAB_AUTH_MODE=proxy   No Authorization header is added; an HTTPS proxy in
                             front of the process injects it (Perplexity
                             Computer sandbox with a vaulted credential).

Live vs dry-run:

    SCHWAB_LIVE_ORDERS=false (default) — every order is risk-checked, priced and
    written to the audit log with status DRY_RUN but never POSTed to Schwab.
    SCHWAB_LIVE_ORDERS=true  — orders are POSTed. There is NO paper account on
    the Schwab API: live means real money.

Futures are not supported by the Schwab Trader API and are not supported here.
The overlay is strictly options.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import httpx
except Exception:  # pragma: no cover - httpx is in requirements.txt
    httpx = None  # type: ignore[assignment]

from .broker_types import (
    DailyTargetManager,
    LiveDashboardState,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PerformanceTracker,
    PortfolioState,
    Position,
    RiskLimiter,
    RiskProfile,
    SignalType,
)

try:
    from config.heartbeat_config import QUOTE_CACHE_TTL as _CFG_QUOTE_TTL
except Exception:  # pragma: no cover
    _CFG_QUOTE_TTL = 5

logger = logging.getLogger(__name__)

TRADER_BASE = "https://api.schwabapi.com/trader/v1"
MARKET_BASE = "https://api.schwabapi.com/marketdata/v1"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
AUTHORIZE_URL = "https://api.schwabapi.com/v1/oauth/authorize"

_EQUITY_INSTRUCTION = {
    OrderSide.BUY: "BUY",
    OrderSide.SELL: "SELL",
    OrderSide.SHORT: "SELL_SHORT",
    OrderSide.COVER: "BUY_TO_COVER",
}

OPTION_INSTRUCTIONS = frozenset({"BUY_TO_OPEN", "SELL_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_CLOSE"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ssl_context() -> Any:
    """SSL context that honours SSL_CERT_FILE and tolerates MITM proxy CAs.

    Python >= 3.13 enables VERIFY_X509_STRICT, which rejects the private CA
    used by credential-injecting HTTPS proxies. We relax only that flag.
    """
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    try:
        ctx = ssl.create_default_context(cafile=cafile if cafile and Path(cafile).exists() else None)
        if hasattr(ssl, "VERIFY_X509_STRICT"):
            ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        return ctx
    except Exception:
        return True


# ---------------------------------------------------------------------------
# OAuth / token management
# ---------------------------------------------------------------------------
class SchwabAuth:
    """Manages Schwab OAuth2 tokens (or a proxy/static bearer)."""

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        callback_url: Optional[str] = None,
        token_path: Optional[str] = None,
        mode: Optional[str] = None,
        static_token: Optional[str] = None,
    ):
        self.app_key = app_key or os.environ.get("SCHWAB_APP_KEY", "")
        self.app_secret = app_secret or os.environ.get("SCHWAB_APP_SECRET", "")
        self.callback_url = callback_url or os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182/callback")
        self.token_path = Path(token_path or os.environ.get("SCHWAB_TOKEN_PATH", str(Path.home() / ".metadron" / "schwab_token.json")))
        self._static_token = static_token or os.environ.get("SCHWAB_ACCESS_TOKEN", "")
        self._tokens: dict = {}
        self.mode = (mode or os.environ.get("SCHWAB_AUTH_MODE", "") or self._detect_mode()).lower()
        if self.mode == "oauth":
            self._load_tokens()

    def _detect_mode(self) -> str:
        if self._static_token:
            return "token"
        if self.app_key and self.app_secret:
            return "oauth"
        if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
            return "proxy"
        return "oauth"

    # -- persistence -----------------------------------------------------
    def _load_tokens(self):
        try:
            if self.token_path.exists():
                self._tokens = json.loads(self.token_path.read_text())
        except Exception as e:
            logger.warning("Schwab token file unreadable (%s): %s", self.token_path, e)
            self._tokens = {}

    def _save_tokens(self):
        try:
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            self.token_path.write_text(json.dumps(self._tokens, indent=2))
            os.chmod(self.token_path, 0o600)
        except Exception as e:
            logger.warning("Schwab token file write failed: %s", e)

    # -- OAuth flow ------------------------------------------------------
    def build_authorize_url(self) -> str:
        """URL the account owner opens in a browser to mint a new refresh token."""
        from urllib.parse import urlencode
        return AUTHORIZE_URL + "?" + urlencode({"client_id": self.app_key, "redirect_uri": self.callback_url})

    def exchange_code(self, redirect_url_or_code: str) -> dict:
        """Exchange the one-time ``code`` from the callback redirect for tokens."""
        from urllib.parse import parse_qs, unquote, urlparse
        code = redirect_url_or_code
        if "code=" in redirect_url_or_code:
            qs = parse_qs(urlparse(redirect_url_or_code).query)
            code = qs.get("code", [""])[0]
        code = unquote(code)
        return self._token_request({"grant_type": "authorization_code", "code": code, "redirect_uri": self.callback_url})

    def refresh(self) -> dict:
        rt = self._tokens.get("refresh_token")
        if not rt:
            raise RuntimeError("Schwab refresh token missing — run the OAuth login flow (build_authorize_url → exchange_code)")
        return self._token_request({"grant_type": "refresh_token", "refresh_token": rt})

    def _token_request(self, form: dict) -> dict:
        if httpx is None:
            raise RuntimeError("httpx not installed")
        basic = base64.b64encode(f"{self.app_key}:{self.app_secret}".encode()).decode()
        headers = {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"}
        with httpx.Client(timeout=30, verify=_ssl_context()) as c:
            r = c.post(TOKEN_URL, data=form, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"Schwab token endpoint {r.status_code}: {r.text[:200]}")
        tok = r.json()
        tok["expires_at"] = time.time() + float(tok.get("expires_in", 1800)) - 30
        tok["refresh_token_expires_at"] = time.time() + 7 * 86400 if "refresh_token" in tok else self._tokens.get("refresh_token_expires_at")
        if "refresh_token" not in tok and "refresh_token" in self._tokens:
            tok["refresh_token"] = self._tokens["refresh_token"]
        self._tokens = tok
        self._save_tokens()
        logger.info("Schwab access token refreshed (expires in %.0fs)", tok["expires_at"] - time.time())
        return tok

    # -- header ----------------------------------------------------------
    def auth_headers(self) -> dict:
        if self.mode == "proxy":
            return {}
        if self.mode == "token":
            return {"Authorization": f"Bearer {self._static_token}"}
        if not self._tokens.get("access_token") or time.time() >= float(self._tokens.get("expires_at", 0)):
            self.refresh()
        return {"Authorization": f"Bearer {self._tokens['access_token']}"}

    def status(self) -> dict:
        exp = float(self._tokens.get("expires_at", 0)) if self._tokens else 0.0
        return {
            "mode": self.mode,
            "has_refresh_token": bool(self._tokens.get("refresh_token")),
            "access_token_seconds_left": max(0.0, exp - time.time()) if exp else None,
            "callback_url": self.callback_url,
        }


# ---------------------------------------------------------------------------
# Normalised option contract
# ---------------------------------------------------------------------------
@dataclass
class OptionQuote:
    """One quoted contract from a Schwab chain, normalised for the options engine."""
    symbol: str
    underlying: str
    put_call: str            # "CALL" | "PUT"
    strike: float
    expiry: str              # YYYY-MM-DD
    dte: int
    bid: float
    ask: float
    last: float
    mark: float
    iv: float                # decimal (0.25 = 25 %) — Schwab returns percent
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    open_interest: int
    volume: int
    underlying_price: float
    in_the_money: bool = False

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last or self.mark

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m if m > 0 and self.ask >= self.bid else 1.0

    @property
    def moneyness(self) -> float:
        return self.strike / self.underlying_price if self.underlying_price > 0 else 1.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["mid"] = self.mid
        d["spread_pct"] = self.spread_pct
        return d


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------
class SchwabBroker:
    """Charles Schwab execution + data broker implementing BrokerProtocol."""

    QUOTE_CACHE_TTL = float(_CFG_QUOTE_TTL or 5)
    CHAIN_CACHE_TTL = 20.0
    HISTORY_CACHE_TTL = 300.0
    ACCOUNT_CACHE_TTL = 15.0
    MAX_ORDER_NOTIONAL_PCT = 0.10      # broker-edge hard cap per order (10 % NAV)

    paper = False                       # BrokerProtocol attr — Schwab has no paper mode

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        log_dir: Optional[str | Path] = None,
        account_number: Optional[str] = None,
        live_orders: Optional[bool] = None,
        daily_target_pct: float = 0.05,
        auth: Optional[SchwabAuth] = None,
        connect: bool = True,
    ):
        self._initial_cash = float(initial_cash or 100_000.0)
        self._log_dir = Path(log_dir or "logs/schwab_broker")
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._account_number = account_number or os.environ.get("SCHWAB_ACCOUNT_NUMBER", "")
        env_live = os.environ.get("SCHWAB_LIVE_ORDERS", "false").lower() in ("1", "true", "yes")
        self.live_orders = env_live if live_orders is None else bool(live_orders)

        self.auth = auth or SchwabAuth()
        self._client = httpx.Client(timeout=30, verify=_ssl_context()) if httpx is not None else None
        self._connected = False
        self._account_hash: Optional[str] = None
        self._accounts: List[dict] = []

        # local book (rebuilt from Schwab on sync)
        self.state = PortfolioState(cash=self._initial_cash, nav=self._initial_cash)
        self._orders: List[Order] = []
        self._trade_history: List[dict] = []
        self._option_positions: Dict[str, dict] = {}
        self._day_start_nav: Optional[float] = None   # anchored to first Schwab NAV read
        self._daily_pnl_today = 0.0
        self._trade_count = 0
        self._day_start_date = date.today()
        self._pending_slices: List[dict] = []

        # caches
        self._quote_cache: Dict[str, Tuple[dict, float]] = {}
        self._chain_cache: Dict[str, Tuple[List[OptionQuote], float]] = {}
        self._history_cache: Dict[str, Tuple[dict, float]] = {}
        self._account_cache: Tuple[Optional[dict], float] = (None, 0.0)

        # risk + performance
        self._risk_limiter = RiskLimiter(
            max_position_pct=0.10, max_sector_pct=0.30, max_single_name_pct=0.10,
            daily_loss_limit_pct=0.03, max_gross_exposure=2.5, max_net_exposure=1.5,
        )
        self._perf_tracker = PerformanceTracker(initial_nav=self._initial_cash)
        self._daily_target = DailyTargetManager(initial_nav=self._initial_cash)
        self._daily_target.DAILY_TARGET_PCT = daily_target_pct
        self._dashboard = LiveDashboardState()

        if connect:
            self._connect()

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------
    def _request(self, method: str, url: str, *, params: dict | None = None, json_body: dict | None = None, retries: int = 2) -> "httpx.Response":
        if self._client is None:
            raise RuntimeError("httpx not installed — pip install httpx")
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            headers = {"Accept": "application/json"}
            headers.update(self.auth.auth_headers())
            try:
                r = self._client.request(method, url, params=params, json=json_body, headers=headers)
            except Exception as e:  # network
                last_exc = e
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code == 401 and self.auth.mode == "oauth" and attempt < retries:
                try:
                    self.auth.refresh()
                    continue
                except Exception as e:
                    last_exc = e
                    break
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return r
        raise RuntimeError(f"Schwab request failed: {method} {url}: {last_exc}")

    def _get_json(self, url: str, params: dict | None = None) -> Any:
        r = self._request("GET", url, params=params)
        if r.status_code >= 400:
            raise RuntimeError(f"Schwab {r.status_code} on {url.split('/v1/')[-1]}: {r.text[:200]}")
        return r.json()

    # ------------------------------------------------------------------
    # Connection / account
    # ------------------------------------------------------------------
    def _connect(self):
        try:
            numbers = self._get_json(f"{TRADER_BASE}/accounts/accountNumbers")
            self._accounts = numbers or []
            chosen = None
            if self._account_number:
                for a in self._accounts:
                    if a.get("accountNumber", "").endswith(self._account_number):
                        chosen = a
                        break
            if chosen is None and self._accounts:
                chosen = self._accounts[0]
            if chosen is None:
                raise RuntimeError("no Schwab accounts returned")
            self._account_hash = chosen["hashValue"]
            self._connected = True
            acct = self.sync_account(force=True)
            self._daily_target.reset_day(self._day_start_nav or acct.get("nav", self._initial_cash))
            self.sync_positions()
            logger.info(
                "SchwabBroker connected: account ****%s | NAV $%.2f | live_orders=%s | auth=%s",
                chosen.get("accountNumber", "")[-4:], self._day_start_nav, self.live_orders, self.auth.mode,
            )
        except Exception as e:
            self._connected = False
            logger.warning("SchwabBroker connect failed: %s — running in DRY_RUN/offline mode", e)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def account_display(self) -> str:
        """Masked selected account, e.g. ``****9565`` (empty when not connected)."""
        for a in self._accounts:
            if a.get("hashValue") == self._account_hash:
                return "****" + a.get("accountNumber", "")[-4:]
        return ""

    def list_accounts(self) -> List[dict]:
        return [{"account_last4": a.get("accountNumber", "")[-4:], "selected": a.get("hashValue") == self._account_hash} for a in self._accounts]

    def sync_account(self, force: bool = False) -> dict:
        """Pull balances for the selected account (cached ACCOUNT_CACHE_TTL)."""
        cached, ts = self._account_cache
        if cached and not force and time.time() - ts < self.ACCOUNT_CACHE_TTL:
            return cached
        if not self._connected or not self._account_hash:
            return {"cash": self.state.cash, "nav": self.state.nav, "connected": False}
        try:
            data = self._get_json(f"{TRADER_BASE}/accounts/{self._account_hash}", params={"fields": "positions"})
            sa = data.get("securitiesAccount", {})
            bal = sa.get("currentBalances", {}) or {}
            nav = float(bal.get("liquidationValue") or bal.get("equity") or self.state.nav)
            cash = float(bal.get("cashBalance") if bal.get("cashBalance") is not None else bal.get("cashAvailableForTrading", nav))
            result = {
                "nav": nav,
                "cash": cash,
                "buying_power": float(bal.get("buyingPower") or bal.get("cashAvailableForTrading") or 0.0),
                "option_buying_power": float(bal.get("buyingPowerNonMarginableTrade") or bal.get("cashAvailableForTrading") or 0.0),
                "long_option_value": float(bal.get("longOptionMarketValue") or 0.0),
                "short_option_value": float(bal.get("shortOptionMarketValue") or 0.0),
                "maintenance_requirement": float(bal.get("maintenanceRequirement") or 0.0),
                "account_type": sa.get("type", ""),
                "is_day_trader": bool(sa.get("isDayTrader", False)),
                "round_trips": int(sa.get("roundTrips", 0) or 0),
                "connected": True,
                "raw_positions": sa.get("positions", []) or [],
            }
            if self._day_start_nav is None:
                # first read of the session: anchor day-start NAV and session baseline to Schwab truth
                self._day_start_nav = nav
                self._initial_cash = nav
                self._perf_tracker = PerformanceTracker(initial_nav=nav)
            self.state.cash = cash
            self.state.nav = nav
            self.state.total_pnl = nav - self._initial_cash
            self._daily_pnl_today = nav - self._day_start_nav
            self._record_nav_once_per_day(nav)
            self._account_cache = (result, time.time())
            return result
        except Exception as e:
            logger.error("Schwab account sync failed: %s", e)
            return {"cash": self.state.cash, "nav": self.state.nav, "connected": False, "error": str(e)}

    def _record_nav_once_per_day(self, nav: float):
        today = datetime.now().strftime("%Y-%m-%d")
        navs = self._perf_tracker._daily_navs
        if navs and navs[-1][0] == today:
            navs[-1] = (today, nav)
            if self._perf_tracker._daily_pnls:
                prev = navs[-2][1] if len(navs) >= 2 else self._perf_tracker.initial_nav
                self._perf_tracker._daily_pnls[-1] = (today, nav - prev)
            self._perf_tracker._high_water_mark = max(self._perf_tracker._high_water_mark, nav)
        else:
            self._perf_tracker.record_nav(nav, today)

    def sync_positions(self) -> Dict[str, Position]:
        """Rebuild the local book from Schwab positions (ground truth)."""
        acct = self.sync_account()
        raw = acct.get("raw_positions", [])
        positions: Dict[str, Position] = {}
        options: Dict[str, dict] = {}
        for p in raw:
            ins = p.get("instrument", {}) or {}
            sym = ins.get("symbol", "")
            qty = float(p.get("longQuantity", 0) or 0) - float(p.get("shortQuantity", 0) or 0)
            mv = float(p.get("marketValue", 0) or 0)
            avg = float(p.get("averagePrice", 0) or 0)
            if ins.get("assetType") == "OPTION":
                options[sym] = {
                    "symbol": sym,
                    "underlying": ins.get("underlyingSymbol", sym.split()[0] if sym else ""),
                    "put_call": ins.get("putCall", ""),
                    "quantity": int(qty),
                    "avg_price": avg,
                    "market_value": mv,
                    "description": ins.get("description", ""),
                    "day_pnl": float(p.get("currentDayProfitLoss", 0) or 0),
                }
                # options are tracked in the equity book too; current_price is the
                # per-CONTRACT value (premium × 100) so Position.market_value == Schwab marketValue
                pos = Position(ticker=sym, quantity=int(qty), avg_cost=avg * 100.0,
                               current_price=(mv / qty) if qty else avg * 100.0, sector="OPTIONS")
                pos.unrealized_pnl = mv - avg * qty * 100
                positions[sym] = pos
            else:
                price = (mv / qty) if qty else avg
                pos = Position(ticker=sym, quantity=int(qty), avg_cost=avg, current_price=price,
                               sector=ins.get("type", "") or "EQUITY")
                pos.unrealized_pnl = mv - avg * qty
                positions[sym] = pos
        self.state.positions = positions
        self._option_positions = options
        self._recompute_exposures()
        return positions

    def _recompute_exposures(self):
        nav = self.state.nav or 1.0
        long_v = sum(p.market_value for p in self.state.positions.values() if p.quantity > 0)
        short_v = sum(abs(p.market_value) for p in self.state.positions.values() if p.quantity < 0)
        self.state.gross_exposure = (long_v + short_v) / nav
        self.state.net_exposure = (long_v - short_v) / nav

    # ------------------------------------------------------------------
    # Market data — quotes
    # ------------------------------------------------------------------
    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, dict]:
        """Batch quotes (bid/ask/last/volume/52w) with TTL cache."""
        tickers = [t.upper() for t in tickers if t]
        now = time.time()
        out: Dict[str, dict] = {}
        missing = []
        for t in tickers:
            c = self._quote_cache.get(t)
            if c and now - c[1] < self.QUOTE_CACHE_TTL:
                out[t] = c[0]
            else:
                missing.append(t)
        if missing and self._connected:
            for i in range(0, len(missing), 300):
                batch = missing[i:i + 300]
                try:
                    data = self._get_json(f"{MARKET_BASE}/quotes", params={"symbols": ",".join(batch), "fields": "quote,reference"})
                except Exception as e:
                    logger.warning("Schwab quotes failed for %d symbols: %s", len(batch), e)
                    continue
                for sym, v in data.items():
                    q = v.get("quote", {}) or {}
                    ref = v.get("reference", {}) or {}
                    rec = {
                        "symbol": sym,
                        "last": float(q.get("lastPrice") or q.get("mark") or 0.0),
                        "bid": float(q.get("bidPrice") or 0.0),
                        "ask": float(q.get("askPrice") or 0.0),
                        "bid_size": int(q.get("bidSize") or 0),
                        "ask_size": int(q.get("askSize") or 0),
                        "mark": float(q.get("mark") or 0.0),
                        "volume": int(q.get("totalVolume") or 0),
                        "open": float(q.get("openPrice") or 0.0),
                        "high": float(q.get("highPrice") or 0.0),
                        "low": float(q.get("lowPrice") or 0.0),
                        "close": float(q.get("closePrice") or 0.0),
                        "net_pct_change": float(q.get("netPercentChange") or 0.0),
                        "high_52": float(q.get("52WeekHigh") or 0.0),
                        "low_52": float(q.get("52WeekLow") or 0.0),
                        "asset_type": v.get("assetMainType", ""),
                        "description": ref.get("description", ""),
                        "ts": now,
                    }
                    self._quote_cache[sym] = (rec, now)
                    out[sym] = rec
        return out

    def get_quote(self, ticker: str) -> Optional[float]:
        q = self.get_quotes([ticker]).get(ticker.upper())
        if not q:
            return None
        return q["last"] or q["mark"] or ((q["bid"] + q["ask"]) / 2 if q["bid"] and q["ask"] else None)

    def _get_current_price(self, ticker: str) -> float:
        return float(self.get_quote(ticker) or 0.0)

    def get_micro_price(self, ticker: str) -> Optional[float]:
        """Size-weighted micro-price from the top of book (WonderTrader input)."""
        q = self.get_quotes([ticker]).get(ticker.upper())
        if not q or not (q["bid"] and q["ask"]):
            return None
        bs, as_ = max(q["bid_size"], 1), max(q["ask_size"], 1)
        return (q["bid"] * as_ + q["ask"] * bs) / (bs + as_)

    # ------------------------------------------------------------------
    # Market data — price history
    # ------------------------------------------------------------------
    def get_price_history(self, ticker: str, days: int = 120, frequency: str = "daily") -> dict:
        """Daily (or minute) candles → dict of numpy arrays (cached 5 min)."""
        key = f"{ticker.upper()}|{days}|{frequency}"
        c = self._history_cache.get(key)
        if c and time.time() - c[1] < self.HISTORY_CACHE_TTL:
            return c[0]
        empty = {"close": np.array([]), "high": np.array([]), "low": np.array([]), "open": np.array([]), "volume": np.array([]), "dates": []}
        if not self._connected:
            return empty
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(days * 1.6) + 5)
        if frequency == "daily":
            params = {"symbol": ticker.upper(), "periodType": "year", "frequencyType": "daily", "frequency": 1,
                      "startDate": int(start.timestamp() * 1000), "endDate": int(end.timestamp() * 1000), "needExtendedHoursData": "false"}
        else:  # intraday minutes
            params = {"symbol": ticker.upper(), "periodType": "day", "period": 10, "frequencyType": "minute", "frequency": int(frequency), "needExtendedHoursData": "false"}
        try:
            data = self._get_json(f"{MARKET_BASE}/pricehistory", params=params)
        except Exception as e:
            logger.warning("Schwab price history failed for %s: %s", ticker, e)
            return empty
        candles = data.get("candles", []) or []
        candles = candles[-days:] if frequency == "daily" else candles
        out = {
            "close": np.array([c["close"] for c in candles], dtype=float),
            "high": np.array([c["high"] for c in candles], dtype=float),
            "low": np.array([c["low"] for c in candles], dtype=float),
            "open": np.array([c["open"] for c in candles], dtype=float),
            "volume": np.array([c["volume"] for c in candles], dtype=float),
            "dates": [datetime.fromtimestamp(c["datetime"] / 1000, tz=timezone.utc).date().isoformat() for c in candles],
        }
        self._history_cache[key] = (out, time.time())
        return out

    def get_returns(self, ticker: str, days: int = 120) -> np.ndarray:
        h = self.get_price_history(ticker, days=days)
        close = h["close"]
        if close.size < 3:
            return np.array([])
        return np.diff(np.log(close))

    # ------------------------------------------------------------------
    # Market data — option chains (1-7 DTE default)
    # ------------------------------------------------------------------
    def get_option_chain(
        self,
        underlying: str,
        dte_min: int = 1,
        dte_max: int = 7,
        strike_count: int = 20,
        contract_type: str = "ALL",
    ) -> List[OptionQuote]:
        """Normalised chain restricted to the [dte_min, dte_max] window.

        Schwab returns IV in percent and Greeks per contract; we convert IV to a
        decimal and drop contracts with no two-sided market.
        """
        key = f"{underlying.upper()}|{dte_min}|{dte_max}|{strike_count}|{contract_type}"
        c = self._chain_cache.get(key)
        if c and time.time() - c[1] < self.CHAIN_CACHE_TTL:
            return c[0]
        if not self._connected:
            return []
        today = date.today()
        params = {
            "symbol": underlying.upper(),
            "contractType": contract_type,
            "strikeCount": strike_count,
            "includeUnderlyingQuote": "true",
            "fromDate": (today + timedelta(days=max(0, dte_min))).isoformat(),
            "toDate": (today + timedelta(days=dte_max)).isoformat(),
        }
        try:
            data = self._get_json(f"{MARKET_BASE}/chains", params=params)
        except Exception as e:
            logger.warning("Schwab chain failed for %s: %s", underlying, e)
            return []
        und_px = float(data.get("underlyingPrice") or (data.get("underlying") or {}).get("last") or 0.0)
        quotes: List[OptionQuote] = []
        for map_key, pc in (("callExpDateMap", "CALL"), ("putExpDateMap", "PUT")):
            for exp_key, strikes in (data.get(map_key) or {}).items():
                exp_date = exp_key.split(":")[0]
                for strike_str, contracts in strikes.items():
                    for o in contracts:
                        dte = int(o.get("daysToExpiration", 0) or 0)
                        if dte < dte_min or dte > dte_max:
                            continue
                        bid, ask = float(o.get("bid") or 0), float(o.get("ask") or 0)
                        if bid <= 0 or ask <= 0:
                            continue
                        iv_pct = float(o.get("volatility") or 0.0)
                        if iv_pct <= 0 or iv_pct > 900:
                            continue
                        quotes.append(OptionQuote(
                            symbol=o.get("symbol", ""), underlying=underlying.upper(), put_call=pc,
                            strike=float(strike_str), expiry=exp_date, dte=dte, bid=bid, ask=ask,
                            last=float(o.get("last") or 0), mark=float(o.get("mark") or 0), iv=iv_pct / 100.0,
                            delta=float(o.get("delta") or 0), gamma=float(o.get("gamma") or 0),
                            theta=float(o.get("theta") or 0), vega=float(o.get("vega") or 0), rho=float(o.get("rho") or 0),
                            open_interest=int(o.get("openInterest") or 0), volume=int(o.get("totalVolume") or 0),
                            underlying_price=und_px, in_the_money=bool(o.get("inTheMoney", False)),
                        ))
        self._chain_cache[key] = (quotes, time.time())
        return quotes

    # ------------------------------------------------------------------
    # Orders — equities
    # ------------------------------------------------------------------
    def _new_order(self, ticker: str, side: OrderSide, quantity: int, signal_type: SignalType,
                   limit_price: Optional[float], reason: str) -> Order:
        return Order(
            id=str(uuid.uuid4())[:12], ticker=ticker.upper(), side=side,
            order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
            quantity=int(quantity), limit_price=limit_price, signal_type=signal_type,
            timestamp=_now_iso(), reason=reason,
        )

    def _broker_edge_risk(self, ticker: str, notional: float, sector: str = "EQUITY") -> Tuple[bool, List[str]]:
        nav = self.state.nav or self._initial_cash
        if self._connected and self._account_cache[0] is None:
            self.sync_account()
        if nav > 0 and notional / nav > self.MAX_ORDER_NOTIONAL_PCT:
            return False, [f"broker-edge: order notional {notional/nav:.1%} > {self.MAX_ORDER_NOTIONAL_PCT:.0%} NAV"]
        return self._risk_limiter.run_all_checks(
            notional, ticker, sector, self.state.positions, nav, self._daily_pnl_today,
            self.state.gross_exposure, self.state.net_exposure,
        )

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        signal_type: SignalType = SignalType.HOLD,
        limit_price: Optional[float] = None,
        reason: str = "",
        sector: str = "EQUITY",
    ) -> Order:
        """Equity order (market or limit). Honors dry-run and broker-edge risk."""
        order = self._new_order(ticker, side, quantity, signal_type, limit_price, reason)
        if quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.reason = "quantity must be > 0"
            return self._finish(order, "EQUITY")
        px = limit_price or self._get_current_price(ticker) or 0.0
        ok, failures = self._broker_edge_risk(ticker, abs(quantity) * px, sector)
        if not ok:
            order.status = OrderStatus.REJECTED
            order.reason = "; ".join(failures)
            return self._finish(order, "EQUITY")
        payload = {
            "orderType": "LIMIT" if limit_price else "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": _EQUITY_INSTRUCTION.get(side, "BUY"),
                "quantity": int(abs(quantity)),
                "instrument": {"symbol": ticker.upper(), "assetType": "EQUITY"},
            }],
        }
        if limit_price:
            payload["price"] = f"{limit_price:.2f}"
        return self._submit(order, payload, px, "EQUITY")

    def place_twap_order(self, ticker: str, side: OrderSide, quantity: int, duration_minutes: int = 30,
                         signal_type: SignalType = SignalType.HOLD, limit_price: Optional[float] = None,
                         reason: str = "", slices: int = 4) -> Order:
        """Schwab has no native TWAP: first slice now, remaining slices on heartbeat()."""
        slices = max(1, min(slices, quantity))
        per = max(1, quantity // slices)
        first = self.place_order(ticker, side, per, signal_type, limit_price, f"TWAP 1/{slices} {reason}".strip())
        remaining = quantity - per
        if remaining > 0 and first.status != OrderStatus.REJECTED:
            interval = max(30.0, duration_minutes * 60.0 / slices)
            for i in range(1, slices):
                q = per if i < slices - 1 else remaining - per * (slices - 2)
                if q <= 0:
                    break
                self._pending_slices.append({"due": time.time() + i * interval, "ticker": ticker, "side": side, "quantity": q,
                                             "signal_type": signal_type, "limit_price": limit_price,
                                             "reason": f"TWAP {i+1}/{slices} {reason}".strip(), "parent": first.id})
        return first

    def place_vwap_order(self, ticker: str, side: OrderSide, quantity: int, duration_minutes: int = 60,
                         max_pct_volume: float = 0.25, signal_type: SignalType = SignalType.HOLD,
                         limit_price: Optional[float] = None, reason: str = "") -> Order:
        """VWAP approximated as participation-capped slicing (no native algo on Schwab)."""
        return self.place_twap_order(ticker, side, quantity, duration_minutes, signal_type, limit_price,
                                     f"VWAP({max_pct_volume:.0%}) {reason}".strip(), slices=6)

    # ------------------------------------------------------------------
    # Orders — options
    # ------------------------------------------------------------------
    def place_option_order(
        self,
        option_symbol: str,
        instruction: str,
        quantity: int,
        limit_price: float,
        underlying: str = "",
        signal_type: SignalType = SignalType.HOLD,
        reason: str = "",
    ) -> Order:
        """Single-leg option order. ``option_symbol`` is the Schwab OCC symbol
        (e.g. ``"SPY   260910C00775000"``). Options are ALWAYS limit orders."""
        instruction = instruction.upper()
        if instruction not in OPTION_INSTRUCTIONS:
            raise ValueError(f"instruction must be one of {sorted(OPTION_INSTRUCTIONS)}")
        side = OrderSide.BUY if instruction.startswith("BUY") else OrderSide.SELL
        order = self._new_order(option_symbol, side, quantity, signal_type, limit_price, reason)
        order.order_type = OrderType.LIMIT
        notional = abs(quantity) * float(limit_price) * 100.0
        ok, failures = self._broker_edge_risk(underlying or option_symbol.split()[0], notional, "OPTIONS")
        if not ok:
            order.status = OrderStatus.REJECTED
            order.reason = "; ".join(failures)
            return self._finish(order, "OPTION")
        payload = {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "duration": "DAY",
            "price": f"{float(limit_price):.2f}",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": instruction,
                "quantity": int(abs(quantity)),
                "instrument": {"symbol": option_symbol, "assetType": "OPTION"},
            }],
        }
        return self._submit(order, payload, float(limit_price), "OPTION")

    def place_option_spread(
        self,
        legs: List[dict],
        net_price: float,
        quantity: int,
        underlying: str = "",
        signal_type: SignalType = SignalType.HOLD,
        reason: str = "",
        strategy: str = "VERTICAL",
        is_debit: bool = True,
    ) -> Order:
        """Multi-leg option order (vertical/strangle/straddle) at a NET limit price.

        ``legs`` = [{"symbol": OCC, "instruction": "BUY_TO_OPEN", "quantity": 1}, ...]
        """
        label = "+".join(l["symbol"].split()[-1] for l in legs)
        order = self._new_order(f"{underlying or legs[0]['symbol'].split()[0]}:{strategy}:{label}",
                                OrderSide.BUY if is_debit else OrderSide.SELL, quantity, signal_type, net_price, reason)
        order.order_type = OrderType.LIMIT
        notional = abs(quantity) * abs(float(net_price)) * 100.0
        ok, failures = self._broker_edge_risk(underlying or legs[0]["symbol"].split()[0], notional, "OPTIONS")
        if not ok:
            order.status = OrderStatus.REJECTED
            order.reason = "; ".join(failures)
            return self._finish(order, "OPTION")
        payload = {
            "orderType": "NET_DEBIT" if is_debit else "NET_CREDIT",
            "session": "NORMAL",
            "duration": "DAY",
            "price": f"{abs(float(net_price)):.2f}",
            "orderStrategyType": "SINGLE",
            "complexOrderStrategyType": strategy,
            "orderLegCollection": [
                {"instruction": l["instruction"], "quantity": int(l.get("quantity", 1)) * int(abs(quantity)),
                 "instrument": {"symbol": l["symbol"], "assetType": "OPTION"}}
                for l in legs
            ],
        }
        return self._submit(order, payload, abs(float(net_price)), "OPTION")

    # ------------------------------------------------------------------
    # Submission core
    # ------------------------------------------------------------------
    def _submit(self, order: Order, payload: dict, ref_price: float, product: str) -> Order:
        if not self.live_orders:
            order.status = OrderStatus.DRY_RUN
            order.fill_price = float(ref_price or 0.0)
            order.fill_timestamp = _now_iso()
            order.reason = (order.reason + " [DRY_RUN: SCHWAB_LIVE_ORDERS=false]").strip()
            return self._finish(order, product, payload)
        if not self._connected or not self._account_hash:
            order.status = OrderStatus.REJECTED
            order.reason = (order.reason + " [Schwab not connected]").strip()
            return self._finish(order, product, payload)
        try:
            r = self._request("POST", f"{TRADER_BASE}/accounts/{self._account_hash}/orders", json_body=payload, retries=0)
            if r.status_code in (200, 201):
                loc = r.headers.get("Location", "")
                order.reason = (order.reason + f" [schwab_order_id={loc.rsplit('/', 1)[-1]}]").strip()
                order.status = OrderStatus.PENDING
                self._trade_count += 1
                # poll once for an immediate fill (marketable limits usually fill instantly)
                time.sleep(0.6)
                self._refresh_order_status(order, loc.rsplit("/", 1)[-1], ref_price)
            else:
                order.status = OrderStatus.REJECTED
                order.reason = (order.reason + f" [Schwab {r.status_code}: {r.text[:160]}]").strip()
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reason = (order.reason + f" [Schwab error: {e}]").strip()
        return self._finish(order, product, payload)

    def _refresh_order_status(self, order: Order, schwab_id: str, ref_price: float):
        if not schwab_id:
            return
        try:
            data = self._get_json(f"{TRADER_BASE}/accounts/{self._account_hash}/orders/{schwab_id}")
            st = data.get("status", "")
            if st == "FILLED":
                fills = [a for a in data.get("orderActivityCollection", []) if a.get("executionType") == "FILL"]
                px = 0.0
                qty = 0.0
                for a in fills:
                    for leg in a.get("executionLegs", []):
                        px += float(leg.get("price", 0)) * float(leg.get("quantity", 0))
                        qty += float(leg.get("quantity", 0))
                order.fill_price = px / qty if qty else float(ref_price)
                order.status = OrderStatus.FILLED
                order.fill_timestamp = _now_iso()
            elif st in ("REJECTED", "CANCELED", "EXPIRED"):
                order.status = OrderStatus.REJECTED if st == "REJECTED" else OrderStatus.CANCELLED
                order.reason = (order.reason + f" [{st}: {data.get('statusDescription', '')}]").strip()
        except Exception as e:
            logger.debug("order status refresh failed: %s", e)

    def _finish(self, order: Order, product: str, payload: Optional[dict] = None) -> Order:
        self._orders.append(order)
        entry = order.to_dict()
        entry.update({"product": product, "live_orders": self.live_orders, "logged_at": _now_iso(), "payload": payload})
        self._trade_history.append(entry)
        self._perf_tracker.record_trade(entry)
        self._log_order(entry)
        if order.status == OrderStatus.FILLED:
            self._account_cache = (None, 0.0)  # force re-sync
        return order

    def _log_order(self, entry: dict):
        f = self._log_dir / f"orders_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(f, "a") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.debug("Schwab order log write failed: %s", e)

    def cancel_order(self, schwab_order_id: str) -> bool:
        if not self.live_orders or not self._connected:
            return False
        r = self._request("DELETE", f"{TRADER_BASE}/accounts/{self._account_hash}/orders/{schwab_order_id}", retries=0)
        return r.status_code in (200, 204)

    def get_open_orders(self) -> List[dict]:
        if not self._connected:
            return []
        now = datetime.now(timezone.utc)
        params = {"fromEnteredTime": (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                  "toEnteredTime": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "status": "WORKING"}
        try:
            return self._get_json(f"{TRADER_BASE}/accounts/{self._account_hash}/orders", params=params) or []
        except Exception as e:
            logger.debug("open orders fetch failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Heartbeat — run pending TWAP slices, refresh book
    # ------------------------------------------------------------------
    def heartbeat(self) -> dict:
        fired = 0
        now = time.time()
        if self._day_start_nav is not None and self._day_start_date != date.today():
            self.reset_daily_target()
        still = []
        for s in self._pending_slices:
            if s["due"] <= now:
                self.place_order(s["ticker"], s["side"], s["quantity"], s["signal_type"], s["limit_price"], s["reason"])
                fired += 1
            else:
                still.append(s)
        self._pending_slices = still
        self.sync_positions()
        self._daily_target.update(self.state.nav)
        return {"slices_fired": fired, "slices_pending": len(still), "nav": self.state.nav, "daily_pnl": self._daily_pnl_today}

    def check_algo_fills(self) -> List[dict]:
        return [o.to_dict() for o in self._orders if o.status == OrderStatus.FILLED][-20:]

    # ------------------------------------------------------------------
    # BrokerProtocol surface
    # ------------------------------------------------------------------
    def get_trade_history(self, last_n: int = 0) -> List[dict]:
        return self._trade_history[-last_n:] if last_n else list(self._trade_history)

    def get_orders(self) -> List[Order]:
        return list(self._orders)

    def get_all_positions(self) -> Dict[str, Position]:
        return dict(self.state.positions)

    def get_positions(self) -> Dict[str, Position]:
        return self.get_all_positions()

    def get_position(self, ticker: str) -> Optional[Position]:
        return self.state.positions.get(ticker.upper())

    def get_option_positions(self) -> Dict[str, dict]:
        return dict(self._option_positions)

    def compute_nav(self) -> float:
        return float(self.sync_account().get("nav", self.state.nav))

    def get_nav(self) -> float:
        return self.compute_nav()

    def compute_exposures(self) -> dict:
        self._recompute_exposures()
        opt_long = sum(p.market_value for p in self.state.positions.values() if p.sector == "OPTIONS" and p.quantity > 0)
        return {"gross": self.state.gross_exposure, "net": self.state.net_exposure,
                "options_long_value": opt_long, "nav": self.state.nav}

    def get_drawdown(self) -> dict:
        return self._perf_tracker.get_drawdown()

    def refresh_prices(self):
        if self.state.positions:
            eq = [t for t, p in self.state.positions.items() if p.sector != "OPTIONS"]
            qs = self.get_quotes(eq)
            for t, q in qs.items():
                if t in self.state.positions and q["last"]:
                    self.state.positions[t].current_price = q["last"]
        self._recompute_exposures()

    def get_risk_profile(self) -> RiskProfile:
        return self._daily_target.profile

    def get_daily_target_state(self) -> dict:
        return self._daily_target.get_state()

    def reset_daily_target(self):
        acct = self.sync_account(force=True)
        self._day_start_nav = acct.get("nav", self.state.nav)
        self._daily_pnl_today = 0.0
        self._day_start_date = date.today()
        self._daily_target.reset_day(self._day_start_nav)

    def get_leverage_multiplier(self) -> float:
        return self._daily_target.get_leverage_multiplier()

    def emit_dashboard_state(self, pipeline_state: Optional[dict] = None) -> dict:
        return self._dashboard.emit(self.get_portfolio_summary(), self.get_daily_target_state(), pipeline_state)

    def get_dashboard_snapshot(self) -> dict:
        return self._dashboard.get_latest()

    def get_dashboard_history(self, n: int = 100) -> List[dict]:
        return self._dashboard.get_history(n)

    def register_dashboard_callback(self, cb):
        self._dashboard.register_callback(cb)

    def get_performance_metrics(self) -> dict:
        return self._perf_tracker.get_all_metrics()

    @property
    def connected(self) -> bool:
        return self._connected

    def get_equity_pnl(self) -> float:
        return sum(p.unrealized_pnl + p.realized_pnl for p in self.state.positions.values() if p.sector != "OPTIONS")

    def get_options_pnl(self) -> float:
        return sum(p.unrealized_pnl + p.realized_pnl for p in self.state.positions.values() if p.sector == "OPTIONS")

    def get_sector_pnl(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p in self.state.positions.values():
            out[p.sector or "EQUITY"] = out.get(p.sector or "EQUITY", 0.0) + p.unrealized_pnl + p.realized_pnl
        return out

    def get_daily_pnl(self) -> float:
        return self._daily_pnl_today

    def reconcile(self) -> dict:
        """Schwab is ground truth — rebuild the book and report what changed."""
        before = {t: p.quantity for t, p in self.state.positions.items()}
        after = {t: p.quantity for t, p in self.sync_positions().items()}
        return {
            "added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(t for t in set(before) & set(after) if before[t] != after[t]),
            "position_count": len(after),
            "nav": self.state.nav,
        }

    def export_positions_csv(self) -> str:
        lines = ["ticker,quantity,avg_cost,current_price,market_value,unrealized_pnl,sector"]
        for p in self.state.positions.values():
            lines.append(f"{p.ticker},{p.quantity},{p.avg_cost:.4f},{p.current_price:.4f},{p.market_value:.2f},{p.unrealized_pnl:.2f},{p.sector}")
        return "\n".join(lines)

    def get_portfolio_summary(self) -> dict:
        acct = self.sync_account()
        nav = acct.get("nav", self.state.nav)
        return {
            "nav": nav,
            "cash": acct.get("cash", self.state.cash),
            "buying_power": acct.get("buying_power", 0.0),
            "option_buying_power": acct.get("option_buying_power", 0.0),
            "total_pnl": nav - self._initial_cash,
            "total_pnl_pct": (nav - self._initial_cash) / self._initial_cash if self._initial_cash else 0.0,
            "daily_pnl": self._daily_pnl_today,
            "position_count": len(self.state.positions),
            "option_position_count": len(self._option_positions),
            "trade_count": self._trade_count,
            "orders_logged": len(self._orders),
            "gross_exposure": self.state.gross_exposure,
            "net_exposure": self.state.net_exposure,
            "broker": "SCHWAB",
            "connected": self._connected,
            "live_orders": self.live_orders,
            "paper": False,
            "account": self.list_accounts(),
            "auth": self.auth.status(),
        }

    def get_status(self) -> dict:
        return self.get_portfolio_summary()

    def __repr__(self) -> str:
        return f"SchwabBroker(connected={self._connected}, live_orders={self.live_orders}, nav={self.state.nav:.2f})"


# ---------------------------------------------------------------------------
# Process-wide shared broker (replaces the old ad-hoc ``PaperBroker()`` calls).
# ExecutionEngine registers its broker here; everything else asks for it.
# ---------------------------------------------------------------------------
_SHARED_BROKER: Any = None


def set_shared_broker(broker: Any) -> Any:
    global _SHARED_BROKER
    _SHARED_BROKER = broker
    return broker


def get_shared_broker() -> Any:
    """Return the engine's Schwab broker (router or single account). If no
    ExecutionEngine has been built yet, returns an offline DRY_RUN SchwabBroker
    so read-only callers (reports, metrics, archives) never fail."""
    global _SHARED_BROKER
    if _SHARED_BROKER is None:
        _SHARED_BROKER = SchwabBroker(connect=False)
    return _SHARED_BROKER
