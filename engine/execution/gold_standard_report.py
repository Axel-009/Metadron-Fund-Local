"""
Gold-standard report — VIEW 1 / VIEW 2 / TRANSACTION LOG / VIEW 3.
==================================================================

Renders one full-scan cycle (or the end-of-day recap) in the operator's
"gold standard" layout:

    VIEW 1 — THINKING OUTPUT        Phase 2 macro + Cube, one block per universe
                                    RUN (SP500 → SP400 → SP600 → ETF_FI), short-DTE
                                    options engine, Phase 5 decisions, Phase 6
                                    learning (in-chat patch), Phase 7 monitoring
    VIEW 2 — SCORECARD + ALLOCATION per-run scorecards, deployment summary,
                                    allocation fill rates vs AllocationRules
    TRANSACTION LOG                 rotation exits + new deployments + net flow
    VIEW 3 — LIVE POSITION PANEL    equities grouped by allocation bucket, short
                                    hedge book, options overlay (1–7 DTE), margin &
                                    money market, grand total, account mandates

Everything is pure rendering over a `GoldStandardContext`; `collect_context()`
builds that context from the live objects (Schwab broker / router, tranche
scan, ShortDTE scan, L7 orders, macro flag, drawdown gate, EOD council).
The same renderer serves three prompts:

    * every 30-minute full scan            → render(ctx)
    * "gold standard report" at any time   → render(ctx)  (no new scan needed)
    * end-of-day recap                     → render(ctx, recap=True)  (positions +
                                             all of today's transactions)

Futures are gone (strictly 1–7 DTE options overlay) so the FUTURES rows of the
original layout are omitted by design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

W1 = 114    # VIEW 1 / VIEW 2 rule width
W3 = 135    # VIEW 3 rule width

# ── allocation buckets (display order, description, AllocationRules attribute) ──
BUCKETS: List[Tuple[str, str, str]] = [
    ("IG_EQUITY", "Investment Grade Single-Name Equities", "ig_equity_pct"),
    ("HY_EQUITY", "High Yield (BB-B Rated)", "hy_equity_pct"),
    ("DISTRESSED", "Distressed / Fallen Angels / Recovery", "distressed_equity_pct"),
    ("TLTW_CASHFLOW", "TLTW / Cashflow ETFs (DRIP Reinvest)", "tltw_cashflow_pct"),
    ("FI_MACRO", "Fixed Income + Macro Signals", "fi_macro_pct"),
    ("EVENT_CVR", "Event-Driven + Contingent Value Rights", "event_driven_cvr_pct"),
]
BUCKET_KEYS = [b[0] for b in BUCKETS]

# every spelling used across the engine → report bucket key
_ALIAS = {
    "IG_EQUITY": "IG_EQUITY", "IG": "IG_EQUITY", "QUALITY": "IG_EQUITY",
    "HY_EQUITY": "HY_EQUITY", "HY": "HY_EQUITY",
    "DISTRESSED": "DISTRESSED", "DISTRESSED_EQUITY": "DISTRESSED",
    "TLTW": "TLTW_CASHFLOW", "TLTW_CASHFLOW": "TLTW_CASHFLOW", "ETF": "TLTW_CASHFLOW",
    "FIXED_INCOME": "FI_MACRO", "FI_MACRO": "FI_MACRO", "FI": "FI_MACRO",
    "CVR": "EVENT_CVR", "EVENT_CVR": "EVENT_CVR", "EVENT_DRIVEN_CVR": "EVENT_CVR",
    "OPTIONS": "OPTIONS", "OPTIONS_IG": "OPTIONS", "OPTIONS_HY": "OPTIONS", "OPTIONS_DISTRESSED": "OPTIONS",
}

try:  # sleeve hints from the universe data (SP500 → IG, SP400/600 → HY)
    from engine.data.cross_asset_universe import SP500_TICKERS, SP400_TICKERS, SP600_TICKERS
except Exception:  # noqa: BLE001
    SP500_TICKERS, SP400_TICKERS, SP600_TICKERS = [], [], []
try:
    from engine.execution.universe_tranche_scan import BOND_ETFS, INCOME_ETFS, ETF_HINTS
except Exception:  # noqa: BLE001
    BOND_ETFS, INCOME_ETFS, ETF_HINTS = set(), set(), set()

_SP500, _SP400, _SP600 = set(SP500_TICKERS), set(SP400_TICKERS), set(SP600_TICKERS)


def norm_bucket(name: str) -> str:
    return _ALIAS.get((name or "").upper().strip(), "")


def classify_bucket(ticker: str, sector_hint: str = "", hints: Optional[Dict[str, str]] = None) -> str:
    """Report bucket for a position (options → 'OPTIONS')."""
    t = (ticker or "").upper()
    if sector_hint == "OPTIONS" or " " in t.strip() or (len(t) > 15 and t[-9] in "CP"):
        return "OPTIONS"
    if hints and t in hints and norm_bucket(hints[t]):
        return norm_bucket(hints[t])
    b = norm_bucket(sector_hint)
    if b:
        return b
    if t in INCOME_ETFS:
        return "TLTW_CASHFLOW"
    if t in BOND_ETFS:
        return "FI_MACRO"
    if t in ("HYG", "JNK"):
        return "DISTRESSED"
    if t in ETF_HINTS:
        return "TLTW_CASHFLOW"
    if t in _SP500:
        return "IG_EQUITY"
    if t in _SP400 or t in _SP600:
        return "HY_EQUITY"
    return "IG_EQUITY"


def fill_status(pct: float) -> str:
    if pct > 120:
        return "⚠ OVER"
    if pct >= 80:
        return "✓ FILLED"
    if pct >= 40:
        return "◐ PARTIAL"
    return "○ LIGHT"


def money(v: float, width: int = 0, plus: bool = False) -> str:
    s = f"{v:+,.2f}" if plus else f"{v:,.2f}"
    return f"$ {s:>{width}}" if width else f"${s}"


# ─────────────────────────────────────────────────────────────────────────────
# Context
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionRow:
    symbol: str
    qty: float
    entry: float
    current: float
    market_value: float
    pnl: float
    pnl_pct: float
    bucket: str
    is_new: bool = False
    is_option: bool = False
    account: str = ""


@dataclass
class GoldStandardContext:
    as_of: datetime
    engine_label: str = "Metadron L7 · Schwab (7-phase pipeline; Phase 6/7 → in-chat patch)"
    mode: str = "DRY_RUN"                 # DRY_RUN | LIVE | OFFLINE
    account_label: str = ""
    cycle_time_s: float = 0.0
    nav: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    day_pnl: float = 0.0
    macro: Dict[str, Any] = field(default_factory=dict)
    runs: List[Any] = field(default_factory=list)            # TrancheResult list
    concurrence: Any = None                                  # ConcurrenceResult
    options_report: Optional[dict] = None                    # ShortDTE last_run
    exits: List[Any] = field(default_factory=list)           # RotationExit list / dicts
    deployments: List[dict] = field(default_factory=list)    # {ticker,$,bucket,signal,alpha,side,status}
    learning: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    positions: List[PositionRow] = field(default_factory=list)
    accounts: Dict[str, dict] = field(default_factory=dict)  # router portfolio_snapshot()["accounts"]
    rules: Any = None
    macro_flag: Optional[dict] = None
    drawdown: Optional[dict] = None
    council: Any = None
    notes: List[str] = field(default_factory=list)

    # ── derived ──────────────────────────────────────────────────────────────
    def target_pct(self, bucket: str) -> float:
        r = self.rules
        for key, _, attr in BUCKETS:
            if key == bucket:
                return float(getattr(r, attr, 0.0)) if r is not None else 0.0
        if bucket == "OPTIONS":
            return float(getattr(r, "options_notional_pct", 0.25)) if r is not None else 0.25
        if bucket == "MARGIN":
            return float(getattr(r, "margin_pct", 0.08)) if r is not None else 0.08
        if bucket == "MONEY_MARKET":
            return float(getattr(r, "money_market_pct", 0.02)) if r is not None else 0.02
        return 0.0

    def equities(self) -> List[PositionRow]:
        return [p for p in self.positions if not p.is_option]

    def options(self) -> List[PositionRow]:
        return [p for p in self.positions if p.is_option]

    def by_bucket(self, bucket: str) -> List[PositionRow]:
        return sorted([p for p in self.equities() if p.bucket == bucket and p.qty > 0], key=lambda p: -abs(p.market_value))

    def shorts(self) -> List[PositionRow]:
        return sorted([p for p in self.equities() if p.qty < 0], key=lambda p: p.market_value)


# ─────────────────────────────────────────────────────────────────────────────
# Collector — live objects → context
# ─────────────────────────────────────────────────────────────────────────────
def _pct(a: float, b: float) -> float:
    return (a / b * 100.0) if b else 0.0


def build_macro(broker: Any, sd: Any = None, macro_flag: Any = None, vix: Optional[float] = None,
                spy_close: Any = None) -> Dict[str, Any]:
    """Phase 2 block from Schwab data: ShortDTE MarketContext + MetadronCube on a Schwab-fed MacroSnapshot."""
    import numpy as np
    m: Dict[str, Any] = {"regime": "NORMAL", "vix": vix, "spy_1m": None, "spy_3m": None, "gmtf": None, "cube": "n/a",
                         "max_leverage": None, "beta_cap": None, "target_beta": None, "corridor": "", "direction_bias": 0.0,
                         "kill_switch": False, "L": None, "R": None, "F": None, "gates": {}}
    ctx = None
    if sd is not None:
        try:
            ctx = sd.market_context()
            m.update(regime=ctx.regime, vix=float(ctx.vix), corridor=ctx.corridor_position,
                     direction_bias=float(ctx.direction_bias), target_beta=float(ctx.target_beta))
        except Exception:  # noqa: BLE001
            pass
    close = None
    try:
        if spy_close is not None and len(spy_close) > 70:
            close = np.asarray(spy_close, dtype=float)
        elif broker is not None and hasattr(broker, "get_price_history"):
            h = broker.get_price_history("SPY", 120) or {}
            c = h.get("close")
            close = np.asarray(c, dtype=float) if c is not None and len(c) > 70 else None
    except Exception:  # noqa: BLE001
        close = None
    if close is not None:
        m["spy_1m"] = float(close[-1] / close[-22] - 1.0)
        m["spy_3m"] = float(close[-1] / close[-64] - 1.0)
    try:
        from engine.signals.metadron_cube import MetadronCube
        from engine.signals.macro_engine import MacroSnapshot
        snap = MacroSnapshot(vix=float(m["vix"] or 20.0), spy_return_1m=float(m["spy_1m"] or 0.0),
                             spy_return_3m=float(m["spy_3m"] or 0.0))
        cube = MetadronCube()
        out = cube.compute(snap)
        m.update(cube=getattr(out.regime, "value", str(out.regime)), max_leverage=float(out.max_leverage),
                 beta_cap=float(out.beta_cap), L=float(out.liquidity.value), R=float(out.risk.value), F=float(out.flow.value))
        ks = cube.get_kill_switch() if hasattr(cube, "get_kill_switch") else None
        m["kill_switch"] = bool(ks.is_active()) if ks is not None and hasattr(ks, "is_active") else False
        m["gmtf"] = float(getattr(snap, "gmtf_score", 0.0) or 0.0) or None
    except Exception as exc:  # noqa: BLE001
        m["cube_error"] = str(exc)
    if macro_flag is not None:
        fl = macro_flag.to_dict() if hasattr(macro_flag, "to_dict") else dict(macro_flag)
        m["flag"] = fl
    return m


def _gates_from_runs(runs: List[Any], macro: Dict[str, Any]) -> Dict[str, float]:
    """4-gate proxies averaged over the finalists of every run (0..1)."""
    import numpy as np
    top = [c for r in runs for c in getattr(r, "top", [])]
    if not top:
        return {}
    mom = float(np.mean([(c.momentum_score + 1) / 2 for c in top]))
    flow = float(np.mean([min(1.0, math.log10(max(c.dollar_volume, 1.0)) / 9.0) for c in top]))
    fund = float(np.mean([c.ensemble for c in top]))
    fl = macro.get("flag") or {}
    macro_g = max(0.0, 1.0 - 0.25 * float(fl.get("score", 0) or 0))
    return {"G1_flow": round(flow, 2), "G2_macro": round(macro_g, 2), "G3_fund": round(fund, 2), "G4_mom": round(mom, 2)}


def collect_context(
    exec_engine: Any,
    *,
    tranche_result: Any = None,
    options_report: Optional[dict] = None,
    orders: Iterable[dict] = (),
    exits: Iterable[Any] = (),
    macro_flag: Any = None,
    drawdown: Optional[dict] = None,
    council: Any = None,
    vix: Optional[float] = None,
    cycle_time_s: float = 0.0,
    mode: Optional[str] = None,
    new_tickers: Iterable[str] = (),
    bucket_hints: Optional[Dict[str, str]] = None,
    learning: Optional[Dict[str, Any]] = None,
    as_of: Optional[datetime] = None,
) -> GoldStandardContext:
    broker = getattr(exec_engine, "broker", None)
    sd = getattr(exec_engine, "options_engine_short_dte", None)
    try:
        from engine.allocation.allocation_engine import AllocationRules
        rules = AllocationRules()
    except Exception:  # noqa: BLE001
        rules = None
    state = getattr(broker, "state", None)
    nav = float(getattr(state, "nav", 0.0) or 0.0)
    cash = float(getattr(state, "cash", 0.0) or 0.0)
    connected = bool(getattr(broker, "is_connected", False)) if broker is not None else False
    live = bool(getattr(broker, "live_orders", False)) if broker is not None else False
    mode = mode or ("LIVE" if live else ("DRY_RUN" if connected else "OFFLINE"))

    accounts: Dict[str, dict] = {}
    bp = 0.0
    day_pnl = 0.0
    if broker is not None and hasattr(broker, "portfolio_snapshot"):
        try:
            snap = broker.portfolio_snapshot()
            accounts = snap.get("accounts", {})
        except Exception:  # noqa: BLE001
            accounts = {}
    brokers = getattr(broker, "brokers", None) or ({"PRIMARY": broker} if broker is not None else {})
    for label, b in brokers.items():
        try:
            acct = b.sync_account() if hasattr(b, "sync_account") else {}
            bp += float(acct.get("buying_power", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            day_pnl += float(b.get_daily_pnl()) if hasattr(b, "get_daily_pnl") else 0.0
        except Exception:  # noqa: BLE001
            pass

    # bucket hints: L7 orders carry the sleeve in `sector`; concurrence finalists carry `sleeve`
    hints: Dict[str, str] = dict(bucket_hints or {})
    if tranche_result is not None:
        for c in getattr(tranche_result, "final", []):
            hints.setdefault(c.ticker, c.sleeve)
    orders = list(orders)
    for o in orders:
        if o.get("sector") and o.get("ticker"):
            hints.setdefault(o["ticker"], o["sector"])
    new_set = set(new_tickers) | {o.get("ticker") for o in orders if str(o.get("side", "")).upper() in ("BUY", "BUY_TO_OPEN")}
    new_set |= {o.get("contract_symbol") for o in orders if o.get("contract_symbol")}

    rows: List[PositionRow] = []
    holder: Dict[str, str] = {}
    for label, b in brokers.items():
        for sym, p in getattr(getattr(b, "state", None), "positions", {}).items():
            holder[sym] = label
    positions = getattr(state, "positions", {}) or {}
    for sym, p in positions.items():
        qty = float(getattr(p, "quantity", 0) or 0)
        entry = float(getattr(p, "avg_cost", 0.0) or 0.0)
        cur = float(getattr(p, "current_price", 0.0) or 0.0)
        mv = float(getattr(p, "market_value", qty * cur) or qty * cur)
        pnl = float(getattr(p, "unrealized_pnl", (cur - entry) * qty) or 0.0)
        is_opt = getattr(p, "sector", "") == "OPTIONS" or " " in sym.strip()
        if is_opt:
            entry, cur = entry / 100.0, cur / 100.0     # per-contract → premium
        pnl_pct = (pnl / abs(entry * qty * (100 if is_opt else 1))) if entry and qty else 0.0
        rows.append(PositionRow(symbol=sym, qty=qty, entry=entry, current=cur, market_value=mv, pnl=pnl, pnl_pct=pnl_pct,
                                bucket="OPTIONS" if is_opt else classify_bucket(sym, getattr(p, "sector", ""), hints),
                                is_new=sym in new_set, is_option=is_opt, account=holder.get(sym, "")))

    # deployments this cycle (BUY-side L7 orders) — options carry notional, equities qty × fill
    deployments: List[dict] = []
    for o in orders:
        side = str(o.get("side", "")).upper()
        if side not in ("BUY", "BUY_TO_OPEN"):
            continue
        qty = float(o.get("fill_quantity") or o.get("quantity") or 0)
        px = float(o.get("fill_price") or o.get("limit_price") or 0.0)
        is_opt = str(o.get("product_type", "")).upper().endswith("OPTION") or bool(o.get("contract_symbol"))
        dollars = qty * px * (100 if is_opt else 1)
        alpha = o.get("alpha") or o.get("composite")
        deployments.append({"ticker": o.get("contract_symbol") or o.get("ticker"), "dollars": dollars,
                            "bucket": "OPTIONS" if is_opt else classify_bucket(o.get("ticker", ""), o.get("sector", ""), hints),
                            "signal": o.get("signal_type", "scan_signal"), "alpha": alpha, "status": str(o.get("status", "")).split(".")[-1],
                            "reason": o.get("reason", "")})

    runs = list(getattr(tranche_result, "tranches", []) or [])
    spy_close = getattr(getattr(exec_engine, "_tranche_scanner", None), "_spy_close", None)
    macro = build_macro(broker, sd, macro_flag, vix=vix, spy_close=spy_close)
    macro["gates"] = _gates_from_runs(runs, macro)

    # Phase 7 — circuit breaker + anomalies from the book
    anomalies = []
    for r in rows:
        if r.pnl_pct <= -0.20:
            anomalies.append(("DRAWDOWN_EXIT", r.symbol, r.pnl_pct))
        if abs(r.pnl_pct) >= 0.20:
            anomalies.append(("EXTREME_MOVE", r.symbol, r.pnl_pct))
    l7 = getattr(exec_engine, "l7", None)
    l7sum = {}
    try:
        l7sum = l7.get_summary() if l7 is not None and hasattr(l7, "get_summary") else {}
    except Exception:  # noqa: BLE001
        l7sum = {}
    monitoring = {"circuit_breaker": bool(l7sum.get("kill_switch")) or bool((drawdown or {}).get("level") in ("CLOSE", "KILL")),
                  "anomalies": anomalies, "l7": l7sum, "drawdown": drawdown or {}}

    ctx = GoldStandardContext(
        as_of=as_of or datetime.now(ET), mode=mode, cycle_time_s=cycle_time_s, nav=nav, cash=cash, buying_power=bp, day_pnl=day_pnl,
        account_label=" / ".join(f"{l}:{a.get('account', '')}" for l, a in accounts.items()) or str(getattr(broker, "account_display", "") or ""),
        macro=macro, runs=runs, concurrence=tranche_result, options_report=options_report, exits=list(exits), deployments=deployments,
        learning=learning or {}, monitoring=monitoring, positions=rows, accounts=accounts, rules=rules,
        macro_flag=(macro_flag.to_dict() if hasattr(macro_flag, "to_dict") else macro_flag), drawdown=drawdown, council=council,
    )
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Renderers
# ─────────────────────────────────────────────────────────────────────────────
def _hdr(title: str, width: int = W1) -> List[str]:
    return ["═" * width, title, "═" * width]


def _fmt_pct(v: Optional[float], plus: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.1%}" if plus else f"{v:.1%}"


def _num(v: Optional[float], fmt: str) -> str:
    return "n/a" if v is None else format(v, fmt)


def render_view1(ctx: GoldStandardContext) -> str:
    L: List[str] = _hdr("VIEW 1 — THINKING OUTPUT  (Per-Run Intelligence Logic)")
    L.append(f"Scan Date: {ctx.as_of:%Y-%m-%d %H:%M} ET  |  Engine: {ctx.engine_label}  |  Mode: {ctx.mode}")
    L.append(f"Cycle Time: {ctx.cycle_time_s:.1f}s  |  Account: {ctx.account_label or 'n/a'}")
    L.append("")
    m = ctx.macro
    L.append("─── PHASE 2: MACRO ENGINE + METADRON CUBE ───")
    L.append(f"  Regime: {m.get('regime', 'n/a')}  │  VIX: {_num(m.get('vix'), '.1f')}  │  SPY 1M: {_fmt_pct(m.get('spy_1m'))}  │  SPY 3M: {_fmt_pct(m.get('spy_3m'))}")
    L.append(f"  GMTF Score: {_num(m.get('gmtf'), '.3f')}  │  Cube: {m.get('cube', 'n/a')}  │  Max Leverage: {_num(m.get('max_leverage'), '.1f')}x  │  β_cap: {_num(m.get('beta_cap'), '.2f')}")
    ks = "ACTIVE ⚠" if m.get("kill_switch") else "OFF ✓"
    L.append(f"  Kill Switch: {ks}  │  L(t)={_num(m.get('L'), '.2f')}  R(t)={_num(m.get('R'), '.2f')}  F(t)={_num(m.get('F'), '.2f')}")
    g = m.get("gates") or {}
    if g:
        L.append("  4-Gate: " + "  ".join(f"{k}={v:.1f}" for k, v in g.items()))
    L.append(f"  β corridor: {m.get('corridor') or 'n/a'}  │  directional fair-value bias: {float(m.get('direction_bias') or 0):+.2f}  │  target β: {_num(m.get('target_beta'), '.2f')}")
    fl = m.get("flag") or ctx.macro_flag or {}
    if fl:
        trig = "; ".join(fl.get("triggers", [])[:3]) or "none"
        L.append(f"  Hourly macro flag: {fl.get('level', 'NONE')} (score {fl.get('score', 0)})  │  opt×{float(fl.get('options_add_scale', 1)):.2f}  eq×{float(fl.get('equities_add_scale', 1)):.2f}  │  {trig}")
    L.append("")

    # ── per-run blocks ──
    if not ctx.runs:
        L.append("─── UNIVERSE RUNS ───")
        L.append("  (no universe runs this cycle — Schwab offline or scan skipped)")
        L.append("")
    for i, r in enumerate(ctx.runs, 1):
        label = getattr(r, "label", r.name)
        L.append(f"─── RUN {i}: {label} ───")
        L.append(f"  Scanned: {r.screened}/{r.universe_size}  │  BUY: {r.buy_n}  │  SELL: {r.sell_n}  │  HOLD: {r.hold_n}  │  avg α: {r.avg_alpha:+.4f}  │  Time: {r.elapsed_s:.1f}s")
        L.append("  Top 5 BUY signals:")
        for j, c in enumerate(r.top[:5], 1):
            L.append(f"    {j}. {c.ticker:<7} α={c.raw_score:+.4f}  score={c.score:.4f}  Sharpe={c.sharpe:.2f}  vol={c.realized_vol:.0%}  ensemble={c.ensemble:.3f}  [{norm_bucket(c.sleeve) or c.sleeve}]")
            L.append(f"       WHY: {c.why()}")
        if not r.top:
            L.append("    (none)")
        L.append("  Top 3 SELL signals:")
        for j, c in enumerate(r.sells[:3], 1):
            L.append(f"    {j}. {c.ticker:<7} α={c.raw_score:+.4f}  mom_10d={c.mom_10d:+.1%}  vol={c.realized_vol:.0%}  → EXIT signal")
        if not r.sells:
            L.append("    (none — no shortlisted name crossed the SELL floor)")
        L.append("")

    if ctx.concurrence is not None:
        fin = list(getattr(ctx.concurrence, "final", []))
        L.append("─── CONCURRENCE (after last run) ───")
        L.append(f"  Final selection: {len(fin)} names  │  need ≥4/6 votes  │  sector cap 30%  │  sleeve caps per allocation file  │  min 2 per run")
        for c in fin[:20]:
            v = sum(c.votes.values()); n = len(c.votes) or 6
            L.append(f"    {c.ticker:<7} {getattr(c, 'tranche', ''):<14} z={c.z_score:+.2f}  votes={v}/{n}  {c.direction:<5} → {norm_bucket(c.sleeve) or c.sleeve}")
        dropped = list(getattr(ctx.concurrence, "dropped", []))
        if dropped:
            L.append("  Dropped: " + "; ".join(f"{t} ({why})" for t, why in dropped[:10]))
        L.append("")

    # ── short-DTE options engine ──
    rep = ctx.options_report or {}
    L.append("─── PHASE 4: SHORT-DTE OPTIONS ENGINE (1–7 DTE · BSM on the chain DTE · Monte Carlo full scan) ───")
    if not rep:
        L.append("  (options scan did not run — offline, or options budget 0 after macro / drawdown scaling)")
    else:
        per = rep.get("per_ticker", {}) or {}
        counts: Dict[str, int] = {}
        for rec in per.values():
            counts[rec.get("status", "?")] = counts.get(rec.get("status", "?"), 0) + 1
        L.append(f"  Universe: {len(rep.get('universe', []))}  │  " + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())) +
                 f"  │  committed: " + ", ".join(f"{k} ${v:,.0f}" for k, v in (rep.get("committed_by_bucket") or {}).items()))
        for it in rep.get("intents", [])[:12]:
            g = it.get("greeks", {}) or {}
            L.append(f"    INTENT {it.get('ticker'):<6} {it.get('direction'):<7} {it.get('put_call')} {it.get('strike')} exp {it.get('expiry')} "
                     f"({it.get('dte')}DTE) ×{it.get('contracts')} @ ${float(it.get('limit_price') or 0):.2f}  notional ${float(it.get('notional') or 0):,.0f}  "
                     f"composite={float(it.get('composite') or 0):.2f}  edge={float(it.get('edge_bps') or 0):+.0f}bp  Δ$={float(g.get('delta_exposure_usd') or 0):,.0f}  [{it.get('bucket')}]")
            if it.get("rationale"):
                L.append(f"       WHY: {'; '.join(str(x) for x in it['rationale'][:3])}")
        skipped = [(t, rec) for t, rec in per.items() if rec.get("status") != "INTENT"]
        for t, rec in skipped[:8]:
            L.append(f"    {rec.get('status', '?'):<11} {t:<6} {'; '.join(rec.get('reasons', [])[:2])}")
    L.append("")

    # ── phase 5 ──
    L.append("─── PHASE 5: EXECUTION DECISIONS ───")
    ex = ctx.exits
    L.append(f"  Rotation sell candidates: {len(ex)}")
    for e in ex[:10]:
        d = e.to_dict() if hasattr(e, "to_dict") else dict(e)
        L.append(f"    EXIT: {d['ticker']:<7} P&L={d['pnl_pct']:+.1%}  MV=${d['market_value']:,.2f}  → {d.get('reason_short', d.get('reason', ''))}")
    buys = [d for d in ctx.deployments if d["bucket"] != "OPTIONS"]
    opts = [d for d in ctx.deployments if d["bucket"] == "OPTIONS"]
    L.append(f"  New buy targets: {len(buys)} equities/ETFs  │  {len(opts)} option contracts  │  mode {ctx.mode}")
    dd = ctx.drawdown or {}
    if dd:
        L.append(f"  20% drawdown gate: {dd.get('level', 'n/a')} ({float(dd.get('drawdown', 0) or 0):.1%})  adds_allowed={dd.get('adds_allowed', True)}  add_scale={dd.get('add_scale', 1)}")
    L.append("")

    # ── phase 6 (in-chat patch) ──
    L.append("─── PHASE 6: LEARNING LOOP (in-chat patch — not built in sandbox) ───")
    lr = ctx.learning or {}
    if lr:
        if "signal_accuracy" in lr:
            L.append(f"  Signal accuracy: {lr['signal_accuracy']:.1%}  │  Avg P&L: {lr.get('avg_pnl', 0):+.2%}  │  Evaluated: {lr.get('evaluated', 0)} positions")
        for line in lr.get("lines", [])[:8]:
            L.append(f"  {line}")
    else:
        L.append("  Learnings + rotation recommendation are delivered in the in-chat run patch after this report.")
    L.append("")

    # ── phase 7 ──
    mon = ctx.monitoring or {}
    L.append("─── PHASE 7: MONITORING ───")
    L.append(f"  Circuit breaker: {'ON ⚠' if mon.get('circuit_breaker') else 'OFF ✓'}")
    l7 = mon.get("l7") or {}
    if l7:
        L.append(f"  L7: orders today {l7.get('total_orders_today', 0)}  fills {l7.get('total_fills_today', 0)}  dry-run {l7.get('total_dry_run_today', 0)}  "
                 f"│  risk {l7.get('risk_level', 'n/a')}  │  TCA {float(l7.get('avg_tca_cost_bps', 0) or 0):.1f}bp ({l7.get('tca_trend', 'n/a')})  │  VaR95 1d ${float(l7.get('var_95_1d', 0) or 0):,.0f}")
    an = mon.get("anomalies") or []
    L.append(f"  Anomalies: {len(an)}")
    for kind, sym, pct in an[:8]:
        L.append(f"    {kind}: {sym} at {pct:+.1%}")
    L.append("")
    return "\n".join(L)


def _bucket_deployed(ctx: GoldStandardContext, bucket: str) -> Tuple[float, int, float]:
    rows = ctx.by_bucket(bucket)
    return sum(r.market_value for r in rows), len(rows), sum(r.pnl for r in rows)


def render_view2(ctx: GoldStandardContext) -> str:
    L: List[str] = _hdr("VIEW 2 — SCORECARD + ALLOCATION  (Per-Run Deployment Tables)")
    L.append(f"Date: {ctx.as_of:%Y-%m-%d %H:%M} ET  |  NAV: {money(ctx.nav)}  |  BP: {money(ctx.buying_power)}  |  Mode: {ctx.mode}")
    L.append("")
    L.append("─── SCAN SCORECARDS ───")
    tot_s = tot_b = tot_sell = 0
    for i, r in enumerate(ctx.runs, 1):
        label = getattr(r, "label", r.name)
        L.append(f"  Run {i} [{label:<6}]: {r.screened:>4} scanned  │  {r.buy_n:>3} BUY ({_pct(r.buy_n, r.screened):.0f}%)  │  {r.sell_n:>3} SELL  │  {r.hold_n:>3} HOLD  │  avg α={r.avg_alpha:+.4f}  │  {r.elapsed_s:.1f}s")
        tot_s += r.screened; tot_b += r.buy_n; tot_sell += r.sell_n
    if not ctx.runs:
        L.append("  (no universe runs this cycle)")
    L.append("  " + "─" * 95)
    L.append(f"  TOTAL:         {tot_s:>4} scanned  │  {tot_b:>3} BUY ({_pct(tot_b, tot_s):.0f}%)  │  {tot_sell:>3} SELL  │  Cycle: {ctx.cycle_time_s:.1f}s")
    L.append("")

    L.append("─── DEPLOYMENT SUMMARY ───")
    L.append(f"  Rotation Exits:  {len(ctx.exits)} positions")
    for e in ctx.exits[:12]:
        d = e.to_dict() if hasattr(e, "to_dict") else dict(e)
        L.append(f"    EXIT {d['ticker']:<7} ~$ {d['market_value']:>9,.2f}  │  {d['reason']}")
    dep = ctx.deployments
    L.append(f"  New Deployments: {len(dep)} positions  ({ctx.mode})")
    by: Dict[str, Tuple[int, float]] = {}
    for d in dep:
        n, s = by.get(d["bucket"], (0, 0.0)); by[d["bucket"]] = (n + 1, s + float(d["dollars"]))
    for b, (n, s) in sorted(by.items()):
        L.append(f"    {b:<20}: {n:>3} positions  $ {s:>10,.2f}")
    L.append("")

    L.append("─── ALLOCATION FILL RATES (vs AllocationRules) ───")
    L.append("")
    L.append(f"  {'Bucket':<22} {'Target':>7} {'Target $':>12} {'Deployed':>12} {'# Pos':>7} {'Fill':>8} Status")
    L.append("  " + "─" * 22 + " " + "─" * 7 + " " + "─" * 12 + " " + "─" * 12 + " " + "─" * 7 + " " + "─" * 8 + " " + "─" * 12)
    eq_t = eq_d = 0.0; eq_n = 0
    for key, _, _ in BUCKETS:
        tp = ctx.target_pct(key); tgt = tp * ctx.nav
        dep_v, n, _ = _bucket_deployed(ctx, key)
        f = _pct(dep_v, tgt)
        L.append(f"  {key:<22} {tp:>6.0%} $ {tgt:>10,.0f} $ {dep_v:>10,.2f} {n:>7} {f:>7.1f}% {fill_status(f)}")
        eq_t += tgt; eq_d += dep_v; eq_n += n
    L.append("  " + "─" * 22 + " " + "─" * 7 + " " + "─" * 12 + " " + "─" * 12 + " " + "─" * 7 + " " + "─" * 8 + " " + "─" * 12)
    L.append(f"  {'EQUITY SUBTOTAL':<22} {eq_t / ctx.nav if ctx.nav else 0:>6.0%} $ {eq_t:>10,.0f} $ {eq_d:>10,.2f} {eq_n:>7} {_pct(eq_d, eq_t):>7.1f}%")
    L.append("")
    opt_rows = ctx.options(); opt_v = sum(abs(r.market_value) for r in opt_rows)
    op = ctx.target_pct("OPTIONS"); mp = ctx.target_pct("MARGIN"); mmp = ctx.target_pct("MONEY_MARKET")
    L.append(f"  {'OPTIONS (notional)':<22} {op:>6.0%} $ {op * ctx.nav:>10,.0f} $ {opt_v:>10,.2f} {len(opt_rows):>7} {_pct(opt_v, op * ctx.nav):>7.1f}%   (1–7 DTE overlay; futures retired)")
    im = sum(abs(r.entry * r.qty * 100) for r in opt_rows)
    L.append(f"  {'MARGIN (8%)':<22} {mp:>6.0%} $ {mp * ctx.nav:>10,.0f} $ {im:>10,.2f} {'':>7} {_pct(im, mp * ctx.nav):>7.1f}%")
    mm = max(ctx.cash, 0.0)
    L.append(f"  {'MONEY MARKET (2%)':<22} {mmp:>6.0%} $ {mmp * ctx.nav:>10,.0f} $ {mm:>10,.2f} {'':>7} {_pct(mm, mmp * ctx.nav):>7.1f}%  {'✓ FLOOR HELD' if mm >= mmp * ctx.nav else '⚠ BELOW FLOOR'}")
    L.append("")
    return "\n".join(L)


def render_transaction_log(ctx: GoldStandardContext) -> str:
    L: List[str] = _hdr("TRANSACTION LOG  (Between View 2 → View 3)")
    n_tr = len(ctx.exits) + len(ctx.deployments)
    L.append(f"Date: {ctx.as_of:%Y-%m-%d %H:%M} ET  |  Trades: {n_tr}  |  Mode: {ctx.mode}" + ("  (DRY_RUN — routed through L7 gates, not sent to Schwab)" if ctx.mode == "DRY_RUN" else ""))
    L.append("")
    L.append(f"  ── ROTATION EXITS ({len(ctx.exits)}) ──")
    L.append(f"    {'#':>1}  {'Ticker':<8} {'Side':<14} {'$ Value':>12}  {'Reason':<40}")
    L.append("  " + "─" * 80)
    freed = 0.0
    for i, e in enumerate(ctx.exits, 1):
        d = e.to_dict() if hasattr(e, "to_dict") else dict(e)
        freed += float(d["market_value"])
        L.append(f"    {i:>1}  {d['ticker']:<8} {d['side']:<14} $ {d['market_value']:>10,.2f}  {d['reason']:<40}")
    if not ctx.exits:
        L.append("    (none)")
    L.append("")
    L.append(f"  ── NEW DEPLOYMENTS ({len(ctx.deployments)}) ──")
    L.append(f"    {'#':>1}  {'Ticker':<22} {'$ Deployed':>12}  {'Bucket':<18} {'Signal':<34}")
    L.append("  " + "─" * 96)
    deployed = 0.0
    for i, d in enumerate(ctx.deployments, 1):
        deployed += float(d["dollars"])
        a = d.get("alpha")
        sig = f"{d.get('signal', 'scan_signal')}" + (f" (α={float(a):+.4f})" if a is not None else "") + (f" [{d.get('status')}]" if d.get("status") else "")
        L.append(f"    {i:>1}  {str(d['ticker']):<22} $ {float(d['dollars']):>10,.2f}  {d['bucket']:<18} {sig:<34}")
    if not ctx.deployments:
        L.append("    (none)")
    L.append("")
    L.append(f"  Capital freed:    $ {freed:>12,.2f}")
    L.append(f"  Capital deployed: $ {deployed:>12,.2f}")
    L.append(f"  Net flow:         $ {deployed - freed:>+12,.2f}")
    L.append("")
    return "\n".join(L)


def _pos_table(rows: List[PositionRow], option: bool = False) -> List[str]:
    L: List[str] = []
    if option:
        L.append(f"│     #  {'Contract':<28} {'Qty':>6} {'Entry':>9} {'Now':>9} {'Mkt Value':>11} {'P&L':>12} {'P&L %':>8}  NEW ")
    else:
        L.append(f"│     #  {'Symbol':<8} {'Side':<6} {'Qty':>12} {'Entry':>10} {'Current':>10} {'Mkt Value':>12} {'P&L':>12} {'P&L %':>8}  NEW ")
    L.append("├" + "─" * (W3 - 1))
    for i, r in enumerate(rows, 1):
        new = "←NEW" if r.is_new else ""
        if option:
            L.append(f"│  {i:>3}  {r.symbol:<28} {r.qty:>6.0f} $ {r.entry:>7.2f} $ {r.current:>7.2f} $ {r.market_value:>9,.2f} $ {r.pnl:>+10,.2f} {r.pnl_pct:>+8.2%}  {new}")
        else:
            side = "long" if r.qty >= 0 else "short"
            L.append(f"│  {i:>3}  {r.symbol:<8} {side:<6} {r.qty:>12.6f} $ {r.entry:>8.2f} $ {r.current:>8.2f} $ {r.market_value:>10,.2f} $ {r.pnl:>+10,.2f} {r.pnl_pct:>+8.2%}  {new}")
    return L


def render_view3(ctx: GoldStandardContext) -> str:
    L: List[str] = _hdr("VIEW 3 — LIVE POSITION PANEL  (Bucket-Grouped by AllocationRules)", W3)
    L.append(f"Pulled: {ctx.as_of:%Y-%m-%d %H:%M} ET  |  NAV: {money(ctx.nav)}  |  BP: {money(ctx.buying_power)}  |  Day P&L: {money(ctx.day_pnl, plus=True)} ({_pct(ctx.day_pnl, ctx.nav):+.2f}%)")
    L.append("")
    L.append("━" * W3); L.append("TABLE 1 — EQUITIES  (Grouped by Allocation Bucket)"); L.append("━" * W3); L.append("")
    eq_long = eq_pnl = 0.0; eq_n = 0
    for key, desc, _ in BUCKETS:
        rows = ctx.by_bucket(key)
        tp = ctx.target_pct(key); tgt = tp * ctx.nav
        dep = sum(r.market_value for r in rows); pnl = sum(r.pnl for r in rows); f = _pct(dep, tgt)
        L.append(f"┌─ {key} — {desc}")
        L.append(f"│  Target: {tp:.0%} = ${tgt:,.0f}  │  Deployed: ${dep:,.2f}  │  Fill: {f:.1f}%  {fill_status(f)}")
        L.append(f"│  Positions: {len(rows)}  │  P&L: {money(pnl, plus=True)}")
        L.append("├" + "─" * (W3 - 1))
        L += _pos_table(rows)
        if not rows:
            L.append("│  (no positions)")
        L.append("└" + "─" * (W3 - 1)); L.append("")
        eq_long += dep; eq_pnl += pnl; eq_n += len(rows)
    sh = ctx.shorts()
    L.append("┌─ SHORT HEDGE BOOK")
    L.append(f"│  Positions: {len(sh)}  │  Short MV: ${sum(r.market_value for r in sh):,.2f}  │  P&L: {money(sum(r.pnl for r in sh), plus=True)}")
    L.append("├" + "─" * (W3 - 1))
    L += _pos_table(sh)
    if not sh:
        L.append("│  (no short hedges)")
    L.append("└" + "─" * (W3 - 1)); L.append("")
    eq_t = sum(ctx.target_pct(k) for k in BUCKET_KEYS) * ctx.nav
    short_mv = sum(r.market_value for r in sh)
    L.append("══ EQUITY SUBTOTAL ══")
    L.append(f"   Long: $ {eq_long:>12,.2f} ({eq_n} pos)  │  Target: {sum(ctx.target_pct(k) for k in BUCKET_KEYS):.0%} = ${eq_t:,.0f}  │  Fill: {_pct(eq_long, eq_t):.1f}%")
    L.append(f"   Short: $ {short_mv:>12,.2f} ({len(sh)} pos)")
    L.append(f"   Net: $ {eq_long + short_mv:>12,.2f}  │  P&L: {money(eq_pnl + sum(r.pnl for r in sh), plus=True)}")
    L.append(""); L.append("")

    L.append("━" * W3); L.append("TABLE 2 — DERIVATIVES OVERLAY  (1–7 DTE Options · β corridor = directional fair value; no futures)"); L.append("━" * W3); L.append("")
    opts = sorted(ctx.options(), key=lambda r: -abs(r.market_value))
    op = ctx.target_pct("OPTIONS"); ov = sum(abs(r.market_value) for r in opts)
    L.append("┌─ OPTIONS — All Contracts (1–7 DTE)")
    L.append(f"│  Target: {op:.0%} notional = ${op * ctx.nav:,.0f}  │  Deployed: ${ov:,.2f}  │  Fill: {_pct(ov, op * ctx.nav):.1f}%")
    L.append("├" + "─" * (W3 - 1))
    L += _pos_table(opts, option=True)
    if not opts:
        L.append("│  (no open contracts)")
    L.append("└" + "─" * (W3 - 1)); L.append("")
    L.append("══ DERIVATIVES SUBTOTAL ══")
    L.append(f"   Options:  $ {ov:>12,.2f} ({len(opts)} ct)  │  Target: {op:.0%} = ${op * ctx.nav:,.0f}  │  Fill: {_pct(ov, op * ctx.nav):.1f}%")
    L.append(f"   Futures:  retired — strictly options overlay")
    L.append("")
    L.append("━" * W3); L.append("MARGIN & MONEY MARKET"); L.append("━" * W3)
    mp = ctx.target_pct("MARGIN"); mmp = ctx.target_pct("MONEY_MARKET")
    im = sum(abs(r.entry * r.qty * 100) for r in opts); mm = max(ctx.cash, 0.0)
    L.append(f"  Margin (8%):  target ${mp * ctx.nav:,.0f}  │  derivatives IM ~${im:,.2f}  │  Fill: {_pct(im, mp * ctx.nav):.1f}%")
    L.append(f"  Money Market (2%): target ${mmp * ctx.nav:,.0f}  │  effective ${mm:,.2f}  │  {'✓ FLOOR HELD' if mm >= mmp * ctx.nav else '⚠ BELOW FLOOR'}")
    L.append("")
    L.append("═" * W3); L.append("PORTFOLIO GRAND TOTAL"); L.append("═" * W3)
    long_mv = eq_long + ov
    lev = (long_mv + abs(short_mv)) / ctx.nav if ctx.nav else 0.0
    book_pnl = sum(r.pnl for r in ctx.positions)
    L.append(f"  NAV: $ {ctx.nav:>12,.2f}  │  Cash: $ {ctx.cash:>12,.2f}  │  BP: $ {ctx.buying_power:>12,.2f}")
    L.append(f"  Long MV: $ {long_mv:>12,.2f}  │  Short MV: $ {short_mv:>12,.2f}  │  Leverage: {lev:.2f}x")
    L.append(f"  Day P&L: $ {ctx.day_pnl:>+12,.2f}  ({_pct(ctx.day_pnl, ctx.nav):+.2f}%)")
    L.append(f"  Total Book P&L: $ {book_pnl:>+12,.2f}")
    L.append(f"  Positions: {len(ctx.positions)} total")
    L.append("")
    L.append("  ALLOCATION CAPACITY SUMMARY")
    L.append(f"  {'Bucket':<22} {'Target':>7} {'Target $':>12} {'Deployed':>12} {'Fill':>8} Status")
    L.append("  " + "─" * 75)
    for key, _, _ in BUCKETS:
        tp = ctx.target_pct(key); tgt = tp * ctx.nav; dep, _, _ = _bucket_deployed(ctx, key); f = _pct(dep, tgt)
        L.append(f"  {key:<22} {tp:>6.0%} $ {tgt:>10,.0f} $ {dep:>10,.2f} {f:>7.1f}% {fill_status(f)}")
    L.append("  " + "─" * 75)
    L.append(f"  {'OPTIONS':<22} {op:>6.0%} $ {op * ctx.nav:>10,.0f} $ {ov:>10,.2f} {_pct(ov, op * ctx.nav):>7.1f}%")
    L.append(f"  {'MARGIN':<22} {mp:>6.0%} $ {mp * ctx.nav:>10,.0f} $ {im:>10,.2f} {_pct(im, mp * ctx.nav):>7.1f}%")
    L.append(f"  {'MONEY MARKET':<22} {mmp:>6.0%} $ {mmp * ctx.nav:>10,.0f} $ {mm:>10,.2f} {_pct(mm, mmp * ctx.nav):>7.1f}%")
    L.append("")

    if ctx.accounts:
        L.append("  ACCOUNT MANDATES (Schwab multi-account router)")
        L.append(f"  {'Account':<12} {'Schwab':<10} {'Mandate':<26} {'NAV':>12} {'Options / cap':>24} {'Equities+ETF / cap':>26} {'DD':>7}  Level")
        L.append("  " + "─" * 125)
        for label, a in ctx.accounts.items():
            m = a.get("mandate", {}); dd = a.get("drawdown", {})
            mand = f"opt {float(m.get('options_pct', 0)):.0%} / eq {float(m.get('equities_pct', 0)):.0%}"
            L.append(f"  {label:<12} {str(a.get('account', ''))[-9:]:<10} {mand:<26} $ {float(a.get('nav', 0)):>10,.0f} "
                     f"$ {float(a.get('options_notional', 0)):>9,.0f} / {float(a.get('options_cap', 0)):>9,.0f} "
                     f"$ {float(a.get('equities_mv', 0)):>10,.0f} / {float(a.get('equities_cap', 0)):>10,.0f} {float(dd.get('drawdown', 0) or 0):>6.1%}  {dd.get('level', '')}")
        L.append("")
    if ctx.council is not None:
        v = ctx.council
        L.append("  EOD EQUITIES/ETF COUNCIL")
        L.append(f"  Execution grade: {getattr(v, 'execution_grade', 'n/a')}  │  next-day allocation: " +
                 ", ".join(f"{k} {float(x):.0%}" for k, x in (getattr(v, "next_day_allocation", {}) or {}).items()))
        for line in (getattr(v, "summary", None) or getattr(v, "notes", None) or [])[:5]:
            L.append(f"    · {line}")
        L.append("")
    return "\n".join(L)


def render(ctx: GoldStandardContext, recap: bool = False) -> str:
    parts = []
    if recap:
        parts.append("═" * W1 + "\n" + f"END-OF-DAY RECAP — positions + all transactions today  ({ctx.as_of:%Y-%m-%d} · {ctx.mode})" + "\n" + "═" * W1 + "\n")
    parts += [render_view1(ctx), render_view2(ctx), render_transaction_log(ctx), render_view3(ctx)]
    return "\n".join(parts)


class GoldStandardReporter:
    """Component wrapper: collect + render in one call (used by wiring manifest / orchestrator)."""

    def __init__(self):
        self.last_text: str = ""

    def build(self, exec_engine: Any, **kw) -> GoldStandardContext:
        return collect_context(exec_engine, **kw)

    def report(self, exec_engine: Any, recap: bool = False, **kw) -> str:
        self.last_text = render(self.build(exec_engine, **kw), recap=recap)
        return self.last_text


__all__ = ["GoldStandardReporter", "GoldStandardContext", "PositionRow", "collect_context", "render", "render_view1", "render_view2",
           "render_transaction_log", "render_view3", "classify_bucket", "norm_bucket", "fill_status", "BUCKETS"]
