#!/usr/bin/env python3
"""
Metadron sandbox run cycle — one 30-minute FULL-SCAN heartbeat, end to end.
==========================================================================

    Schwab (data)  →  universe + allocation slate  →  ShortDTE options scan (1–7 DTE)
                   →  20 % drawdown gate  →  hourly macro flag  →  L7 (DRY_RUN unless
                   SCHWAB_LIVE_ORDERS=true)  →  EOD council (if --close)  →  in-chat patch

Everything the live loop does every 30 minutes, compressed into one command so the
system can be exercised in the sandbox and the learning / rotation patch printed
in chat. Nothing is sent to Schwab unless SCHWAB_LIVE_ORDERS=true.

    SCHWAB_AUTH_MODE=proxy SCHWAB_ACCOUNT_ROTH=9565 SCHWAB_ACCOUNT_LLC=0514 \\
    SCHWAB_ACCOUNT_INDIVIDUAL=4806 PYTHONPATH=. python3 run_sandbox_cycle.py --close

Options:
    --universe SPY,QQQ,...  tickers to scan (default: SPY,QQQ,AAPL,NVDA,TSLA,AMD,MSFT,XLE,HYG,TLT)
    --close                 also convene the EOD equities/ETF council
    --offline               do not connect to Schwab (structure test only)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"), format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sandbox_cycle")

DEFAULT_UNIVERSE = "SPY,QQQ,AAPL,NVDA,TSLA,AMD,MSFT,XLE,HYG,TLT"

# ticker → (options bucket, equity sleeve) — follows the allocation file
BUCKET_MAP = {
    "SPY": ("OPTIONS_IG", "IG_EQUITY"), "QQQ": ("OPTIONS_IG", "IG_EQUITY"), "AAPL": ("OPTIONS_IG", "IG_EQUITY"),
    "MSFT": ("OPTIONS_IG", "IG_EQUITY"), "NVDA": ("OPTIONS_HY", "HY_EQUITY"), "AMD": ("OPTIONS_HY", "HY_EQUITY"),
    "TSLA": ("OPTIONS_HY", "HY_EQUITY"), "XLE": ("OPTIONS_HY", "HY_EQUITY"), "HYG": ("OPTIONS_DISTRESSED", "DISTRESSED"),
    "TLT": ("OPTIONS_IG", "FIXED_INCOME"), "TLTW": (None, "TLTW"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=DEFAULT_UNIVERSE)
    ap.add_argument("--close", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-tranche", action="store_true", help="skip the 3-tranche universe scan and use the static --universe list")
    ap.add_argument("--equity-slate-pct", type=float, default=0.02, help="per-name equity add as fraction of NAV for the slate")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.universe.split(",") if t.strip()]

    from engine.execution.execution_engine import ExecutionEngine
    from engine.execution.macro_event_flagger import MacroEventFlagger
    from engine.execution.eod_allocation_council import EODAllocationCouncil
    from engine.execution.run_patch_report import build_run_patch
    from engine.execution.account_mandates import DrawdownGuard
    from engine.allocation.allocation_engine import AllocationRules
    from engine.execution.broker_types import OrderSide

    t0 = datetime.now()
    print(f"# Sandbox cycle {t0:%Y-%m-%d %H:%M:%S}  universe={','.join(tickers)}  mode={'OFFLINE' if args.offline else 'SCHWAB'}")

    # ── 1. Engine + Schwab broker (router when SCHWAB_ACCOUNT_* set) ─────────
    eng = ExecutionEngine(connect_broker=not args.offline)
    broker = eng.broker
    status = eng.get_broker_status()
    print(f"broker={status['broker']} connected={status['connected']} live={status['is_live']} account={status['account']}")
    if status["connected"] and hasattr(broker, "sync_positions"):
        try:
            broker.sync_positions()
        except Exception as exc:  # noqa: BLE001
            print(f"  position sync failed: {exc}")
    nav = float(broker.state.nav or 0.0)
    print(f"NAV ${nav:,.2f}  cash ${broker.state.cash:,.2f}  positions={len(broker.state.positions)}")

    # ── 2. hourly macro flag (basket moves come from Schwab quotes) ───────────
    flagger = MacroEventFlagger(broker=broker if status["connected"] else None)
    quotes = {}
    if status["connected"]:
        try:
            quotes = broker.get_quotes(tickers + ["$VIX"]) or {}
        except Exception as exc:  # noqa: BLE001
            print(f"  quotes failed: {exc}")
    vix = None
    for k in ("$VIX", "^VIX", "VIX"):
        q = quotes.get(k)
        if isinstance(q, dict) and (q.get("last") or q.get("mark")):
            vix = float(q.get("last") or q.get("mark")); break
    flag = flagger.evaluate(headlines=[], regime="NORMAL", vix_level=vix)
    print(f"macro flag: {flag.level} score={flag.score:.2f} opt×{flag.options_add_scale:.2f} eq×{flag.equities_add_scale:.2f} "
          + ("triggers=" + "; ".join(flag.triggers[:3]) if flag.triggers else ""))

    # ── 3. 20 % drawdown gate BEFORE any add ─────────────────────────────────
    if hasattr(broker, "portfolio_snapshot"):
        snap = broker.portfolio_snapshot(); dd = snap["drawdown"]
        for label, a in snap["accounts"].items():
            print(f"  {label:<10} …{a['mandate']['account_last4']} NAV ${a['nav']:,.0f} opt {a['options_notional']:,.0f}/{a['options_cap']:,.0f} "
                  f"eq {a['equities_mv']:,.0f}/{a['equities_cap']:,.0f} DD {a['drawdown']['drawdown']:.1%} {a['drawdown']['level']}")
    else:
        g = DrawdownGuard(); g.seed_peak("PORTFOLIO", nav); dd = g.check("PORTFOLIO", nav).to_dict()
    print(f"drawdown: {dd['level']} ({dd.get('drawdown', 0):.1%}) adds_allowed={dd['adds_allowed']} scale={dd['add_scale']}")
    add_scale = float(dd["add_scale"])

    # ── 3b. Universe scan in THREE separate tranches → concurrence ───────────
    #   Scan 1 = S&P 500 · Scan 2 = S&P SmallCap 600 · Scan 3 = remaining ~400 (MidCap 400 + extras)
    #   Each tranche is scored on its own distribution and reported on its own;
    #   only after Scan 3 does the concurrence vote build the allocation slate.
    sd = eng.options_engine_short_dte
    tranche = None
    eq_slate: list = []
    opt_universe = [(t, BUCKET_MAP.get(t, ("OPTIONS_HY", "HY_EQUITY"))[0]) for t in tickers if BUCKET_MAP.get(t, ("OPTIONS_HY",))[0]]
    if status["connected"] and not args.no_tranche:
        try:
            from engine.execution.universe_tranche_scan import UniverseTrancheScanner
            bias = None
            try:
                bias = float(sd.market_context().direction_bias)
            except Exception:  # noqa: BLE001
                pass
            scanner = UniverseTrancheScanner(broker, options_engine=sd)
            tranche = scanner.run(corridor_bias=bias)
            for tr in tranche.tranches:
                print(f"{tr.name}: universe {tr.universe_size} → quoted {tr.quoted} → screened {tr.screened} → "
                      f"shortlist {tr.shortlisted} → top {len(tr.top)} ({tr.elapsed_s:.0f}s): "
                      + ", ".join(f"{c.ticker} z{c.z_score:+.1f}" for c in tr.top[:8]))
            print(f"CONCURRENCE → {len(tranche.final)} names: " + ", ".join(f"{c.ticker}[{c.sleeve}]" for c in tranche.final))
            eq_slate = tranche.equity_slate()
            opt_universe = tranche.options_universe() + [u for u in opt_universe if u[0] in ("SPY", "QQQ")]
        except Exception as exc:  # noqa: BLE001
            print(f"  tranche scan failed: {exc}")
    else:
        print("tranche scan: skipped (" + ("offline" if not status["connected"] else "--no-tranche") + ") — static universe used")

    # ── 4. ShortDTE options scan (1–7 DTE): momentum/RSI + beta corridor + BSM/VolSurface + MC gate ──
    opt_nav = nav
    if hasattr(broker, "options_budget"):
        budget = sum(broker.options_budget().values()); opt_nav = budget / 0.25 if budget > 0 else 0.0
    opt_nav *= flag.options_add_scale * add_scale
    report, intents = None, []
    if sd is not None and opt_nav > 0 and status["connected"]:
        try:
            scan = sd.scan(opt_universe, nav=opt_nav)
            report, intents = scan["report"], scan["intents"]
            ctx = scan["context"]
            print(f"options scan: regime={ctx.regime} VIX={ctx.vix:.2f} corridor={ctx.corridor_position} bias={ctx.direction_bias:+.2f} "
                  f"intents={len(intents)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  options scan failed: {exc}")
    else:
        print("options scan: skipped (" + ("offline" if not status["connected"] else "options budget 0 after macro/drawdown scaling") + ")")

    # ── 5. L7 execution — options intents then equity slate (all through L7.submit) ──
    orders = []
    for it in intents:
        o = eng.l7_submit_option_intent(it, regime=getattr(sd, "regime", "NORMAL"))
        if o:
            orders.append(o.to_dict() if hasattr(o, "to_dict") else dict(o))
    if dd["adds_allowed"]:
        rules = AllocationRules()
        if eq_slate:
            # concurrence slate: sleeve % (allocation file) × NAV ÷ names in sleeve, capped at G1 10 % NAV
            sleeve_pct = {"IG_EQUITY": rules.ig_equity_pct, "HY_EQUITY": rules.hy_equity_pct, "DISTRESSED": rules.distressed_equity_pct,
                          "TLTW": rules.tltw_cashflow_pct, "FIXED_INCOME": rules.fi_macro_pct, "CVR": rules.event_driven_cvr_pct}
            n_by = {}
            for r in eq_slate:
                n_by[r["bucket"]] = n_by.get(r["bucket"], 0) + 1
            eq_rows = [(r["ticker"], r["bucket"], min(0.10, sleeve_pct.get(r["bucket"], 0.05) / n_by[r["bucket"]])) for r in eq_slate]
            quotes.update(broker.get_quotes([t for t, _, _ in eq_rows]) or {})
        else:
            eq_rows = [(t, BUCKET_MAP.get(t, (None, "HY_EQUITY"))[1], args.equity_slate_pct) for t in tickers]
        for t, sleeve, pct in eq_rows:
            dollars = nav * pct * add_scale * flag.equities_add_scale
            q = quotes.get(t) if isinstance(quotes.get(t), dict) else None
            px = float((q or {}).get("last") or (q or {}).get("mark") or 0.0)
            qty = int(dollars // px) if px > 0 else 0
            if qty <= 0:
                continue
            o = eng.l7_submit(ticker=t, side="BUY", quantity=qty, signal_type="QUALITY_BUY", regime="NORMAL",
                              sector=sleeve, reason=f"{'concurrence' if eq_slate else 'sandbox'} slate {sleeve} {pct:.1%} NAV")
            orders.append(o if isinstance(o, dict) else o.to_dict())
    else:
        print("equity slate: BLOCKED by 20% drawdown rule — rotate-or-close only")
    print(f"L7 orders this cycle: {len(orders)}  " + ", ".join(f"{o.get('ticker')}:{str(o.get('status')).split('.')[-1]}" for o in orders[:12]))

    # ── 6. EOD council (equities/ETF) ────────────────────────────────────────
    verdict = None
    if args.close:
        l7 = eng.l7
        l7_orders = list(getattr(l7, "_filled_orders", [])) + list(getattr(l7, "_dry_run_orders", []))
        close_px = {t: float(q.get("last") or q.get("mark") or 0) for t, q in quotes.items() if isinstance(q, dict)}
        council = EODAllocationCouncil(AllocationRules())
        verdict = council.convene(l7_orders=l7_orders, nav=nav, positions=dict(broker.state.positions), close_prices=close_px,
                                  drawdown=dd, macro_flag=flag.to_dict(), momentum={}, regime="NORMAL", vix=vix,
                                  account_router=broker if hasattr(broker, "mandates") else None)
        print(f"EOD council: grade {verdict.execution_grade}  next-day {verdict.next_day_allocation}")

    # ── 7. in-chat patch ─────────────────────────────────────────────────────
    md = build_run_patch(exec_engine=eng, options_report=report, orders=orders, macro_flag=flag,
                         council_verdict=verdict, drawdown=dd, tranche_result=tranche)
    print("\n" + md)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/last_sandbox_cycle.md").write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
