#!/usr/bin/env python3
"""
Metadron sandbox run cycle — one 30-minute FULL-SCAN heartbeat, end to end.
==========================================================================

    Schwab (data)  →  universe + allocation slate  →  ShortDTE options scan (1–30 DTE, shorter preferred)
                   →  20 % drawdown gate  →  hourly macro flag  →  L7 (DRY_RUN unless
                   SCHWAB_LIVE_ORDERS=true)  →  EOD council (if --close)  →  in-chat patch

Everything the live loop does every 30 minutes, compressed into one command so the
system can be exercised in the sandbox and the learning / rotation patch printed
in chat. Nothing is sent to Schwab unless SCHWAB_LIVE_ORDERS=true.

    SCHWAB_AUTH_MODE=proxy SCHWAB_ACCOUNT_ROTH=0514 SCHWAB_ACCOUNT_LLC=9565 \\
    SCHWAB_ACCOUNT_INDIVIDUAL=4806 PYTHONPATH=. python3 run_sandbox_cycle.py --close

Options:
    --universe SPY,QQQ,...  tickers to scan (default: SPY,QQQ,AAPL,NVDA,TSLA,AMD,MSFT,XLE,HYG,TLT)
    --close                 also convene the EOD equities/ETF council
    --offline               do not connect to Schwab (structure test only)
    --no-tranche            skip the 4 universe runs and use the static --universe list
    --gold / --no-gold      print the gold-standard report (VIEW 1/2/TX LOG/VIEW 3); default on
    --recap                 render as the end-of-day recap (positions + all of today's transactions)
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


SLEEVE_PCT_KEYS = {"IG_EQUITY": "ig_equity_pct", "HY_EQUITY": "hy_equity_pct", "DISTRESSED": "distressed_equity_pct",
                   "TLTW": "tltw_cashflow_pct", "FIXED_INCOME": "fi_macro_pct", "CVR": "event_driven_cvr_pct"}


def proposed_allocation(tr, nav: float, rules, accounts: dict, add_scale: float = 1.0) -> str:
    """Per-run PROPOSED allocation printed right after the run (before concurrence).

    Sizing follows the allocation file: sleeve % × NAV ÷ names in that sleeve (this run),
    capped at L7 G1 = 10 % NAV, scaled by the 20 % drawdown gate / macro flag.  Then split
    across the account mandates: ROTH 75 % equities, LLC 100 % equities+ETF, INDIVIDUAL
    0 % equities (options only).  This is a proposal — the concurrence vote after RUN 4
    builds the slate actually sent to L7.
    """
    L = [f"  ── PROPOSED ALLOCATION after {getattr(tr, 'label', tr.name)} (pre-concurrence) ──"]
    if not tr.top:
        L.append("    (no LONG names cleared the composite floor in this run — nothing proposed)")
        return "\n".join(L)
    n_by = {}
    for c in tr.top:
        n_by[c.sleeve] = n_by.get(c.sleeve, 0) + 1
    tot = 0.0
    L.append(f"    {'Ticker':<7} {'Sleeve':<14} {'Sleeve %':>8} {'Per-name $':>11} {'% NAV':>6}  {'Signal':<5} {'α':>7}  Sector")
    for c in tr.top:
        pct = min(0.10, float(getattr(rules, SLEEVE_PCT_KEYS.get(c.sleeve, "hy_equity_pct"))) / n_by[c.sleeve]) * add_scale
        d = nav * pct; tot += d
        L.append(f"    {c.ticker:<7} {c.sleeve:<14} {float(getattr(rules, SLEEVE_PCT_KEYS.get(c.sleeve, 'hy_equity_pct'))):>7.0%} $ {d:>9,.0f} {pct:>6.1%}  {c.signal:<5} {c.raw_score:>+7.3f}  {c.sector}")
    L.append(f"    Total proposed this run: $ {tot:,.0f} ({tot / nav if nav else 0:.1%} NAV)  │  by sleeve: " +
             ", ".join(f"{k} {v}" for k, v in n_by.items()))
    if accounts:
        L.append("    Account split (mandates):")
        for label, a in accounts.items():
            eq_pct = float(a.get("mandate", {}).get("equities_pct", 0.0)); anav = float(a.get("nav", 0.0))
            share = anav / nav if nav else 0.0
            head = float(a.get("equities_headroom", a.get("equities_cap", 0.0)))
            want = tot * share * eq_pct
            L.append(f"      {label:<11} …{a.get('mandate', {}).get('account_last4', '')}  eq mandate {eq_pct:.0%}  → $ {min(want, head):>9,.0f}"
                     + (f"  (capped by headroom $ {head:,.0f})" if want > head else "") + ("  — options only, no equities" if eq_pct == 0 else ""))
    opt = [c for c in tr.top if c.raw_score > 0][:4]
    if opt:
        L.append("    Options candidates forwarded to the 1–7 DTE engine: " + ", ".join(c.ticker for c in opt))
    return "\n".join(L)


def _p(*a):
    print(*a, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=DEFAULT_UNIVERSE)
    ap.add_argument("--close", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-tranche", action="store_true", help="skip the 3-tranche universe scan and use the static --universe list")
    ap.add_argument("--equity-slate-pct", type=float, default=0.02, help="per-name equity add as fraction of NAV for the slate")
    ap.add_argument("--gold", dest="gold", action="store_true", default=True)
    ap.add_argument("--no-gold", dest="gold", action="store_false")
    ap.add_argument("--recap", action="store_true", help="render the gold-standard output as the EOD recap")
    ap.add_argument("--fill-per-run", dest="fill_per_run", action="store_true",
                    default=os.environ.get("METADRON_FILL_PER_RUN", "1") == "1",
                    help="fill each run's trades (INDIVIDUAL options → LLC equities → ROTH) right after that run's chains, then move to the next run")
    ap.add_argument("--fill-at-end", dest="fill_per_run", action="store_false")
    ap.add_argument("--start-from", default="", help="resume the tranche sequence at this run (e.g. SCAN_2_SP400); earlier runs are skipped")
    ap.add_argument("--options-only", action="store_true", help="operator: NO equity fills this stage — scan → chain sweep → options (INDIVIDUAL → ROTH) only")
    ap.add_argument("--only-run", default="", help="stage mode: run exactly this tranche (scan → equities → chains → options) and exit; "
                                                 "used to keep each live stage inside the credential-proxy window")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.universe.split(",") if t.strip()]

    from engine.execution.execution_engine import ExecutionEngine
    from engine.execution.macro_event_flagger import MacroEventFlagger
    from engine.execution.eod_allocation_council import EODAllocationCouncil
    from engine.execution.run_patch_report import build_run_patch
    from engine.execution.account_mandates import DrawdownGuard
    from engine.allocation.allocation_engine import AllocationRules
    from engine.execution.broker_types import OrderSide
    from engine.execution.rotation_exits import compute_rotation_exits
    from engine.execution.gold_standard_report import collect_context, render, classify_bucket
    import time
    t_start = time.monotonic()

    t0 = datetime.now()
    _p(f"# Sandbox cycle {t0:%Y-%m-%d %H:%M:%S}  universe={','.join(tickers)}  mode={'OFFLINE' if args.offline else 'SCHWAB'}")

    # ── 1. Engine + Schwab broker (router when SCHWAB_ACCOUNT_* set) ─────────
    eng = ExecutionEngine(connect_broker=not args.offline)
    broker = eng.broker
    status = eng.get_broker_status()
    _p(f"broker={status['broker']} connected={status['connected']} live={status['is_live']} account={status['account']}")
    if status["connected"] and hasattr(broker, "sync_positions"):
        try:
            broker.sync_positions()
        except Exception as exc:  # noqa: BLE001
            print(f"  position sync failed: {exc}")
    nav = float(broker.state.nav or 0.0)
    _p(f"NAV ${nav:,.2f}  cash ${broker.state.cash:,.2f}  positions={len(broker.state.positions)}")

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
    _p(f"macro flag: {flag.level} score={flag.score:.2f} opt×{flag.options_add_scale:.2f} eq×{flag.equities_add_scale:.2f} "
          + ("triggers=" + "; ".join(flag.triggers[:3]) if flag.triggers else ""))

    # ── 3. 20 % drawdown gate BEFORE any add ─────────────────────────────────
    if hasattr(broker, "portfolio_snapshot"):
        snap = broker.portfolio_snapshot(); dd = snap["drawdown"]
        for label, a in snap["accounts"].items():
            print(f"  {label:<10} …{a['mandate']['account_last4']} NAV ${a['nav']:,.0f} opt {a['options_notional']:,.0f}/{a['options_cap']:,.0f} "
                  f"eq {a['equities_mv']:,.0f}/{a['equities_cap']:,.0f} DD {a['drawdown']['drawdown']:.1%} {a['drawdown']['level']}")
    else:
        g = DrawdownGuard(); g.seed_peak("PORTFOLIO", nav); dd = g.check("PORTFOLIO", nav).to_dict()
    _p(f"drawdown: {dd['level']} ({dd.get('drawdown', 0):.1%}) adds_allowed={dd['adds_allowed']} scale={dd['add_scale']}")
    add_scale = float(dd["add_scale"])

    # ── 3b. Universe scan in SEPARATE tranches (allocation-guide UNIVERSE_ORDER) → concurrence ──
    #   Run 1 = SP500 · Run 2 = SP400 (+extras) · Run 3 = SP600 · Run 4 = ETF_FI
    #   Each run is scored on its own distribution and reported on its own;
    #   only after the last run does the concurrence vote build the allocation slate.
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
            rules_ = AllocationRules()
            acct_snap = snap["accounts"] if hasattr(broker, "portfolio_snapshot") else {}
            run_no = {"i": 0}
            per_run_orders: list = []          # every L7 order placed run-by-run (feeds the report / patch)
            per_run_intents: list = []
            filled_eq: dict = {}               # ticker → {account: $}
            seeded: set = set()                # (account, ticker) holdings already counted into the sleeve ledger
            sleeve_memo: dict = {}             # ticker → sleeve, remembered across runs for held names
            filled_opt: set = set()            # option underlyings already submitted this cycle
            run_used_ledger: dict = {}     # (account, run) → $ placed by that run today
            sleeve_used: dict = {}             # (account, sleeve) → $ committed so far (locked-sleeve tracking)
            opt_used: dict = {}                # account → option notional committed so far
            # share of each LOCKED sleeve budget that each universe run may consume (Σ over runs = 1.0):
            RUN_SLEEVE_SHARE = {
                "SCAN_1_SP500":  {"IG_EQUITY": 1.00, "HY_EQUITY": 0.50},
                "SCAN_2_SP400":  {"HY_EQUITY": 0.50, "DISTRESSED": 0.50},
                "SCAN_3_SP600":  {"DISTRESSED": 0.50},
                "SCAN_4_ETF_FI": {"TLTW": 1.00, "FIXED_INCOME": 1.00},
            }
            MIN_NAMES = {"DISTRESSED": 3}     # operator rule: the distressed sleeve carries at least 3 names
            # operator 2026-09-04 (equities): FORGET the IG/HY/distressed buckets — budget per universe run as a share
            # of the account's equity mandate, filled with the BEST names of that run in 5–10 % of ACCOUNT VALUE
            # sleeves (strongest composite: momentum + RSI breakout + Sharpe/ensemble + 52w dislocation).
            RUN_EQ_SHARE = {"SCAN_1_SP500": 0.40, "SCAN_2_SP400": 0.25, "SCAN_3_SP600": 0.20, "SCAN_4_ETF_FI": 0.15}
            EQ_NAME_MIN, EQ_NAME_MAX, EQ_NAME_TARGET = 0.05, 0.10, 0.075     # of account NAV per name
            # per-position options premium cap as a fraction of ACCOUNT NAV; default (None) = engine 10 %.
            # operator 2026-09-04: Individual raised to 20 % so 7–30 DTE large-cap contracts fit a $5.7k account.
            OPTION_POS_CAP = {"INDIVIDUAL": 0.20}
            # operator 2026-09-04: Roth options ALWAYS come from RUN 2 & 3 (SP400 HY / SP600 distressed pools);
            # in RUN 2/3 the chain sweep + INDIVIDUAL options go in BEFORE the equities, then LLC/Roth equities, then Roth options.
            ROTH_OPTION_RUNS = {"SCAN_2_SP400", "SCAN_3_SP600"}
            OPTIONS_FIRST_RUNS = {"SCAN_2_SP400", "SCAN_3_SP600"}
            SLEEVE_PCT = {"IG_EQUITY": rules_.ig_equity_pct, "HY_EQUITY": rules_.hy_equity_pct, "DISTRESSED": rules_.distressed_equity_pct,
                          "TLTW": rules_.tltw_cashflow_pct, "FIXED_INCOME": rules_.fi_macro_pct, "CVR": rules_.event_driven_cvr_pct}

            def _fill_run_options(label: str, tr):
                """Size THIS run's chain pool for one account and route each intent through L7 (market DAY)."""
                a = acct_snap.get(label, {})
                headroom = float(a.get("options_headroom", 0.0)) * add_scale * flag.options_add_scale - opt_used.get(label, 0.0)
                if sd is None or headroom <= 0:
                    _p(f"     {label}: no options headroom left (${headroom:,.0f})"); return
                sd._pool_pending = [t for t in sd._pool_pending if t[0] not in filled_opt]
                try:
                    scan = sd.scan([], nav=headroom / 0.25, l7_nav=nav, account_nav=float(a.get("nav", 0.0)),
                                   delta_used_usd=float(getattr(eng.l7, "_options_delta_exposure", 0.0) or 0.0), use_pool=True,
                                   position_cap_pct=OPTION_POS_CAP.get(label))
                except Exception as exc:  # noqa: BLE001
                    _p(f"     {label}: options sizing failed: {exc}"); return
                its = scan["intents"]
                _p(f"     {label}: {len(its)} option intents on ${headroom:,.0f} headroom (regime {scan['context'].regime}, VIX {scan['context'].vix:.1f})")
                for it in its:
                    o = eng.l7_submit_option_intent(it, regime=getattr(sd, "regime", "NORMAL"))
                    if not o:
                        continue
                    od = o.to_dict() if hasattr(o, "to_dict") else dict(o); od["account"] = label
                    per_run_orders.append(od); per_run_intents.append(it)
                    st = str(od.get("status")).split(".")[-1]
                    _p(f"       {it.ticker:<6} {it.direction:<6} {it.put_call} {it.strike} {it.expiry} {it.dte}DTE ×{it.contracts} mkt≈${it.limit_price:.2f} "
                       f"notional ${it.notional:,.0f} comp={it.composite:.2f} edge={it.edge_bps:+.0f}bp Δ${it.greeks.get('delta_exposure_usd', 0):+,.0f} {it.structure} → {st}"
                       + ("" if "REJECT" not in st.upper() else f"  ← {str(od.get('reason') or od.get('broker_reason') or '')[:120]}"))
                    if "REJECT" not in st.upper():
                        opt_used[label] = opt_used.get(label, 0.0) + float(it.notional)
                        filled_opt.add(it.ticker)

            def _fill_run_equities(label: str, tr):
                """This run's longs → LOCKED sleeves × this run's share × account equity mandate, ≤ G1 10 % NAV."""
                a = acct_snap.get(label, {})
                eq_pct = float(a.get("mandate", {}).get("equities_pct", 0.0))
                if eq_pct <= 0 or not tr.top:
                    _p(f"     {label}: no equities mandate" if eq_pct <= 0 else f"     {label}: run has no longs"); return
                acct_nav_ = float(a.get("nav", 0.0))
                base = acct_nav_ * eq_pct                      # equity mandate of THIS account
                by_sleeve: dict = {}
                # per-ACCOUNT held check (router keeps one SchwabBroker per account); never double-buy on relaunch
                sub = getattr(broker, "brokers", {}).get(label) if hasattr(broker, "brokers") else broker
                held_pos = getattr(getattr(sub, "state", None), "positions", {}) or {}
                held_now = set(held_pos)
                # seed the sleeve ledger with what this account ALREADY holds (relaunch / earlier heartbeat), so the
                # locked sleeve budgets are net of existing positions rather than re-spent
                sleeve_of_run = {c.ticker: c.sleeve for c in list(tr.top) + list(tr.candidates)}
                for t_, p_ in held_pos.items():
                    sl_ = sleeve_of_run.get(t_) or sleeve_memo.get(t_)
                    if not sl_ or (label, t_) in seeded:
                        continue
                    sleeve_memo[t_] = sl_
                    mv = abs(float(getattr(p_, "market_value", 0.0) or 0.0))
                    sleeve_used[(label, sl_)] = sleeve_used.get((label, sl_), 0.0) + mv
                    seeded.add((label, t_))
                # ── run budget = account equity mandate × this run's share, net of what this run already put on
                run_budget = base * RUN_EQ_SHARE.get(tr.name, 0.0) * add_scale * flag.equities_add_scale
                run_used = sum(mv_ for (lb_, rn_), mv_ in run_used_ledger.items() if lb_ == label and rn_ == tr.name)
                budget = max(0.0, run_budget - run_used)
                if budget < EQ_NAME_MIN * acct_nav_:
                    _p(f"     {label}: {tr.name} run budget {RUN_EQ_SHARE.get(tr.name, 0.0):.0%} of equity mandate = ${run_budget:,.0f} "
                       f"already used (${run_used:,.0f}) — no room for a ≥ {EQ_NAME_MIN:.0%} sleeve"); return
                # ── best of the run: composite score, breakout / dislocation tie-break; sleeves of 5–10 % of account NAV
                def _rank(c):
                    bonus = (0.05 if "break" in str(getattr(c, "breakout", "")).lower() else 0.0) \
                          + (0.03 if 55 <= float(getattr(c, "rsi", 50.0)) <= 72 else 0.0) \
                          + (0.02 if float(getattr(c, "pos_52w", 0.5)) < 0.85 else 0.0)     # room to run vs 52w high
                    return -(float(c.score) + bonus)
                cands = [c for c in sorted(tr.top, key=_rank) if c.signal == "BUY"
                         and not filled_eq.get(c.ticker, {}).get(label) and c.ticker not in held_now]
                skipped_held = [c.ticker for c in tr.top if c.ticker in held_now]
                if skipped_held:
                    _p(f"     {label}: already held — skip {', '.join(skipped_held)}")
                n_names = max(1, min(len(cands), int(round(budget / (EQ_NAME_TARGET * acct_nav_)))))
                per = min(EQ_NAME_MAX * acct_nav_, max(EQ_NAME_MIN * acct_nav_, budget / max(1, n_names)))
                picks = cands[:n_names]
                _p(f"     {label}: run budget ${run_budget:,.0f} ({RUN_EQ_SHARE.get(tr.name, 0.0):.0%} of ${base:,.0f} equity mandate), "
                   f"used ${run_used:,.0f} → {len(picks)} sleeves × ${per:,.0f} ({per / acct_nav_:.1%} of account)")
                rows = [(c.ticker, c.sleeve, per, c) for c in picks]
                if not rows:
                    _p(f"     {label}: no eligible BUY names left in this run"); return
                qmap = broker.get_quotes([t for t, *_ in rows]) or {}
                tot = 0.0
                for t, sleeve, dollars, c in rows:
                    q = qmap.get(t) if isinstance(qmap.get(t), dict) else {}
                    px = float(q.get("last") or q.get("mark") or c.price or 0.0)
                    qty = int(dollars // px) if px > 0 else 0
                    if qty <= 0:
                        continue
                    o = eng.l7_submit(ticker=t, side="BUY", quantity=qty, signal_type="QUALITY_BUY", regime="NORMAL",
                                      sector=(getattr(c, "sector", "") or sleeve),
                                      reason=f"{label} {tr.name} run slate {sleeve} ${dollars:,.0f} α{c.raw_score:+.3f}")
                    od = o if isinstance(o, dict) else o.to_dict(); od["alpha"] = c.raw_score; od["account"] = label
                    per_run_orders.append(od)
                    st = str(od.get("status")).split(".")[-1]
                    fp = float(od.get("fill_price") or px)
                    why = "" if "REJECT" not in st.upper() else f"  ← {str(od.get('reason') or od.get('reject_reason') or '')[:110]}"
                    _p(f"       {t:<6} BUY {qty:>4} @ {fp:.2f} ${qty * fp:>9,.0f} [{sleeve}] α{c.raw_score:+.3f} → {st}{why}")
                    if "REJECT" not in st.upper():
                        tot += qty * fp
                        sleeve_used[(label, sleeve)] = sleeve_used.get((label, sleeve), 0.0) + qty * fp
                        run_used_ledger[(label, tr.name)] = run_used_ledger.get((label, tr.name), 0.0) + qty * fp
                        filled_eq.setdefault(t, {})[label] = qty * fp
                _p(f"     {label}: {tot:,.0f} filled this run │ run ledger " +
                   ", ".join(f"{rn} ${v:,.0f}" for (lb, rn), v in run_used_ledger.items() if lb == label))

            def _fill_run_eq(tr):
                """Equities first (right after the run's proposed allocation): LLC → ROTH."""
                if not dd["adds_allowed"]:
                    _p("  ── FILLS SKIPPED: adds blocked by 20% drawdown rule"); return
                mode = "LIVE" if getattr(broker, "live_orders", False) else "DRY_RUN"
                _p(f"  ── FILL RUN {run_no['i']} ({tr.name}) [{mode}] EQUITIES — LLC → ROTH ──")
                if hasattr(broker, "prefer"): broker.prefer("LLC")
                _fill_run_equities("LLC", tr)
                if hasattr(broker, "prefer"): broker.prefer("ROTH")
                _fill_run_equities("ROTH", tr)
                if hasattr(broker, "prefer"): broker.prefer(None)

            def _fill_run(tr, accounts=("INDIVIDUAL", "ROTH")):
                """Options after the tranche's chain sweep: INDIVIDUAL → ROTH (Roth only from RUN 2/3 pools)."""
                if not dd["adds_allowed"]:
                    _p("  ── FILLS SKIPPED: adds blocked by 20% drawdown rule"); return
                mode = "LIVE" if getattr(broker, "live_orders", False) else "DRY_RUN"
                accts = [a for a in accounts if not (a == "ROTH" and tr.name not in ROTH_OPTION_RUNS)]
                if not accts:
                    return
                _p(f"  ── FILL RUN {run_no['i']} ({tr.name}) [{mode}] OPTIONS — {' → '.join(accts)} ──")
                if "ROTH" in accounts and "ROTH" not in accts:
                    _p("     ROTH: skipped — Roth options come from RUN 2 & 3 only (operator rule)")
                for a in accts:
                    if hasattr(broker, "prefer"): broker.prefer(a)
                    _fill_run_options(a, tr)
                if hasattr(broker, "prefer"): broker.prefer(None)

            def on_run(tr):
                run_no["i"] += 1
                _p(f"\n═══ RUN {run_no['i']}: {getattr(tr, 'label', tr.name)} ═══")
                _p(f"  Scanned {tr.screened}/{tr.universe_size} (quoted {tr.quoted}) → shortlist {tr.shortlisted} → "
                   f"BUY {tr.buy_n} │ SELL {tr.sell_n} │ HOLD {tr.hold_n} │ avg α {tr.avg_alpha:+.4f} │ {tr.elapsed_s:.0f}s")
                for j, c in enumerate(tr.top[:5], 1):
                    _p(f"    {j}. {c.ticker:<7} α={c.raw_score:+.4f} score={c.score:.3f} Sharpe={c.sharpe:.2f} vol={c.realized_vol:.0%} [{c.sleeve}]  WHY: {c.why()}")
                for j, c in enumerate(tr.sells[:3], 1):
                    _p(f"    SELL {j}. {c.ticker:<7} α={c.raw_score:+.4f} mom_10d={c.mom_10d:+.1%} → EXIT signal")
                _p(proposed_allocation(tr, nav, rules_, acct_snap, add_scale * flag.equities_add_scale))
                Path("logs").mkdir(exist_ok=True)
                Path(f"logs/run_{run_no['i']}_{tr.name}.txt").write_text(proposed_allocation(tr, nav, rules_, acct_snap, add_scale * flag.equities_add_scale))
                options_first = tr.name in OPTIONS_FIRST_RUNS
                if args.options_only:
                    _p("  OPTIONS-ONLY stage (operator): equities dropped this stage — chain sweep → INDIVIDUAL → ROTH options")
                if args.fill_per_run and not options_first and not args.options_only:
                    try:
                        _fill_run_eq(tr)          # RUN 1/4: equities first, then chains + options
                    except Exception as exc:  # noqa: BLE001
                        _p(f"  equity fill after {tr.name} failed: {exc}")
                elif options_first:
                    _p("  (RUN 2/3 order: chain sweep → INDIVIDUAL options → LLC/ROTH equities → ROTH options)")
                # ── options chains for EVERY name of this tranche, right after its run (sequential: run 1 →
                #    chains 1 → run 2 → chains 2 → …). Sized later, once all tranches are in and concurred.
                try:
                    from engine.execution.universe_tranche_scan import ConcurrenceResult as _CR
                    bkt = _CR.TRANCHE_OPTIONS_BUCKET.get(tr.name, "OPTIONS_HY")
                    by_t = {c.ticker: (c.options_bucket or bkt) for c in list(tr.top) + list(tr.candidates) + list(tr.sells)}
                    uni = [(t, by_t.get(t, bkt)) for t in sorted(set(tr.universe_tickers) | set(by_t))]
                    sm = sd.sweep_tranche(tr.name, uni)
                    _p(f"  ── OPTIONS CHAINS RUN {run_no['i']} ({tr.name}): {sm['names']} names → {sm['chains']} chains pulled, "
                       f"{sm['no_chain']} no listed options in window, {sm['illiquid']} illiquid, {sm['errors']} fetch errors, "
                       f"{sm['passed']} PASS BSM/MC → contract found │ {sm['elapsed_s']:.0f}s")
                    _p("     status: " + ", ".join(f"{k} {v}" for k, v in sm["status_hist"].items()))
                    for j, t in enumerate(sm["top"], 1):
                        _p(f"     {j}. {t['ticker']:<6} {t['direction']:<7} {t['structure']:<8} {t['put_call']} {t['strike']:g} "
                           f"{t['dte']}DTE mid ${t['mid']:.2f} composite {t['composite']:.2f} edge {t['edge_bps']:+d}bp [{t['bucket']}]")
                    if not sm["top"]:
                        _p("     no contract in this tranche cleared composite ≥ 0.55 / MC / liquidity")
                except Exception as exc:  # noqa: BLE001
                    _p(f"  options chain sweep for {tr.name} failed: {exc}")
                if args.fill_per_run:
                    try:
                        if args.options_only:
                            _fill_run(tr)
                        elif options_first:
                            _fill_run(tr, accounts=("INDIVIDUAL",))
                            _fill_run_eq(tr)
                            _fill_run(tr, accounts=("ROTH",))
                        else:
                            _fill_run(tr)
                    except Exception as exc:  # noqa: BLE001
                        _p(f"  fill after {tr.name} failed: {exc}")

            sd.reset_pool()
            if args.only_run:
                names_ = [n for n, _ in scanner.tranches]
                if args.only_run in names_:
                    run_no["i"] = names_.index(args.only_run)
                    scanner.tranches = [(n, t) for n, t in scanner.tranches if n == args.only_run]
                    _p(f"  STAGE MODE: {args.only_run} only (run {run_no['i'] + 1} of {len(names_)}) — concurrence/council run in the final stage")
            if args.start_from:
                names_ = [n for n, _ in scanner.tranches]
                if args.start_from in names_:
                    skipped = names_[: names_.index(args.start_from)]
                    scanner.tranches = [(n, t) for n, t in scanner.tranches if n not in skipped]
                    run_no["i"] = len(skipped)
                    _p(f"  resuming at {args.start_from} — skipping {', '.join(skipped)} (already run/filled on the previous token)")
            tranche = scanner.run(corridor_bias=bias, on_run=on_run)
            if args.only_run and args.only_run != "SCAN_4_ETF_FI":
                _p(f"\n  STAGE {args.only_run} complete — equities and options for this tranche are placed; launch the next stage.")
                _p(f"  positions now: {len(getattr(broker.state, 'positions', {}) or {})}  NAV ${broker.state.nav:,.0f}")
                return 0
            _p(f"\n═══ CONCURRENCE (after RUN {run_no['i']}) → {len(tranche.final)} names ═══")
            pool = getattr(sd, "_pool_runs", {}) or {}
            if pool:
                _p("  options pool for final sizing: " + " │ ".join(f"{k.split('_', 1)[-1]} {v['passed']}/{v['names']}" for k, v in pool.items())
                   + f" → {sum(v['passed'] for v in pool.values())} contracts carried into allocation")
            _p("  " + ", ".join(f"{c.ticker}[{c.sleeve}] z{c.z_score:+.1f} {sum(c.votes.values())}/{len(c.votes) or 6}" for c in tranche.final))
            for note in getattr(tranche, "backfill_notes", []) or []:
                _p(f"  LOCKED SLEEVE → {note}")
            if getattr(tranche, "dropped", None):
                _p("  dropped: " + "; ".join(f"{t} ({w})" for t, w in list(tranche.dropped)[:8]))
            eq_slate = tranche.equity_slate()
            opt_universe = tranche.options_universe()   # EVERY tranche name (longs, sells, shortlist) + core ETFs
            by_run = tranche.options_universe_by_run()
            _p("  options universe: " + ", ".join(f"{k} {len(v)}" for k, v in by_run.items()) + f" + core ETFs → {len(opt_universe)} names, chains pulled for all")
            sd.backfill_candidates = tranche.options_backfill()   # empty HY/Dist buckets ← SP400/SP600 names
        except Exception as exc:  # noqa: BLE001
            print(f"  tranche scan failed: {exc}")
    else:
        print("tranche scan: skipped (" + ("offline" if not status["connected"] else "--no-tranche") + ") — static universe used")

    # ── 4/5. FILL ORDER: INDIVIDUAL (options) → LLC (equities/ETF) → ROTH (composite) ──
    #   Rotation exits go first (20 % drawdown rule / SELL signal in a universe run), then:
    #     Phase A  INDIVIDUAL  100 % options — 1–7 DTE engine sized to that account's options headroom
    #     Phase B  LLC         equities + ETFs from the concurrence slate, sized on LLC NAV
    #     Phase C  ROTH        composite: 25 % options (same contracts, re-sized) + 75 % equities (same slate)
    #   The router pins the destination per phase; mandate permission + headroom still enforced.
    orders = []
    exits = compute_rotation_exits(dict(broker.state.positions), tranche_result=tranche,
                                   bucket_of={s_: classify_bucket(s_, getattr(p_, "sector", "")) for s_, p_ in broker.state.positions.items()})
    for e in exits:
        o = eng.l7_submit(ticker=e.ticker, side="SELL" if e.side == "SELL" else "BUY", quantity=e.quantity,
                          signal_type="ROTATION_EXIT", regime="NORMAL", sector=e.bucket, reason=e.reason)
        od = o if isinstance(o, dict) else o.to_dict()
        e.status = str(od.get("status", "")).split(".")[-1]
        orders.append(od)
    _p(f"\n═══ PHASE 5 · ROTATION EXITS: {len(exits)} ═══  " + ", ".join(f"{e.ticker} {e.pnl_pct:+.1%} ({e.reason_short}) → {e.status}" for e in exits[:8]))

    rules = AllocationRules()
    accounts = snap["accounts"] if hasattr(broker, "portfolio_snapshot") else {}
    scale = add_scale * flag.options_add_scale
    report, intents = None, []

    def _acct(label):
        return accounts.get(label, {})

    def _equity_rows(acct_nav: float, eq_pct: float):
        """Slate sized on ONE account: sleeve % × (account NAV × equities mandate) ÷ names in sleeve, ≤ 10 % NAV (G1)."""
        sleeve_pct = {"IG_EQUITY": rules.ig_equity_pct, "HY_EQUITY": rules.hy_equity_pct, "DISTRESSED": rules.distressed_equity_pct,
                      "TLTW": rules.tltw_cashflow_pct, "FIXED_INCOME": rules.fi_macro_pct, "CVR": rules.event_driven_cvr_pct}
        if eq_slate:
            n_by = {}
            for r in eq_slate:
                n_by[r["bucket"]] = n_by.get(r["bucket"], 0) + 1
            rows = [(r["ticker"], r["bucket"], sleeve_pct.get(r["bucket"], 0.05) / n_by[r["bucket"]]) for r in eq_slate]
        else:
            rows = [(t, BUCKET_MAP.get(t, (None, "HY_EQUITY"))[1], args.equity_slate_pct) for t in tickers]
        base = acct_nav * eq_pct / 0.90        # sleeves sum to 90 % → scale so the equity mandate is fully used
        return [(t, b, min(0.10 * nav, base * pct) * add_scale * flag.equities_add_scale) for t, b, pct in rows]

    def _submit_equities(label: str, rows) -> list:
        out = []
        alpha_by = {c.ticker: c.z_score for c in tranche.final} if tranche is not None else {}
        quotes.update(broker.get_quotes([t for t, _, _ in rows]) or {})
        for t, sleeve, dollars in rows:
            q = quotes.get(t) if isinstance(quotes.get(t), dict) else None
            px = float((q or {}).get("last") or (q or {}).get("mark") or 0.0)
            qty = int(dollars // px) if px > 0 else 0
            if qty <= 0:
                continue
            o = eng.l7_submit(ticker=t, side="BUY", quantity=qty, signal_type="QUALITY_BUY", regime="NORMAL", sector=sleeve,
                              reason=f"{label} {'concurrence' if eq_slate else 'sandbox'} slate {sleeve} ${dollars:,.0f}")
            od = o if isinstance(o, dict) else o.to_dict(); od["alpha"] = alpha_by.get(t); od["account"] = label
            out.append(od)
        return out

    extended_proposals: list = []

    def _submit_options(label: str, opt_headroom: float, acct_nav: float = 0.0):
        """Run the 1–7 DTE engine sized to this account's options headroom and route every intent through L7."""
        nonlocal report
        if sd is None or opt_headroom <= 0 or not status["connected"]:
            _p(f"  {label}: options skipped (" + ("offline" if not status["connected"] else "no options headroom after mandate/DD/macro scaling") + ")")
            return [], []
        try:
            # engine caps 10/10/5 of nav → Σ ≤ headroom; l7_nav = portfolio NAV for the G9 Σ|Δ$| ≤ 20 % budget
            scan = sd.scan(opt_universe, nav=opt_headroom / 0.25, l7_nav=nav, account_nav=acct_nav,
                           delta_used_usd=float(getattr(eng.l7, "_options_delta_exposure", 0.0) or 0.0),
                           use_pool=tranche is not None,      # pass 1 already done per tranche → size the pool
                           position_cap_pct=OPTION_POS_CAP.get(str(getattr(broker, "_preferred", "") or "").upper()))
        except Exception as exc:  # noqa: BLE001
            _p(f"  {label}: options scan failed: {exc}"); return [], []
        report = report or scan["report"]
        ctx = scan["context"]; its = scan["intents"]
        _p(f"  {label}: options scan regime={ctx.regime} VIX={ctx.vix:.2f} corridor={ctx.corridor_position} bias={ctx.direction_bias:+.2f} "
           f"→ {len(its)} intents on ${opt_headroom:,.0f} headroom")
        bf = scan["report"].get("backfill") or {}
        for bkt, names in bf.items():
            _p(f"    BUCKET BACK-FILL {bkt} had no eligible name → filled from {'SP400' if bkt == 'OPTIONS_HY' else 'SP600'}: {', '.join(names)}")
        if getattr(sd, "backfill_candidates", None) and not bf:
            per_t = (scan.get("per_ticker") or scan["report"].get("per_ticker") or {})
            # per-tranche chain coverage: every name in each run must have had its chain pulled
            if tranche is not None:
                for run_name, names in tranche.options_universe_by_run().items():
                    st = [str(per_t.get(t, {}).get("status", "MISSING")) for t in names]
                    no_chain = sum(1 for t in names if any("DTE chain" in r for r in per_t.get(t, {}).get("reasons", [])))
                    err = sum(1 for x in st if x == "ERROR"); missing = sum(1 for x in st if x == "MISSING")
                    ok = len(names) - no_chain - err - missing
                    _p(f"    chains {run_name:<7} {len(names):>3} names → {ok} chains scanned, {no_chain} no listed options in window, {err} fetch errors, {missing} not reached")
            tried = [t for t, r in per_t.items() if r.get("backfill")]
            if tried:
                _p(f"    BUCKET BACK-FILL tried {len(tried)} SP400/SP600 names, none eligible: " + ", ".join(tried[:12]))
        out = []
        for it in its:
            o = eng.l7_submit_option_intent(it, regime=getattr(sd, "regime", "NORMAL"))
            if o:
                od = o.to_dict() if hasattr(o, "to_dict") else dict(o); od["account"] = label; out.append(od)
                wing = next((l for l in (it.legs or []) if l.get("instruction") == "SELL_TO_OPEN"), None)
                struct = f"{it.structure}" + (f" (short wing {wing['symbol'][-8:]})" if wing else "")
                _p(f"    {it.ticker:<6} {it.direction:<6} {it.put_call} {it.strike} {it.expiry} {it.dte}DTE ×{it.contracts} @ ${it.limit_price:.2f} "
                   f"notional ${it.notional:,.0f} comp={it.composite:.2f} edge={it.edge_bps:+.0f}bp Δ${it.greeks.get('delta_exposure_usd', 0):+,.0f} "
                   f"structure={struct} → {str(od.get('status')).split('.')[-1]}")
        # Extended tenor (dte_max+1 … 30 DTE) — proposal only; skipped when the auto window already covers 30 DTE
        ext = []
        if sd.cfg.dte_max < 30:
            try:
                ext = sd.extended_watch(opt_universe, dte_min=sd.cfg.dte_max + 1, dte_max=30)
            except Exception as exc:  # noqa: BLE001
                _p(f"  {label}: extended-tenor scan failed: {exc}")
        if sd.cfg.dte_max >= 30:
            pass
        elif ext:
            _p(f"  {label}: EXTENDED TENOR 8–30 DTE — {len(ext)} attractive proposal(s), NOT executed (need your OK):")
            for pr in ext:
                wing = f"/{pr['wing_strike']:g}" if pr.get("wing_strike") else ""
                _p(f"    ▶ {pr['ticker']:<5} {pr['direction']:<7} {pr['structure']} {pr['put_call']} {pr['strike']:g}{wing} exp {pr['expiry']} ({pr['dte']}DTE) "
                   f"mid ${pr['mid']:.2f} vs BSM ${pr['bsm_fair']:.2f} edge {pr['edge_bps']:+d}bp comp {pr['composite']:.2f} "
                   f"MC conf {pr['mc_conf']:.2f} P(ITM) {pr['p_itm']:.0%} Δ{pr['delta']:+.2f} Γ{pr['gamma']:.3f} Θ{pr['theta']:+.2f}")
                _p(f"        why: {'; '.join(pr['why'])}")
            extended_proposals.extend(dict(pr, account=label) for pr in ext)
        else:
            _p(f"  {label}: extended tenor 8–30 DTE — no proposal clears the 0.60 composite bar")
        return out, its

    filled_per_run = bool(args.fill_per_run and tranche is not None)
    if filled_per_run:
        orders += per_run_orders; intents += per_run_intents
        acc = [o for o in per_run_orders if "REJECT" not in str(o.get("status", "")).upper()]
        _p(f"\n═══ FILLS ALREADY PLACED RUN-BY-RUN: {len(acc)} accepted / {len(per_run_orders) - len(acc)} rejected by L7 ═══")
        for label in ("INDIVIDUAL", "LLC", "ROTH"):
            eq = sum(v[label] for v in filled_eq.values() if label in v)
            _p(f"  {label:<10} options ${opt_used.get(label, 0.0):,.0f} │ equities ${eq:,.0f} │ " +
               ", ".join(f"{sl} ${v:,.0f}" for (lb, sl), v in sleeve_used.items() if lb == label))
        # minimum-names top-up (e.g. DISTRESSED ≥ 3) from the tranche candidates, SP600 first, then SP400, then SP500
        for sleeve, need in MIN_NAMES.items():
            sleeve_of = {c.ticker: c.sleeve for tr_ in tranche.tranches for c in list(tr_.top) + list(tr_.candidates)}
            have = sorted({t for t in filled_eq if sleeve_of.get(t) == sleeve})
            if len(have) >= need:
                continue
            order = ["SCAN_3_SP600", "SCAN_2_SP400", "SCAN_1_SP500"]
            pool_c = []
            for rn in order:
                tr_ = next((x for x in tranche.tranches if x.name == rn), None)
                if tr_ is None:
                    continue
                pool_c += sorted([c for c in list(tr_.top) + list(tr_.candidates) if c.ticker not in filled_eq and c.signal == "BUY"],
                                 key=lambda c: (0 if c.sleeve == sleeve else 1, -c.score))
            picks = pool_c[: need - len(have)]
            _p(f"  MIN-NAMES {sleeve}: {len(have)} filled < {need} → topping up with " + (", ".join(f"{c.ticker}[{c.tranche}]" for c in picks) or "nothing eligible"))
            for label in ("LLC", "ROTH"):
                a = acct_snap.get(label, {}); eq_pct = float(a.get("mandate", {}).get("equities_pct", 0.0))
                if eq_pct <= 0 or not picks:
                    continue
                base = float(a.get("nav", 0.0)) * eq_pct / 0.90
                budget = max(0.0, base * SLEEVE_PCT.get(sleeve, 0.10) * add_scale * flag.equities_add_scale - sleeve_used.get((label, sleeve), 0.0))
                per = min(0.10 * nav, budget / len(picks)) if budget > 0 else 0.0
                if per <= 0:
                    _p(f"     {label}: {sleeve} budget exhausted"); continue
                if hasattr(broker, "prefer"): broker.prefer(label)
                qmap = broker.get_quotes([c.ticker for c in picks]) or {}
                for c in picks:
                    q = qmap.get(c.ticker) if isinstance(qmap.get(c.ticker), dict) else {}
                    px = float(q.get("last") or q.get("mark") or c.price or 0.0); qty = int(per // px) if px > 0 else 0
                    if qty <= 0:
                        continue
                    o = eng.l7_submit(ticker=c.ticker, side="BUY", quantity=qty, signal_type="QUALITY_BUY", regime="NORMAL",
                                      sector=(getattr(c, "sector", "") or sleeve),
                                      reason=f"{label} min-names top-up {sleeve} from {c.tranche} ${per:,.0f}")
                    od = o if isinstance(o, dict) else o.to_dict(); od["account"] = label; od["alpha"] = c.raw_score; orders.append(od)
                    st = str(od.get("status")).split(".")[-1]; fp = float(od.get("fill_price") or px)
                    _p(f"       {c.ticker:<6} BUY {qty:>4} @ {fp:.2f} ${qty * fp:>9,.0f} [{sleeve}] → {st}")
                    if "REJECT" not in st.upper():
                        sleeve_used[(label, sleeve)] = sleeve_used.get((label, sleeve), 0.0) + qty * fp
                        filled_eq.setdefault(c.ticker, {})[label] = qty * fp
                if hasattr(broker, "prefer"): broker.prefer(None)
        _p("  concurrence review → rotation exits above; sleeves left empty are back-filled per the locked-sleeve rule on the next heartbeat")
        if report is None and sd is not None:
            report = getattr(sd, "last_run", None)
    if not dd["adds_allowed"]:
        _p("adds BLOCKED by 20% portfolio drawdown rule — rotate-or-close only")
    elif filled_per_run:
        pass
    else:
        # ── Phase A: INDIVIDUAL — fill with options first ──
        a = _acct("INDIVIDUAL")
        _p(f"\n═══ PHASE A · INDIVIDUAL …{a.get('mandate', {}).get('account_last4', '')}  (100% options)  NAV ${float(a.get('nav', 0)):,.0f}  "
           f"options headroom ${float(a.get('options_headroom', 0)):,.0f} ═══")
        if hasattr(broker, "prefer"): broker.prefer("INDIVIDUAL")
        o_ind, i_ind = _submit_options("INDIVIDUAL", float(a.get("options_headroom", 0.0)) * scale, float(a.get("nav", 0.0)))
        orders += o_ind; intents += i_ind
        acc = [o for o in o_ind if "REJECT" not in str(o.get("status", "")).upper()]
        _p(f"  INDIVIDUAL result: {len(acc)} option orders accepted ({len(o_ind) - len(acc)} rejected by L7), "
           f"notional ${sum(i.notional for i in i_ind):,.0f}")

        # ── Phase B: LLC — equities + ETFs from the same universe runs ──
        b = _acct("LLC")
        _p(f"\n═══ PHASE B · LLC …{b.get('mandate', {}).get('account_last4', '')}  (equities + ETFs only)  NAV ${float(b.get('nav', 0)):,.0f}  "
           f"equities headroom ${float(b.get('equities_headroom', 0)):,.0f} ═══")
        if hasattr(broker, "prefer"): broker.prefer("LLC")
        rows_llc = _equity_rows(float(b.get("nav", 0.0)), 1.0) if b else _equity_rows(nav, 0.90)
        o_llc = _submit_equities("LLC", rows_llc); orders += o_llc
        for od in o_llc:
            _p(f"    {od.get('ticker'):<6} BUY {od.get('quantity')} @ {float(od.get('fill_price') or 0):.2f} [{od.get('sector')}] → {str(od.get('status')).split('.')[-1]}")
        _p(f"  LLC result: {len(o_llc)} equity/ETF orders, ${sum(float(o.get('fill_quantity') or o.get('quantity') or 0) * float(o.get('fill_price') or 0) for o in o_llc):,.0f}")

        # ── Phase C: ROTH — composite of both (25 % options / 75 % equities) ──
        r = _acct("ROTH")
        _p(f"\n═══ PHASE C · ROTH …{r.get('mandate', {}).get('account_last4', '')}  (composite 25% options / 75% equities)  NAV ${float(r.get('nav', 0)):,.0f} ═══")
        if hasattr(broker, "prefer"): broker.prefer("ROTH")
        o_ro, i_ro = _submit_options("ROTH", float(r.get("options_headroom", 0.0)) * scale, float(r.get("nav", 0.0)))
        orders += o_ro; intents += i_ro
        rows_roth = _equity_rows(float(r.get("nav", 0.0)), float(r.get("mandate", {}).get("equities_pct", 0.75)))
        o_roe = _submit_equities("ROTH", rows_roth); orders += o_roe
        for od in o_roe:
            _p(f"    {od.get('ticker'):<6} BUY {od.get('quantity')} @ {float(od.get('fill_price') or 0):.2f} [{od.get('sector')}] → {str(od.get('status')).split('.')[-1]}")
        acc_ro = [o for o in o_ro if "REJECT" not in str(o.get("status", "")).upper()]
        _p(f"  ROTH result: {len(acc_ro)} option orders accepted ({len(o_ro) - len(acc_ro)} rejected) (${sum(i.notional for i in i_ro):,.0f}) + {len(o_roe)} equity orders")
        if hasattr(broker, "prefer"): broker.prefer(None)
    _p(f"\nL7 orders this cycle: {len(orders)}  " + ", ".join(f"{o.get('account', '')}:{o.get('ticker')}:{str(o.get('status')).split('.')[-1]}" for o in orders[:16]))

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
        _p(f"EOD council: grade {verdict.execution_grade}  next-day {verdict.next_day_allocation}")

    # ── 7. gold-standard report (VIEW 1 / VIEW 2 / TRANSACTION LOG / VIEW 3) + in-chat patch ──
    Path("logs").mkdir(exist_ok=True)
    cycle_s = time.monotonic() - t_start
    if args.gold:
        gctx = collect_context(eng, tranche_result=tranche, options_report=report, orders=orders, exits=exits, macro_flag=flag,
                               drawdown=dd, council=verdict, vix=vix, cycle_time_s=cycle_s,
                               mode="OFFLINE" if args.offline else None)
        gold = render(gctx, recap=args.recap)
        print("\n" + gold)
        Path("logs/last_gold_standard_report.txt").write_text(gold)
    md = build_run_patch(exec_engine=eng, options_report=report, orders=orders, macro_flag=flag,
                         council_verdict=verdict, drawdown=dd, tranche_result=tranche)
    print("\n" + md)
    Path("logs/last_sandbox_cycle.md").write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
