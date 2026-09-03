"""
Run Patch Report — in-chat learning + rotation patch after every run.
=====================================================================

In the sandbox the persistent Phase 6 (learning) and Phase 7 (reconciliation /
weight updates) are NOT built. Instead, after every run this module renders a
markdown "patch" that states:

    1. WHAT WAS LEARNED  — options scan (1–7 DTE): why each ticker was/wasn't
       traded (momentum / RSI breakout, beta-corridor fair value, MC gate,
       BSM/VolSurface edge, composite, sizing), equities slate outcome, L7
       gate rejections, execution quality (slippage / IS).
    2. WHERE THE ROTATION IS RECOMMENDED — sleeve utilisation vs the
       allocation file (IG 40 / HY 10 / Dist 10 / TLTW 15 / FI 5 / CVR 10 /
       Margin 8 / MM 2; options 10/10/5), per-account mandate usage
       (ROTH 25/75 · LLC equities+ETF · INDIVIDUAL 100% options), 20 %
       drawdown status, hourly macro flag, EOD council verdict.

It never trades. It only reads the objects the run already produced.

Usage (after a run)::

    from engine.execution.run_patch_report import build_run_patch
    print(build_run_patch(orchestrator=orch, exec_engine=orch.get("execution_engine")))
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("metadron.run_patch")

ALLOCATION_FILE_SLEEVES = {   # from ARCHITECTURE / AllocationRules (equities + fixed income legs)
    "IG_EQUITY": 0.40, "HY_EQUITY": 0.10, "DISTRESSED": 0.10, "TLTW": 0.15,
    "FIXED_INCOME": 0.05, "CVR": 0.10, "MARGIN": 0.08, "MONEY_MARKET": 0.02,
}
# sleeve → AllocationRules.to_dict() key
_RULE_KEY = {
    "IG_EQUITY": "ig_equity_pct", "HY_EQUITY": "hy_equity_pct", "DISTRESSED": "distressed_equity_pct",
    "TLTW": "tltw_cashflow_pct", "FIXED_INCOME": "fi_macro_pct", "CVR": "event_driven_cvr_pct",
    "MARGIN": "margin_pct", "MONEY_MARKET": "money_market_pct",
}
# broker position sector labels that roll into each sleeve
_SECTOR_TO_SLEEVE = {
    "IG_EQUITY": "IG_EQUITY", "IG": "IG_EQUITY", "HY_EQUITY": "HY_EQUITY", "HY": "HY_EQUITY",
    "DISTRESSED": "DISTRESSED", "DISTRESSED_EQUITY": "DISTRESSED", "TLTW": "TLTW", "TLTW_CASHFLOW": "TLTW",
    "FIXED_INCOME": "FIXED_INCOME", "FI_MACRO": "FIXED_INCOME", "CVR": "CVR", "EVENT_DRIVEN_CVR": "CVR",
    "MARGIN": "MARGIN", "MONEY_MARKET": "MONEY_MARKET",
}
OPTIONS_BUCKET_CAPS = {"OPTIONS_IG": 0.10, "OPTIONS_HY": 0.10, "OPTIONS_DISTRESSED": 0.05}


def _e(x: Any) -> str:
    """Enum-safe display (ProductType.EQUITY → EQUITY)."""
    return str(getattr(x, "value", x)).split(".")[-1] if x is not None else ""


def _f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return d


def _pct(x: Any) -> str:
    return f"{_f(x) * 100:.1f}%"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def _options_section(report: Optional[dict]) -> List[str]:
    out = ["## 1. Options scan (1–7 DTE) — what was learned"]
    if not report:
        out.append("_No ShortDTE options scan ran this cycle._")
        return out
    mkt = report.get("market") or {}
    out.append(
        f"- Market context: proxy **{mkt.get('proxy', 'SPY')}** spot {_f(mkt.get('spot')):.2f} · "
        f"VIX {_f(mkt.get('vix')):.2f} · regime **{report.get('regime', mkt.get('regime', '?'))}** · "
        f"σ₂₀ {_pct(mkt.get('sigma_m'))} · Rm(ann) {_pct(mkt.get('rm_annual'))}"
    )
    out.append(
        f"- Beta corridor fair value: target β {_f(mkt.get('target_beta')):.2f} vs base {_f(mkt.get('base_beta')):.2f} → "
        f"position **{mkt.get('corridor_position', '?')}**, directional bias {_f(mkt.get('direction_bias')):+.2f} "
        f"(used as the fair-value tilt on every direction score)"
    )
    per = report.get("per_ticker") or {}
    if per:
        out.append("")
        out.append("| Ticker | Bucket | RSI | Breakout | Mom 5d/21d | Dir score | MC conf (best DTE) | Status | Why |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for t, rec in per.items():
            mom = rec.get("momentum") or {}
            mc = rec.get("monte_carlo") or {}
            best_mc = max((_f(v.get("confidence")) for v in mc.values()), default=0.0) if mc else 0.0
            why = "; ".join(str(r) for r in (rec.get("reasons") or [])[:2]) or "—"
            out.append(
                f"| {t} | {rec.get('bucket', '')} | {_f(mom.get('rsi')):.0f} | {mom.get('breakout') or mom.get('divergence') or '—'} | "
                f"{_pct(mom.get('mom_5d'))}/{_pct(mom.get('mom_21d'))} | {_f(rec.get('direction_score')):+.2f} | "
                f"{best_mc:.2f} | **{rec.get('status', '')}** | {why} |"
            )
    intents = report.get("intents") or []
    out.append("")
    if intents:
        out.append(f"**{len(intents)} option intent(s) produced** (composite ≥ 0.55, edge ≥ 200 bps, MC gate passed):")
        for i in intents:
            g = i.get("greeks") or {}
            out.append(
                f"- {i.get('ticker')} {i.get('direction')} {i.get('structure')} `{i.get('contract_symbol')}` "
                f"{i.get('put_call')} {_f(i.get('strike')):.2f} exp {i.get('expiry')} ({i.get('dte')} DTE) × {i.get('contracts')} @ "
                f"{_f(i.get('limit_price')):.2f} · notional ${_f(i.get('notional')):,.0f} · composite {_f(i.get('composite')):.2f} · "
                f"edge {_f(i.get('edge_bps')):.0f} bps · Δ {_f(g.get('delta')):+.2f} Γ {_f(g.get('gamma')):.3f} Θ {_f(g.get('theta')):.2f} ν {_f(g.get('vega')):.2f}"
            )
    else:
        statuses: Dict[str, int] = {}
        for rec in per.values():
            statuses[rec.get("status", "?")] = statuses.get(rec.get("status", "?"), 0) + 1
        out.append(
            "**No option contract deployed.** Gate outcomes: "
            + ", ".join(f"{k}×{v}" for k, v in sorted(statuses.items()))
            + ". Learned: "
            + ("1-DTE Monte-Carlo confidence rarely clears 0.20 in a low-vol tape — the scan should lean to 3–7 DTE when VIX < 16; "
               if _f(mkt.get("vix")) and _f(mkt.get("vix")) < 16 else "")
            + "no contract had ≥200 bps VolSurface edge at acceptable delta quality."
        )
    committed = report.get("committed_by_bucket") or {}
    nav = _f(report.get("nav"), 1.0) or 1.0
    out.append("")
    out.append("Options bucket utilisation vs allocation file (10 / 10 / 5 % notional):")
    for b, cap in OPTIONS_BUCKET_CAPS.items():
        used = _f(committed.get(b)) / nav
        out.append(f"- {b}: {_pct(used)} used of {_pct(cap)} cap")
    lad = report.get("ladder") or {}
    if lad:
        legs = lad.get("legs") or lad.get("puts") or []
        out.append(f"- Convexity put ladder: {len(legs)} leg(s) proposed" + (f" — {lad.get('note')}" if lad.get("note") else ""))
    return out


def _execution_section(exec_engine: Any, orders: List[dict]) -> List[str]:
    out = ["## 2. Execution (L7 → Schwab) — what happened"]
    l7 = getattr(exec_engine, "l7", None)
    summ = {}
    try:
        summ = l7.get_execution_summary() if l7 is not None else {}
    except Exception as exc:  # noqa: BLE001
        summ = {"error": str(exc)}
    status = {}
    try:
        status = exec_engine.get_broker_status() if exec_engine is not None else {}
    except Exception as exc:  # noqa: BLE001
        status = {"error": str(exc)}
    mode = "LIVE" if status.get("is_live") else ("DRY_RUN — connected, SCHWAB_LIVE_ORDERS=false" if status.get("connected") else "OFFLINE DRY_RUN")
    out.append(f"- Broker: **{status.get('broker', '?')}** · mode **{mode}** · account {status.get('account')}")
    if summ:
        out.append(
            f"- L7 today: {summ.get('total_orders_today', 0)} orders · fills {summ.get('total_fills_today', 0)} · "
            f"dry-run {summ.get('total_dry_run_today', 0)} · risk level {summ.get('risk_level')} · kill-switch {summ.get('kill_switch')} · "
            f"avg TCA cost {_f(summ.get('avg_tca_cost_bps')):.1f} bps ({summ.get('tca_trend', '—')}) · VaR₉₅ {_pct(summ.get('var_95_1d'))}"
        )
    rejects = [o for o in orders if str(o.get("status", "")).upper() == "REJECTED"]
    if rejects:
        out.append(f"- **{len(rejects)} rejected by L7 gates**:")
        for o in rejects[:8]:
            out.append(f"  - {o.get('ticker')} {o.get('side')} ×{o.get('quantity')} — {str(o.get('reason', ''))[:140]}")
    placed = [o for o in orders if str(o.get("status", "")).upper() in ("FILLED", "DRY_RUN", "SUBMITTED", "PENDING")]
    if placed:
        out.append(f"- {len(placed)} order(s) routed:")
        for o in placed[:12]:
            out.append(
                f"  - {o.get('ticker')} {o.get('side')} ×{o.get('quantity')} @ {_f(o.get('fill_price') or o.get('limit_price') or o.get('arrival_price')):.2f} "
                f"[{_e(o.get('status'))}] {_e(o.get('product_type'))} {o.get('sector') or ''} · route {_e(o.get('routing'))} · "
                f"slip {_f(o.get('slippage_bps')):.1f} bps" + (f" · {o.get('contract_symbol')}" if o.get('contract_symbol') else "")
            )
    if not orders:
        out.append("- No orders were generated this run.")
    return out


def _mandate_section(broker: Any, drawdown: Optional[dict]) -> List[str]:
    out = ["## 3. Account mandates + 20% drawdown rule"]
    if broker is None or not hasattr(broker, "portfolio_snapshot"):
        out.append("_Single-account mode (set SCHWAB_ACCOUNT_ROTH / LLC / INDIVIDUAL to enable mandates)._")
        if drawdown:
            out.append(f"- Portfolio drawdown {_pct(drawdown.get('drawdown'))} → **{drawdown.get('level')}** {drawdown.get('directive', '')}")
        return out
    try:
        snap = broker.portfolio_snapshot()
    except Exception as exc:  # noqa: BLE001
        out.append(f"_snapshot unavailable: {exc}_")
        return out
    out.append("| Account | Mandate | NAV | Options used | Equities used | Drawdown | Adds |")
    out.append("|---|---|---|---|---|---|---|")
    for label, a in (snap.get("accounts") or {}).items():
        m = a.get("mandate") or {}
        dd = a.get("drawdown") or {}
        out.append(
            f"| {label} (…{m.get('account_last4', '?')}) | opt {_pct(m.get('options_pct'))} / eq {_pct(m.get('equities_pct'))} | "
            f"${_f(a.get('nav')):,.0f} | {_pct(a.get('options_used_pct'))} | {_pct(a.get('equities_used_pct'))} | "
            f"{_pct(dd.get('drawdown'))} {dd.get('level', '')} | {'✅' if dd.get('adds_allowed', True) else '⛔ ROTATE/CLOSE'} |"
        )
    pdd = snap.get("drawdown") or {}
    out.append(f"- Portfolio: NAV ${_f(snap.get('nav')):,.0f} · drawdown {_pct(pdd.get('drawdown'))} → **{pdd.get('level', 'OK')}**"
               + (f" — {pdd.get('directive')}" if pdd.get("directive") else ""))
    plan = (drawdown or {}).get("rotation_plan") or []
    if plan:
        out.append("- **Rotate-or-close plan** (20% rule tripped):")
        for p in plan[:10]:
            out.append(f"  - {p.get('action', 'CLOSE')} {p.get('symbol')} — {p.get('reason', '')}")
    log = []
    try:
        log = broker.get_routing_log(last_n=10)
    except Exception:  # noqa: BLE001
        pass
    if log:
        out.append(f"- Routing decisions (last {len(log)}): " + "; ".join(
            f"{r.get('symbol', r.get('ticker'))}→{r.get('account')}" + (f" ({r.get('reason')})" if r.get("reason") else "") for r in log))
    return out


def _macro_section(flag: Any) -> List[str]:
    out = ["## 4. Hourly macro / market-event flag"]
    if flag is None:
        out.append("_No macro flag evaluated yet this session._")
        return out
    d = flag.to_dict() if hasattr(flag, "to_dict") else dict(flag)
    out.append(f"- Level **{d.get('level')}** (score {_f(d.get('score')):.2f}) · options add scale ×{_f(d.get('options_add_scale'), 1):.2f} · "
               f"equities add scale ×{_f(d.get('equities_add_scale'), 1):.2f}"
               + (" · **forces rotation review**" if d.get("force_rotation_review") else ""))
    for t in (d.get("triggers") or [])[:6]:
        out.append(f"  - {t}")
    for e in (d.get("events") or [])[:6]:
        out.append(f"  - event: {e}")
    return out


def _allocation_section(rules: Any, verdict: Any, positions: Dict[str, Any], nav: float) -> List[str]:
    out = ["## 5. Allocation vs the allocation file → rotation recommendation"]
    used: Dict[str, float] = {}
    for sym, p in (positions or {}).items():
        sec = str(getattr(p, "sector", None) or (p.get("sector") if isinstance(p, dict) else "") or "UNASSIGNED").upper()
        sec = _SECTOR_TO_SLEEVE.get(sec, sec)
        mv = _f(getattr(p, "market_value", None) if not isinstance(p, dict) else p.get("market_value"))
        used[sec] = used.get(sec, 0.0) + mv
    nav = nav or 1.0
    out.append("| Sleeve | Allocation file | Deployed | Gap | Action |")
    out.append("|---|---|---|---|---|")
    recs: List[str] = []
    for sleeve, target in ALLOCATION_FILE_SLEEVES.items():
        cur = _f(rules.to_dict().get(_RULE_KEY[sleeve]), target) if rules is not None and hasattr(rules, "to_dict") else target
        dep = used.get(sleeve, 0.0) / nav
        gap = cur - dep
        hard = sleeve in ("MARGIN", "MONEY_MARKET")
        if hard:
            action = "HARD — hold"
        elif gap > 0.03:
            action = f"ADD up to {_pct(gap)}"
            recs.append(f"Rotate **into {sleeve}** (+{_pct(gap)} to reach {_pct(cur)})")
        elif gap < -0.03:
            action = f"TRIM {_pct(-gap)}"
            recs.append(f"Rotate **out of {sleeve}** (−{_pct(-gap)}, over the {_pct(cur)} cap)")
        else:
            action = "on target"
        out.append(f"| {sleeve} | {_pct(cur)} | {_pct(dep)} | {gap:+.1%} | {action} |")
    out.append("")
    if verdict is not None:
        v = verdict.to_dict() if hasattr(verdict, "to_dict") else dict(verdict)
        out.append(f"**EOD council** (equities/ETF only): execution grade **{v.get('execution_grade', '?')}**, "
                   f"votes {v.get('votes')}. Next-day allocation:")
        for k, val in (v.get("next_day_allocation") or {}).items():
            out.append(f"- {k}: {_pct(val)}")
        for n in (v.get("notes") or [])[:5]:
            out.append(f"- note: {n}")
    else:
        out.append("_EOD council has not convened yet (runs at 16:00 ET)._")
    out.append("")
    out.append("### Rotation recommendation")
    if recs:
        out += [f"- {r}" for r in recs]
    else:
        out.append("- Sleeves are within ±3 pts of the allocation file — no rotation required from allocation drift.")
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_run_patch(
    orchestrator: Any = None,
    exec_engine: Any = None,
    *,
    options_report: Optional[dict] = None,
    orders: Optional[List[dict]] = None,
    macro_flag: Any = None,
    council_verdict: Any = None,
    drawdown: Optional[dict] = None,
    tranche_result: Any = None,
    save_dir: Optional[str] = "logs/run_patch",
) -> str:
    """Render the in-chat learning + rotation patch as markdown."""
    if orchestrator is not None:
        exec_engine = exec_engine or orchestrator._get("execution_engine")
        tranche_result = tranche_result or getattr(orchestrator, "_last_tranche_result", None)
        options_report = options_report or getattr(orchestrator, "_last_options_report", None)
        orders = orders if orders is not None else list(getattr(orchestrator, "_last_execution_orders", []) or [])
        macro_flag = macro_flag or getattr(orchestrator, "_last_macro_flag", None)
        council_verdict = council_verdict or getattr(orchestrator, "_last_council_verdict", None)
        if drawdown is None:
            try:
                drawdown = orchestrator._drawdown_gate(exec_engine, orchestrator._live_nav(exec_engine))
            except Exception:  # noqa: BLE001
                drawdown = None
    if isinstance(options_report, dict) and "report" in options_report and "per_ticker" not in options_report:
        options_report = options_report["report"]
    orders = orders or []
    l7 = getattr(exec_engine, "l7", None)
    if not orders and l7 is not None:
        try:
            orders = l7.get_dry_run_orders(last_n=50)
        except Exception:  # noqa: BLE001
            orders = []
    broker = getattr(exec_engine, "broker", None)
    positions = dict(getattr(getattr(broker, "state", None), "positions", {}) or {})
    nav = _f(getattr(getattr(broker, "state", None), "nav", None)) or _f((options_report or {}).get("nav"))
    alloc = orchestrator._get("allocation_engine") if orchestrator is not None else None
    rules = getattr(alloc, "rules", None)
    if rules is None:
        try:
            from engine.allocation.allocation_engine import AllocationRules
            rules = AllocationRules()
        except Exception:  # noqa: BLE001
            rules = None

    lines: List[str] = [
        f"# Metadron run patch — {datetime.now().strftime('%Y-%m-%d %H:%M %Z').strip()}",
        "_Sandbox replacement for Phase 6/7: nothing below is persisted as a weight update; it is the learning + rotation patch for this run._",
        "",
    ]
    lines.append("## 0. Universe scan — Scan 1 (S&P 500) → Scan 2 (SmallCap 600) → Scan 3 (remaining ~400) → concurrence")
    if tranche_result is not None and hasattr(tranche_result, "markdown"):
        lines += tranche_result.markdown().splitlines()
    else:
        lines.append("_No tranche universe scan ran this cycle (Schwab offline or scan skipped)._")
    lines.append("")
    lines += _options_section(options_report); lines.append("")
    lines += _execution_section(exec_engine, orders); lines.append("")
    lines += _mandate_section(broker, drawdown); lines.append("")
    lines += _macro_section(macro_flag); lines.append("")
    lines += _allocation_section(rules, council_verdict, positions, nav)
    md = "\n".join(lines)

    if save_dir:
        try:
            d = Path(save_dir); d.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            (d / f"patch_{stamp}.md").write_text(md)
            (d / f"patch_{stamp}.json").write_text(json.dumps({
                "options_report": options_report, "orders": orders, "drawdown": drawdown,
                "macro_flag": macro_flag.to_dict() if hasattr(macro_flag, "to_dict") else macro_flag,
                "council": council_verdict.to_dict() if hasattr(council_verdict, "to_dict") else council_verdict,
            }, default=str, indent=1))
        except Exception as exc:  # noqa: BLE001
            logger.debug("patch save failed: %s", exc)
    return md


__all__ = ["build_run_patch", "ALLOCATION_FILE_SLEEVES", "OPTIONS_BUCKET_CAPS"]
