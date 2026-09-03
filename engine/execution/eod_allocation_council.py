"""
EOD Allocation Council — equities + ETFs ONLY (options overlay is out of scope).

Runs once at market close (16:00 ET) from LiveLoopOrchestrator._run_market_close.
It reviews how the day's equity/ETF allocation performed *from price execution*
(arrival → fill slippage, implementation shortfall, fill-to-close drift) and how
each sleeve did versus its target weight, then convenes five council members —
each an independent, rule-based perspective — that vote on next-day sleeve
tilts. Votes are aggregated, clipped to ±5 pts per sleeve around the
``AllocationRules`` targets, renormalised, and the hard rules are preserved:

    Margin 8% (HARD)      Money-market 2% (HARD FLOOR)
    20% drawdown          → council may only recommend de-risking / rotation

Council members
    EXECUTION   — slippage, fill quality, arrival-vs-close drift per sleeve
    RISK        — sleeve P&L dispersion, drawdown status, gross exposure
    MOMENTUM    — day return by sleeve, RSI/momentum read passed in from engine
    MACRO       — macro-event flag, regime, VIX
    DISCIPLINE  — drift back to the allocation file (mean reversion to targets)

Output: ``CouncilVerdict`` (dict + markdown) saved to logs/council/eod_YYYYMMDD.{json,md}.
The markdown is what gets posted in chat as the Phase 6/7 replacement patch.
The JSON is the exact brief to hand to an external multi-model council if a
second opinion is requested.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EQUITY_SLEEVES = ["IG_EQUITY", "HY_EQUITY", "DISTRESSED_EQUITY", "TLTW_CASHFLOW", "FI_MACRO", "EVENT_DRIVEN_CVR"]
HARD_SLEEVES = {"MARGIN": 0.08, "MONEY_MARKET": 0.02}
MAX_TILT = 0.05                     # ±5 pts per sleeve per day
ETF_SLEEVE_HINTS = {
    "TLTW": "TLTW_CASHFLOW", "JEPI": "TLTW_CASHFLOW", "JEPQ": "TLTW_CASHFLOW", "XYLD": "TLTW_CASHFLOW", "QYLD": "TLTW_CASHFLOW",
    "TLT": "FI_MACRO", "IEF": "FI_MACRO", "SHY": "FI_MACRO", "AGG": "FI_MACRO", "BND": "FI_MACRO", "LQD": "FI_MACRO",
    "HYG": "HY_EQUITY", "JNK": "HY_EQUITY",
}


@dataclass
class SleeveDayStat:
    sleeve: str
    target_pct: float
    deployed_pct: float = 0.0
    orders: int = 0
    filled: int = 0
    dry_run: int = 0
    rejected: int = 0
    notional: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_shortfall_usd: float = 0.0
    fill_to_close_bps: float = 0.0     # + means price moved in our favour after fill
    day_pnl: float = 0.0
    day_return_pct: float = 0.0
    tickers: List[str] = field(default_factory=list)


@dataclass
class CouncilVote:
    member: str
    tilts: Dict[str, float]            # sleeve → +/- pts (fraction of NAV)
    confidence: float
    rationale: List[str]


@dataclass
class CouncilVerdict:
    as_of: str
    nav: float
    equity_orders_reviewed: int
    sleeves: Dict[str, dict]
    votes: List[dict]
    next_day_allocation: Dict[str, float]
    current_targets: Dict[str, float]
    drawdown_level: str
    execution_grade: str
    summary: List[str]
    per_account: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def markdown(self) -> str:
        L = [f"## EOD Allocation Council — equities & ETFs ({self.as_of[:10]})",
             f"NAV ${self.nav:,.0f} · {self.equity_orders_reviewed} equity/ETF orders reviewed · "
             f"execution grade **{self.execution_grade}** · drawdown {self.drawdown_level}", "",
             "| Sleeve | Target | Deployed | Orders (fill/dry/rej) | Slippage bps | Fill→close bps | Day P&L | Next-day |",
             "|---|---|---|---|---|---|---|---|"]
        for s, d in self.sleeves.items():
            L.append(f"| {s} | {d['target_pct']:.1%} | {d['deployed_pct']:.1%} | {d['orders']} ({d['filled']}/{d['dry_run']}/{d['rejected']}) "
                     f"| {d['avg_slippage_bps']:+.1f} | {d['fill_to_close_bps']:+.1f} | ${d['day_pnl']:,.0f} | **{self.next_day_allocation.get(s, 0):.1%}** |")
        L.append(f"| MARGIN (hard) | 8.0% | — | — | — | — | — | **8.0%** |")
        L.append(f"| MONEY_MARKET (hard floor) | 2.0% | — | — | — | — | — | **2.0%** |")
        L += ["", "### Council votes"]
        for v in self.votes:
            tilt_txt = ", ".join(f"{k} {val:+.1%}" for k, val in v["tilts"].items() if abs(val) >= 0.0005) or "no change"
            L.append(f"- **{v['member']}** (conf {v['confidence']:.2f}): {tilt_txt}")
            for r in v["rationale"]:
                L.append(f"  - {r}")
        if self.per_account:
            L += ["", "### Next-day sleeve caps per account (fraction of account NAV)"]
            for acct, caps in self.per_account.items():
                L.append(f"- **{acct}**: " + ", ".join(f"{k} {v:.1%}" for k, v in caps.items()))
        L += ["", "### Summary"] + [f"- {s}" for s in self.summary]
        return "\n".join(L)


class EODAllocationCouncil:
    def __init__(self, rules: Any, log_dir: Optional[str | Path] = "logs/council"):
        self.rules = rules
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _targets(self) -> Dict[str, float]:
        r = self.rules
        return {
            "IG_EQUITY": r.ig_equity_pct, "HY_EQUITY": r.hy_equity_pct, "DISTRESSED_EQUITY": r.distressed_equity_pct,
            "TLTW_CASHFLOW": r.tltw_cashflow_pct, "FI_MACRO": r.fi_macro_pct, "EVENT_DRIVEN_CVR": r.event_driven_cvr_pct,
        }

    @staticmethod
    def _sleeve_of(order: Any) -> Optional[str]:
        sector = str(getattr(order, "sector", "") or "").upper()
        if sector in EQUITY_SLEEVES:
            return sector
        t = str(getattr(order, "ticker", "")).upper()
        if t in ETF_SLEEVE_HINTS:
            return ETF_SLEEVE_HINTS[t]
        st = str(getattr(order, "signal_type", "")).upper()
        if "CVR" in st or "EVENT" in st:
            return "EVENT_DRIVEN_CVR"
        if "FALLEN" in st or "DISTRESS" in st:
            return "DISTRESSED_EQUITY"
        if "HY" in st:
            return "HY_EQUITY"
        if "MACRO" in st or "FI_" in st:
            return "FI_MACRO"
        return "IG_EQUITY"

    # ------------------------------------------------------------------
    def convene(
        self,
        l7_orders: List[Any],
        nav: float,
        positions: Dict[str, Any],
        close_prices: Optional[Dict[str, float]] = None,
        drawdown: Optional[dict] = None,
        macro_flag: Optional[dict] = None,
        momentum: Optional[Dict[str, float]] = None,      # ticker → momentum score (-1..1) / RSI-based
        regime: str = "",
        vix: Optional[float] = None,
        account_router: Any = None,
    ) -> CouncilVerdict:
        targets = self._targets()
        stats = {s: SleeveDayStat(sleeve=s, target_pct=t) for s, t in targets.items()}
        close_prices = close_prices or {}
        nav = float(nav or 0.0) or 1.0

        # ---- 1. execution review on equity/ETF orders only ----------------
        reviewed = 0
        for o in l7_orders:
            pt = str(getattr(o, "product_type", "")).upper()
            if "OPTION" in pt or "FUTURE" in pt:
                continue
            reviewed += 1
            s = stats[self._sleeve_of(o)]
            s.orders += 1
            status = str(getattr(o, "status", "")).upper()
            if status == "FILLED":
                s.filled += 1
            elif status == "DRY_RUN":
                s.dry_run += 1
            elif status == "REJECTED":
                s.rejected += 1
                continue
            px = float(getattr(o, "fill_price", 0.0) or getattr(o, "limit_price", 0.0) or getattr(o, "arrival_price", 0.0) or 0.0)
            qty = int(getattr(o, "fill_quantity", 0) or getattr(o, "quantity", 0) or 0)
            notional = abs(qty) * px
            s.notional += notional
            if o.ticker not in s.tickers:
                s.tickers.append(o.ticker)
            n = s.filled + s.dry_run
            slip = float(getattr(o, "slippage_bps", 0.0) or 0.0)
            s.avg_slippage_bps += (slip - s.avg_slippage_bps) / max(1, n)
            s.avg_shortfall_usd += (float(getattr(o, "implementation_shortfall", 0.0) or 0.0) - s.avg_shortfall_usd) / max(1, n)
            close = close_prices.get(o.ticker)
            if close and px:
                sign = 1.0 if str(getattr(o, "side", "BUY")).upper() in ("BUY", "COVER") else -1.0
                drift = sign * (close / px - 1.0) * 1e4
                s.fill_to_close_bps += (drift - s.fill_to_close_bps) / max(1, n)

        # ---- 2. deployment + P&L by sleeve from positions -----------------
        for sym, p in positions.items():
            sector = str(getattr(p, "sector", "") or (p.get("sector") if isinstance(p, dict) else "")).upper()
            if sector == "OPTIONS" or " " in sym:
                continue
            mv = float(getattr(p, "market_value", 0.0) or (p.get("market_value", 0.0) if isinstance(p, dict) else 0.0))
            pnl = float(getattr(p, "unrealized_pnl", 0.0) or (p.get("unrealized_pnl", 0.0) if isinstance(p, dict) else 0.0))
            sleeve = ETF_SLEEVE_HINTS.get(sym.upper()) or (sector if sector in EQUITY_SLEEVES else "IG_EQUITY")
            stats[sleeve].deployed_pct += mv / nav
            stats[sleeve].day_pnl += pnl
        for s in stats.values():
            base = s.deployed_pct * nav
            s.day_return_pct = (s.day_pnl / base) if base > 0 else 0.0

        dd_level = str((drawdown or {}).get("level", "OK")).upper()
        dd_val = float((drawdown or {}).get("drawdown", 0.0) or 0.0)

        # ---- 3. council votes ---------------------------------------------
        votes: List[CouncilVote] = []
        zero = {s: 0.0 for s in EQUITY_SLEEVES}

        # EXECUTION member: penalise sleeves with bad slippage / adverse drift, reward clean fills
        t, r = dict(zero), []
        for s in stats.values():
            if s.orders == 0:
                continue
            q = -s.avg_slippage_bps / 100.0 + s.fill_to_close_bps / 200.0    # bps → pts
            q = max(-0.02, min(0.02, q * 0.01))
            t[s.sleeve] = q
            r.append(f"{s.sleeve}: slippage {s.avg_slippage_bps:+.1f}bps, fill→close {s.fill_to_close_bps:+.1f}bps → {q:+.2%}")
        votes.append(CouncilVote("EXECUTION", t, 0.7 if reviewed else 0.2, r or ["no equity/ETF orders today — neutral"]))

        # RISK member
        t, r = dict(zero), []
        if dd_level == "ROTATE_OR_CLOSE":
            for s in EQUITY_SLEEVES:
                t[s] = -MAX_TILT if s in ("DISTRESSED_EQUITY", "HY_EQUITY", "EVENT_DRIVEN_CVR") else -0.01
            r.append(f"drawdown {dd_val:.1%} ≥ 20% — de-risk: cut Distressed/HY/CVR, no new adds until rotation completes")
        elif dd_level == "WARN":
            t["DISTRESSED_EQUITY"] = -0.02; t["HY_EQUITY"] = -0.01; t["TLTW_CASHFLOW"] = +0.02; t["FI_MACRO"] = +0.01
            r.append(f"drawdown {dd_val:.1%} in warn band — tilt toward cashflow / FI")
        else:
            losers = [s for s in stats.values() if s.day_pnl < 0 and s.deployed_pct > 0.02]
            for s in losers:
                t[s.sleeve] = -0.01
            r.append("drawdown OK; trim sleeves with negative day P&L: " + (", ".join(s.sleeve for s in losers) or "none"))
        votes.append(CouncilVote("RISK", t, 0.8, r))

        # MOMENTUM member: day return by sleeve + external momentum scores
        t, r = dict(zero), []
        mom = momentum or {}
        for s in stats.values():
            score = s.day_return_pct * 2.0
            names = [mom[x] for x in s.tickers if x in mom]
            if names:
                score += sum(names) / len(names) * 0.02
            score = max(-0.02, min(0.02, score))
            if abs(score) >= 0.0025:
                t[s.sleeve] = score
                r.append(f"{s.sleeve}: day ret {s.day_return_pct:+.2%}" + (f", momentum {sum(names)/len(names):+.2f}" if names else "") + f" → {score:+.2%}")
        votes.append(CouncilVote("MOMENTUM", t, 0.6, r or ["flat — no momentum tilt"]))

        # MACRO member
        t, r = dict(zero), []
        lvl = str((macro_flag or {}).get("level", "NONE")).upper()
        if lvl in ("IMPORTANT", "CRITICAL"):
            k = 1.0 if lvl == "IMPORTANT" else 2.0
            t["IG_EQUITY"] = -0.01 * k; t["DISTRESSED_EQUITY"] = -0.01 * k; t["HY_EQUITY"] = -0.005 * k
            t["FI_MACRO"] = +0.01 * k; t["TLTW_CASHFLOW"] = +0.015 * k
            r.append(f"macro flag {lvl}: " + "; ".join((macro_flag or {}).get("triggers", [])[:4]))
        if vix is not None and vix >= 22:
            t["TLTW_CASHFLOW"] += 0.01; t["DISTRESSED_EQUITY"] -= 0.01
            r.append(f"VIX {vix:.1f} ≥ 22 — favour covered-call cashflow over distressed beta")
        if regime:
            r.append(f"regime {regime}")
        votes.append(CouncilVote("MACRO", t, 0.65, r or ["no macro-moving event flagged — neutral"]))

        # DISCIPLINE member: pull toward allocation file targets
        t, r = dict(zero), []
        for s in stats.values():
            gap = s.target_pct - s.deployed_pct
            if abs(gap) > 0.02:
                t[s.sleeve] = max(-MAX_TILT, min(MAX_TILT, gap * 0.5))
                r.append(f"{s.sleeve}: deployed {s.deployed_pct:.1%} vs target {s.target_pct:.1%} → pull {t[s.sleeve]:+.2%}")
        votes.append(CouncilVote("DISCIPLINE", t, 0.9, r or ["deployment within 2 pts of the allocation file on every sleeve"]))

        # ---- 4. aggregate: confidence-weighted tilts, clip ±5, renormalise ---
        wsum = sum(v.confidence for v in votes) or 1.0
        agg = {s: sum(v.tilts.get(s, 0.0) * v.confidence for v in votes) / wsum for s in EQUITY_SLEEVES}
        next_alloc = {}
        for s, tgt in targets.items():
            tilt = max(-MAX_TILT, min(MAX_TILT, agg[s]))
            if dd_level == "ROTATE_OR_CLOSE":
                tilt = min(tilt, 0.0)                     # only de-risk when in 20% DD
            next_alloc[s] = max(0.0, tgt + tilt)
        equity_budget = 1.0 - sum(HARD_SLEEVES.values()) - (self.rules.options_notional_pct * 0.0)  # options are notional overlay, not cash
        equity_budget = 0.90                              # sleeves sum to 90% in the allocation file
        tot = sum(next_alloc.values()) or 1.0
        if dd_level != "ROTATE_OR_CLOSE":
            next_alloc = {s: round(v / tot * equity_budget, 4) for s, v in next_alloc.items()}
        else:
            next_alloc = {s: round(v, 4) for s, v in next_alloc.items()}   # freed capital parks in MM
        next_alloc.update(HARD_SLEEVES)
        if dd_level == "ROTATE_OR_CLOSE":
            next_alloc["MONEY_MARKET"] = round(1.0 - sum(v for k, v in next_alloc.items() if k != "MONEY_MARKET"), 4)
        else:
            # absorb rounding residue in the largest soft sleeve so the book sums to exactly 1.0
            soft = [k for k in next_alloc if k not in HARD_SLEEVES]
            if soft:
                big = max(soft, key=lambda k: next_alloc[k])
                next_alloc[big] = round(next_alloc[big] + (1.0 - sum(next_alloc.values())), 4)

        # ---- 5. grade + per-account projection ----------------------------
        slips = [s.avg_slippage_bps for s in stats.values() if s.orders]
        avg_slip = sum(slips) / len(slips) if slips else 0.0
        grade = "A" if avg_slip <= 3 else "B" if avg_slip <= 8 else "C" if avg_slip <= 15 else "D"
        if reviewed == 0:
            grade = "n/a"
        per_account: Dict[str, Dict[str, float]] = {}
        if account_router is not None and hasattr(account_router, "mandates"):
            try:
                from .account_mandates import scale_sleeves_for_mandate
                class _R:  # shadow rules with next-day sleeves
                    pass
                shadow = _R()
                for k, v in vars(self.rules).items():
                    setattr(shadow, k, v)
                shadow.ig_equity_pct = next_alloc["IG_EQUITY"]; shadow.hy_equity_pct = next_alloc["HY_EQUITY"]
                shadow.distressed_equity_pct = next_alloc["DISTRESSED_EQUITY"]; shadow.tltw_cashflow_pct = next_alloc["TLTW_CASHFLOW"]
                shadow.fi_macro_pct = next_alloc["FI_MACRO"]; shadow.event_driven_cvr_pct = next_alloc["EVENT_DRIVEN_CVR"]
                for label, m in account_router.mandates.items():
                    if m.allow_equities or m.allow_etfs:
                        caps = scale_sleeves_for_mandate(shadow, m)
                        per_account[label] = {k: v for k, v in caps.items() if k in EQUITY_SLEEVES}
            except Exception as exc:  # noqa: BLE001
                logger.debug("per-account projection failed: %s", exc)

        summary = [
            f"Execution: {reviewed} equity/ETF orders, avg slippage {avg_slip:+.1f} bps → grade {grade}.",
            "Biggest tilt: " + (max(((s, next_alloc[s] - targets[s]) for s in EQUITY_SLEEVES), key=lambda x: abs(x[1]))[0]
                                if EQUITY_SLEEVES else "none")
            + f" ({max((next_alloc[s] - targets[s] for s in EQUITY_SLEEVES), key=abs):+.2%}).",
            "Hard rules preserved: Margin 8%, Money-market 2% floor; options overlay untouched (not in council scope).",
        ]
        if dd_level == "ROTATE_OR_CLOSE":
            summary.insert(0, f"DRAWDOWN {dd_val:.1%} ≥ 20%: council in de-risk-only mode — rotate/close before any add tomorrow.")

        verdict = CouncilVerdict(
            as_of=datetime.now().isoformat(), nav=nav, equity_orders_reviewed=reviewed,
            sleeves={s: asdict(v) for s, v in stats.items()}, votes=[asdict(v) for v in votes],
            next_day_allocation=next_alloc, current_targets=targets, drawdown_level=dd_level,
            execution_grade=grade, summary=summary, per_account=per_account,
        )
        self._persist(verdict)
        return verdict

    def _persist(self, verdict: CouncilVerdict):
        if self.log_dir is None:
            return
        try:
            stem = self.log_dir / f"eod_{verdict.as_of[:10]}"
            stem.with_suffix(".json").write_text(json.dumps(verdict.to_dict(), indent=2, default=str))
            stem.with_suffix(".md").write_text(verdict.markdown())
        except Exception as exc:  # noqa: BLE001
            logger.debug("council persist failed: %s", exc)

    @staticmethod
    def apply_to_rules(rules: Any, verdict: CouncilVerdict) -> Any:
        """Write the council's next-day sleeves back onto AllocationRules (equity sleeves only)."""
        a = verdict.next_day_allocation
        rules.ig_equity_pct = a["IG_EQUITY"]; rules.hy_equity_pct = a["HY_EQUITY"]
        rules.distressed_equity_pct = a["DISTRESSED_EQUITY"]; rules.tltw_cashflow_pct = a["TLTW_CASHFLOW"]
        rules.fi_macro_pct = a["FI_MACRO"]; rules.event_driven_cvr_pct = a["EVENT_DRIVEN_CVR"]
        rules.margin_pct = HARD_SLEEVES["MARGIN"]
        rules.money_market_pct = max(HARD_SLEEVES["MONEY_MARKET"], a.get("MONEY_MARKET", HARD_SLEEVES["MONEY_MARKET"]))
        return rules
