"""
Account mandates + drawdown guard — Metadron Capital (Schwab, multi-account).

Three Schwab accounts, three mandates (per Metadron allocation instructions):

    ROTH        25% options overlay (1-7 DTE) / 75% equities + ETFs
    LLC         100% equities + ETFs — options NOT permitted
    INDIVIDUAL  100% options (1-7 DTE) — equities NOT permitted

Inside each account the sleeve percentages of ``AllocationRules`` (IG 40 / HY 10 /
Dist 10 / TLTW 15 / FI 5 / CVR 10 / Margin 8 HARD / MM 2 HARD FLOOR; options
10 / 10 / 5) are applied to the part of the account that the mandate opens to
that product class. E.g. ROTH equities sleeves are scaled onto 75% of Roth NAV,
Roth options buckets onto 25% of Roth NAV; INDIVIDUAL options buckets are
scaled 10/10/5 → 40/40/20 of Individual NAV.

Drawdown rule (applies BEFORE every add, not only on monitoring ticks):
    account or portfolio drawdown from peak NAV >= 20%  →  ``ROTATE_OR_CLOSE``
    - no new risk may be added to that account
    - the guard emits a rotate/close directive (close losers, rotate to cash/MM
      or to the best-scoring sleeve) that the execution phase acts on
    - the portfolio-level 20% DD also feeds the Cube kill-switch ordering
      (portfolio DD ≥20% → Cube kill-switch → margin breach).

Configuration (env or explicit):
    SCHWAB_ACCOUNT_ROTH=0514
    SCHWAB_ACCOUNT_LLC=9565
    SCHWAB_ACCOUNT_INDIVIDUAL=4806
    SCHWAB_ACCOUNT_MANDATES='{"0514":"ROTH","9565":"LLC","4806":"INDIVIDUAL"}'  (alternative)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DRAWDOWN_ROTATE_PCT = 0.20          # 20% from peak → rotate or close, block adds
DRAWDOWN_WARN_PCT = 0.15            # 15% → warn, halve new adds

ETF_HINTS = {"SPY", "QQQ", "IWM", "DIA", "TLT", "TLTW", "HYG", "LQD", "JEPI", "JEPQ", "XYLD", "QYLD",
             "GLD", "SLV", "USO", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
             "VTI", "VOO", "IVV", "EFA", "EEM", "AGG", "BND", "SHY", "IEF", "VNQ", "ARKK", "SMH", "SOXX"}


@dataclass
class AccountMandate:
    label: str                         # ROTH | LLC | INDIVIDUAL
    account_last4: str = ""
    options_pct: float = 0.0           # share of account NAV that may be options notional
    equities_pct: float = 0.0          # share of account NAV that may be equities + ETFs
    allow_options: bool = False
    allow_equities: bool = False
    allow_etfs: bool = False
    drawdown_rotate_pct: float = DRAWDOWN_ROTATE_PCT
    priority_options: int = 99         # lower = preferred destination for option orders
    priority_equities: int = 99        # lower = preferred destination for equity/ETF orders

    def to_dict(self) -> dict:
        return asdict(self)

    def permits(self, product: str, ticker: str = "") -> bool:
        p = (product or "").upper()
        if p == "OPTION":
            return self.allow_options
        if p in ("EQUITY", "ETF", "STOCK"):
            is_etf = p == "ETF" or ticker.upper() in ETF_HINTS
            return self.allow_etfs if is_etf else self.allow_equities
        return False


DEFAULT_MANDATES: Dict[str, AccountMandate] = {
    "ROTH": AccountMandate(
        label="ROTH", options_pct=0.25, equities_pct=0.75,
        allow_options=True, allow_equities=True, allow_etfs=True,
        priority_options=2, priority_equities=2,
    ),
    "LLC": AccountMandate(
        label="LLC", options_pct=0.0, equities_pct=1.0,
        allow_options=False, allow_equities=True, allow_etfs=True,
        priority_options=99, priority_equities=1,
    ),
    "INDIVIDUAL": AccountMandate(
        label="INDIVIDUAL", options_pct=1.0, equities_pct=0.0,
        allow_options=True, allow_equities=False, allow_etfs=False,
        priority_options=1, priority_equities=99,
    ),
}


def load_mandates(env: Optional[dict] = None) -> Dict[str, AccountMandate]:
    """Resolve mandate → account last-4 mapping from env. Unmapped mandates are dropped."""
    env = env if env is not None else os.environ
    out: Dict[str, AccountMandate] = {}
    raw = env.get("SCHWAB_ACCOUNT_MANDATES", "")
    mapping: Dict[str, str] = {}
    if raw:
        try:
            mapping = {str(v).upper(): str(k)[-4:] for k, v in json.loads(raw).items()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("SCHWAB_ACCOUNT_MANDATES unparsable: %s", exc)
    for label in DEFAULT_MANDATES:
        last4 = mapping.get(label) or env.get(f"SCHWAB_ACCOUNT_{label}", "")
        if last4:
            m = AccountMandate(**DEFAULT_MANDATES[label].to_dict())
            m.account_last4 = str(last4)[-4:]
            out[label] = m
    return out


def scale_sleeves_for_mandate(rules, mandate: AccountMandate) -> Dict[str, float]:
    """Project AllocationRules sleeve percentages onto one account's mandate.

    Equity sleeves (IG/HY/Dist/TLTW/FI/CVR = 90% of a full portfolio) are
    renormalised onto ``equities_pct`` of the account; options buckets (10/10/5)
    onto ``options_pct``. Margin (8%) and MM (2%) hard rules are preserved as a
    share of the account. Returns fractions of *account* NAV.
    """
    eq = {
        "IG_EQUITY": rules.ig_equity_pct, "HY_EQUITY": rules.hy_equity_pct,
        "DISTRESSED_EQUITY": rules.distressed_equity_pct, "TLTW_CASHFLOW": rules.tltw_cashflow_pct,
        "FI_MACRO": rules.fi_macro_pct, "EVENT_DRIVEN_CVR": rules.event_driven_cvr_pct,
    }
    opt = {
        "OPTIONS_IG": rules.options_ig_pct, "OPTIONS_HY": rules.options_hy_pct,
        "OPTIONS_DISTRESSED": rules.options_distressed_pct,
    }
    eq_total = sum(eq.values()) or 1.0
    opt_total = sum(opt.values()) or 1.0
    reserve = rules.margin_pct + rules.money_market_pct        # 10% hard reserve per account
    investable = max(0.0, 1.0 - reserve)
    out: Dict[str, float] = {}
    if mandate.allow_equities or mandate.allow_etfs:
        share = investable * mandate.equities_pct
        for k, v in eq.items():
            out[k] = round(share * v / eq_total, 6)
    if mandate.allow_options:
        share = investable * mandate.options_pct if mandate.equities_pct > 0 else investable * mandate.options_pct
        for k, v in opt.items():
            out[k] = round(share * v / opt_total, 6)
    out["MARGIN"] = rules.margin_pct
    out["MONEY_MARKET"] = rules.money_market_pct
    return out


@dataclass
class DrawdownStatus:
    scope: str                     # account label or "PORTFOLIO"
    nav: float
    peak_nav: float
    drawdown: float                # positive fraction, 0.12 = 12% below peak
    level: str                     # OK | WARN | ROTATE_OR_CLOSE
    adds_allowed: bool
    add_scale: float               # 1.0 normal, 0.5 warn, 0.0 blocked
    directive: str = ""
    as_of: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class DrawdownGuard:
    """Tracks peak NAV per scope and enforces the 20% rotate-or-close rule.

    Called by the execution surface BEFORE every position add and by the
    monitoring phase. Peaks persist for the process lifetime and can be seeded
    from the broker performance tracker's high-water mark.
    """

    def __init__(self, rotate_pct: float = DRAWDOWN_ROTATE_PCT, warn_pct: float = DRAWDOWN_WARN_PCT):
        self.rotate_pct = rotate_pct
        self.warn_pct = warn_pct
        self._peaks: Dict[str, float] = {}
        self._history: List[DrawdownStatus] = []

    def seed_peak(self, scope: str, peak_nav: float):
        if peak_nav and peak_nav > 0:
            self._peaks[scope] = max(self._peaks.get(scope, 0.0), float(peak_nav))

    def check(self, scope: str, nav: float) -> DrawdownStatus:
        nav = float(nav or 0.0)
        peak = max(self._peaks.get(scope, 0.0), nav)
        self._peaks[scope] = peak
        dd = (peak - nav) / peak if peak > 0 else 0.0
        if dd >= self.rotate_pct:
            level, allowed, scale = "ROTATE_OR_CLOSE", False, 0.0
            directive = (f"{scope}: drawdown {dd:.1%} >= {self.rotate_pct:.0%} — NO new risk; "
                         f"close losers / rotate to money-market or best-scoring sleeve before any add")
        elif dd >= self.warn_pct:
            level, allowed, scale = "WARN", True, 0.5
            directive = f"{scope}: drawdown {dd:.1%} — new adds halved, rotation review required"
        else:
            level, allowed, scale, directive = "OK", True, 1.0, ""
        st = DrawdownStatus(scope=scope, nav=nav, peak_nav=peak, drawdown=round(dd, 4), level=level,
                            adds_allowed=allowed, add_scale=scale, directive=directive,
                            as_of=datetime.now().isoformat())
        self._history.append(st)
        if len(self._history) > 2000:
            self._history = self._history[-2000:]
        if level != "OK":
            logger.warning("[DrawdownGuard] %s", directive)
        return st

    def rotation_plan(self, scope: str, positions: Dict[str, dict], keep_top_n: int = 3) -> List[dict]:
        """Rotate-or-close plan for a scope in ROTATE_OR_CLOSE: close the worst
        P&L positions first, keep at most ``keep_top_n`` best performers."""
        rows = []
        for sym, p in positions.items():
            pnl = float(p.get("unrealized_pnl", 0.0) or 0.0) + float(p.get("realized_pnl", 0.0) or 0.0)
            rows.append({"symbol": sym, "quantity": p.get("quantity", 0), "pnl": pnl,
                         "sector": p.get("sector", "")})
        rows.sort(key=lambda r: r["pnl"])
        plan = []
        for i, r in enumerate(rows):
            action = "CLOSE" if (r["pnl"] < 0 or i < max(0, len(rows) - keep_top_n)) else "HOLD"
            plan.append({**r, "action": action, "scope": scope})
        return plan

    def latest(self) -> Dict[str, DrawdownStatus]:
        out: Dict[str, DrawdownStatus] = {}
        for st in self._history:
            out[st.scope] = st
        return out
