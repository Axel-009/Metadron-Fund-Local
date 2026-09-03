"""
MacroEventFlagger — hourly "is something macro-moving happening?" flag.

Runs on the live-loop hourly cadence (and on demand). Uses Schwab quotes for a
macro basket + whatever the engine already knows (macro regime snapshot,
News+Miro output, options-engine market context) to produce a single flag:

    NONE       nothing unusual
    WATCH      one soft trigger (e.g. VIX +8%, TLT ±0.8%)
    IMPORTANT  ≥2 soft triggers or one hard trigger (SPY ±1.5%, VIX +15%, HYG ±1%)
    CRITICAL   ≥2 hard triggers or VIX +25% / SPY ±2.5% (kill-switch adjacent)

Effect on the allocation / execution path (consumed by LiveLoopOrchestrator):
    WATCH      → note in patch report, no sizing change
    IMPORTANT  → options adds scaled ×0.5, equities adds ×0.75, rotation review forced on next 30-min full scan
    CRITICAL   → no new options adds this hour, equities adds ×0.5, drawdown guard re-checked on every order

Known scheduled macro events are matched by keyword when a calendar feed or
News+Miro headline mentions them (FOMC, CPI, PCE, NFP/payrolls, GDP, PPI, ISM,
retail sales, Treasury refunding, Fed speak, OPEX, quad witching).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MACRO_BASKET = {
    "SPY": ("equities", 0.010, 0.015, 0.025),     # (label, soft, hard, critical) abs % move
    "QQQ": ("tech", 0.012, 0.018, 0.030),
    "IWM": ("small caps", 0.012, 0.018, 0.030),
    "^VIX": ("vol", 0.08, 0.15, 0.25),           # relative change in VIX level (up only)
    "TLT": ("long bonds", 0.008, 0.012, 0.020),
    "HYG": ("high yield credit", 0.005, 0.010, 0.015),
    "UUP": ("dollar", 0.005, 0.008, 0.012),
    "GLD": ("gold", 0.010, 0.015, 0.025),
    "USO": ("oil", 0.020, 0.035, 0.050),
}
VIX_SYMBOLS = ("$VIX", "VIX", "^VIX", "$VIX.X")

EVENT_KEYWORDS = {
    "FOMC": ["fomc", "fed decision", "rate decision", "powell", "fed funds"],
    "CPI": ["cpi", "consumer price", "inflation print"],
    "PCE": ["pce", "core pce"],
    "NFP": ["nonfarm", "non-farm", "payrolls", "jobs report", "unemployment rate"],
    "GDP": ["gdp"],
    "PPI": ["ppi", "producer price"],
    "ISM": ["ism manufacturing", "ism services", "pmi"],
    "RETAIL_SALES": ["retail sales"],
    "TREASURY": ["treasury auction", "refunding", "10-year auction", "30-year auction"],
    "FED_SPEAK": ["fed governor", "fed president", "waller", "williams", "jefferson"],
    "OPEX": ["opex", "options expiration", "quad witching", "triple witching"],
    "GEOPOLITICAL": ["tariff", "sanction", "strike on", "missile", "ceasefire", "invasion"],
    "EARNINGS_MEGA": ["nvidia earnings", "apple earnings", "microsoft earnings", "amazon earnings", "tesla earnings"],
}


@dataclass
class MacroEventFlag:
    as_of: str
    level: str = "NONE"                       # NONE | WATCH | IMPORTANT | CRITICAL
    score: int = 0                            # soft=1, hard=2, critical=3 (summed)
    triggers: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    basket: Dict[str, float] = field(default_factory=dict)   # symbol → % change used
    regime: str = ""
    options_add_scale: float = 1.0
    equities_add_scale: float = 1.0
    force_rotation_review: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def markdown(self) -> str:
        head = f"**Macro event flag: {self.level}** (score {self.score}) — {self.as_of[:16]}"
        lines = [head]
        if self.events:
            lines.append("- Scheduled/macro events detected: " + ", ".join(self.events))
        for t in self.triggers:
            lines.append(f"- {t}")
        if self.level != "NONE":
            lines.append(f"- Sizing response: options adds ×{self.options_add_scale:.2f}, equities adds ×{self.equities_add_scale:.2f}"
                         + (", rotation review forced on next full scan" if self.force_rotation_review else ""))
        for n in self.notes:
            lines.append(f"- {n}")
        return "\n".join(lines)


class MacroEventFlagger:
    """Hourly macro-moving-event detector. ``broker`` must expose ``get_quotes``."""

    def __init__(self, broker: Any = None, basket: Optional[Dict[str, tuple]] = None):
        self.broker = broker
        self.basket = basket or MACRO_BASKET
        self.history: List[MacroEventFlag] = []

    # ------------------------------------------------------------------
    def _pct_change(self, q: dict) -> Optional[float]:
        """Schwab quote → intraday % change (last vs previous close)."""
        if not q:
            return None
        for k in ("netPercentChange", "net_pct_change", "pct_change"):
            v = q.get(k)
            if v is not None:
                v = float(v)
                return v / 100.0 if abs(v) > 1.0 else v
        last = q.get("last") or q.get("lastPrice") or q.get("mark") or q.get("price")
        prev = q.get("closePrice") or q.get("prev_close") or q.get("previousClose")
        if last and prev:
            return float(last) / float(prev) - 1.0
        return None

    def _fetch_basket(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.broker is None or not hasattr(self.broker, "get_quotes"):
            return out
        symbols = [s for s in self.basket if s != "^VIX"] + list(VIX_SYMBOLS[:2])
        try:
            quotes = self.broker.get_quotes(symbols) or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("macro basket quotes failed: %s", exc)
            return out
        for sym, q in quotes.items():
            pc = self._pct_change(q if isinstance(q, dict) else {})
            if pc is None:
                continue
            key = "^VIX" if sym.upper().strip("$^").replace(".X", "") == "VIX" else sym.upper()
            out[key] = pc
        return out

    @staticmethod
    def _scan_events(texts: List[str]) -> List[str]:
        found: List[str] = []
        for t in texts:
            tl = (t or "").lower()
            for ev, kws in EVENT_KEYWORDS.items():
                if ev not in found and any(k in tl for k in kws):
                    found.append(ev)
        return found

    # ------------------------------------------------------------------
    def evaluate(
        self,
        basket_moves: Optional[Dict[str, float]] = None,
        headlines: Optional[List[str]] = None,
        regime: str = "",
        vix_level: Optional[float] = None,
        extra_notes: Optional[List[str]] = None,
    ) -> MacroEventFlag:
        moves = basket_moves if basket_moves is not None else self._fetch_basket()
        flag = MacroEventFlag(as_of=datetime.now().isoformat(), regime=regime, basket=dict(moves))
        score = 0
        hard_hits = 0
        for sym, (label, soft, hard, crit) in self.basket.items():
            pc = moves.get(sym)
            if pc is None:
                continue
            mag = pc if sym == "^VIX" else abs(pc)     # VIX: spikes UP matter
            if mag >= crit:
                score += 3; hard_hits += 2
                flag.triggers.append(f"CRITICAL {label} ({sym}) {pc:+.2%}")
            elif mag >= hard:
                score += 2; hard_hits += 1
                flag.triggers.append(f"HARD {label} ({sym}) {pc:+.2%}")
            elif mag >= soft:
                score += 1
                flag.triggers.append(f"soft {label} ({sym}) {pc:+.2%}")
        if vix_level is not None:
            if vix_level >= 30:
                score += 2; hard_hits += 1
                flag.triggers.append(f"HARD VIX level {vix_level:.1f} ≥ 30 (stress regime)")
            elif vix_level >= 22:
                score += 1
                flag.triggers.append(f"soft VIX level {vix_level:.1f} ≥ 22")
        events = self._scan_events(headlines or [])
        flag.events = events
        if events:
            score += 1 if len(events) == 1 else 2
            if any(e in ("FOMC", "CPI", "NFP") for e in events):
                hard_hits += 1
        if regime and regime.upper() in ("CRISIS", "STRESS", "HIGH_VOL", "RISK_OFF"):
            score += 2; hard_hits += 1
            flag.triggers.append(f"HARD macro regime {regime}")

        flag.score = score
        if hard_hits >= 2 or score >= 6:
            flag.level, flag.options_add_scale, flag.equities_add_scale, flag.force_rotation_review = "CRITICAL", 0.0, 0.5, True
        elif hard_hits >= 1 or score >= 3:
            flag.level, flag.options_add_scale, flag.equities_add_scale, flag.force_rotation_review = "IMPORTANT", 0.5, 0.75, True
        elif score >= 1:
            flag.level = "WATCH"
        flag.notes = list(extra_notes or [])
        self.history.append(flag)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        if flag.level in ("IMPORTANT", "CRITICAL"):
            logger.warning("[MacroEventFlagger] %s: %s", flag.level, "; ".join(flag.triggers + flag.events))
        else:
            logger.info("[MacroEventFlagger] %s (score %d)", flag.level, flag.score)
        return flag

    def latest(self) -> Optional[MacroEventFlag]:
        return self.history[-1] if self.history else None
