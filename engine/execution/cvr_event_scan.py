"""CVR (Contingent Value Right) event scan — news-driven sourcing for the CVR sleeve.

Operator rule (Sept 2026): the CVR sleeve is EVENT-BASED — never proxy it with an ETF.
Each cycle: scan news / SEC 8-K flow for live merger agreements paying "cash per share +
one contingent value right", verify the TARGET is quotable at Schwab, and hold the target
equity (merger-arb style: collect the cash spread, receive the CVR optionality at close).
Closed / terminated deals are excluded; CVRs are usually non-transferable, so the listed
target is the only tradable instrument.

Pipeline: web search (4 query variants) → LLM extraction → PENDING + listed + cash known →
broker quote → spread = cash/price − 1 (≥ −1 %) → rank by spread + 0.5·CVR upside → top N.
Degrades gracefully: no search library / network → [] and the caller reports it.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

QUERIES = [
    "contingent value right merger agreement announced",
    "to be acquired for cash per share plus contingent value right CVR",
    "CVR contingent value right tender offer pending shareholder approval",
    "acquisition cash plus non-tradeable CVR milestone payment definitive agreement",
]

EXTRACT_PROMPT = (
    "This page concerns a merger/acquisition that includes a Contingent Value Right (CVR). "
    "Return ONLY a JSON object with keys: target (company name), ticker (US listed ticker of the "
    "TARGET, or null), acquirer, cash_per_share (number or null), cvr_max_per_share (number or null; "
    "maximum CVR payment per share), cvr_terms (one sentence), status (PENDING, CLOSED, TERMINATED or "
    "UNKNOWN — PENDING = announced but not closed), definitive (true if a DEFINITIVE merger agreement / "
    "tender offer has been signed, false for non-binding proposals, indications of interest or letters of "
    "intent), announced (YYYY-MM-DD or null), expected_close (text or null). If the page is not about a "
    "specific CVR deal return {\"target\": null}."
)


@dataclass
class CVREvent:
    ticker: str
    target: str
    acquirer: str
    cash_per_share: float
    cvr_max_per_share: float
    cvr_terms: str
    status: str
    announced: str
    expected_close: str
    source_url: str
    price: float = 0.0
    spread_pct: float = 0.0
    cvr_upside_pct: float = 0.0
    score: float = 0.0
    why: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _num(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
        return float(m.group()) if m else 0.0
    except Exception:
        return 0.0


def _parse_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


class CVREventScanner:
    def __init__(self, broker=None, max_hits: int = 24, max_names: int = 3,
                 min_spread_pct: float = -0.01, max_spread_pct: float = 0.35, recency_days: int = 150):
        self.max_spread_pct = max_spread_pct     # > this and the market says the deal is broken/at risk
        self.last_verify: Dict[str, List[Dict[str, Any]]] = {}
        self.broker = broker
        self.max_hits = max_hits
        self.max_names = max_names
        self.min_spread_pct = min_spread_pct
        self.recency_days = recency_days
        self.last_deals: List[Dict[str, Any]] = []
        self.last_notes: List[str] = []

    # The Schwab credential proxy sets HTTPS_PROXY / SSL_CERT_FILE for the engine process, which
    # breaks the search library's own TLS path. Run the news scan in a child process with a
    # scrubbed environment so both pipes work side by side.
    _SCRUB = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS")

    _WORKER = r"""
import json, sys, datetime as dt
import pplx_sdk
cfg = json.loads(sys.stdin.read())
today = dt.date.today(); months = []
for k in range(3):
    m = (today.month - 1 - k) % 12 + 1; y = today.year if today.month - k > 0 else today.year - 1
    months.append(dt.date(y, m, 1).strftime("%B %Y"))
queries = (cfg["queries"] + [f"to be acquired cash per share plus CVR {mo}" for mo in months])[:10] if cfg["queries"] else []
by_q = pplx_sdk.search.web_by_query(queries, limit_per_query=8) if queries else {}
seen, hits = set(), []
for hs in by_q.values():
    for h in hs:
        if h.url not in seen:
            seen.add(h.url); hits.append({"url": h.url, "date": str(h.date or ""), "title": h.title})
cutoff = (today - dt.timedelta(days=cfg["recency_days"])).isoformat()
dated = [h for h in hits if h["date"] and h["date"][:10] >= cutoff]; undated = [h for h in hits if not h["date"]]
hits = (dated + undated)[: cfg["max_hits"]]
deals, n_err = [], 0
if hits:
    for pg in pplx_sdk.content.fetch([h["url"] for h in hits], prompt=cfg["prompt"]):
        if pg.error: n_err += 1; continue
        deals.append({"url": pg.url, "content": pg.content or ""})
verify = {}
for name in cfg.get("verify", []):
    try:
        vh = pplx_sdk.search.web(f"{name} merger agreement status", reformulations=[f"{name} acquisition completed terminated", f"{name} tender offer results"])
        urls = [h.url for h in vh[:4]]
        vp = pplx_sdk.content.fetch(urls, prompt=cfg["prompt"]) if urls else []
        verify[name] = [{"url": pg.url, "content": pg.content or ""} for pg in vp if not pg.error]
    except Exception as exc:
        verify[name] = [{"url": "", "content": json.dumps({"target": None, "error": str(exc)})}]
print(json.dumps({"hits": len(hits), "n_err": n_err, "deals": deals, "verify": verify}))
"""

    def _fetch_deals(self, verify: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        import os, subprocess, sys
        env = {k: v for k, v in os.environ.items() if k not in self._SCRUB}
        payload = json.dumps({"queries": QUERIES if not verify else [], "recency_days": self.recency_days,
                              "max_hits": self.max_hits if not verify else 0, "prompt": EXTRACT_PROMPT, "verify": verify or []})
        try:
            r = subprocess.run([sys.executable, "-c", self._WORKER], input=payload, capture_output=True, text=True, env=env, timeout=240)
        except Exception as exc:  # noqa: BLE001
            self.last_notes.append(f"CVR news scan failed to launch: {exc}")
            return []
        if r.returncode != 0:
            self.last_notes.append(f"CVR news scan failed: {(r.stderr or '').strip().splitlines()[-1][:160] if r.stderr else 'exit ' + str(r.returncode)}")
            return []
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            self.last_notes.append("CVR news scan returned no parsable output")
            return []
        if verify:
            self.last_verify = {}
            for name, recs in (out.get("verify") or {}).items():
                self.last_verify[name] = [d for d in (_parse_json(r.get("content", "")) for r in recs) if d and d.get("target")]
            return []
        deals = []
        for rec in out.get("deals", []):
            d = _parse_json(rec.get("content", ""))
            if d and d.get("target"):
                d["source_url"] = rec["url"]; deals.append(d)
        self.last_notes.append(f"CVR news scan: {out.get('hits', 0)} hits → {len(deals)} deal records ({out.get('n_err', 0)} fetch errors)")
        return deals

    def scan(self) -> List[CVREvent]:
        self.last_notes = []
        raw = self._fetch_deals()
        by_ticker: Dict[str, Dict[str, Any]] = {}
        for d in raw:
            t = (d.get("ticker") or "").strip().upper()
            if not re.fullmatch(r"[A-Z]{1,5}", t):
                continue
            st = str(d.get("status") or "UNKNOWN").upper()
            prev = by_ticker.get(t)
            if prev is None or st in ("CLOSED", "TERMINATED") or (prev.get("status") == "UNKNOWN" and st == "PENDING"):
                by_ticker[t] = {**(prev or {}), **{k: v for k, v in d.items() if v not in (None, "", 0)}, "status": st}
        self.last_deals = list(by_ticker.values())
        events: List[CVREvent] = []
        for t, d in by_ticker.items():
            if d["status"] != "PENDING":
                self.last_notes.append(f"{t}: {d['status']} — skipped"); continue
            if d.get("definitive") is False:
                self.last_notes.append(f"{t}: non-binding proposal, no definitive agreement — skipped"); continue
            cash = _num(d.get("cash_per_share"))
            if cash <= 0:
                self.last_notes.append(f"{t}: cash consideration unknown — skipped"); continue
            price = 0.0
            if self.broker is not None:
                try:
                    price = float(self.broker.get_quote(t) or 0.0)
                except Exception:
                    price = 0.0
            if price <= 0:
                self.last_notes.append(f"{t}: no Schwab quote — skipped"); continue
            spread = cash / price - 1.0
            cvr_max = _num(d.get("cvr_max_per_share"))
            upside = cvr_max / price if cvr_max > 0 else 0.0
            if spread < self.min_spread_pct:
                self.last_notes.append(f"{t}: trading {abs(spread):.1%} through cash ${cash:.2f} (px {price:.2f}) — skipped"); continue
            if spread > self.max_spread_pct:
                self.last_notes.append(f"{t}: spread {spread:+.0%} vs cash ${cash:.2f} (px {price:.2f}) — market says deal broken/at risk, skipped"); continue
            ev = CVREvent(t, str(d.get("target") or t), str(d.get("acquirer") or ""), cash, cvr_max,
                          str(d.get("cvr_terms") or ""), "PENDING", str(d.get("announced") or ""),
                          str(d.get("expected_close") or ""), str(d.get("source_url") or ""),
                          price=price, spread_pct=spread, cvr_upside_pct=upside)
            ev.score = spread + 0.5 * upside
            ev.why = (f"{ev.acquirer or 'buyer'} pays ${cash:.2f} cash + CVR (max ${cvr_max:.2f}); px {price:.2f} → "
                      f"spread {spread:+.1%}, CVR upside {upside:.1%}; close {ev.expected_close or 'TBD'}")
            events.append(ev)
        events.sort(key=lambda e: -e.score)
        # second stage: fresh status check per candidate (an old 8-K amendment can still read PENDING)
        shortlist = events[: self.max_names + 2]
        if shortlist:
            self._fetch_deals(verify=[f"{e.target} {e.acquirer}".strip() for e in shortlist])
        kept: List[CVREvent] = []
        for e in shortlist:
            recs = self.last_verify.get(f"{e.target} {e.acquirer}".strip(), [])
            verdicts = [str(r.get("status") or "").upper() for r in recs]
            bad = [v for v in verdicts if v in ("CLOSED", "TERMINATED")]
            if bad:
                self.last_notes.append(f"{e.ticker}: verification says {bad[0]} ({len(bad)}/{len(verdicts)} sources) — dropped"); continue
            if any(r.get("definitive") is False for r in recs) and not any(r.get("definitive") is True for r in recs):
                self.last_notes.append(f"{e.ticker}: verification finds no definitive agreement — dropped"); continue
            e.why += f" · verified PENDING ({len(verdicts)} sources)" if verdicts else " · unverified"
            kept.append(e)
        return kept[: self.max_names]


def scan_cvr_events(broker=None, max_names: int = 3) -> List[CVREvent]:
    return CVREventScanner(broker, max_names=max_names).scan()
