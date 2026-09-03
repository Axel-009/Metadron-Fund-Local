"""
Universe Tranche Scan — SEPARATE scans per universe run, then a concurrence vote.
================================================================================

The full universe is NEVER scanned in one pass. Following the allocation guide
(engine/allocation/universe_scan.py UNIVERSE_ORDER) each 30-minute full scan runs
the universe runs independently, reports each on its own, and only then concurs:

    Run 1  →  SP500     S&P 500 large cap          (IG sleeve heavy)
    Run 2  →  SP400     S&P MidCap 400 + extras    (HY / IG mix)
    Run 3  →  SP600     S&P SmallCap 600           (HY / distressed tilt)
    Run 4  →  ETF_FI    ETF + fixed income         (TLTW / FI_MACRO sleeves)
    ─────────────────────────────────────────────────────────────────────
    Concur  →  z-score each tranche on its own distribution, then vote:
               momentum/RSI + relative strength vs SPY + beta-corridor
               direction + 20-bar breakout + liquidity, subject to the
               allocation-file sleeve caps, the L7 G2 30 % sector cap and a
               minimum representation per tranche.

Data source is Schwab only (batched quotes for the screen, daily candles for
the shortlist) so a full multi-tranche pass fits the Schwab rate budget.
Every run also reports BUY / SELL / HOLD counts, avg α, top-5 BUY and top-3 SELL
signals so the gold-standard report (VIEW 1 / VIEW 2) can be rendered per run.

The output feeds: (a) the equity slate for L7, (b) the ShortDTE options
universe (top names per options bucket), (c) the in-chat run patch.
"""
from __future__ import annotations

import logging
import math
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Callable, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("metadron.tranche_scan")

try:
    from engine.data.cross_asset_universe import (
        SP500_TICKERS, SP400_TICKERS, SP600_TICKERS, EXTRA_TICKERS, get_sector_for_ticker,
    )
except Exception:  # noqa: BLE001 — never hard-fail the engine on a data import
    SP500_TICKERS, SP400_TICKERS, SP600_TICKERS, EXTRA_TICKERS = [], [], [], []

    def get_sector_for_ticker(t: str) -> str:  # type: ignore[misc]
        return "Unknown"

try:
    from engine.allocation.universe_scan import ETF_TICKERS, FI_TICKERS
except Exception:  # noqa: BLE001
    ETF_TICKERS = ["TLTW", "QQQ", "SPY", "IWM", "HYG", "LQD", "TLT", "GLD", "XLE", "XLF", "XLK", "XLV", "DIA", "JEPI", "JEPQ"]
    FI_TICKERS = ["TLT", "IEF", "SHY", "HYG", "LQD", "EMB", "AGG", "BND", "TIP", "MBB"]

try:
    from engine.execution.short_dte_options_engine import ShortDTEOptionsEngine
except Exception:  # noqa: BLE001
    ShortDTEOptionsEngine = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tranche definitions
# ---------------------------------------------------------------------------
ETF_HINTS = {"SPY", "QQQ", "IWM", "DIA", "MDY", "VTI", "RSP", "TLT", "TLTW", "HYG", "LQD", "JNK", "XLE", "XLF",
             "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "GLD", "USO", "UUP", "SLV", "EEM", "EFA"}


RUN_LABELS = {"SCAN_1_SP500": "SP500", "SCAN_2_SP400": "SP400", "SCAN_3_SP600": "SP600", "SCAN_4_ETF_FI": "ETF_FI"}
BOND_ETFS = {"TLT", "IEF", "SHY", "LQD", "AGG", "BND", "MBB", "VMBS", "TIP", "TIPS", "MUB", "EMB", "BKLN", "VCIT", "VCSH",
             "GOVT", "FLOT", "SCHO", "SCHR", "IGIB", "IGSB", "USIG", "STIP", "SCHP", "BNDX", "IAGG"}
INCOME_ETFS = {"TLTW", "JEPI", "JEPQ", "DIVO", "XYLD", "QYLD", "RYLD", "SCHD", "DVY", "HDV", "VIG"}


def default_tranches() -> List[Tuple[str, List[str]]]:
    """Universe runs in allocation-guide order: SP500 → SP400 (+extras) → SP600 → ETF_FI. No overlap."""
    etf_fi = sorted(set(ETF_TICKERS) | set(FI_TICKERS))
    sp500 = sorted(set(SP500_TICKERS) - set(etf_fi))
    sp400 = sorted((set(SP400_TICKERS) | set(EXTRA_TICKERS)) - set(sp500) - set(etf_fi))
    sp600 = sorted(set(SP600_TICKERS) - set(sp500) - set(sp400) - set(etf_fi))
    return [("SCAN_1_SP500", sp500), ("SCAN_2_SP400", sp400), ("SCAN_3_SP600", sp600), ("SCAN_4_ETF_FI", etf_fi)]


def run_label(tranche_name: str) -> str:
    return RUN_LABELS.get(tranche_name, tranche_name.split("_", 2)[-1])


@dataclass
class TrancheConfig:
    cvr_max_names: int = 3               # CVR sleeve: max live cash+CVR deal targets held
    cvr_event_scan: bool = os.environ.get("METADRON_CVR_SCAN", "1") != "0"   # news/8-K scan each concurrence
    shortlist_per_tranche: int = 40      # names that get daily candles after the quote screen
    history_days: int = 90
    min_price: float = 5.0
    min_dollar_volume: float = 2_000_000.0
    top_per_tranche: int = 12            # names each tranche forwards to the concurrence vote
    final_max_names: int = 20
    sector_cap: float = 0.30             # L7 G2
    min_per_tranche: int = 2             # every tranche keeps a voice in the final selection
    sleeve_targets: Dict[str, float] = field(default_factory=lambda: {
        "IG_EQUITY": 0.40, "HY_EQUITY": 0.10, "DISTRESSED": 0.10, "TLTW": 0.15, "FIXED_INCOME": 0.05, "CVR": 0.10})


@dataclass
class Candidate:
    ticker: str
    tranche: str
    sector: str
    price: float
    dollar_volume: float
    pct_change: float
    pos_52w: float                        # 0 = at 52w low, 1 = at 52w high
    rsi: float = 50.0
    breakout: str = ""
    mom_5d: float = 0.0
    mom_21d: float = 0.0
    mom_63d: float = 0.0
    rel_strength_63d: float = 0.0         # vs SPY
    realized_vol: float = 0.0
    momentum_score: float = 0.0           # [-1, 1] from ShortDTE momentum_read
    raw_score: float = 0.0                # α — composite alpha score (signed)
    z_score: float = 0.0
    mom_10d: float = 0.0
    sharpe: float = 0.0                   # 21d annualised return / realised vol
    ensemble: float = 0.0                 # 0..1 agreement of the sub-signals
    signal: str = "HOLD"                  # BUY / SELL / HOLD
    fully_scored: bool = False            # True once candles were pulled (shortlist)
    direction: str = "LONG"
    sleeve: str = "HY_EQUITY"
    options_bucket: str = "OPTIONS_HY"
    notes: List[str] = field(default_factory=list)
    votes: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def score(self) -> float:
        """Confidence-style score used by the slate and the gold-standard report."""
        return round(min(1.0, 0.5 + abs(float(self.z_score)) / 4), 4)

    def why(self) -> str:
        """One-line WHY string in the gold-standard style."""
        bits: List[str] = []
        m10 = self.mom_10d
        if abs(m10) >= 0.05:
            bits.append(f"{'strong' if m10 > 0 else 'weak'} 10d mom ({m10:+.1%})")
        elif m10 != 0:
            bits.append(f"{'positive' if m10 > 0 else 'negative'} 10d mom ({m10:+.1%})")
        if self.sharpe >= 2.0:
            bits.append(f"high Sharpe ({self.sharpe:.2f})")
        elif self.sharpe <= -2.0:
            bits.append(f"negative Sharpe ({self.sharpe:.2f})")
        if self.breakout:
            bits.append(f"RSI {self.rsi:.0f} {self.breakout.lower().replace('_', ' ')}")
        elif self.rsi >= 70 or self.rsi <= 30:
            bits.append(f"RSI {self.rsi:.0f} {'overbought' if self.rsi >= 70 else 'oversold'}")
        if self.realized_vol >= 0.45:
            bits.append(f"high vol ({self.realized_vol:.0%})")
        elif self.realized_vol and self.realized_vol <= 0.25:
            bits.append("low vol")
        if self.ensemble >= 0.60:
            bits.append(f"strong ensemble ({self.ensemble:.3f})")
        if abs(self.rel_strength_63d) >= 0.05:
            bits.append(f"RS vs SPY {self.rel_strength_63d:+.1%}")
        return ", ".join(bits[:4]) if bits else "quote-screen only (not shortlisted for candles)"


@dataclass
class TrancheResult:
    name: str
    universe_size: int
    quoted: int
    screened: int
    shortlisted: int
    top: List[Candidate]
    rejected: Dict[str, int]
    elapsed_s: float
    buy_n: int = 0
    sell_n: int = 0
    hold_n: int = 0
    avg_alpha: float = 0.0
    sells: List[Candidate] = field(default_factory=list)   # strongest SELL / EXIT signals
    candidates: List[Candidate] = field(default_factory=list)  # every fully-scored name (for locked-sleeve backfill)

    @property
    def label(self) -> str:
        return run_label(self.name)

    def to_dict(self) -> dict:
        d = asdict(self); d["top"] = [c.to_dict() for c in self.top]; d["sells"] = [c.to_dict() for c in self.sells]
        d["label"] = self.label; return d


@dataclass
class ConcurrenceResult:
    tranches: List[TrancheResult]
    final: List[Candidate]
    dropped: List[Tuple[str, str]]         # (ticker, reason)
    corridor_bias: float
    spy_mom_63d: float
    as_of: str

    def to_dict(self) -> dict:
        return {"as_of": self.as_of, "corridor_bias": self.corridor_bias, "spy_mom_63d": self.spy_mom_63d,
                "tranches": [t.to_dict() for t in self.tranches], "final": [c.to_dict() for c in self.final],
                "dropped": self.dropped}

    LOCKED_SLEEVES = ("IG_EQUITY", "HY_EQUITY", "DISTRESSED", "TLTW", "FIXED_INCOME", "CVR")
    # last-resort sleeve placeholders when NO run produced a candidate for a locked sleeve
    SLEEVE_FALLBACK = {"TLTW": "TLTW", "FIXED_INCOME": "TLT", "DISTRESSED": "HYG", "HY_EQUITY": "IWM", "IG_EQUITY": "SPY"}   # CVR: event-based, no proxy

    cvr_events: List[Any] = field(default_factory=list)   # CVREvent rows from the news scan
    cvr_notes: List[str] = field(default_factory=list)

    def lock_cvr_from_events(self, events: List[Any], notes: List[str]) -> bool:
        """CVR sleeve is EVENT-BASED (operator rule): hold the listed TARGET of live merger
        deals that pay cash + a Contingent Value Right (news / 8-K scan). Returns True when
        at least one event name was placed, so lock_sleeves() must not use the placeholder."""
        self.cvr_events, self.cvr_notes = list(events), list(notes)
        placed = False
        taken = {c.ticker for c in self.final}
        for ev in events:
            if ev.ticker in taken:
                continue
            c = Candidate(ticker=ev.ticker, tranche="CVR_EVENT", sector="Event-Driven", price=float(ev.price),
                          dollar_volume=0.0, pct_change=0.0, pos_52w=0.5)
            c.sleeve, c.direction, c.signal, c.fully_scored = "CVR", "LONG", "BUY", True
            c.raw_score = float(ev.score); c.z_score = float(max(0.5, min(3.0, 10.0 * ev.score)))
            c.notes = [f"CVR event: {ev.why}", f"source {ev.source_url}"]
            self.final.append(c); taken.add(ev.ticker); placed = True
        return placed

    def lock_sleeves(self) -> List[str]:
        """LOCKED ALLOCATION (operator rule): every equity sleeve MUST receive at least one name
        from the universe runs. Missing sleeves are back-filled with the best LONG candidate for
        that sleeve across all runs (ranked by composite z), else the sleeve placeholder ETF.
        Returns the list of back-fill notes."""
        have = {c.sleeve for c in self.final}
        taken = {c.ticker for c in self.final}
        notes = []
        for sleeve in self.LOCKED_SLEEVES:
            if sleeve in have:
                continue
            pool = [c for tr in self.tranches for c in tr.candidates
                    if c.sleeve == sleeve and c.direction == "LONG" and c.ticker not in taken]
            pool.sort(key=lambda c: -c.z_score)
            if pool:
                c = pool[0]; c.notes = list(c.notes) + [f"locked-sleeve backfill {sleeve}"]
                self.final.append(c); taken.add(c.ticker)
                notes.append(f"{sleeve}: back-filled with {c.ticker} (best {sleeve} candidate, z={c.z_score:+.2f}, {c.tranche})")
            elif sleeve == "CVR":
                notes.append("CVR: no live cash+CVR merger event passed the news scan → sleeve left UNFILLED this cycle "
                             "(event-based sleeve, no ETF proxy)" + (f"; scan notes: {'; '.join(self.cvr_notes[:4])}" if self.cvr_notes else ""))
            else:
                t = self.SLEEVE_FALLBACK[sleeve]
                c = Candidate(ticker=t, tranche="LOCKED_SLEEVE", sector="ETF", price=0.0, dollar_volume=0.0, pct_change=0.0, pos_52w=0.5)
                c.sleeve, c.direction, c.z_score, c.raw_score, c.signal = sleeve, "LONG", 0.0, 0.0, "BUY"
                c.notes = [f"locked-sleeve placeholder {sleeve} — no scan candidate this cycle"]
                self.final.append(c); taken.add(t)
                notes.append(f"{sleeve}: no candidate in any run → placeholder {t}")
        self.backfill_notes = notes
        return notes

    def equity_slate(self) -> List[dict]:
        """Approved-trade-shaped rows for Phase 5 / L7 (after lock_sleeves → every sleeve present)."""
        if not getattr(self, "backfill_notes", None) and {c.sleeve for c in self.final} < set(self.LOCKED_SLEEVES):
            self.lock_sleeves()
        return [{"ticker": c.ticker, "signal": None, "side": "BUY" if c.direction == "LONG" else "SELL",
                 "decision": {"source": "TRANCHE_CONCURRENCE", "bucket": c.sleeve, "type": "EQUITY"},
                 "bucket": c.sleeve, "instrument_type": "ETF" if c.ticker in ETF_HINTS else "EQUITY",
                 "confidence": round(min(1.0, 0.5 + abs(float(c.z_score)) / 4), 3), "alpha_score": round(float(c.raw_score), 4),
                 "tranche": c.tranche, "sector": c.sector, "reason": "; ".join(c.notes[:3])} for c in self.final]

    # liquid underlyings that ALWAYS carry 1–7 DTE chains (index / sector / macro ETFs)
    OPTIONS_CORE = (("SPY", "OPTIONS_IG"), ("QQQ", "OPTIONS_IG"), ("IWM", "OPTIONS_HY"), ("XLE", "OPTIONS_IG"),
                    ("XLF", "OPTIONS_IG"), ("XLV", "OPTIONS_IG"), ("GLD", "OPTIONS_IG"), ("TLT", "OPTIONS_IG"),
                    ("HYG", "OPTIONS_DISTRESSED"))

    def options_universe(self, max_names: int = 30, longs: int = 12, shorts: int = 9) -> List[Tuple[str, str]]:
        """(ticker, OPTIONS_* bucket) pairs for the 1–7 DTE scan.

        Both sides of every run are handed to the engine — the strongest LONG concurrence names
        (CALL candidates) AND each run's strongest SELL / EXIT names (PUT candidates) — plus the
        liquid core ETFs.  The engine's own direction score (momentum ⊕ beta-corridor fair value)
        decides call vs put, so when the corridor is BELOW fair value the put side is where the
        momentum and corridor tilt agree.  Names without a 1–7 DTE chain are SKIPped by the engine.
        """
        out: List[Tuple[str, str]] = []
        seen = set()
        def add(t, b):
            if t not in seen:
                seen.add(t); out.append((t, b))
        for c in sorted([c for c in self.final if c.tranche != "LOCKED_SLEEVE"], key=lambda c: -abs(c.z_score))[:longs]:
            add(c.ticker, c.options_bucket)
        sell_pool = sorted([c for tr in self.tranches for c in tr.sells], key=lambda c: c.raw_score)
        for c in sell_pool[:shorts]:
            add(c.ticker, c.options_bucket or "OPTIONS_HY")
        for t, b in self.OPTIONS_CORE:
            add(t, b)
        return out[:max_names]

    def options_backfill(self, per_run: int = 20) -> Dict[str, List[str]]:
        """Operator rule: when the HY / Distressed option buckets have no eligible name, the
        options engine fills the remainder from the S&P 400 / S&P 600 runs (regardless of
        HY/distressed classification). Returns {"SP400": [...], "SP600": [...]} ranked by |z|
        across each run's full scored candidate list (longs first, then sells)."""
        out: Dict[str, List[str]] = {}
        for tr in self.tranches:
            key = "SP400" if "400" in tr.name.upper() else ("SP600" if "600" in tr.name.upper() else None)
            if not key:
                continue
            ranked = sorted(list(tr.candidates) + list(tr.sells), key=lambda c: -abs(c.z_score))
            names: List[str] = []
            for c in ranked:
                if c.ticker not in names:
                    names.append(c.ticker)
            out[key] = names[:per_run]
        return out

    def markdown(self) -> str:
        out = [f"### Universe scan — {len(self.tranches)} tranches → concurrence ({self.as_of})",
               f"- Beta-corridor directional bias {self.corridor_bias:+.2f} · SPY 63d momentum {self.spy_mom_63d:+.1%}", ""]
        for t in self.tranches:
            out.append(f"**{t.name}** — universe {t.universe_size} · quoted {t.quoted} · passed screen {t.screened} · "
                       f"shortlist {t.shortlisted} · top {len(t.top)} · BUY {t.buy_n} / SELL {t.sell_n} / HOLD {t.hold_n} · "
                       f"avg α {t.avg_alpha:+.4f} · {t.elapsed_s:.0f}s")
            if t.rejected:
                out.append("  rejects: " + ", ".join(f"{k}×{v}" for k, v in sorted(t.rejected.items(), key=lambda x: -x[1])[:5]))
            for c in t.top[:6]:
                out.append(f"  - {c.ticker} ({c.sector}) z {c.z_score:+.2f} · RSI {c.rsi:.0f} {c.breakout or ''} · "
                           f"mom 21d {c.mom_21d:+.1%} · RS {c.rel_strength_63d:+.1%} → {c.sleeve}")
        out.append("")
        out.append(f"**Concurrence — final {len(self.final)} names**")
        out.append("| # | Ticker | Tranche | Sector | Sleeve | z | Dir | Votes | Why |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for i, c in enumerate(self.final, 1):
            v = sum(c.votes.values()); n = len(c.votes)
            out.append(f"| {i} | {c.ticker} | {c.tranche.split('_', 2)[-1]} | {c.sector} | {c.sleeve} | {c.z_score:+.2f} | {c.direction} | "
                       f"{v}/{n} | {'; '.join(c.notes[:2])} |")
        if self.dropped:
            out.append("")
            out.append("Dropped at concurrence: " + "; ".join(f"{t} ({r})" for t, r in self.dropped[:12]))
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------
class UniverseTrancheScanner:
    """Three separate tranche scans + concurrence. Schwab is the only data source."""

    def __init__(self, broker: Any, cfg: Optional[TrancheConfig] = None,
                 tranches: Optional[List[Tuple[str, List[str]]]] = None, options_engine: Any = None):
        self.broker = broker
        self.cfg = cfg or TrancheConfig()
        self.tranches = tranches or default_tranches()
        self._sd = options_engine
        if self._sd is None and ShortDTEOptionsEngine is not None:
            try:
                self._sd = ShortDTEOptionsEngine(broker, nav=100_000.0)
            except Exception:  # noqa: BLE001
                self._sd = None
        self.last_result: Optional[ConcurrenceResult] = None
        self._spy_close: np.ndarray = np.array([])

    # ---- data ------------------------------------------------------------
    def _quotes(self, tickers: List[str]) -> Dict[str, dict]:
        try:
            return self.broker.get_quotes(tickers) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("quotes failed: %s", exc)
            return {}

    def _close(self, ticker: str) -> np.ndarray:
        try:
            h = self.broker.get_price_history(ticker, days=self.cfg.history_days)
            return np.asarray(h.get("close", []), dtype=float)
        except Exception:  # noqa: BLE001
            return np.array([])

    @staticmethod
    def _mom(close: np.ndarray, n: int) -> float:
        return float(close[-1] / close[-n - 1] - 1.0) if close.size > n and close[-n - 1] > 0 else 0.0

    # ---- classification ----------------------------------------------------
    @staticmethod
    def _sleeve_for(c: Candidate) -> Tuple[str, str]:
        # ETF_FI run → allocation-file ETF routing (mirrors AllocationEngine._infer_bucket)
        if c.ticker in INCOME_ETFS:
            return "TLTW", "OPTIONS_IG"
        if c.ticker in BOND_ETFS:
            return "FIXED_INCOME", "OPTIONS_IG"
        if c.ticker in ("HYG", "JNK"):
            return "DISTRESSED", "OPTIONS_DISTRESSED"
        if c.tranche == "SCAN_4_ETF_FI" or c.ticker in ETF_HINTS:
            return "TLTW", "OPTIONS_IG"          # broad / sector / commodity ETFs → cashflow-ETF sleeve
        # distressed: deep 52w-low + high vol; HY: small/mid or high vol; IG: large-cap core, low vol
        if c.pos_52w < 0.15 and c.realized_vol > 0.45:
            return "DISTRESSED", "OPTIONS_DISTRESSED"
        if c.tranche == "SCAN_1_SP500" and c.realized_vol < 0.40:
            return "IG_EQUITY", "OPTIONS_IG"
        if c.tranche == "SCAN_2_SP400" and c.realized_vol < 0.30 and c.pos_52w > 0.6:
            return "IG_EQUITY", "OPTIONS_IG"     # quality mid-cap trending near highs
        return "HY_EQUITY", "OPTIONS_HY"

    # ---- one tranche -------------------------------------------------------
    def scan_tranche(self, name: str, tickers: List[str], corridor_bias: float) -> TrancheResult:
        t0 = time.monotonic()
        cfg = self.cfg
        rejected: Dict[str, int] = {}

        def rej(k: str):
            rejected[k] = rejected.get(k, 0) + 1

        quotes = self._quotes(tickers)
        pre: List[Candidate] = []
        for t in tickers:
            q = quotes.get(t)
            if not isinstance(q, dict):
                rej("no_quote"); continue
            px = float(q.get("last") or q.get("mark") or 0.0)
            vol = float(q.get("volume") or 0.0)
            if px < cfg.min_price:
                rej("price<min"); continue
            if px * vol < cfg.min_dollar_volume:
                rej("illiquid"); continue
            hi, lo = float(q.get("high_52") or 0.0), float(q.get("low_52") or 0.0)
            pos = (px - lo) / (hi - lo) if hi > lo > 0 else 0.5
            pre.append(Candidate(ticker=t, tranche=name, sector=(get_sector_for_ticker(t) or "Unknown"), price=px,
                                 dollar_volume=px * vol, pct_change=float(q.get("net_pct_change") or 0.0), pos_52w=pos))
        screened = len(pre)
        # quote-based pre-rank: today's move in the corridor direction, 52w position, liquidity
        sgn = 1.0 if corridor_bias >= 0 else -1.0
        for c in pre:
            c.raw_score = sgn * c.pct_change / 2.0 + (c.pos_52w - 0.5) * sgn + math.log10(max(c.dollar_volume, 1.0)) / 20.0
        pre.sort(key=lambda c: -c.raw_score)
        short = pre[:cfg.shortlist_per_tranche]
        for c in pre:  # quote-screen classification (refined below for the shortlist)
            c.mom_10d = c.pct_change / 100.0
            c.signal = "BUY" if c.raw_score > 0.05 else ("SELL" if c.raw_score < -0.05 else "HOLD")

        # candles → momentum / RSI / breakout / relative strength
        scored: List[Candidate] = []
        spy = self._spy_close
        spy63 = self._mom(spy, 63) if spy.size else 0.0
        for c in short:
            close = self._close(c.ticker)
            if close.size < 30:
                rej("history<30d"); continue
            rets = np.diff(np.log(close[-22:]))
            c.realized_vol = float(np.std(rets) * math.sqrt(252)) if rets.size else 0.0
            c.sharpe = float(np.mean(rets) * 252 / c.realized_vol) if rets.size and c.realized_vol > 0 else 0.0
            c.mom_5d, c.mom_21d, c.mom_63d = self._mom(close, 5), self._mom(close, 21), self._mom(close, 63)
            c.mom_10d = self._mom(close, 10)
            c.fully_scored = True
            c.rel_strength_63d = c.mom_63d - spy63
            if self._sd is not None:
                try:
                    m = self._sd.momentum_read(close)
                    c.rsi, c.breakout, c.momentum_score = m.rsi, (m.breakout or m.divergence), m.direction_score
                    c.notes += m.notes[:2]
                except Exception:  # noqa: BLE001
                    pass
            # composite: momentum read (RSI breakout/patterns) + RS + 21d trend + corridor agreement − vol penalty
            agree = 1.0 if (c.momentum_score >= 0) == (corridor_bias >= 0) else 0.6
            c.raw_score = float(agree * (0.45 * c.momentum_score + 0.25 * np.clip(c.rel_strength_63d / 0.15, -1, 1)
                                         + 0.20 * np.clip(c.mom_21d / 0.10, -1, 1) + 0.10 * (c.pos_52w - 0.5) * 2)
                                - 0.10 * max(0.0, c.realized_vol - 0.60))
            c.direction = "LONG" if c.raw_score >= 0 else "SHORT"
            c.signal = "BUY" if c.raw_score > 0.10 else ("SELL" if c.raw_score < -0.10 else "HOLD")
            subs = [c.momentum_score > 0, c.rel_strength_63d > 0, c.mom_21d > 0, c.mom_10d > 0, c.pos_52w > 0.5]
            agree_n = sum(subs) if c.raw_score >= 0 else len(subs) - sum(subs)
            c.ensemble = round(agree_n / len(subs), 3)
            c.sleeve, c.options_bucket = self._sleeve_for(c)
            if corridor_bias < -0.3 and c.direction == "LONG":
                c.notes.append("corridor ABOVE fair value: long sized down")
            scored.append(c)
        # z-score INSIDE the tranche so tranches compete fairly at concurrence
        if scored:
            arr = np.array([c.raw_score for c in scored]); mu, sd = float(arr.mean()), float(arr.std() or 1.0)
            for c in scored:
                c.z_score = float((c.raw_score - mu) / sd)
        scored.sort(key=lambda c: -c.z_score)
        top = [c for c in scored if c.direction == "LONG"][:cfg.top_per_tranche] or scored[:cfg.top_per_tranche]
        sells = sorted([c for c in scored if c.signal == "SELL"], key=lambda c: c.raw_score)[:5]
        buy_n = sum(1 for c in pre if c.signal == "BUY"); sell_n = sum(1 for c in pre if c.signal == "SELL")
        hold_n = len(pre) - buy_n - sell_n
        avg_alpha = float(np.mean([c.raw_score for c in scored])) if scored else 0.0
        res = TrancheResult(name=name, universe_size=len(tickers), quoted=len(quotes), screened=screened,
                            shortlisted=len(short), top=top, rejected=rejected, elapsed_s=time.monotonic() - t0,
                            buy_n=buy_n, sell_n=sell_n, hold_n=hold_n, avg_alpha=avg_alpha, sells=sells, candidates=scored)
        logger.info("%s: %d universe → %d screened → %d shortlisted → top %d (%.0fs)", name, len(tickers), screened, len(short), len(top), res.elapsed_s)
        return res

    # ---- concurrence -------------------------------------------------------
    def concur(self, results: List[TrancheResult], corridor_bias: float, spy63: float) -> ConcurrenceResult:
        cfg = self.cfg
        pool: List[Candidate] = [c for r in results for c in r.top]
        dropped: List[Tuple[str, str]] = []
        for c in pool:
            c.votes = {
                "momentum": c.momentum_score > 0.10 if c.direction == "LONG" else c.momentum_score < -0.10,
                "rel_strength": c.rel_strength_63d > 0 if c.direction == "LONG" else c.rel_strength_63d < 0,
                "corridor": (corridor_bias >= -0.3) if c.direction == "LONG" else (corridor_bias <= 0.3),
                "trend_21d": c.mom_21d > 0 if c.direction == "LONG" else c.mom_21d < 0,
                "liquidity": c.dollar_volume >= cfg.min_dollar_volume * 5,
                "tranche_rank": c.z_score >= 0.5,
            }
        # need ≥4 of 6 votes; then fill by z-score with sector / sleeve caps + min per tranche
        eligible = [c for c in pool if sum(c.votes.values()) >= 4]
        for c in pool:
            if c not in eligible:
                dropped.append((c.ticker, f"{sum(c.votes.values())}/6 votes"))
        eligible.sort(key=lambda c: -c.z_score)
        final: List[Candidate] = []
        sector_n: Dict[str, int] = {}
        sleeve_n: Dict[str, int] = {}
        max_per_sector = max(1, int(cfg.final_max_names * cfg.sector_cap))
        sleeve_cap_n = {k: max(1, round(v / sum(cfg.sleeve_targets.values()) * cfg.final_max_names)) for k, v in cfg.sleeve_targets.items()}

        tried: set = set()

        def admit(c: Candidate, reserved: bool = False) -> bool:
            if c.ticker in tried:
                return False
            tried.add(c.ticker)
            if c.sector != "Unknown" and sector_n.get(c.sector, 0) >= max_per_sector:
                dropped.append((c.ticker, f"sector cap {c.sector}")); return False
            # reserved seats (min per tranche) bypass the sleeve cap so every tranche keeps a voice
            if not reserved and sleeve_n.get(c.sleeve, 0) >= sleeve_cap_n.get(c.sleeve, 2):
                dropped.append((c.ticker, f"sleeve cap {c.sleeve}")); return False
            final.append(c); sector_n[c.sector] = sector_n.get(c.sector, 0) + 1; sleeve_n[c.sleeve] = sleeve_n.get(c.sleeve, 0) + 1
            return True

        # 1. minimum representation per tranche
        for r in results:
            n = 0
            for c in [x for x in eligible if x.tranche == r.name]:
                if n >= cfg.min_per_tranche: break
                if admit(c, reserved=True): n += 1
        # 2. fill by z-score
        for c in eligible:
            if len(final) >= cfg.final_max_names: break
            admit(c)
        final.sort(key=lambda c: -c.z_score)
        from datetime import datetime
        res = ConcurrenceResult(tranches=results, final=final, dropped=dropped, corridor_bias=corridor_bias,
                                spy_mom_63d=spy63, as_of=datetime.now().isoformat(timespec="minutes"))
        # CVR sleeve: event-based — news/8-K scan for live cash + CVR merger deals (target equity)
        try:
            if self.cfg.cvr_event_scan:
                from engine.execution.cvr_event_scan import CVREventScanner
                cvr = CVREventScanner(self.broker, max_names=self.cfg.cvr_max_names)
                res.lock_cvr_from_events(cvr.scan(), cvr.last_notes)
            else:
                res.cvr_notes = ["CVR event scan disabled (METADRON_CVR_SCAN=0)"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("CVR event scan failed: %s", exc)
            res.cvr_notes = [f"CVR event scan failed: {exc}"]
        # LOCKED ALLOCATION: every equity sleeve must be represented from the runs
        for note in res.lock_sleeves():
            logger.info("locked sleeve: %s", note)
        self.last_result = res
        return res

    # ---- full run ----------------------------------------------------------
    def run(self, corridor_bias: Optional[float] = None, on_run: Optional[Callable[[TrancheResult], None]] = None) -> ConcurrenceResult:
        """Run 1 → Run 2 → Run 3 → Run 4 separately (each reported on its own), then concur.

        `on_run(result)` is invoked right after each run finishes so the caller can print
        that run's result + proposed allocation before the next run starts.
        """
        self._spy_close = self._close("SPY")
        spy63 = self._mom(self._spy_close, 63) if self._spy_close.size else 0.0
        if corridor_bias is None:
            corridor_bias = 0.0
            if self._sd is not None:
                try:
                    ctx = self._sd.market_context()
                    corridor_bias = float(getattr(ctx, "direction_bias", 0.0))
                except Exception:  # noqa: BLE001
                    pass
        results: List[TrancheResult] = []
        for name, tickers in self.tranches:
            try:
                results.append(self.scan_tranche(name, tickers, corridor_bias))
            except Exception as exc:  # noqa: BLE001
                logger.error("%s failed: %s", name, exc)
                results.append(TrancheResult(name, len(tickers), 0, 0, 0, [], {"error": 1}, 0.0))
            if on_run is not None:
                try:
                    on_run(results[-1])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("on_run callback failed: %s", exc)
        return self.concur(results, corridor_bias, spy63)


__all__ = ["UniverseTrancheScanner", "TrancheConfig", "TrancheResult", "ConcurrenceResult", "Candidate", "default_tranches",
           "run_label", "RUN_LABELS"]
