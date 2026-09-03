"""
Universe Tranche Scan — three SEPARATE scans, then a concurrence vote.
=====================================================================

The full universe (S&P 500 + S&P MidCap 400 + S&P SmallCap 600 + extras) is
NEVER scanned in one pass. Each 30-minute full scan runs three independent
tranche scans and only then concurs on a final selection:

    Scan 1  →  S&P 500                     (large-cap core: IG sleeve heavy)
    Scan 2  →  S&P SmallCap 600            (small-cap: HY / distressed tilt)
    Scan 3  →  Remaining ~400              (S&P MidCap 400 + extras/ETFs)
    ─────────────────────────────────────────────────────────────────────
    Concur  →  z-score each tranche on its own distribution, then vote:
               momentum/RSI + relative strength vs SPY + beta-corridor
               direction + 20-bar breakout + liquidity, subject to the
               allocation-file sleeve caps, the L7 G2 30 % sector cap and a
               minimum representation per tranche.

Data source is Schwab only (batched quotes for the screen, daily candles for
the shortlist) so a full three-tranche pass fits the Schwab rate budget.

The output feeds: (a) the equity slate for L7, (b) the ShortDTE options
universe (top names per options bucket), (c) the in-chat run patch.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    from engine.execution.short_dte_options_engine import ShortDTEOptionsEngine
except Exception:  # noqa: BLE001
    ShortDTEOptionsEngine = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tranche definitions
# ---------------------------------------------------------------------------
ETF_HINTS = {"SPY", "QQQ", "IWM", "DIA", "MDY", "VTI", "RSP", "TLT", "TLTW", "HYG", "LQD", "JNK", "XLE", "XLF",
             "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "GLD", "USO", "UUP", "SLV", "EEM", "EFA"}


def default_tranches() -> List[Tuple[str, List[str]]]:
    sp500 = sorted(set(SP500_TICKERS))
    sp600 = sorted(set(SP600_TICKERS) - set(sp500))
    remaining = sorted((set(SP400_TICKERS) | set(EXTRA_TICKERS)) - set(sp500) - set(sp600))
    return [("SCAN_1_SP500", sp500), ("SCAN_2_SP600", sp600), ("SCAN_3_REMAINING_400", remaining)]


@dataclass
class TrancheConfig:
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
    raw_score: float = 0.0
    z_score: float = 0.0
    direction: str = "LONG"
    sleeve: str = "HY_EQUITY"
    options_bucket: str = "OPTIONS_HY"
    notes: List[str] = field(default_factory=list)
    votes: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


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

    def to_dict(self) -> dict:
        d = asdict(self); d["top"] = [c.to_dict() for c in self.top]; return d


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

    def equity_slate(self) -> List[dict]:
        """Approved-trade-shaped rows for Phase 5 / L7."""
        return [{"ticker": c.ticker, "signal": None, "side": "BUY" if c.direction == "LONG" else "SELL",
                 "decision": {"source": "TRANCHE_CONCURRENCE", "bucket": c.sleeve, "type": "EQUITY"},
                 "bucket": c.sleeve, "instrument_type": "ETF" if c.ticker in ETF_HINTS else "EQUITY",
                 "confidence": round(min(1.0, 0.5 + abs(float(c.z_score)) / 4), 3), "alpha_score": round(float(c.raw_score), 4),
                 "tranche": c.tranche, "sector": c.sector, "reason": "; ".join(c.notes[:3])} for c in self.final]

    def options_universe(self, max_names: int = 8) -> List[Tuple[str, str]]:
        """(ticker, OPTIONS_* bucket) pairs for the 1–7 DTE scan — strongest names only."""
        ranked = sorted(self.final, key=lambda c: -abs(c.z_score))
        return [(c.ticker, c.options_bucket) for c in ranked[:max_names]]

    def markdown(self) -> str:
        out = [f"### Universe scan — 3 tranches → concurrence ({self.as_of})",
               f"- Beta-corridor directional bias {self.corridor_bias:+.2f} · SPY 63d momentum {self.spy_mom_63d:+.1%}", ""]
        for t in self.tranches:
            out.append(f"**{t.name}** — universe {t.universe_size} · quoted {t.quoted} · passed screen {t.screened} · "
                       f"shortlist {t.shortlisted} · top {len(t.top)} · {t.elapsed_s:.0f}s")
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
        if c.ticker in ("TLT", "TLTW", "LQD", "IEF", "SHY", "AGG", "BND"):
            return ("TLTW" if c.ticker == "TLTW" else "FIXED_INCOME"), "OPTIONS_IG"
        if c.ticker in ("HYG", "JNK"):
            return "DISTRESSED", "OPTIONS_DISTRESSED"
        # distressed: deep 52w-low + high vol; HY: small/mid or high vol; IG: large-cap core, low vol
        if c.pos_52w < 0.15 and c.realized_vol > 0.45:
            return "DISTRESSED", "OPTIONS_DISTRESSED"
        if c.tranche == "SCAN_1_SP500" and c.realized_vol < 0.40:
            return "IG_EQUITY", "OPTIONS_IG"
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
            c.mom_5d, c.mom_21d, c.mom_63d = self._mom(close, 5), self._mom(close, 21), self._mom(close, 63)
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
        top = scored[:cfg.top_per_tranche]
        res = TrancheResult(name=name, universe_size=len(tickers), quoted=len(quotes), screened=screened,
                            shortlisted=len(short), top=top, rejected=rejected, elapsed_s=time.monotonic() - t0)
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
        self.last_result = res
        return res

    # ---- full run ----------------------------------------------------------
    def run(self, corridor_bias: Optional[float] = None) -> ConcurrenceResult:
        """Scan 1 → Scan 2 → Scan 3 separately, then concur."""
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
        return self.concur(results, corridor_bias, spy63)


__all__ = ["UniverseTrancheScanner", "TrancheConfig", "TrancheResult", "ConcurrenceResult", "Candidate", "default_tranches"]
