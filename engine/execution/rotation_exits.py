"""
Rotation exits — Phase 5 sell decisions for the equities / ETF book.
====================================================================

Runs BEFORE any new add in a full scan (allocation guide: "20 % drawdown →
rotate or close, even when adding a position"). Two triggers:

    1. DRAWDOWN        position P&L ≤ −20 % vs cost      → close (hard rule)
    2. ALPHA_FLOOR     held name shows up as a SELL in its universe run
                       (negative composite α / momentum roll-over)  → rotate

Options contracts are NOT rotated here — the ShortDTE engine owns its own
1–7 DTE exits (theta / MC gate).  The returned rows are shaped for the
gold-standard report (VIEW 1 PHASE 5, VIEW 2 DEPLOYMENT SUMMARY, TRANSACTION LOG).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

DRAWDOWN_EXIT_PCT = 0.20


@dataclass
class RotationExit:
    ticker: str
    side: str                  # SELL (long) | BUY_TO_COVER (short)
    quantity: int
    market_value: float
    pnl_pct: float
    reason: str                # drawdown_exit (-21.7%) | alpha_floor_breach (-8.8%)
    reason_short: str          # "20% drawdown rule" | "alpha floor breach"
    bucket: str = ""
    status: str = "PENDING"    # set after L7 submit

    def to_dict(self) -> dict:
        return asdict(self)


def _pnl_pct(pos: Any) -> float:
    cost = float(getattr(pos, "avg_cost", 0.0) or 0.0)
    px = float(getattr(pos, "current_price", 0.0) or 0.0)
    qty = float(getattr(pos, "quantity", 0) or 0)
    if cost <= 0 or px <= 0:
        return 0.0
    raw = px / cost - 1.0
    return raw if qty >= 0 else -raw


def projected_pnl_pct(pnl_pct: float, annual_vol: float, horizon_days: int = 5, z: float = 1.65) -> float:
    """Where the position's P&L lands if it takes a 1-week 95 % adverse move (vol from the scan)."""
    if not annual_vol or annual_vol <= 0:
        return pnl_pct
    return pnl_pct - z * annual_vol * math.sqrt(horizon_days / 252.0)


def compute_rotation_exits(
    positions: Dict[str, Any],
    tranche_result: Any = None,
    drawdown_pct: float = DRAWDOWN_EXIT_PCT,
    bucket_of: Optional[Dict[str, str]] = None,
    extra_sell_tickers: Iterable[str] = (),
    vol_by: Optional[Dict[str, float]] = None,
) -> List[RotationExit]:
    """positions: ticker → Position (equities/ETFs only; option symbols are skipped).

    Leverage is flexible (Cube max_leverage governs) BECAUSE this rule is strict: a position is
    rotated when it hits −20 % OR when its 1-week 95 % adverse projection would take it through
    −20 % (projected_drawdown), OR when its universe run flags it SELL (alpha floor).
    """
    sells: Dict[str, float] = {}
    vols: Dict[str, float] = dict(vol_by or {})
    if tranche_result is not None:
        for tr in getattr(tranche_result, "tranches", []):
            for c in getattr(tr, "sells", []):
                sells[c.ticker] = float(c.raw_score)
            for c in getattr(tr, "candidates", []) or []:
                if getattr(c, "realized_vol", 0):
                    vols.setdefault(c.ticker, float(c.realized_vol))
    for t in extra_sell_tickers:
        sells.setdefault(t, -1.0)
    out: List[RotationExit] = []
    for sym, pos in positions.items():
        if getattr(pos, "sector", "") == "OPTIONS" or " " in sym:
            continue
        qty = int(getattr(pos, "quantity", 0) or 0)
        if qty == 0:
            continue
        pnl = _pnl_pct(pos)
        mv = float(getattr(pos, "market_value", 0.0) or (qty * float(getattr(pos, "current_price", 0.0) or 0.0)))
        side = "SELL" if qty > 0 else "BUY_TO_COVER"
        b = (bucket_of or {}).get(sym, "")
        proj = projected_pnl_pct(pnl, vols.get(sym, 0.0))
        if pnl <= -abs(drawdown_pct):
            out.append(RotationExit(sym, side, abs(qty), abs(mv), pnl, f"drawdown_exit ({pnl:+.1%})", "20% drawdown rule", b))
        elif proj <= -abs(drawdown_pct):
            out.append(RotationExit(sym, side, abs(qty), abs(mv), pnl, f"projected_drawdown ({pnl:+.1%} → {proj:+.1%} 1wk/95%)",
                                    "projected 20% drawdown", b))
        elif sym in sells:
            out.append(RotationExit(sym, side, abs(qty), abs(mv), pnl, f"alpha_floor_breach ({pnl:+.1%})", "alpha floor breach", b))
    # worst first (most negative P&L) — matches the gold-standard ordering
    out.sort(key=lambda e: e.pnl_pct)
    return out


class RotationExitEngine:
    """Thin component wrapper (wiring manifest) around compute_rotation_exits."""

    def __init__(self, drawdown_pct: float = DRAWDOWN_EXIT_PCT):
        self.drawdown_pct = drawdown_pct
        self.last: List[RotationExit] = []

    def evaluate(self, positions: Dict[str, Any], tranche_result: Any = None, **kw) -> List[RotationExit]:
        self.last = compute_rotation_exits(positions, tranche_result, drawdown_pct=self.drawdown_pct, **kw)
        return self.last


__all__ = ["RotationExit", "RotationExitEngine", "compute_rotation_exits", "projected_pnl_pct", "DRAWDOWN_EXIT_PCT"]
