"""The 11 GICS sectors — the rotation destinations out of crashed / stressed themes (owner rule).

Rotation trades the Select Sector SPDR ETFs (one per GICS sector), never hand-picked names. Which
sectors receive the freed weight is decided by the research and the regime, not by a fixed list:

    sector score = mean edge of its market-map sub-sectors (macro question tree + scenarios + fundamentals)
                 + regime tilt   STRESS / CRASH: defensives +0.30, cyclicals −0.20
                                 TRENDING:       cyclicals +0.10, growth +0.10
                 + bias tilt     BEARISH: defensives +0.20 · BULLISH: cyclicals / growth +0.10
                 + escalation    (score ≥ 0.5): Energy +0.30, Industrials (aerospace & defense) +0.15
                 − theme penalty a sector whose own sub-sector is the crashing theme gets −0.50
    rotation     freed weight → the top ROTATION_TOP sectors with a positive score, ∝ score;
                 no positive score → the defensive trio (Health Care, Staples, Utilities) equally.
"""

from __future__ import annotations

from typing import Dict, List, Optional

GICS = {   # ETF: (GICS sector, style)
    "XLK": ("Information Technology", "GROWTH"),
    "XLC": ("Communication Services", "GROWTH"),
    "XLY": ("Consumer Discretionary", "CYCLICAL"),
    "XLF": ("Financials", "CYCLICAL"),
    "XLI": ("Industrials", "CYCLICAL"),
    "XLB": ("Materials", "CYCLICAL"),
    "XLRE": ("Real Estate", "CYCLICAL"),
    "XLE": ("Energy", "DEFENSIVE"),
    "XLV": ("Health Care", "DEFENSIVE"),
    "XLP": ("Consumer Staples", "DEFENSIVE"),
    "XLU": ("Utilities", "DEFENSIVE"),
}
SECTOR_ETFS = list(GICS)
DEFENSIVE_FALLBACK = ["XLV", "XLP", "XLU"]

SUBSECTOR_TO_ETF = {
    "AI_PLATFORMS": "XLC", "AI_CHIP_DESIGN": "XLK", "SEMI_EQUIPMENT": "XLK", "FOUNDRY_REGIONAL": "XLK",
    "MEMORY": "XLK", "NETWORKING": "XLK", "DATA_CENTERS": "XLRE", "POWER_GRID": "XLU",
    "AI_NEOCLOUD_CREDIT": "XLK", "AI_HOSTING_PIVOT": "XLK", "LOW_END_SAAS": "XLK", "SOFTWARE": "XLK",
    "LABOR_INTENSIVE_SERVICES": "XLI", "MATERIALS": "XLB", "INDUSTRIALS": "XLI", "FINANCIALS": "XLF",
    "CONSUMER_DISCRETIONARY": "XLY", "REAL_ESTATE": "XLRE", "COMMODITY_PRODUCERS": "XLE",
    "HEALTHCARE_PHARMA": "XLV", "MEDIA_TELECOM": "XLC", "STAPLES": "XLP", "UTILITIES": "XLU",
    "ENERGY_INTEGRATED": "XLE", "REFINERS": "XLE", "DEFENSE": "XLI",
}

ROTATION_TOP = 3
REGIME_TILT = {
    "STRESS": {"DEFENSIVE": 0.30, "CYCLICAL": -0.20, "GROWTH": -0.10},
    "CRASH": {"DEFENSIVE": 0.30, "CYCLICAL": -0.20, "GROWTH": -0.10},
    "TRENDING": {"DEFENSIVE": 0.0, "CYCLICAL": 0.10, "GROWTH": 0.10},
    "RANGE": {},
}
BIAS_TILT = {"BEARISH": {"DEFENSIVE": 0.20}, "BULLISH": {"CYCLICAL": 0.10, "GROWTH": 0.10}}
ESCALATION_TILT = {"XLE": 0.30, "XLI": 0.15}
THEME_PENALTY = 0.50


def sector_scores(subsector_edges: Optional[Dict[str, float]] = None, regime: str = "RANGE",
                  bias: str = "NEUTRAL", escalation: float = 0.0,
                  crashing_subsectors: Optional[List[str]] = None) -> Dict[str, float]:
    edges = subsector_edges or {}
    by_etf: Dict[str, List[float]] = {e: [] for e in SECTOR_ETFS}
    for sub, edge in edges.items():
        etf = SUBSECTOR_TO_ETF.get(sub)
        if etf:
            by_etf[etf].append(edge)
    scores = {}
    for etf, (_, style) in GICS.items():
        vals = by_etf[etf]
        s = sum(vals) / len(vals) if vals else 0.0
        s += REGIME_TILT.get(str(regime).upper(), {}).get(style, 0.0)
        s += BIAS_TILT.get(str(bias).upper(), {}).get(style, 0.0)
        if escalation >= 0.5:
            s += ESCALATION_TILT.get(etf, 0.0)
        scores[etf] = s
    for sub in crashing_subsectors or []:
        etf = SUBSECTOR_TO_ETF.get(sub)
        if etf:
            scores[etf] -= THEME_PENALTY
    return scores


def rotation_weights(scores: Dict[str, float], amount: float, top: int = ROTATION_TOP) -> Dict[str, float]:
    """Split `amount` (fraction of NAV) over the best sectors ∝ positive score."""
    if amount <= 0:
        return {}
    best = [(s, e) for e, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0][:top]
    if not best:
        return {e: amount / len(DEFENSIVE_FALLBACK) for e in DEFENSIVE_FALLBACK}
    tot = sum(s for s, _ in best)
    return {e: amount * s / tot for s, e in best}
