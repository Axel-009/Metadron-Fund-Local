"""Theme-crash rotation — the "strong market, crashing theme" playbook (owner-approved paradigm).

Four levels of "crash" are kept separate (they trigger different responses):

    MARKET     S&P regime / Rm (corridor.py)        → sizes market beta (MES overlay)
    THEME      a cluster of related names crashing  → THIS module: rotate the equity sleeve,
               while the market is fine                hedge escalation, harvest the drawdown
    POSITION   one holding's own drawdown            → position protection (stops / trims)
    PORTFOLIO  the book's NAV drawdown               → hard stops in risk_budget.py

Example paradigm (Oct–Nov): SPY trailing > 12% (ROTATION zone, max leverage) while AI names with
heavy debt / private-credit exposure (e.g. CRWV, NBIS) and over-spenders (e.g. IREN, HIVE) crash.
Response, all sized inside the 2.5% stock-specific VaR budget (risk_budget.check_alpha_trade):

    1. EXIT / CUT   longs in the crashed theme (CRASH → 0, STRESS → cap 2% NAV per name). Market
                    beta untouched. Targets are idempotent: re-running the plan daily never compounds.
    2. ROTATE       freed equity weight into defensives — defense (e.g. LMT) weighted up when the
                    geopolitical-escalation score is high, healthcare (e.g. LLY) otherwise.
    3. ESCALATION   predictive posture when escalation is likely: long oil via call options
                    (premium budget) + refiners (e.g. VLO).
    4. HARVEST      benefit from the drawdown:
                      - discounted LEAPS calls on crashed names that are still SOLVENT
                        (capital-structure check passes) — call SPREADS when implied vol is rich;
                      - short-dated straddles where realised vol > implied vol or an event is near.

Tickers in THEMES / DEFENSIVES are configuration examples supplied by the owner, not
recommendations; agents may propose changes, the owner approves (CLAUDE.md design rule 3).
Pure Python + math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

# --- configuration (owner-editable) -----------------------------------------
THEMES: Dict[str, List[str]] = {
    "AI_CREDIT_HUBS": ["CRWV", "NBIS"],         # AI infra financed with heavy debt / private credit
    "AI_OVERSPEND": ["IREN", "HIVE"],           # capex over-spenders
}
DEFENSIVES: Dict[str, List[str]] = {
    "DEFENSE": ["LMT", "NOC", "RTX", "GD"],
    "HEALTHCARE": ["LLY", "JNJ", "MRK", "ABBV"],
}
ESCALATION_SLEEVE = {
    "oil_option_underlyings": ["USO", "XLE"],   # long calls
    "refiners": ["VLO", "MPC"],
}

# Theme crash thresholds (equal-weight theme index)
STRESS_NAME_CAP = 0.02         # STRESS theme: cap each name at 2% NAV (idempotent, not "halve daily")
CRASH_DRAWDOWN = 0.30          # ≥ 30% below its 63-day high …
CRASH_REL_SPY = -0.15          # … and ≥ 15 pts under SPY over 21 days
STRESS_DRAWDOWN = 0.15
STRESS_REL_SPY = -0.08

# Escalation score components (each 0/1, averaged)
ESC_OIL_21D = 0.10             # oil proxy up ≥ 10% in 21 days
ESC_GOLD_21D = 0.05            # gold up ≥ 5%
ESC_DEFENSE_REL_21D = 0.05     # defense basket beats SPY by ≥ 5 pts
ESC_VIX = 25.0
ESCALATION_THRESHOLD = 0.5

# Budgets (fractions of NAV)
DEFENSE_SHARE_ESCALATION = 0.60   # share of freed weight to defense when escalating …
DEFENSE_SHARE_CALM = 0.25         # … otherwise
OIL_CALL_PREMIUM = 0.010          # total premium for oil calls
REFINER_WEIGHT = 0.03             # total weight across refiners
LEAPS_PREMIUM_PER_NAME = 0.005
LEAPS_MIN_DTE = 270
LEAPS_MIN_DISCOUNT = 0.30         # price ≥ 30% below its 252-day high
LEAPS_RICH_IV_RATIO = 1.3         # IV / RV above this → call spread instead of outright
STRADDLE_PREMIUM_PER_NAME = 0.003
STRADDLE_MAX_NAMES = 3
STRADDLE_RV_IV_RATIO = 1.1        # buy straddles when RV ≥ 1.1 × IV …
STRADDLE_EVENT_DAYS = 14          # … or an event is within 14 days
STRADDLE_DTE = (7, 30)

# Solvency (from the capital-structure monitor)
SOLVENT_MAX_PD = 0.05             # Merton 1-year default probability
SOLVENT_MIN_COVERAGE = 2.0        # interest coverage (EBIT / interest)


# --- primitives ---------------------------------------------------------------
def _ret(closes: List[float], n: int) -> float:
    if len(closes) <= n or closes[-n - 1] <= 0:
        return 0.0
    return closes[-1] / closes[-n - 1] - 1.0


def _drawdown(closes: List[float], window: int) -> float:
    w = closes[-window:] if len(closes) >= window else closes
    peak = max(w) if w else 0.0
    return 1.0 - closes[-1] / peak if peak > 0 else 0.0


def realised_vol(closes: List[float], n: int = 20) -> float:
    w = closes[-(n + 1):]
    rets = [math.log(b / a) for a, b in zip(w[:-1], w[1:]) if a > 0 and b > 0]
    if len(rets) < 2:
        return 0.0
    m = sum(rets) / len(rets)
    return math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1) * 252)


def theme_index(prices: Dict[str, List[float]], tickers: Iterable[str]) -> List[float]:
    """Equal-weight index (rebased to 1) of the tickers that have data."""
    series = [prices[t] for t in tickers if t in prices and len(prices[t]) > 1 and prices[t][0] > 0]
    if not series:
        return []
    n = min(len(s) for s in series)
    series = [s[-n:] for s in series]
    return [sum(s[i] / s[0] for s in series) / len(series) for i in range(n)]


# --- state detection ----------------------------------------------------------
@dataclass
class ThemeState:
    theme: str
    state: str              # NORMAL | STRESS | CRASH
    drawdown: float         # from 63-day high
    rel_spy_21d: float      # theme return − SPY return, 21 days
    names: List[str] = field(default_factory=list)


def theme_state(theme: str, tickers: List[str], prices: Dict[str, List[float]],
                spy: List[float]) -> ThemeState:
    idx = theme_index(prices, tickers)
    if len(idx) < 22 or len(spy) < 22:
        return ThemeState(theme, "NORMAL", 0.0, 0.0, list(tickers))
    dd = _drawdown(idx, 63)
    rel = _ret(idx, 21) - _ret(spy, 21)
    if dd >= CRASH_DRAWDOWN and rel <= CRASH_REL_SPY:
        st = "CRASH"
    elif dd >= STRESS_DRAWDOWN or rel <= STRESS_REL_SPY:
        st = "STRESS"
    else:
        st = "NORMAL"
    return ThemeState(theme, st, dd, rel, list(tickers))


def escalation_score(oil: Optional[List[float]] = None, gold: Optional[List[float]] = None,
                     defense_idx: Optional[List[float]] = None, spy: Optional[List[float]] = None,
                     vix: Optional[float] = None, agent_flag: Optional[bool] = None) -> float:
    """0..1 geopolitical-escalation score; an agent flag (events agent) forces 1.0 / 0.0."""
    if agent_flag is not None:
        return 1.0 if agent_flag else 0.0
    parts = []
    if oil:
        parts.append(1.0 if _ret(oil, 21) >= ESC_OIL_21D else 0.0)
    if gold:
        parts.append(1.0 if _ret(gold, 21) >= ESC_GOLD_21D else 0.0)
    if defense_idx and spy:
        parts.append(1.0 if _ret(defense_idx, 21) - _ret(spy, 21) >= ESC_DEFENSE_REL_21D else 0.0)
    if vix is not None:
        parts.append(1.0 if vix >= ESC_VIX else 0.0)
    return sum(parts) / len(parts) if parts else 0.0


def is_solvent(cap_structure: Optional[dict]) -> bool:
    """Capital-structure check; unknown → NOT solvent (no LEAPS on missing data)."""
    if not cap_structure:
        return False
    pd = cap_structure.get("merton_pd")
    cov = cap_structure.get("interest_coverage")
    if pd is None or cov is None:
        return False
    return pd <= SOLVENT_MAX_PD and cov >= SOLVENT_MIN_COVERAGE


# --- plan -------------------------------------------------------------------------
@dataclass
class RotationAction:
    kind: str               # CUT | ROTATE | ESCALATION_CALLS | ESCALATION_EQUITY | LEAPS | LEAPS_SPREAD | STRADDLE
    ticker: str
    target_weight: float = 0.0      # equity weight (fraction of NAV), or 0 for options
    premium: float = 0.0            # option premium budget (fraction of NAV)
    dte: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class RotationPlan:
    market_zone: str
    escalation: float
    themes: List[ThemeState]
    actions: List[RotationAction]

    @property
    def active(self) -> bool:
        return any(t.state != "NORMAL" for t in self.themes)


def build_plan(market_zone: str,
               prices: Dict[str, List[float]],
               spy: List[float],
               holdings: Dict[str, float],
               escalation: float = 0.0,
               cap_structure: Optional[Dict[str, dict]] = None,
               implied_vol: Optional[Dict[str, float]] = None,
               days_to_event: Optional[Dict[str, int]] = None,
               themes: Optional[Dict[str, List[str]]] = None,
               defensives: Optional[Dict[str, List[str]]] = None) -> RotationPlan:
    """Rotation plan for the equity / options sleeves. Market beta is NOT changed here.

    holdings: {ticker: current equity weight (fraction of NAV)}.
    The plan proposes target weights / premium budgets; final sizes go through the
    stock-specific VaR budget (risk_budget.check_alpha_trade) and the decision gates.
    """
    themes = themes or THEMES
    defensives = defensives or DEFENSIVES
    cap_structure = cap_structure or {}
    implied_vol = implied_vol or {}
    days_to_event = days_to_event or {}
    states = [theme_state(name, tks, prices, spy) for name, tks in themes.items()]
    actions: List[RotationAction] = []
    freed = 0.0

    # 1. cut longs in stressed / crashed themes
    for ts in states:
        if ts.state == "NORMAL":
            continue
        cap = 0.0 if ts.state == "CRASH" else STRESS_NAME_CAP
        for t in ts.names:
            w = holdings.get(t, 0.0)
            if w > cap:
                actions.append(RotationAction("CUT", t, target_weight=cap,
                                              reason=f"{ts.theme} {ts.state}: dd {ts.drawdown:.0%}, vs SPY {ts.rel_spy_21d:+.0%}"))
                freed += w - cap

    if not any(ts.state != "NORMAL" for ts in states):
        return RotationPlan(market_zone, escalation, states, actions)

    # 2. rotate freed weight into defensives (equal weight within each basket)
    if freed > 0:
        d_share = DEFENSE_SHARE_ESCALATION if escalation >= ESCALATION_THRESHOLD else DEFENSE_SHARE_CALM
        split = {"DEFENSE": d_share, "HEALTHCARE": 1 - d_share}
        for basket, share in split.items():
            names = [t for t in defensives.get(basket, []) if t in prices] or defensives.get(basket, [])
            for t in names:
                add = freed * share / max(len(names), 1)
                actions.append(RotationAction("ROTATE", t, target_weight=holdings.get(t, 0.0) + add,
                                              reason=f"defensive rotation ({basket}, escalation {escalation:.2f})"))

    # 3. escalation posture
    if escalation >= ESCALATION_THRESHOLD:
        und = ESCALATION_SLEEVE["oil_option_underlyings"]
        for t in und:
            actions.append(RotationAction("ESCALATION_CALLS", t, premium=OIL_CALL_PREMIUM / len(und), dte=60,
                                          reason=f"long oil calls — escalation {escalation:.2f}"))
        ref = ESCALATION_SLEEVE["refiners"]
        for t in ref:
            actions.append(RotationAction("ESCALATION_EQUITY", t,
                                          target_weight=max(holdings.get(t, 0.0), REFINER_WEIGHT / len(ref)),
                                          reason="refiner exposure — escalation"))

    # 4. harvest the drawdown
    straddles = []
    for ts in states:
        if ts.state == "NORMAL":
            continue
        for t in ts.names:
            px = prices.get(t, [])
            if len(px) < 22:
                continue
            rv = realised_vol(px, 20)
            iv = implied_vol.get(t)
            discount = _drawdown(px, 252)
            if ts.state == "CRASH" and discount >= LEAPS_MIN_DISCOUNT and is_solvent(cap_structure.get(t)):
                rich = iv is not None and rv > 0 and iv / rv > LEAPS_RICH_IV_RATIO
                actions.append(RotationAction("LEAPS_SPREAD" if rich else "LEAPS", t,
                                              premium=LEAPS_PREMIUM_PER_NAME, dte=LEAPS_MIN_DTE,
                                              reason=f"solvent, {discount:.0%} below 1y high"
                                                     + (f", IV/RV {iv / rv:.2f} rich → call spread" if rich else "")))
            ev = days_to_event.get(t)
            vol_cheap = iv is not None and iv > 0 and rv >= STRADDLE_RV_IV_RATIO * iv
            if vol_cheap or (ev is not None and 0 <= ev <= STRADDLE_EVENT_DAYS):
                score = (rv / iv if iv else 1.0) + (1.0 if ev is not None and ev <= STRADDLE_EVENT_DAYS else 0.0)
                straddles.append((score, t, rv, iv, ev))
    for score, t, rv, iv, ev in sorted(straddles, reverse=True)[:STRADDLE_MAX_NAMES]:
        why = []
        if iv:
            why.append(f"RV {rv:.0%} vs IV {iv:.0%}")
        if ev is not None:
            why.append(f"event in {ev}d")
        actions.append(RotationAction("STRADDLE", t, premium=STRADDLE_PREMIUM_PER_NAME,
                                      dte=STRADDLE_DTE[1], reason="; ".join(why)))

    return RotationPlan(market_zone, escalation, states, actions)
