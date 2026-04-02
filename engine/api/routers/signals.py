"""
Signals router — ARB, FUTURES, ETF, FIXED INC, OPENBB tabs
Wraps: StatArbEngine, ContagionEngine, EventDrivenEngine, CVREngine,
       DistressedAssetEngine, SecurityAnalysisEngine, PatternDiscoveryEngine
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.signals")
router = APIRouter()

_stat_arb = None
_contagion = None
_event = None
_cvr = None
_distress = None
_security = None
_pattern = None
_social = None


def _get_stat_arb():
    global _stat_arb
    if _stat_arb is None:
        from engine.signals.stat_arb_engine import StatArbEngine
        _stat_arb = StatArbEngine()
    return _stat_arb


def _get_contagion():
    global _contagion
    if _contagion is None:
        from engine.signals.contagion_engine import ContagionEngine
        _contagion = ContagionEngine()
    return _contagion


def _get_event():
    global _event
    if _event is None:
        from engine.signals.event_driven_engine import EventDrivenEngine
        _event = EventDrivenEngine()
    return _event


def _get_cvr():
    global _cvr
    if _cvr is None:
        from engine.signals.cvr_engine import CVREngine
        _cvr = CVREngine()
    return _cvr


def _get_distress():
    global _distress
    if _distress is None:
        from engine.signals.distressed_asset_engine import DistressedAssetEngine
        _distress = DistressedAssetEngine()
    return _distress


def _get_security():
    global _security
    if _security is None:
        from engine.signals.security_analysis_engine import SecurityAnalysisEngine
        _security = SecurityAnalysisEngine()
    return _security


def _get_pattern():
    global _pattern
    if _pattern is None:
        from engine.signals.pattern_discovery_engine import PatternDiscoveryEngine
        _pattern = PatternDiscoveryEngine()
    return _pattern


def _get_social():
    global _social
    if _social is None:
        from engine.signals.social_prediction_engine import SocialPredictionEngine
        _social = SocialPredictionEngine()
    return _social


# ─── ARB tab ───────────────────────────────────────────────

@router.get("/stat-arb/pairs")
async def stat_arb_pairs(max_pairs: int = Query(30, ge=1, le=100)):
    """All RV pairs with cointegration stats and z-scores."""
    try:
        sa = _get_stat_arb()
        pairs = sa.scan_pairs(max_pairs=max_pairs)
        result = []
        for p in pairs:
            result.append({
                "ticker_a": p.ticker_a if hasattr(p, "ticker_a") else "",
                "ticker_b": p.ticker_b if hasattr(p, "ticker_b") else "",
                "correlation": p.correlation if hasattr(p, "correlation") else 0,
                "cointegration_pvalue": p.cointegration_pvalue if hasattr(p, "cointegration_pvalue") else 1,
                "half_life": p.half_life if hasattr(p, "half_life") else 0,
                "spread_zscore": p.spread_zscore if hasattr(p, "spread_zscore") else 0,
                "hedge_ratio": p.hedge_ratio if hasattr(p, "hedge_ratio") else 0,
                "signal_strength": p.signal_strength if hasattr(p, "signal_strength") else 0,
                "status": p.status.value if hasattr(p.status, "value") else str(getattr(p, "status", "")),
            })
        return {"pairs": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"signals/stat-arb error: {e}")
        return {"pairs": [], "error": str(e)}


@router.get("/stat-arb/pair")
async def stat_arb_pair(ticker_a: str, ticker_b: str):
    """Single pair signal detail."""
    try:
        sa = _get_stat_arb()
        p = sa.get_pair_signal(ticker_a, ticker_b)
        if p is None:
            return {"status": "not_found"}
        return {
            "ticker_a": p.ticker_a, "ticker_b": p.ticker_b,
            "correlation": p.correlation, "cointegration_pvalue": p.cointegration_pvalue,
            "half_life": p.half_life, "spread_zscore": p.spread_zscore,
            "hedge_ratio": p.hedge_ratio, "signal_strength": p.signal_strength,
            "status": p.status.value if hasattr(p.status, "value") else str(p.status),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"signals/stat-arb/pair error: {e}")
        return {"error": str(e)}


@router.get("/contagion")
async def contagion_state():
    """Cross-asset contagion graph: 21 nodes, shock propagation."""
    try:
        con = _get_contagion()
        # Try different method names the engine might expose
        state = {}
        if hasattr(con, "get_state"):
            state = con.get_state()
        elif hasattr(con, "analyze"):
            state = con.analyze()
        elif hasattr(con, "run"):
            state = con.run()
        return {**state, "timestamp": datetime.utcnow().isoformat()} if isinstance(state, dict) else {"data": str(state), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"signals/contagion error: {e}")
        return {"error": str(e)}


# ─── Event-driven signals ──────────────────────────────────

@router.get("/events")
async def event_signals():
    """Active event-driven signals: M&A arb, PEAD, catalysts."""
    try:
        ev = _get_event()
        if hasattr(ev, "get_active_events"):
            events = ev.get_active_events()
        elif hasattr(ev, "scan"):
            events = ev.scan()
        else:
            events = []

        result = []
        for e in (events if isinstance(events, list) else []):
            result.append({
                "ticker": getattr(e, "ticker", ""),
                "event_type": getattr(e, "event_type", ""),
                "signal": getattr(e, "signal", ""),
                "confidence": getattr(e, "confidence", 0),
                "expected_return": getattr(e, "expected_return", 0),
                "days_to_event": getattr(e, "days_to_event", 0),
            })
        return {"events": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"signals/events error: {e}")
        return {"events": [], "error": str(e)}


# ─── CVR signals ───────────────────────────────────────────

@router.get("/cvr")
async def cvr_signals():
    """CVR valuation: 5-model, 4 live instruments."""
    try:
        cvr = _get_cvr()
        if hasattr(cvr, "get_valuations"):
            vals = cvr.get_valuations()
        elif hasattr(cvr, "scan"):
            vals = cvr.scan()
        else:
            vals = {}
        return {**(vals if isinstance(vals, dict) else {"data": vals}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"signals/cvr error: {e}")
        return {"error": str(e)}


# ─── Distressed asset signals ──────────────────────────────

@router.get("/distressed")
async def distressed_signals():
    """Distressed asset scores: Altman Z, Merton, fallen angels."""
    try:
        dis = _get_distress()
        if hasattr(dis, "scan"):
            results = dis.scan()
        elif hasattr(dis, "get_signals"):
            results = dis.get_signals()
        else:
            results = {}
        return {**(results if isinstance(results, dict) else {"data": results}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"signals/distressed error: {e}")
        return {"error": str(e)}


# ─── Security analysis (Graham-Dodd) ──────────────────────

@router.get("/security-analysis")
async def security_analysis(ticker: str = Query(...)):
    """Graham-Dodd-Klarman analysis for a ticker."""
    try:
        sa = _get_security()
        if hasattr(sa, "analyze"):
            result = sa.analyze(ticker)
        elif hasattr(sa, "evaluate"):
            result = sa.evaluate(ticker)
        else:
            result = {}
        return {**(result if isinstance(result, dict) else {"data": str(result)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"signals/security-analysis error: {e}")
        return {"error": str(e)}


# ─── Pattern discovery ─────────────────────────────────────

@router.get("/patterns")
async def pattern_signals():
    """AI-Newton + MiroFish pattern discovery bus."""
    try:
        pat = _get_pattern()
        if hasattr(pat, "get_patterns"):
            patterns = pat.get_patterns()
        elif hasattr(pat, "scan"):
            patterns = pat.scan()
        else:
            patterns = {}
        return {**(patterns if isinstance(patterns, dict) else {"data": patterns}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"signals/patterns error: {e}")
        return {"error": str(e)}


# ─── Social prediction ─────────────────────────────────────

@router.get("/social")
async def social_signals():
    """MiroFish social prediction: agent sentiment, narrative regime."""
    try:
        soc = _get_social()
        if hasattr(soc, "get_snapshot"):
            snap = soc.get_snapshot()
        elif hasattr(soc, "analyze"):
            snap = soc.analyze()
        else:
            snap = {}
        return {**(snap if isinstance(snap, dict) else {"data": str(snap)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"signals/social error: {e}")
        return {"error": str(e)}
