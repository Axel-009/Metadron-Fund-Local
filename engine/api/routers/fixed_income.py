"""
Fixed Income router — FI dashboard tab
Wraps: FixedIncomeEngine (treasury rates, credit quality, duration, spreads)
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.fixed-income")
router = APIRouter()

_fi_engine = None


def _get_fi():
    global _fi_engine
    if _fi_engine is None:
        from engine.signals.fixed_income_engine import FixedIncomeEngine
        _fi_engine = FixedIncomeEngine()
    return _fi_engine


# ─── Fixed Income tab endpoints ───────────────────────────

@router.get("/summary")
async def fi_summary():
    """Portfolio-level FI summary: total AUM, avg yield, duration, rating."""
    try:
        fi = _get_fi()
        result = fi.get_summary()
        return {**result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/summary error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/holdings")
async def fi_holdings():
    """Bond holdings list from broker positions."""
    try:
        fi = _get_fi()
        holdings = fi.get_holdings()
        return {"holdings": holdings, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/holdings error: {e}")
        return {"holdings": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/yield-curve")
async def fi_yield_curve():
    """Real yield curve from FRED treasury rates across all tenors."""
    try:
        fi = _get_fi()
        curve = fi.get_yield_curve()
        return {"curve": curve, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/yield-curve error: {e}")
        return {"curve": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/credit-quality")
async def fi_credit_quality():
    """Credit quality distribution."""
    try:
        fi = _get_fi()
        quality = fi.get_credit_quality()
        return {"quality": quality, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/credit-quality error: {e}")
        return {"quality": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/duration-ladder")
async def fi_duration_ladder():
    """Duration/maturity bucket distribution."""
    try:
        fi = _get_fi()
        ladder = fi.get_duration_ladder()
        return {"ladder": ladder, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/duration-ladder error: {e}")
        return {"ladder": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/spread-history")
async def fi_spread_history(days: int = Query(default=90, ge=1, le=365)):
    """Historical IG and HY credit spread data."""
    try:
        fi = _get_fi()
        history = fi.get_spread_history(days=days)
        return {**history, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"fixed-income/spread-history error: {e}")
        return {"dates": [], "ig_spread": [], "hy_spread": [], "error": str(e), "timestamp": datetime.utcnow().isoformat()}
