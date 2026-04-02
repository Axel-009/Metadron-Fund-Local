"""
Universe router — OPENBB, ETF tabs
Wraps: UniverseEngine, OpenBBData
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.universe")
router = APIRouter()

_universe = None
_openbb = None


def _get_universe():
    global _universe
    if _universe is None:
        from engine.data.universe_engine import UniverseEngine
        _universe = UniverseEngine()
        _universe.load_universe()
    return _universe


def _get_openbb():
    global _openbb
    if _openbb is None:
        from engine.data.openbb_data import OpenBBData
        _openbb = OpenBBData()
    return _openbb


# ─── Universe ──────────────────────────────────────────────

@router.get("/securities")
async def universe_securities(
    sector: str = Query(None),
    limit: int = Query(100, ge=1, le=2000),
):
    """Universe securities with optional sector filter."""
    try:
        uni = _get_universe()
        if sector:
            secs = uni.get_by_sector(sector)
        else:
            secs = uni.get_all()

        result = []
        for s in secs[:limit]:
            result.append({
                "ticker": s.ticker if hasattr(s, "ticker") else "",
                "name": s.name if hasattr(s, "name") else "",
                "sector": s.sector if hasattr(s, "sector") else "",
                "gics_sector": s.gics_sector if hasattr(s, "gics_sector") else "",
                "industry_group": s.industry_group if hasattr(s, "industry_group") else "",
                "market_cap": s.market_cap if hasattr(s, "market_cap") else 0,
                "pe_ratio": s.pe_ratio if hasattr(s, "pe_ratio") else 0,
                "beta": s.beta if hasattr(s, "beta") else 0,
                "roe": s.roe if hasattr(s, "roe") else 0,
                "sharpe_12m": s.sharpe_12m if hasattr(s, "sharpe_12m") else 0,
                "momentum_3m": s.momentum_3m if hasattr(s, "momentum_3m") else 0,
                "quality_tier": s.quality_tier if hasattr(s, "quality_tier") else "?",
                "is_tradeable": s.is_fallen_angel if hasattr(s, "is_fallen_angel") else False,
                "options_eligible": s.options_eligible if hasattr(s, "options_eligible") else False,
            })

        return {
            "securities": result,
            "total": uni.size(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"universe/securities error: {e}")
        return {"securities": [], "error": str(e)}


@router.get("/sectors")
async def universe_sectors():
    """Sector list with counts and momentum."""
    try:
        uni = _get_universe()
        counts = uni.get_sector_counts()
        momentum = uni.get_sector_momentum()

        sectors = []
        for sector, count in counts.items():
            sectors.append({
                "sector": sector,
                "count": count,
                "momentum": momentum.get(sector, 0),
            })

        return {"sectors": sectors, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/sectors error: {e}")
        return {"sectors": [], "error": str(e)}


@router.get("/rv-pairs")
async def universe_rv_pairs():
    """Relative value pair candidates."""
    try:
        uni = _get_universe()
        pairs = uni.scan_rv_pairs()
        return {"pairs": pairs, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/rv-pairs error: {e}")
        return {"pairs": [], "error": str(e)}


@router.get("/morning-scan")
async def universe_morning_scan():
    """Daily universe scan: momentum, volume anomalies, sector flows."""
    try:
        uni = _get_universe()
        scan = uni.run_morning_scan()
        return {**(scan if isinstance(scan, dict) else {"data": str(scan)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/morning-scan error: {e}")
        return {"error": str(e)}


# ─── OpenBB data ───────────────────────────────────────────

@router.get("/openbb/search")
async def openbb_search(query: str = Query(..., min_length=1)):
    """Search securities via OpenBB."""
    try:
        obb = _get_openbb()
        if hasattr(obb, "search"):
            results = obb.search(query)
        elif hasattr(obb, "search_equity"):
            results = obb.search_equity(query)
        else:
            results = []
        return {"results": results if isinstance(results, list) else [], "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/openbb/search error: {e}")
        return {"results": [], "error": str(e)}


@router.get("/openbb/quote")
async def openbb_quote(ticker: str = Query(...)):
    """Get live quote via OpenBB."""
    try:
        obb = _get_openbb()
        if hasattr(obb, "get_quote"):
            quote = obb.get_quote(ticker)
        elif hasattr(obb, "quote"):
            quote = obb.quote(ticker)
        else:
            quote = {}
        return {**(quote if isinstance(quote, dict) else {"data": str(quote)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/openbb/quote error: {e}")
        return {"error": str(e)}


@router.get("/openbb/historical")
async def openbb_historical(
    ticker: str = Query(...),
    days: int = Query(252, ge=5, le=1260),
):
    """Get historical OHLCV via OpenBB."""
    try:
        obb = _get_openbb()
        if hasattr(obb, "get_historical"):
            data = obb.get_historical(ticker, lookback_days=days)
        elif hasattr(obb, "historical"):
            data = obb.historical(ticker, days=days)
        else:
            data = {}

        # Convert DataFrame to records if needed
        if hasattr(data, "to_dict"):
            records = data.reset_index().to_dict(orient="records")
            for r in records:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
            return {"data": records, "ticker": ticker, "timestamp": datetime.utcnow().isoformat()}

        return {"data": data if isinstance(data, (list, dict)) else str(data), "ticker": ticker, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/openbb/historical error: {e}")
        return {"data": [], "error": str(e)}


@router.get("/openbb/fred")
async def openbb_fred(series_id: str = Query(...)):
    """Get FRED economic data series via OpenBB."""
    try:
        obb = _get_openbb()
        if hasattr(obb, "get_fred"):
            data = obb.get_fred(series_id)
        elif hasattr(obb, "fred"):
            data = obb.fred(series_id)
        else:
            data = {}

        if hasattr(data, "to_dict"):
            records = data.reset_index().to_dict(orient="records")
            for r in records:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
            return {"data": records, "series_id": series_id, "timestamp": datetime.utcnow().isoformat()}

        return {"data": data if isinstance(data, (list, dict)) else str(data), "series_id": series_id, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"universe/openbb/fred error: {e}")
        return {"data": [], "error": str(e)}
