"""
Execution router — FUTURES, TCA tabs
Wraps: ExecutionEngine, L7UnifiedExecutionSurface, WondertraderEngine, DecisionMatrix
"""
from fastapi import APIRouter
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.execution")
router = APIRouter()

_exec_engine = None
_l7 = None
_wonder = None


def _get_exec():
    global _exec_engine
    if _exec_engine is None:
        from engine.execution.execution_engine import ExecutionEngine
        _exec_engine = ExecutionEngine()
    return _exec_engine


def _get_l7():
    global _l7
    if _l7 is None:
        try:
            from engine.execution.l7_unified_execution_surface import L7UnifiedExecutionSurface
            _l7 = L7UnifiedExecutionSurface()
        except Exception:
            _l7 = None
    return _l7


def _get_wonder():
    global _wonder
    if _wonder is None:
        try:
            from engine.execution.wondertrader_engine import WondertraderEngine
            _wonder = WondertraderEngine()
        except Exception:
            _wonder = None
    return _wonder


@router.get("/pipeline-status")
async def pipeline_status():
    """Current pipeline execution status."""
    try:
        eng = _get_exec()
        broker_status = eng.get_broker_status()
        nav = eng.get_nav()
        return {
            "nav": nav,
            "broker": broker_status if isinstance(broker_status, dict) else str(broker_status),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"execution/pipeline-status error: {e}")
        return {"error": str(e)}


@router.get("/reconciliation")
async def reconciliation():
    """Position reconciliation: engine vs broker ground truth."""
    try:
        eng = _get_exec()
        recon = eng.reconcile_positions()
        return {**(recon if isinstance(recon, dict) else {"data": str(recon)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/reconciliation error: {e}")
        return {"error": str(e)}


@router.get("/l7/status")
async def l7_status():
    """L7 unified execution surface status."""
    try:
        l7 = _get_l7()
        if l7 is None:
            return {"status": "not_available", "timestamp": datetime.utcnow().isoformat()}
        state = {}
        if hasattr(l7, "get_status"):
            state = l7.get_status()
        elif hasattr(l7, "status"):
            state = l7.status()
        return {**(state if isinstance(state, dict) else {"data": str(state)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/l7/status error: {e}")
        return {"error": str(e)}


@router.get("/wondertrader/status")
async def wondertrader_status():
    """WonderTrader CTA/HFT engine status."""
    try:
        wt = _get_wonder()
        if wt is None:
            return {"status": "not_available", "timestamp": datetime.utcnow().isoformat()}
        state = {}
        if hasattr(wt, "get_status"):
            state = wt.get_status()
        elif hasattr(wt, "status"):
            state = wt.status()
        return {**(state if isinstance(state, dict) else {"data": str(state)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/wondertrader error: {e}")
        return {"error": str(e)}


@router.get("/tca")
async def tca_metrics():
    """Transaction cost analysis: slippage, fill rates, cost trends."""
    try:
        eng = _get_exec()
        broker = eng.broker

        trades = broker.get_trades(limit=200)
        if not trades:
            return {"trades": [], "summary": {}, "timestamp": datetime.utcnow().isoformat()}

        total_slippage = 0
        total_trades = len(trades)
        filled = 0
        records = []

        for t in trades:
            slippage = getattr(t, "slippage", 0) or 0
            total_slippage += abs(slippage)
            if getattr(t, "status", "") == "FILLED" or hasattr(t, "fill_price"):
                filled += 1

            records.append({
                "ticker": getattr(t, "ticker", ""),
                "side": t.side.value if hasattr(t.side, "value") else str(getattr(t, "side", "")),
                "quantity": getattr(t, "quantity", 0),
                "fill_price": getattr(t, "fill_price", 0),
                "slippage": slippage,
                "signal_type": t.signal_type.value if hasattr(t.signal_type, "value") else str(getattr(t, "signal_type", "")),
                "timestamp": t.fill_timestamp.isoformat() if hasattr(t, "fill_timestamp") and t.fill_timestamp else "",
            })

        summary = {
            "total_trades": total_trades,
            "fill_rate": filled / total_trades if total_trades > 0 else 0,
            "avg_slippage": total_slippage / total_trades if total_trades > 0 else 0,
            "total_cost": total_slippage,
        }

        return {"trades": records[-100:], "summary": summary, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/tca error: {e}")
        return {"trades": [], "summary": {}, "error": str(e)}


@router.get("/spread-data")
async def spread_data():
    """Bid-ask spread estimates from L1 data layer (OpenBB price data → high-low range proxy).

    Fits in system: L1 Data → L7 Execution surface uses these for slippage estimation.
    """
    try:
        from engine.data.openbb_data import get_prices
        from datetime import timedelta
        import numpy as np

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
        df = get_prices("SPY", start=start, end=end, interval="1d")

        spreads = []
        if not df.empty:
            # Derive spread proxy from high-low range
            if hasattr(df.columns, "levels"):
                high = df["High"].iloc[:, 0] if "High" in df.columns.get_level_values(0) else None
                low = df["Low"].iloc[:, 0] if "Low" in df.columns.get_level_values(0) else None
                close = df["Close"].iloc[:, 0] if "Close" in df.columns.get_level_values(0) else None
            else:
                high = df.get("High") or df.get("high")
                low = df.get("Low") or df.get("low")
                close = df.get("Close") or df.get("close")

            if high is not None and low is not None and close is not None:
                for i in range(len(close)):
                    hl_range = float(high.iloc[i] - low.iloc[i])
                    spread_bps = (hl_range / float(close.iloc[i])) * 100 if float(close.iloc[i]) > 0 else 0
                    spreads.append({"time": str(i), "spread": round(spread_bps, 4)})

        return {"spreads": spreads, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/spread-data error: {e}")
        return {"spreads": [], "error": str(e)}


@router.get("/depth-data")
async def depth_data():
    """Market depth from L1 data layer (OpenBB volume profile → depth proxy).

    Fits in system: L1 Data → L7 ExchangeCore ring buffer uses depth for order matching.
    """
    try:
        from engine.data.openbb_data import get_prices
        from datetime import timedelta
        import numpy as np

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        df = get_prices("SPY", start=start, end=end)

        depth = []
        if not df.empty:
            if hasattr(df.columns, "levels"):
                close = df["Close"].iloc[:, 0] if "Close" in df.columns.get_level_values(0) else None
                volume = df["Volume"].iloc[:, 0] if "Volume" in df.columns.get_level_values(0) else None
            else:
                close = df.get("Close") or df.get("close")
                volume = df.get("Volume") or df.get("volume")

            if close is not None and volume is not None:
                prices = close.values.astype(float)
                volumes = volume.values.astype(float)
                current = prices[-1]
                # Build volume profile around current price
                step = current * 0.002  # 0.2% steps
                for i in range(20):
                    level = current - (10 - i) * step
                    # Volume at this level proportional to how often price was near it
                    near = np.sum(volumes[np.abs(prices - level) < step])
                    if i < 10:
                        depth.append({"price": f"{level:.0f}", "bidDepth": float(near), "askDepth": 0})
                    else:
                        depth.append({"price": f"{level:.0f}", "bidDepth": 0, "askDepth": float(near)})

        return {"depth": depth, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/depth-data error: {e}")
        return {"depth": [], "error": str(e)}


@router.get("/liquidity-data")
async def liquidity_data():
    """Bid-ask liquidity from L1 data layer (OpenBB volume → bid/ask estimate).

    Fits in system: L1 Data → L7 WonderTrader micro-price uses liquidity for execution routing.
    """
    try:
        from engine.data.openbb_data import get_prices
        from datetime import timedelta

        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        df = get_prices("SPY", start=start, end=end)

        liq = []
        if not df.empty:
            if hasattr(df.columns, "levels"):
                high = df["High"].iloc[:, 0] if "High" in df.columns.get_level_values(0) else None
                low = df["Low"].iloc[:, 0] if "Low" in df.columns.get_level_values(0) else None
                volume = df["Volume"].iloc[:, 0] if "Volume" in df.columns.get_level_values(0) else None
            else:
                high = df.get("High") or df.get("high")
                low = df.get("Low") or df.get("low")
                volume = df.get("Volume") or df.get("volume")

            if high is not None and low is not None and volume is not None:
                for i in range(min(30, len(high))):
                    h = float(high.iloc[i])
                    l = float(low.iloc[i])
                    v = float(volume.iloc[i])
                    mid = (h + l) / 2
                    liq.append({
                        "time": f"{mid:.1f}",
                        "bid": v * 0.48,  # ~48% buy volume estimate
                        "ask": v * 0.52,  # ~52% sell volume estimate
                    })

        return {"liquidity": liq, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"execution/liquidity-data error: {e}")
        return {"liquidity": [], "error": str(e)}
