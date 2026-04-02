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
