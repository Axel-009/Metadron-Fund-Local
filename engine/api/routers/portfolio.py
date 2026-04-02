"""
Portfolio router — LIVE, ALLOC, TXLOG tabs
Wraps: PaperBroker, AlpacaBroker, AlphaOptimizer, BetaCorridor
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging
import traceback

logger = logging.getLogger("metadron-api.portfolio")
router = APIRouter()

# ─── Lazy engine singletons ────────────────────────────────
_broker = None
_alpha = None
_beta = None


def _get_broker():
    global _broker
    if _broker is None:
        try:
            from engine.execution.execution_engine import ExecutionEngine
            eng = ExecutionEngine()
            _broker = eng.broker
        except Exception:
            from engine.execution.paper_broker import PaperBroker
            _broker = PaperBroker()
    return _broker


def _get_alpha():
    global _alpha
    if _alpha is None:
        from engine.ml.alpha_optimizer import AlphaOptimizer
        _alpha = AlphaOptimizer()
    return _alpha


def _get_beta():
    global _beta
    if _beta is None:
        from engine.portfolio.beta_corridor import BetaCorridor
        _beta = BetaCorridor()
    return _beta


# ─── LIVE tab endpoints ────────────────────────────────────

@router.get("/live")
async def portfolio_live():
    """Current portfolio state: NAV, P&L, positions, exposure."""
    try:
        broker = _get_broker()
        state = broker.get_portfolio_state()
        return {
            "nav": state.nav if hasattr(state, "nav") else getattr(state, "cash", 0),
            "cash": state.cash if hasattr(state, "cash") else 0,
            "total_pnl": state.total_pnl if hasattr(state, "total_pnl") else 0,
            "gross_exposure": state.gross_exposure if hasattr(state, "gross_exposure") else 0,
            "net_exposure": state.net_exposure if hasattr(state, "net_exposure") else 0,
            "positions_count": len(state.positions) if hasattr(state, "positions") else 0,
            "win_count": state.win_count if hasattr(state, "win_count") else 0,
            "loss_count": state.loss_count if hasattr(state, "loss_count") else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"portfolio/live error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/positions")
async def portfolio_positions():
    """All current positions with P&L."""
    try:
        broker = _get_broker()
        positions = broker.get_positions()
        result = []
        for ticker, pos in positions.items():
            result.append({
                "ticker": ticker,
                "quantity": pos.quantity if hasattr(pos, "quantity") else 0,
                "avg_cost": pos.avg_cost if hasattr(pos, "avg_cost") else 0,
                "current_price": pos.current_price if hasattr(pos, "current_price") else 0,
                "unrealized_pnl": pos.unrealized_pnl if hasattr(pos, "unrealized_pnl") else 0,
                "realized_pnl": pos.realized_pnl if hasattr(pos, "realized_pnl") else 0,
                "sector": pos.sector if hasattr(pos, "sector") else "Unknown",
            })
        return {"positions": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/positions error: {e}")
        return {"positions": [], "error": str(e)}


@router.get("/trades")
async def portfolio_trades(limit: int = Query(50, ge=1, le=500)):
    """Recent trade history."""
    try:
        broker = _get_broker()
        trades = broker.get_trades(limit=limit)
        result = []
        for t in trades:
            result.append({
                "id": str(t.id) if hasattr(t, "id") else "",
                "ticker": t.ticker if hasattr(t, "ticker") else "",
                "side": t.side.value if hasattr(t.side, "value") else str(getattr(t, "side", "")),
                "quantity": t.quantity if hasattr(t, "quantity") else 0,
                "fill_price": t.fill_price if hasattr(t, "fill_price") else 0,
                "signal_type": t.signal_type.value if hasattr(t.signal_type, "value") else str(getattr(t, "signal_type", "")),
                "timestamp": t.fill_timestamp.isoformat() if hasattr(t, "fill_timestamp") and t.fill_timestamp else "",
                "reason": t.reason if hasattr(t, "reason") else "",
            })
        return {"trades": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/trades error: {e}")
        return {"trades": [], "error": str(e)}


# ─── ALLOC tab endpoints ───────────────────────────────────

@router.get("/allocation")
async def portfolio_allocation():
    """Current sleeve allocation and sector weights."""
    try:
        alpha = _get_alpha()
        last = alpha.get_last()
        if last is None:
            return {"allocation": {}, "weights": {}, "status": "no_optimization_run"}

        signals = []
        for s in (last.signals or []):
            signals.append({
                "ticker": s.ticker,
                "weight": s.weight if hasattr(s, "weight") else 0,
                "quality_tier": s.quality_tier if hasattr(s, "quality_tier") else "?",
                "alpha_pred": s.alpha_pred if hasattr(s, "alpha_pred") else 0,
                "sharpe_estimate": s.sharpe_estimate if hasattr(s, "sharpe_estimate") else 0,
                "momentum_3m": s.momentum_3m if hasattr(s, "momentum_3m") else 0,
            })

        return {
            "signals": signals,
            "optimal_weights": last.optimal_weights if hasattr(last, "optimal_weights") else {},
            "expected_return": last.expected_annual_return if hasattr(last, "expected_annual_return") else 0,
            "annual_volatility": last.annual_volatility if hasattr(last, "annual_volatility") else 0,
            "sharpe_ratio": last.sharpe_ratio if hasattr(last, "sharpe_ratio") else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"portfolio/allocation error: {e}")
        return {"error": str(e)}


@router.get("/beta")
async def portfolio_beta():
    """Beta corridor state and analytics."""
    try:
        beta = _get_beta()
        analytics = beta.get_corridor_analytics()
        history = beta.get_history()
        latest = history[-1] if history else None

        state = {}
        if latest:
            state = {
                "current_beta": latest.current_beta if hasattr(latest, "current_beta") else 0,
                "target_beta": latest.target_beta if hasattr(latest, "target_beta") else 0,
                "corridor_position": latest.corridor_position if hasattr(latest, "corridor_position") else "UNKNOWN",
                "vol_adjustment": latest.vol_adjustment if hasattr(latest, "vol_adjustment") else 0,
            }

        return {
            "state": state,
            "analytics": analytics,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"portfolio/beta error: {e}")
        return {"error": str(e)}


@router.get("/exposures")
async def portfolio_exposures():
    """Gross/net exposure breakdown."""
    try:
        broker = _get_broker()
        exposures = broker.compute_exposures()
        return {**exposures, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/exposures error: {e}")
        return {"error": str(e)}
