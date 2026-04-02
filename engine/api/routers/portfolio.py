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


# ─── Market indices (LIVE + ALLOC tabs) ────────────────────

@router.get("/indices")
async def portfolio_indices():
    """Live market index quotes via OpenBB."""
    try:
        from engine.data.openbb_data import get_adj_close
        from datetime import timedelta
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")

        tickers = ["SPY", "QQQ", "IWM", "DIA"]
        indices = []
        for t in tickers:
            try:
                df = get_adj_close(t, start=start, end=end)
                if df.empty or len(df) < 2:
                    continue
                col = df.iloc[:, 0] if df.ndim > 1 else df
                price = float(col.iloc[-1])
                prev = float(col.iloc[-2])
                change = price - prev
                change_pct = (change / prev * 100) if prev else 0
                spark = [float(v) for v in col.tail(8).values]
                indices.append({
                    "ticker": t, "price": round(price, 2),
                    "change": round(change_pct, 2),
                    "data": spark,
                })
            except Exception:
                continue

        return {"indices": indices, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/indices error: {e}")
        return {"indices": [], "error": str(e)}


@router.get("/movers")
async def portfolio_movers():
    """Top movers from universe based on momentum."""
    try:
        from engine.data.universe_engine import UniverseEngine
        uni = UniverseEngine()
        uni.load_universe()
        all_secs = uni.get_all()

        # Sort by momentum_3m for top/bottom movers
        with_momentum = [s for s in all_secs if hasattr(s, "momentum_3m") and s.momentum_3m != 0]
        with_momentum.sort(key=lambda s: s.momentum_3m, reverse=True)

        movers = []
        for s in with_momentum[:5]:
            movers.append({
                "ticker": s.ticker,
                "change": round(s.momentum_3m * 100, 1),
                "momentum": "strong" if s.momentum_3m > 0.05 else "moderate" if s.momentum_3m > 0 else "weak",
            })
        for s in with_momentum[-3:]:
            movers.append({
                "ticker": s.ticker,
                "change": round(s.momentum_3m * 100, 1),
                "momentum": "weak",
            })

        return {"movers": movers, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/movers error: {e}")
        return {"movers": [], "error": str(e)}


@router.get("/sector-allocation")
async def portfolio_sector_allocation():
    """Sector allocation from current positions."""
    try:
        broker = _get_broker()
        positions = broker.get_positions()
        sector_weights: dict[str, float] = {}
        total_value = 0

        for ticker, pos in positions.items():
            val = abs(pos.quantity * pos.current_price) if hasattr(pos, "quantity") else 0
            sector = pos.sector if hasattr(pos, "sector") else "Unknown"
            sector_weights[sector] = sector_weights.get(sector, 0) + val
            total_value += val

        colors = ["#00d4aa", "#58a6ff", "#3fb950", "#bc8cff", "#f0883e", "#d29922", "#484f58", "#f85149", "#4ecdc4", "#da3633", "#a855f7"]
        allocation = []
        for i, (sector, val) in enumerate(sorted(sector_weights.items(), key=lambda x: -x[1])):
            pct = (val / total_value * 100) if total_value > 0 else 0
            allocation.append({
                "name": sector,
                "value": round(pct, 1),
                "color": colors[i % len(colors)],
            })

        return {"allocation": allocation, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/sector-allocation error: {e}")
        return {"allocation": [], "error": str(e)}


@router.get("/pnl-series")
async def portfolio_pnl_series():
    """Intraday P&L time-series from trade history."""
    try:
        broker = _get_broker()
        trades = broker.get_trades(limit=500)
        if not trades:
            return {"series": [], "timestamp": datetime.utcnow().isoformat()}

        # Aggregate P&L by time bucket (15-min intervals)
        from collections import defaultdict
        buckets: dict[str, float] = defaultdict(float)
        cumulative = 0
        for t in reversed(trades):
            ts = t.fill_timestamp if hasattr(t, "fill_timestamp") and t.fill_timestamp else None
            if not ts:
                continue
            bucket = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)[:5]
            pnl = getattr(t, "realized_pnl", 0) or (t.fill_price * t.quantity * (1 if str(getattr(t, "side", "")).upper() in ("SELL", "SHORT") else -1)) * 0.001
            cumulative += pnl
            buckets[bucket] = cumulative

        series = [{"time": k, "value": round(v, 2)} for k, v in sorted(buckets.items())]
        return {"series": series, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"portfolio/pnl-series error: {e}")
        return {"series": [], "error": str(e)}
