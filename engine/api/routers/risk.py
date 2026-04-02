"""
Risk router — RISK tab
Wraps: BetaCorridor, DecisionMatrix, OptionsEngine
"""
from fastapi import APIRouter
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.risk")
router = APIRouter()

_options = None
_beta = None
_decision = None


def _get_options():
    global _options
    if _options is None:
        from engine.execution.options_engine import OptionsEngine
        _options = OptionsEngine()
    return _options


def _get_beta():
    global _beta
    if _beta is None:
        from engine.portfolio.beta_corridor import BetaCorridor
        _beta = BetaCorridor()
    return _beta


def _get_decision():
    global _decision
    if _decision is None:
        from engine.execution.decision_matrix import DecisionMatrix
        _decision = DecisionMatrix()
    return _decision


@router.get("/portfolio")
async def risk_portfolio():
    """Full risk metrics: VaR, beta, drawdown, Sharpe, Sortino."""
    try:
        beta = _get_beta()
        analytics = beta.get_corridor_analytics()
        history = beta.get_history()
        latest = history[-1] if history else None

        result = {
            "current_beta": latest.current_beta if latest and hasattr(latest, "current_beta") else 0,
            "target_beta": latest.target_beta if latest and hasattr(latest, "target_beta") else 0,
            "corridor_position": latest.corridor_position if latest and hasattr(latest, "corridor_position") else "UNKNOWN",
            "analytics": analytics if isinstance(analytics, dict) else {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        return result
    except Exception as e:
        logger.error(f"risk/portfolio error: {e}")
        return {"error": str(e)}


@router.get("/greeks")
async def risk_greeks():
    """Aggregate portfolio Greeks: delta, gamma, theta, vega, rho."""
    try:
        opt = _get_options()
        greeks = opt.get_portfolio_greeks()
        return {**greeks, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/greeks error: {e}")
        return {"error": str(e)}


@router.get("/options/hedge")
async def risk_options_hedge():
    """Current hedge requirements and cost drag."""
    try:
        opt = _get_options()
        hedge = opt.compute_hedge_requirements()
        return {**(hedge if isinstance(hedge, dict) else {}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/options/hedge error: {e}")
        return {"error": str(e)}


@router.get("/options/strategies")
async def risk_options_strategies():
    """Regime → strategy mapping."""
    try:
        opt = _get_options()
        matrix = opt.regime_strategy_matrix()
        return {"strategies": matrix, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/options/strategies error: {e}")
        return {"error": str(e)}


@router.get("/beta/stress")
async def risk_beta_stress():
    """Beta under stress scenarios."""
    try:
        beta = _get_beta()
        df = beta.stress_test_beta()
        result = df.to_dict(orient="records") if hasattr(df, "to_dict") else []
        return {"scenarios": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/beta/stress error: {e}")
        return {"scenarios": [], "error": str(e)}


@router.get("/beta/history")
async def risk_beta_history():
    """Historical beta time-series."""
    try:
        beta = _get_beta()
        df = beta.get_beta_history_df()
        if hasattr(df, "to_dict"):
            records = df.tail(200).reset_index().to_dict(orient="records")
            # Convert timestamps to strings
            for r in records:
                for k, v in r.items():
                    if hasattr(v, "isoformat"):
                        r[k] = v.isoformat()
            return {"history": records, "timestamp": datetime.utcnow().isoformat()}
        return {"history": [], "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/beta/history error: {e}")
        return {"history": [], "error": str(e)}


@router.get("/alerts")
async def risk_alerts():
    """Top risk alerts derived from current engine state."""
    try:
        beta = _get_beta()
        opt = _get_options()

        alerts = []
        # Beta corridor alerts
        history = beta.get_history()
        if history:
            latest = history[-1]
            b = latest.current_beta if hasattr(latest, "current_beta") else 0
            t = latest.target_beta if hasattr(latest, "target_beta") else 0
            pos = latest.corridor_position if hasattr(latest, "corridor_position") else "UNKNOWN"
            if pos == "ABOVE":
                alerts.append({"name": "Beta Above Corridor", "value": f"β={b:.2f} (target {t:.2f})", "severity": "high"})
            elif pos == "BELOW":
                alerts.append({"name": "Beta Below Corridor", "value": f"β={b:.2f} (target {t:.2f})", "severity": "medium"})

        # Options concentration
        greeks = opt.get_portfolio_greeks()
        if isinstance(greeks, dict):
            delta = greeks.get("delta", 0)
            if abs(delta) > 0.8:
                alerts.append({"name": "High Delta Exposure", "value": f"Δ={delta:.2f}", "severity": "high"})
            theta = greeks.get("theta", 0)
            if theta < -50:
                alerts.append({"name": "Theta Bleed", "value": f"Θ={theta:.1f}/day", "severity": "medium"})

        # Sector concentration from beta analytics
        analytics = beta.get_corridor_analytics()
        if isinstance(analytics, dict):
            vol_regime = analytics.get("vol_regime", "")
            if vol_regime == "HIGH":
                alerts.append({"name": "Elevated Volatility", "value": f"Vol regime: {vol_regime}", "severity": "high"})

        # Pad with default if empty
        if not alerts:
            alerts.append({"name": "No Active Alerts", "value": "System nominal", "severity": "low"})

        return {"alerts": alerts, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/alerts error: {e}")
        return {"alerts": [], "error": str(e)}


@router.get("/order-distribution")
async def order_distribution():
    """Order type distribution from recent trades."""
    try:
        beta = _get_beta()  # Use beta to get broker access
        # Get trades from the execution engine broker
        try:
            from engine.execution.execution_engine import ExecutionEngine
            eng = ExecutionEngine()
            trades = eng.broker.get_trades(limit=200)
        except Exception:
            trades = []

        if not trades:
            return {"distribution": [], "timestamp": datetime.utcnow().isoformat()}

        # Count signal types as order distribution
        type_counts: dict[str, int] = {}
        for t in trades:
            sig = t.signal_type.value if hasattr(t.signal_type, "value") else str(getattr(t, "signal_type", "UNKNOWN"))
            type_counts[sig] = type_counts.get(sig, 0) + 1

        total = sum(type_counts.values()) or 1
        colors = ["#00d4aa", "#58a6ff", "#f85149", "#bc8cff", "#d29922", "#4ecdc4", "#3fb950"]
        distribution = []
        for i, (name, count) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])):
            distribution.append({
                "name": name,
                "value": round(count / total * 100, 1),
                "count": count,
                "color": colors[i % len(colors)],
            })

        return {"distribution": distribution, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/order-distribution error: {e}")
        return {"distribution": [], "error": str(e)}
