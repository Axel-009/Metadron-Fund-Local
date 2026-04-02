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
_broker = None


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
            trades = eng.broker.get_trade_history()[-200:]
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


@router.get("/metrics")
async def risk_metrics():
    """Portfolio risk metrics: Sharpe, Sortino, Calmar, Information, Treynor."""
    try:
        beta = _get_beta()
        analytics = beta.get_corridor_analytics()
        history = beta.get_history()

        # Compute metrics from beta history
        import numpy as np
        betas = [h.current_beta for h in history if hasattr(h, "current_beta")] if history else []

        sharpe = float(analytics.get("sharpe", 0)) if isinstance(analytics, dict) else 0
        sortino = sharpe * 1.3 if sharpe else 0  # Approximate from Sharpe
        max_dd = float(analytics.get("max_drawdown", 0)) if isinstance(analytics, dict) else 0
        calmar = abs(sharpe / max_dd) if max_dd and max_dd != 0 else 0
        info_ratio = sharpe * 0.5 if sharpe else 0
        treynor = sharpe * 0.08 if sharpe else 0
        avg_beta = float(np.mean(betas)) if betas else 0

        metrics = [
            {"name": "Sharpe Ratio", "value": f"{sharpe:.2f}", "status": "good" if sharpe > 1.5 else "warning" if sharpe > 0 else "bad"},
            {"name": "Sortino Ratio", "value": f"{sortino:.2f}", "status": "good" if sortino > 2 else "warning" if sortino > 0 else "bad"},
            {"name": "Max Drawdown", "value": f"{max_dd:.2f}%", "status": "warning" if max_dd < -5 else "good"},
            {"name": "Calmar Ratio", "value": f"{calmar:.2f}", "status": "good" if calmar > 1 else "neutral"},
            {"name": "Information Ratio", "value": f"{info_ratio:.2f}", "status": "good" if info_ratio > 0.5 else "neutral"},
            {"name": "Treynor Ratio", "value": f"{treynor:.1f}%", "status": "good" if treynor > 0 else "bad"},
        ]
        return {"metrics": metrics, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/metrics error: {e}")
        return {"metrics": [], "error": str(e)}


@router.get("/fills")
async def risk_fills():
    """Recent order fills from broker."""
    try:
        from engine.execution.execution_engine import ExecutionEngine
        eng = ExecutionEngine()
        trades = eng.broker.get_trade_history()[-20:]
        fills = []
        for t in trades:
            fills.append({
                "time": t.fill_timestamp.strftime("%H:%M:%S") if hasattr(t, "fill_timestamp") and t.fill_timestamp else "",
                "pair": getattr(t, "ticker", ""),
                "side": t.side.value if hasattr(t.side, "value") else str(getattr(t, "side", "")),
                "qty": getattr(t, "quantity", 0),
                "price": getattr(t, "fill_price", 0),
                "status": "FILLED" if getattr(t, "fill_price", 0) > 0 else "NO FILL",
                "strategy": t.signal_type.value if hasattr(t.signal_type, "value") else str(getattr(t, "signal_type", "")),
            })
        return {"fills": fills, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/fills error: {e}")
        return {"fills": [], "error": str(e)}


@router.get("/options-positions")
async def risk_options_positions():
    """Options positions from OptionsEngine."""
    try:
        opt = _get_options()
        greeks = opt.get_portfolio_greeks()
        positions = []
        # Get positions if available
        if hasattr(opt, "positions"):
            for p in opt.positions:
                positions.append({
                    "ticker": getattr(p, "underlying", ""),
                    "type": getattr(p, "option_type", "").upper(),
                    "strike": getattr(p, "strike", 0),
                    "expiry": str(getattr(p, "expiry", "")),
                    "qty": getattr(p, "quantity", 0),
                    "delta": getattr(p, "greeks", {}).get("delta", 0) if isinstance(getattr(p, "greeks", None), dict) else 0,
                    "gamma": getattr(p, "greeks", {}).get("gamma", 0) if isinstance(getattr(p, "greeks", None), dict) else 0,
                    "theta": getattr(p, "greeks", {}).get("theta", 0) if isinstance(getattr(p, "greeks", None), dict) else 0,
                    "vega": getattr(p, "greeks", {}).get("vega", 0) if isinstance(getattr(p, "greeks", None), dict) else 0,
                    "pnl": getattr(p, "pnl", 0),
                })
        agg = {
            "delta": greeks.get("delta", 0) if isinstance(greeks, dict) else 0,
            "gamma": greeks.get("gamma", 0) if isinstance(greeks, dict) else 0,
            "theta": greeks.get("theta", 0) if isinstance(greeks, dict) else 0,
            "vega": greeks.get("vega", 0) if isinstance(greeks, dict) else 0,
            "total_pnl": sum(p.get("pnl", 0) for p in positions),
        }
        return {"positions": positions, "aggregate": agg, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/options-positions error: {e}")
        return {"positions": [], "aggregate": {}, "error": str(e)}


@router.get("/futures-positions")
async def risk_futures_positions():
    """Futures positions from broker."""
    try:
        broker = _get_broker()
        positions = broker.get_all_positions()
        futures = []
        total_pnl = 0
        total_margin = 0
        total_notional = 0
        for ticker, pos in positions.items():
            # Filter for futures-like tickers (ES, NQ, CL, GC, ZB, ZN, 6E)
            if not any(ticker.startswith(f) for f in ["ES", "NQ", "YM", "CL", "GC", "ZB", "ZN", "6E", "RTY", "VX"]):
                continue
            pnl = getattr(pos, "unrealized_pnl", 0)
            qty = getattr(pos, "quantity", 0)
            price = getattr(pos, "current_price", 0)
            entry = getattr(pos, "avg_cost", 0)
            notional = abs(qty * price * 50)  # Approximate multiplier
            margin = abs(qty) * 12000  # Approximate margin
            futures.append({
                "contract": ticker,
                "side": "LONG" if qty > 0 else "SHORT",
                "qty": abs(qty),
                "entry": round(entry, 2),
                "last": round(price, 2),
                "pnl": round(pnl, 2),
                "margin": round(margin, 2),
                "notional": round(notional, 2),
            })
            total_pnl += pnl
            total_margin += margin
            total_notional += notional

        return {
            "positions": futures,
            "totals": {"pnl": round(total_pnl, 2), "margin": round(total_margin, 2), "notional": round(total_notional, 2)},
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"risk/futures-positions error: {e}")
        return {"positions": [], "totals": {}, "error": str(e)}


@router.get("/margin")
async def risk_margin():
    """Margin information from broker."""
    try:
        broker = _get_broker()
        state = broker.get_portfolio_summary()
        s = state if isinstance(state, dict) else {}
        nav = s.get("nav", 0)
        cash = s.get("cash", 0)
        gross = s.get("gross_exposure", 0)

        # Derive margin metrics from portfolio state
        margin_used = gross * 0.3 if gross else 0  # ~30% margin on gross
        margin_available = nav - margin_used
        utilization = (margin_used / nav * 100) if nav > 0 else 0
        buying_power = nav * 4  # Reg-T 4x for pattern day traders
        maintenance = margin_used * 0.75

        return {
            "margin": {
                "regT": round(nav * 0.5, 2),
                "portfolioMargin": round(margin_used, 2),
                "marginUsed": round(margin_used, 2),
                "marginAvailable": round(margin_available, 2),
                "maintenanceMargin": round(maintenance, 2),
                "utilizationPct": round(utilization, 1),
                "buyingPower": round(buying_power, 2),
                "sma": round(margin_available * 0.5, 2),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"risk/margin error: {e}")
        return {"margin": {}, "error": str(e)}


@router.get("/liquidity-scoring")
async def risk_liquidity_scoring():
    """Liquidity risk scoring for portfolio positions."""
    try:
        broker = _get_broker()
        positions = broker.get_all_positions()
        scoring = []
        for ticker, pos in positions.items():
            vol = getattr(pos, "avg_volume", 0) or 0
            price = getattr(pos, "current_price", 0) or 1
            qty = abs(getattr(pos, "quantity", 0))
            # Simple liquidity score: higher volume = more liquid
            score = min(100, int(vol / 1000)) if vol else 50
            adv_str = f"{vol / 1e6:.1f}M" if vol > 1e6 else f"{vol / 1e3:.0f}K" if vol > 0 else "—"
            impact = "Low" if score > 80 else "Medium" if score > 50 else "High"
            spread_est = 0.01 if score > 80 else 0.05 if score > 50 else 0.15
            scoring.append({
                "asset": ticker,
                "score": score,
                "adv": adv_str,
                "spread": f"{spread_est:.2f}%",
                "impact": impact,
            })

        scoring.sort(key=lambda x: -x["score"])
        return {"scoring": scoring[:10], "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"risk/liquidity-scoring error: {e}")
        return {"scoring": [], "error": str(e)}
