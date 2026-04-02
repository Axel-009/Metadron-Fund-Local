"""
Cube router — CUBE tab
Wraps: MetadronCube
"""
from fastapi import APIRouter
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.cube")
router = APIRouter()

_cube = None


def _get_cube():
    global _cube
    if _cube is None:
        from engine.signals.metadron_cube import MetadronCube
        _cube = MetadronCube()
    return _cube


@router.get("/state")
async def cube_state():
    """Current C(t) = f(L,R,F) state — regime, tensors, sleeves."""
    try:
        cube = _get_cube()
        last = cube.get_last()
        if last is None:
            return {"status": "no_computation", "timestamp": datetime.utcnow().isoformat()}

        # Extract liquidity tensor
        liq = {}
        if hasattr(last, "liquidity") and last.liquidity:
            l = last.liquidity
            liq = {
                "score": l.sofr_signal if hasattr(l, "sofr_signal") else 0,
                "credit_impulse": l.credit_impulse if hasattr(l, "credit_impulse") else 0,
                "m2_velocity": l.m2_velocity if hasattr(l, "m2_velocity") else 0,
                "hy_spread_z": l.hy_spread_z if hasattr(l, "hy_spread_z") else 0,
                "fed_funds_impact": l.fed_funds_impact if hasattr(l, "fed_funds_impact") else 0,
                "reverse_repo_signal": l.reverse_repo_signal if hasattr(l, "reverse_repo_signal") else 0,
                "tga_balance_signal": l.tga_balance_signal if hasattr(l, "tga_balance_signal") else 0,
                "reserve_flow": l.reserve_flow if hasattr(l, "reserve_flow") else 0,
            }

        # Extract risk state
        risk = {}
        if hasattr(last, "risk") and last.risk:
            r = last.risk
            risk = {
                "score": r.vix_component if hasattr(r, "vix_component") else 0,
                "realized_vol": r.realized_vol if hasattr(r, "realized_vol") else 0,
                "credit_spread": r.credit_spread_component if hasattr(r, "credit_spread_component") else 0,
                "vix_term_structure": r.vix_term_structure if hasattr(r, "vix_term_structure") else 0,
                "correlation_stress": r.correlation_stress if hasattr(r, "correlation_stress") else 0,
                "tail_risk": r.tail_risk if hasattr(r, "tail_risk") else 0,
            }

        # Extract sleeve allocation
        sleeves = {}
        if hasattr(last, "sleeves") and last.sleeves:
            s = last.sleeves
            sleeves = {
                "p1_directional_equity": s.p1_directional_equity if hasattr(s, "p1_directional_equity") else 0.40,
                "p2_factor_rotation": s.p2_factor_rotation if hasattr(s, "p2_factor_rotation") else 0.10,
                "p3_commodities_macro": s.p3_commodities_macro if hasattr(s, "p3_commodities_macro") else 0.10,
                "p4_options_convexity": s.p4_options_convexity if hasattr(s, "p4_options_convexity") else 0.25,
                "p5_hedges_volatility": s.p5_hedges_volatility if hasattr(s, "p5_hedges_volatility") else 0.10,
            }

        return {
            "regime": last.regime if hasattr(last, "regime") else "UNKNOWN",
            "liquidity": liq,
            "risk": risk,
            "sleeves": sleeves,
            "target_beta": last.target_beta if hasattr(last, "target_beta") else 0,
            "max_leverage": last.max_leverage if hasattr(last, "max_leverage") else 1.0,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"cube/state error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/history")
async def cube_history():
    """Historical cube states for time-series charts."""
    try:
        cube = _get_cube()
        history = cube.get_history()
        if not history:
            return {"history": [], "timestamp": datetime.utcnow().isoformat()}

        entries = getattr(history, "entries", []) if hasattr(history, "entries") else history
        result = []
        for h in entries[-100:]:  # Last 100 entries
            result.append({
                "regime": h.regime if hasattr(h, "regime") else "UNKNOWN",
                "target_beta": h.target_beta if hasattr(h, "target_beta") else 0,
                "max_leverage": h.max_leverage if hasattr(h, "max_leverage") else 1.0,
            })

        return {"history": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"cube/history error: {e}")
        return {"history": [], "error": str(e)}


@router.get("/gates")
async def cube_gates():
    """4-gate entry logic scores."""
    try:
        cube = _get_cube()
        last = cube.get_last()
        if last is None:
            return {"gates": {}, "status": "no_computation"}

        gates = {}
        if hasattr(last, "gate_scores"):
            gates = last.gate_scores
        elif hasattr(last, "gates"):
            gates = last.gates

        return {"gates": gates, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"cube/gates error: {e}")
        return {"gates": {}, "error": str(e)}


@router.get("/stress-tests")
async def cube_stress_tests():
    """Run cube stress test scenarios."""
    try:
        cube = _get_cube()
        results = cube.run_stress_tests()
        return {"stress_tests": results, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"cube/stress-tests error: {e}")
        return {"stress_tests": {}, "error": str(e)}


@router.get("/kill-switch")
async def cube_kill_switch():
    """Kill-switch status."""
    try:
        cube = _get_cube()
        last = cube.get_last()
        if last is None:
            return {"active": False, "status": "no_computation"}

        return {
            "active": getattr(last, "kill_switch_active", False),
            "triggers": getattr(last, "kill_switch_triggers", []),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"cube/kill-switch error: {e}")
        return {"active": False, "error": str(e)}
