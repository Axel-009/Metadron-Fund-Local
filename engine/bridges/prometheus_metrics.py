"""
Metadron Capital — Prometheus Metrics Endpoint

A FastAPI router that exports Prometheus-format metrics for Grafana,
Datadog, or any Prometheus-compatible monitoring stack.

Mount this router on the engine API server:
    from engine.bridges.prometheus_metrics import create_metrics_router
    app.include_router(create_metrics_router(app))

The /metrics endpoint returns metrics in Prometheus text exposition format.

Requires: prometheus_client (pip install prometheus-client)
"""

import os
import sys
import time
import logging
from typing import Optional

logger = logging.getLogger("prometheus-metrics")

# ─── Prometheus Client Availability ────────────────────────────────

_prometheus_available = False
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _prometheus_available = True
except ImportError:
    logger.warning("prometheus_client not installed — metrics endpoint will return 503")


# ─── Metric Definitions ───────────────────────────────────────────

def _create_metrics(registry: "CollectorRegistry"):
    """Create all Prometheus metric objects in the given registry."""
    metrics = {}

    # Engine health
    metrics["engine_up"] = Gauge(
        "metadron_engine_up",
        "Whether the Metadron engine API is running (1=up, 0=down)",
        registry=registry,
    )

    # API request counters and latency
    metrics["api_requests_total"] = Counter(
        "metadron_api_requests_total",
        "Total API requests by endpoint, method, and status",
        ["endpoint", "method", "status"],
        registry=registry,
    )
    metrics["api_duration_seconds"] = Histogram(
        "metadron_api_duration_seconds",
        "API request duration in seconds by endpoint",
        ["endpoint"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        registry=registry,
    )

    # Portfolio metrics
    metrics["portfolio_nav"] = Gauge(
        "metadron_portfolio_nav",
        "Current portfolio net asset value",
        registry=registry,
    )
    metrics["portfolio_pnl_daily"] = Gauge(
        "metadron_portfolio_pnl_daily",
        "Daily portfolio profit and loss",
        registry=registry,
    )
    metrics["positions_count"] = Gauge(
        "metadron_positions_count",
        "Number of open positions",
        registry=registry,
    )

    # Cube/regime metrics
    metrics["cube_signal_score"] = Gauge(
        "metadron_cube_signal_score",
        "MetadronCube composite signal score",
        registry=registry,
    )
    metrics["cube_regime"] = Gauge(
        "metadron_cube_regime",
        "MetadronCube regime state (1=active for that regime)",
        ["regime_name"],
        registry=registry,
    )

    # Trade metrics
    metrics["trades_total"] = Counter(
        "metadron_trades_total",
        "Total trades executed by side",
        ["side"],
        registry=registry,
    )

    # OpenBB data metrics
    metrics["openbb_requests_total"] = Counter(
        "metadron_openbb_requests_total",
        "Total OpenBB data requests by endpoint",
        ["endpoint"],
        registry=registry,
    )
    metrics["openbb_errors_total"] = Counter(
        "metadron_openbb_errors_total",
        "Total OpenBB data errors by endpoint",
        ["endpoint"],
        registry=registry,
    )

    # LLM metrics
    metrics["llm_requests_total"] = Counter(
        "metadron_llm_requests_total",
        "Total LLM inference requests by backend",
        ["backend"],
        registry=registry,
    )
    metrics["llm_duration_seconds"] = Histogram(
        "metadron_llm_duration_seconds",
        "LLM inference duration in seconds by backend",
        ["backend"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        registry=registry,
    )

    # STRAT engine health gauges (1=healthy, 0=degraded/offline)
    metrics["strat_engine_health"] = Gauge(
        "metadron_strat_engine_health",
        "STRAT engine health (1=healthy, 0=degraded)",
        ["engine"],
        registry=registry,
    )

    # VolatilitySurface metrics
    metrics["vol_surface_iv"] = Gauge(
        "metadron_vol_surface_iv",
        "VolatilitySurface current implied volatility",
        registry=registry,
    )
    metrics["vol_surface_skew"] = Gauge(
        "metadron_vol_surface_skew",
        "VolatilitySurface skew value",
        registry=registry,
    )
    metrics["vol_surface_term_structure"] = Gauge(
        "metadron_vol_surface_term_structure",
        "VolatilitySurface term-structure slope",
        registry=registry,
    )

    # StatArbEngine metrics
    metrics["stat_arb_pairs_count"] = Gauge(
        "metadron_stat_arb_pairs_count",
        "Number of cointegrated pairs tracked by StatArbEngine",
        registry=registry,
    )
    metrics["stat_arb_active_trades"] = Gauge(
        "metadron_stat_arb_active_trades",
        "Number of active stat-arb trades",
        registry=registry,
    )
    metrics["stat_arb_portfolio_beta"] = Gauge(
        "metadron_stat_arb_portfolio_beta",
        "StatArbEngine portfolio beta",
        registry=registry,
    )
    metrics["stat_arb_mean_zscore"] = Gauge(
        "metadron_stat_arb_mean_zscore",
        "Mean z-score across all stat-arb pairs",
        registry=registry,
    )

    # MLVoteEnsemble metrics
    metrics["ml_ensemble_vote_bullish"] = Gauge(
        "metadron_ml_ensemble_vote_bullish",
        "MLVoteEnsemble bullish tier count",
        registry=registry,
    )
    metrics["ml_ensemble_vote_bearish"] = Gauge(
        "metadron_ml_ensemble_vote_bearish",
        "MLVoteEnsemble bearish tier count",
        registry=registry,
    )
    metrics["ml_ensemble_confidence"] = Gauge(
        "metadron_ml_ensemble_confidence",
        "MLVoteEnsemble aggregate confidence score",
        registry=registry,
    )

    # DecisionMatrix metrics
    metrics["decision_matrix_gates_passed"] = Gauge(
        "metadron_decision_matrix_gates_passed",
        "DecisionMatrix gates currently passing",
        registry=registry,
    )
    metrics["decision_matrix_gates_total"] = Gauge(
        "metadron_decision_matrix_gates_total",
        "DecisionMatrix total configured gates",
        registry=registry,
    )
    metrics["decision_matrix_approval_rate"] = Gauge(
        "metadron_decision_matrix_approval_rate",
        "DecisionMatrix trade approval rate (0-1)",
        registry=registry,
    )
    metrics["decision_matrix_evaluations_total"] = Counter(
        "metadron_decision_matrix_evaluations_total",
        "Total DecisionMatrix evaluations by result",
        ["result"],
        registry=registry,
    )

    # MetadronCube confidence & sleeve allocation
    metrics["cube_regime_confidence"] = Gauge(
        "metadron_cube_regime_confidence",
        "MetadronCube regime confidence score (0-1)",
        registry=registry,
    )
    metrics["cube_sleeve_weight"] = Gauge(
        "metadron_cube_sleeve_weight",
        "MetadronCube sleeve allocation weight",
        ["sleeve"],
        registry=registry,
    )

    # PatternRecognitionEngine metrics
    metrics["pattern_recognition_patterns_detected"] = Gauge(
        "metadron_pattern_recognition_patterns_detected",
        "Number of active patterns detected",
        registry=registry,
    )
    metrics["pattern_recognition_confidence"] = Gauge(
        "metadron_pattern_recognition_confidence",
        "Mean pattern recognition confidence",
        registry=registry,
    )

    # PM2 process metrics
    metrics["pm2_process_memory_bytes"] = Gauge(
        "metadron_pm2_process_memory_bytes",
        "PM2 process memory usage in bytes",
        ["process"],
        registry=registry,
    )
    metrics["pm2_process_restarts"] = Gauge(
        "metadron_pm2_process_restarts",
        "PM2 process restart count",
        ["process"],
        registry=registry,
    )

    # TXLOG / Trade execution metrics
    metrics["txlog_orders_total"] = Gauge(
        "metadron_txlog_orders_total",
        "Total orders in current session",
        registry=registry,
    )
    metrics["txlog_fill_rate"] = Gauge(
        "metadron_txlog_fill_rate",
        "Order fill rate (0-1)",
        registry=registry,
    )
    metrics["txlog_reject_rate"] = Gauge(
        "metadron_txlog_reject_rate",
        "Order rejection rate (0-1)",
        registry=registry,
    )
    metrics["txlog_avg_latency_ms"] = Gauge(
        "metadron_txlog_avg_latency_ms",
        "Average order fill latency in milliseconds",
        registry=registry,
    )
    metrics["txlog_avg_slippage_bps"] = Gauge(
        "metadron_txlog_avg_slippage_bps",
        "Average order slippage in basis points",
        registry=registry,
    )
    metrics["txlog_notional_volume"] = Gauge(
        "metadron_txlog_notional_volume",
        "Total notional volume of executed orders",
        registry=registry,
    )
    metrics["txlog_orders_by_side"] = Gauge(
        "metadron_txlog_orders_by_side",
        "Order count by side (BUY/SELL/SHORT/COVER)",
        ["side"],
        registry=registry,
    )

    return metrics


# ─── Collector: pull live data into gauges ─────────────────────────

def _collect_live_metrics(metrics: dict):
    """Populate gauge metrics from live engine state.

    Called on each /metrics scrape to ensure values are current.
    """
    # Engine up
    metrics["engine_up"].set(1)

    # Portfolio — try to read from engine singletons
    try:
        from engine.execution.execution_engine import ExecutionEngine
        engine = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
        if engine and hasattr(engine, "broker"):
            broker = engine.broker
            summary = broker.get_portfolio_summary() if hasattr(broker, "get_portfolio_summary") else {}
            metrics["portfolio_nav"].set(summary.get("nav", 0))
            metrics["portfolio_pnl_daily"].set(summary.get("total_pnl", 0))
            metrics["positions_count"].set(summary.get("positions_count", 0))
    except Exception:
        pass

    # Cube regime
    try:
        from engine.signals.metadron_cube import MetadronCube
        # Look for a cached cube state file
        import json
        from pathlib import Path
        cache_path = Path(__file__).resolve().parent.parent.parent / "data" / "cube_state_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                state = json.load(f)
            regime = state.get("regime", "RANGE")
            for r in ["TRENDING", "RANGE", "STRESS", "CRASH"]:
                metrics["cube_regime"].labels(regime_name=r).set(1 if r == regime else 0)
            # Composite score from liquidity
            metrics["cube_signal_score"].set(state.get("liquidity", 0))
    except Exception:
        pass

    # ── STRAT Engine Health Collectors ─────────────────────────────

    # VolatilitySurface
    try:
        from engine.execution.options_engine import VolatilitySurface
        vs = VolatilitySurface()
        metrics["strat_engine_health"].labels(engine="VolatilitySurface").set(1)
        # Pull live surface data if available
        if hasattr(vs, "current_iv"):
            metrics["vol_surface_iv"].set(vs.current_iv or 0)
        if hasattr(vs, "skew"):
            metrics["vol_surface_skew"].set(vs.skew or 0)
        if hasattr(vs, "term_structure_slope"):
            metrics["vol_surface_term_structure"].set(vs.term_structure_slope or 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="VolatilitySurface").set(0)

    # StatArbEngine
    try:
        from engine.signals.stat_arb_engine import StatArbEngine
        sa = StatArbEngine()
        metrics["strat_engine_health"].labels(engine="StatArbEngine").set(1)
        if hasattr(sa, "pairs") and sa.pairs:
            metrics["stat_arb_pairs_count"].set(len(sa.pairs))
            zscores = [p.z_score for p in sa.pairs if hasattr(p, "z_score") and p.z_score is not None]
            if zscores:
                metrics["stat_arb_mean_zscore"].set(sum(zscores) / len(zscores))
        if hasattr(sa, "active_trades"):
            metrics["stat_arb_active_trades"].set(len(sa.active_trades) if sa.active_trades else 0)
        if hasattr(sa, "portfolio_beta"):
            metrics["stat_arb_portfolio_beta"].set(sa.portfolio_beta or 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="StatArbEngine").set(0)

    # MLVoteEnsemble
    try:
        from engine.execution.execution_engine import MLVoteEnsemble
        ens = MLVoteEnsemble()
        metrics["strat_engine_health"].labels(engine="MLVoteEnsemble").set(1)
        if hasattr(ens, "tiers") and ens.tiers:
            bullish = sum(1 for t in ens.tiers if getattr(t, "vote", None) == "BUY")
            bearish = sum(1 for t in ens.tiers if getattr(t, "vote", None) == "SELL")
            metrics["ml_ensemble_vote_bullish"].set(bullish)
            metrics["ml_ensemble_vote_bearish"].set(bearish)
        if hasattr(ens, "confidence"):
            metrics["ml_ensemble_confidence"].set(ens.confidence or 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="MLVoteEnsemble").set(0)

    # DecisionMatrix
    try:
        from engine.execution.decision_matrix import DecisionMatrix, GATE_CONFIGS
        dm = DecisionMatrix()
        metrics["strat_engine_health"].labels(engine="DecisionMatrix").set(1)
        total_gates = len(GATE_CONFIGS) if GATE_CONFIGS else 6
        metrics["decision_matrix_gates_total"].set(total_gates)
        if hasattr(dm, "gates") and dm.gates:
            passing = sum(1 for g in dm.gates if getattr(g, "passing", False))
            metrics["decision_matrix_gates_passed"].set(passing)
            metrics["decision_matrix_approval_rate"].set(passing / total_gates if total_gates else 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="DecisionMatrix").set(0)

    # MetadronCube extended — confidence + sleeve weights
    try:
        cache_path2 = Path(__file__).resolve().parent.parent.parent / "data" / "cube_state_cache.json"
        if cache_path2.exists():
            with open(cache_path2) as f2:
                cs = json.load(f2)
            metrics["cube_regime_confidence"].set(cs.get("confidence", cs.get("liquidity", 0)))
            sleeves = cs.get("sleeves", cs.get("sleeve_allocation", {}))
            if isinstance(sleeves, dict):
                for name, weight in sleeves.items():
                    metrics["cube_sleeve_weight"].labels(sleeve=name).set(weight or 0)
        metrics["strat_engine_health"].labels(engine="MetadronCube").set(1)
    except Exception:
        metrics["strat_engine_health"].labels(engine="MetadronCube").set(0)

    # PatternRecognitionEngine
    try:
        from engine.signals.pattern_recognition import PatternRecognitionEngine
        pre = PatternRecognitionEngine()
        metrics["strat_engine_health"].labels(engine="PatternRecognition").set(1)
        if hasattr(pre, "detected_patterns"):
            patterns = pre.detected_patterns or []
            metrics["pattern_recognition_patterns_detected"].set(len(patterns))
            if patterns:
                confs = [p.get("confidence", 0) for p in patterns if isinstance(p, dict)]
                metrics["pattern_recognition_confidence"].set(sum(confs) / len(confs) if confs else 0)
    except Exception:
        metrics["strat_engine_health"].labels(engine="PatternRecognition").set(0)

    # PM2 process metrics via /proc or psutil
    try:
        import subprocess
        result = subprocess.run(
            ["pm2", "jlist"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            processes = json.loads(result.stdout)
            for proc in processes:
                name = proc.get("name", "unknown")
                monit = proc.get("monit", {})
                metrics["pm2_process_memory_bytes"].labels(process=name).set(
                    monit.get("memory", 0)
                )
                pm2_env = proc.get("pm2_env", {})
                metrics["pm2_process_restarts"].labels(process=name).set(
                    pm2_env.get("restart_time", 0)
                )
    except Exception:
        pass

    # ── TXLOG: Trade Execution Metrics ─────────────────────────────
    try:
        from engine.execution.execution_engine import ExecutionEngine
        eng = ExecutionEngine._instance if hasattr(ExecutionEngine, "_instance") else None
        if eng is None:
            from engine.api.routers.execution import _get_exec
            eng = _get_exec()
        if eng is not None:
            broker = eng.broker
            trades = broker.get_trade_history()
            total = len(trades)
            metrics["txlog_orders_total"].set(total)
            if total > 0:
                filled = [t for t in trades if t.get("fill_price", 0) > 0]
                rejected = total - len(filled)
                metrics["txlog_fill_rate"].set(len(filled) / total if total else 0)
                metrics["txlog_reject_rate"].set(rejected / total if total else 0)
                # Notional volume
                notional = sum(t.get("fill_price", 0) * t.get("quantity", 0) for t in filled)
                metrics["txlog_notional_volume"].set(notional)
                # Slippage (if tracked)
                slippages = [t.get("slippage", t.get("slippage_bps", 0)) or 0 for t in filled]
                if slippages:
                    metrics["txlog_avg_slippage_bps"].set(sum(slippages) / len(slippages))
                # Orders by side
                from collections import Counter
                side_counts = Counter(str(t.get("side", "UNKNOWN")).upper() for t in trades)
                for side_name in ["BUY", "SELL", "SHORT", "COVER"]:
                    metrics["txlog_orders_by_side"].labels(side=side_name).set(side_counts.get(side_name, 0))
    except Exception:
        pass


# ─── Middleware for automatic request tracking ─────────────────────

def create_metrics_middleware(app, metrics: dict):
    """Add middleware to automatically track API request metrics."""
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
    except ImportError:
        logger.warning("starlette not available — skipping request tracking middleware")
        return

    class MetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.time()
            response = await call_next(request)
            duration = time.time() - start

            endpoint = request.url.path
            method = request.method
            status = str(response.status_code)

            metrics["api_requests_total"].labels(
                endpoint=endpoint, method=method, status=status,
            ).inc()
            metrics["api_duration_seconds"].labels(endpoint=endpoint).observe(duration)

            return response

    app.add_middleware(MetricsMiddleware)


# ─── Router Factory ────────────────────────────────────────────────

def create_metrics_router(app=None):
    """Create a FastAPI router that serves /metrics in Prometheus format.

    Args:
        app: Optional FastAPI app instance. If provided, request-tracking
             middleware is automatically installed.

    Returns:
        APIRouter with GET /metrics endpoint.
    """
    try:
        from fastapi import APIRouter
        from fastapi.responses import Response
    except ImportError:
        logger.error("FastAPI not installed — cannot create metrics router")
        return None

    if not _prometheus_available:
        router = APIRouter()

        @router.get("/metrics")
        async def metrics_unavailable():
            return Response(
                content="# prometheus_client not installed\n",
                media_type="text/plain",
                status_code=503,
            )
        return router

    # Create a dedicated registry (avoids default process collector noise)
    registry = CollectorRegistry()
    metrics = _create_metrics(registry)

    # Install middleware if app provided
    if app is not None:
        create_metrics_middleware(app, metrics)

    router = APIRouter()

    @router.get("/metrics")
    async def prometheus_metrics():
        _collect_live_metrics(metrics)
        body = generate_latest(registry)
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)

    return router


# ─── Standalone helper functions for instrumenting other modules ───

_global_metrics = None
_global_registry = None


def get_metrics():
    """Get or create the global metrics dict for use by other modules.

    Example usage in engine code:
        from engine.bridges.prometheus_metrics import get_metrics
        metrics = get_metrics()
        if metrics:
            metrics["openbb_requests_total"].labels(endpoint="get_prices").inc()
    """
    global _global_metrics, _global_registry

    if not _prometheus_available:
        return None

    if _global_metrics is None:
        _global_registry = CollectorRegistry()
        _global_metrics = _create_metrics(_global_registry)

    return _global_metrics
