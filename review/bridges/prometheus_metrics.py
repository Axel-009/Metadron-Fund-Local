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
