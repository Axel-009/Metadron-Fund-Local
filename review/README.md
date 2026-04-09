# Review Files — Missing Service Wrappers & Infrastructure

**Date:** 2026-04-09
**Purpose:** Fill gaps identified in the comprehensive audit (see `/home/user/workspace/audit-results.md`)

These files are ready for review and integration into the main codebase.
No existing files were modified. All new files are in `review/`.

---

## Files Created

### 1. `bridges/airllm_model_server.py`

**Target location:** `engine/bridges/airllm_model_server.py`

Standalone FastAPI server for Air-LLM inference on port 8003. Previously, the PM2 config referenced this file but it didn't exist, causing PM2 to fail on startup.

| Feature | Detail |
|---------|--------|
| Port | 8003 (configurable via `AIRLLM_PORT`) |
| Model | `meta-llama/Llama-3.1-70B` (configurable via `AIRLLM_MODEL_PATH`) |
| GPU | Via `CUDA_VISIBLE_DEVICES` env var (default: device 1) |
| Endpoints | `GET /health`, `POST /generate`, `GET /model-info` |
| Pattern | `create_app()` factory for uvicorn `--factory` |
| Fallback | Starts in degraded mode if `airllm` library not installed |
| Style | Matches `llm_inference_bridge.py` — same logging, signal handling, CORS, lazy loading |

**Integration:**
```bash
cp review/bridges/airllm_model_server.py engine/bridges/airllm_model_server.py
```

---

### 2. `bridges/ainewton_service.py`

**Target location:** `engine/bridges/ainewton_service.py`

PM2 service wrapper for the existing `AINewtonDiscoveryWorker`. The PM2 config referenced `ainewton_service.py` but the actual worker code was in `ainewton_discovery_worker.py`. This wrapper bridges the gap.

| Feature | Detail |
|---------|--------|
| Worker | Imports and wraps `AINewtonDiscoveryWorker` from `ainewton_discovery_worker.py` |
| Threading | Worker runs in background thread; main thread handles signals |
| Health | Optional HTTP health endpoint on `AINEWTON_HEALTH_PORT` (off by default) |
| Degraded mode | If worker import fails, runs in health-only idle loop |
| Lifecycle | Graceful shutdown on SIGINT/SIGTERM, worker thread join with 30s timeout |

**Integration:**
```bash
cp review/bridges/ainewton_service.py engine/bridges/ainewton_service.py
```

---

### 3. `bridges/metadron_cube_service.py`

**Target location:** `engine/bridges/metadron_cube_service.py`

Continuous PM2 service that runs MetadronCube regime detection in a loop. Previously, the PM2 config referenced this file but it didn't exist.

| Feature | Detail |
|---------|--------|
| Modes | `continuous` (default, loop forever), `once` (compute and exit) |
| Interval | Market hours: 60s, Off-hours: 300s (configurable via env vars) |
| State cache | Writes `data/cube_state_cache.json` for consumption by other processes |
| Dependencies | Imports `MetadronCube` from `engine.signals.metadron_cube`, `MacroEngine` from `engine.signals.macro_engine` |
| Fallback | If MacroEngine unavailable, uses default `MacroSnapshot()` |
| Recovery | If initialization fails, retries on next cycle (doesn't crash) |

**Integration:**
```bash
cp review/bridges/metadron_cube_service.py engine/bridges/metadron_cube_service.py
```

---

### 4. `bridges/prometheus_metrics.py`

**Target location:** `engine/bridges/prometheus_metrics.py`

Prometheus metrics endpoint for Grafana/Datadog/any Prometheus-compatible stack. Exports 15 metric families covering engine health, API requests, portfolio state, cube regime, trades, OpenBB data, LLM inference, and PM2 processes.

| Feature | Detail |
|---------|--------|
| Dependency | `prometheus_client` (pip install prometheus-client) |
| Mount | `app.include_router(create_metrics_router(app))` on the engine FastAPI server |
| Endpoint | `GET /metrics` in Prometheus text exposition format |
| Auto-tracking | Optional middleware for automatic API request/latency counting |
| Live collection | Scrapes live engine state on each `/metrics` request |
| PM2 metrics | Calls `pm2 jlist` to collect process memory and restart counts |
| Degraded | Returns 503 if `prometheus_client` not installed |

**Metrics exported:**

| Metric | Type | Labels |
|--------|------|--------|
| `metadron_engine_up` | Gauge | - |
| `metadron_api_requests_total` | Counter | endpoint, method, status |
| `metadron_api_duration_seconds` | Histogram | endpoint |
| `metadron_portfolio_nav` | Gauge | - |
| `metadron_portfolio_pnl_daily` | Gauge | - |
| `metadron_positions_count` | Gauge | - |
| `metadron_cube_signal_score` | Gauge | - |
| `metadron_cube_regime` | Gauge | regime_name |
| `metadron_trades_total` | Counter | side |
| `metadron_openbb_requests_total` | Counter | endpoint |
| `metadron_openbb_errors_total` | Counter | endpoint |
| `metadron_llm_requests_total` | Counter | backend |
| `metadron_llm_duration_seconds` | Histogram | backend |
| `metadron_pm2_process_memory_bytes` | Gauge | process |
| `metadron_pm2_process_restarts` | Gauge | process |

**Integration:**
```bash
cp review/bridges/prometheus_metrics.py engine/bridges/prometheus_metrics.py
```

Then add to `engine/api/server.py`:
```python
from engine.bridges.prometheus_metrics import create_metrics_router
metrics_router = create_metrics_router(app)
if metrics_router:
    app.include_router(metrics_router)
```

---

### 5. `ecosystem.config.cjs`

**Target location:** `ecosystem.config.cjs` (replace root file)

Corrected PM2 configuration with these fixes:

| # | Fix | Before | After |
|---|-----|--------|-------|
| 1 | `airllm-model-server` | Referenced missing file | Now exists (`engine/bridges/airllm_model_server.py`) |
| 2 | `ainewton-service` | Referenced missing `ainewton_service.py` | Now points to wrapper that delegates to `ainewton_discovery_worker.py` |
| 3 | `metadron-cube` | Referenced missing `metadron_cube_service.py` | Now points to wrapper that runs `MetadronCube` continuously |
| 4 | `express-frontend` | Production still runs `npm run dev` | Comment documenting production should use `node dist/index.cjs` |
| 5 | `news-engine` | Already correct | Confirmed path: `News engine/index.js` |

**Integration:**
```bash
cp review/ecosystem.config.cjs ecosystem.config.cjs
```

---

## Integration Checklist

```bash
# 1. Copy the 3 missing service files
cp review/bridges/airllm_model_server.py    engine/bridges/airllm_model_server.py
cp review/bridges/ainewton_service.py        engine/bridges/ainewton_service.py
cp review/bridges/metadron_cube_service.py   engine/bridges/metadron_cube_service.py

# 2. Copy Prometheus metrics (optional but recommended)
cp review/bridges/prometheus_metrics.py      engine/bridges/prometheus_metrics.py

# 3. Replace ecosystem.config.cjs
cp review/ecosystem.config.cjs              ecosystem.config.cjs

# 4. Install prometheus_client (optional, for metrics)
pip install prometheus-client

# 5. Create data directory for cube state cache
mkdir -p data

# 6. Test PM2 startup
pm2 start ecosystem.config.cjs --only engine-api
pm2 start ecosystem.config.cjs --only airllm-model-server
pm2 start ecosystem.config.cjs --only ainewton-service
pm2 start ecosystem.config.cjs --only metadron-cube
pm2 logs --lines 20
```

## Dependencies

| File | Required | Optional |
|------|----------|----------|
| `airllm_model_server.py` | `fastapi`, `uvicorn` | `airllm` (in intelligence_platform/Air-LLM), `transformers` |
| `ainewton_service.py` | (none beyond stdlib) | `numpy`, `pysr` (for actual symbolic regression) |
| `metadron_cube_service.py` | `numpy`, `pandas` (via metadron_cube.py) | - |
| `prometheus_metrics.py` | `fastapi` | `prometheus_client` |

All files follow the existing codebase pattern of try/except on imports and graceful degradation when optional dependencies are missing.
