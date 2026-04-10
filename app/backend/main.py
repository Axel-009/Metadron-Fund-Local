"""
Metadron Capital API — FastAPI application entry point.

Provides the core REST API for the Metadron Capital hedge fund platform,
including flow orchestration, signal pipelines, portfolio management,
and real-time streaming endpoints.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("metadron.api")

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    raise RuntimeError(
        "FastAPI is required. Install it with: pip install fastapi[all]"
    )

# ---------------------------------------------------------------------------
# Router imports (graceful fallback if modules not yet wired)
# ---------------------------------------------------------------------------
try:
    from app.backend.api.flows import router as flows_router
except Exception:
    flows_router = None

try:
    from app.backend.api.flow_runs import router as flow_runs_router
except Exception:
    flow_runs_router = None

try:
    from app.backend.api.hedge_fund import router as hedge_fund_router
except Exception:
    hedge_fund_router = None

try:
    from app.backend.api.api_keys import router as api_keys_router
except Exception:
    api_keys_router = None

try:
    from app.backend.api.streaming import router as streaming_router
except Exception:
    streaming_router = None

try:
    from app.backend.api.allocation import router as allocation_router
except Exception:
    allocation_router = None

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Metadron Capital API",
    description="Hedge-fund orchestration, signal pipeline, and portfolio management API.",
    version="0.1.0",
)

# CORS — allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
_routers = [
    (flows_router, "/flows", ["Flows"]),
    (flow_runs_router, "/flow-runs", ["Flow Runs"]),
    (hedge_fund_router, "/api/hedge-fund", ["Hedge Fund"]),
    (api_keys_router, "/api-keys", ["API Keys"]),
    (streaming_router, "/api/stream", ["Streaming"]),
    (allocation_router, "/api/allocation", ["Allocation"]),
]

for router, prefix, tags in _routers:
    if router is not None:
        app.include_router(router, prefix=prefix, tags=tags)


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize engine components on application startup."""
    logger.info("Metadron Capital API starting up ...")
    try:
        from app.backend.models.database import Base, engine

        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified / created.")
    except Exception as exc:
        logger.warning("Database initialization skipped: %s", exc)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Metadron Capital API shutting down ...")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    """Return a lightweight health-check response."""
    return {
        "status": "ok",
        "service": "Metadron Capital API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
