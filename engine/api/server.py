"""
Metadron Capital — Engine API Bridge Server
Runs on port 8001, proxied by Express on port 5000.
Exposes all engine modules as REST endpoints for the frontend.
"""
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Ensure engine is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from engine.api.routers import (
    portfolio,
    cube,
    macro,
    signals,
    risk,
    execution,
    agents,
    ml,
    monitoring,
    universe,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [API] %(message)s")
logger = logging.getLogger("metadron-api")

app = FastAPI(
    title="Metadron Capital Engine API",
    version="1.0.0",
    docs_url="/api/engine/docs",
    openapi_url="/api/engine/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Register routers ──────────────────────────────────────
app.include_router(portfolio.router, prefix="/api/engine/portfolio", tags=["Portfolio"])
app.include_router(cube.router, prefix="/api/engine/cube", tags=["MetadronCube"])
app.include_router(macro.router, prefix="/api/engine/macro", tags=["Macro"])
app.include_router(signals.router, prefix="/api/engine/signals", tags=["Signals"])
app.include_router(risk.router, prefix="/api/engine/risk", tags=["Risk"])
app.include_router(execution.router, prefix="/api/engine/execution", tags=["Execution"])
app.include_router(agents.router, prefix="/api/engine/agents", tags=["Agents"])
app.include_router(ml.router, prefix="/api/engine/ml", tags=["ML"])
app.include_router(monitoring.router, prefix="/api/engine/monitoring", tags=["Monitoring"])
app.include_router(universe.router, prefix="/api/engine/universe", tags=["Universe"])


@app.get("/api/engine/health")
async def health():
    return {
        "status": "ok",
        "service": "Metadron Capital Engine API",
        "timestamp": datetime.utcnow().isoformat(),
        "engines_loaded": True,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "timestamp": datetime.utcnow().isoformat()},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("ENGINE_API_PORT", "8001"))
    logger.info(f"Starting Metadron Engine API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
