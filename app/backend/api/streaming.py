"""
Metadron Capital — Server-Sent Events (SSE) streaming endpoints.

Endpoints
---------
GET /api/stream/signals    — live signal updates via EventSource
GET /api/stream/portfolio  — live portfolio state via EventSource
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("metadron.api.streaming")

try:
    from fastapi import APIRouter
except ImportError:
    raise RuntimeError("FastAPI is required.")

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    EventSourceResponse = None
    logger.warning(
        "sse_starlette is not installed — streaming endpoints will return 501. "
        "Install it with: pip install sse-starlette"
    )

router = APIRouter()


async def _signal_event_generator():
    """Yield signal events as JSON-encoded SSE data.

    In production this would read from a message queue or internal
    pub/sub channel. The stub emits a heartbeat every 5 seconds.
    """
    while True:
        payload = {
            "type": "heartbeat",
            "message": "No live signals — pipeline idle.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        yield {"event": "signal", "data": json.dumps(payload)}
        await asyncio.sleep(5)


async def _portfolio_event_generator():
    """Yield portfolio-state events as JSON-encoded SSE data.

    Stub implementation: emits a placeholder snapshot every 10 seconds.
    """
    while True:
        payload = {
            "nav": 0.0,
            "cash": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        yield {"event": "portfolio", "data": json.dumps(payload)}
        await asyncio.sleep(10)


@router.get("/signals")
async def stream_signals():
    """Stream live trading signals via Server-Sent Events.

    Requires the ``sse_starlette`` package.
    """
    if EventSourceResponse is None:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=501,
            content={"detail": "SSE streaming not available — install sse-starlette."},
        )
    return EventSourceResponse(_signal_event_generator())


@router.get("/portfolio")
async def stream_portfolio():
    """Stream live portfolio state via Server-Sent Events.

    Requires the ``sse_starlette`` package.
    """
    if EventSourceResponse is None:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=501,
            content={"detail": "SSE streaming not available — install sse-starlette."},
        )
    return EventSourceResponse(_portfolio_event_generator())
