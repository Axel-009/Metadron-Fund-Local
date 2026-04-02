"""Centralized error logger for Metadron Capital.

All engine modules should use this to log errors. The TECH tab reads from
this to display real-time errors and alerts.

Usage in any engine module:
    from engine.ops.error_logger import log_engine_error
    log_engine_error("MacroEngine", "FRED data fetch failed", severity="ERROR")

Usage in TECH tab API:
    from engine.ops.error_logger import get_session_errors, get_recent_errors
"""

import logging
import threading
from datetime import datetime
from collections import deque
from typing import Optional

logger = logging.getLogger("metadron.errors")

# Thread-safe in-memory error buffer (last 500 errors per session)
_error_buffer: deque = deque(maxlen=500)
_lock = threading.Lock()


def log_engine_error(
    engine: str,
    message: str,
    severity: str = "ERROR",
    detail: Optional[str] = None,
) -> None:
    """Log an engine error to the centralized buffer.

    Args:
        engine: Engine name (e.g., "MacroEngine", "BetaCorridor")
        message: Error description
        severity: ERROR, WARN, CRITICAL
        detail: Optional stack trace or extended info
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "engine": engine,
        "severity": severity,
        "message": message,
        "detail": detail,
    }
    with _lock:
        _error_buffer.append(entry)

    # Also log to Python logging
    if severity == "CRITICAL":
        logger.critical(f"[{engine}] {message}")
    elif severity == "ERROR":
        logger.error(f"[{engine}] {message}")
    else:
        logger.warning(f"[{engine}] {message}")


def get_session_errors() -> list:
    """Get all errors from the current session."""
    with _lock:
        return list(_error_buffer)


def get_recent_errors(limit: int = 20) -> list:
    """Get the most recent N errors."""
    with _lock:
        return list(_error_buffer)[-limit:]


def get_error_counts() -> dict:
    """Get error counts by engine and severity."""
    with _lock:
        by_engine: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for e in _error_buffer:
            eng = e.get("engine", "unknown")
            sev = e.get("severity", "ERROR")
            by_engine[eng] = by_engine.get(eng, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1
        return {
            "by_engine": by_engine,
            "by_severity": by_severity,
            "total": len(_error_buffer),
        }


def clear_session_errors() -> int:
    """Clear the error buffer (e.g., on session start). Returns count cleared."""
    with _lock:
        count = len(_error_buffer)
        _error_buffer.clear()
        return count


# ─── Install as a Python logging handler ───────────────────
# This captures ALL Python logging errors across the engine

class EngineErrorHandler(logging.Handler):
    """Python logging handler that captures ERROR+ to the centralized buffer."""

    def emit(self, record: logging.LogRecord):
        # Skip our own logger to prevent recursion
        if record.name.startswith("metadron.errors"):
            return
        if record.levelno >= logging.ERROR:
            log_engine_error(
                engine=record.name,
                message=record.getMessage(),
                severity="CRITICAL" if record.levelno >= logging.CRITICAL else "ERROR",
                detail=self.format(record) if record.exc_info else None,
            )


def install_global_handler():
    """Install the centralized error handler on the root logger.

    Call this once at startup (e.g., in API server or run_open.py).
    """
    handler = EngineErrorHandler()
    handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(handler)
    logger.info("Centralized error handler installed on root logger")
