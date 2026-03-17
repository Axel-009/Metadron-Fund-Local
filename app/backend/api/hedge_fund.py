"""
Metadron Capital — Core hedge-fund API endpoints.

Endpoints
---------
GET /api/hedge-fund/pipeline   — run the full signal pipeline
GET /api/hedge-fund/macro      — macro-economic snapshot
GET /api/hedge-fund/signals    — latest trading signals
GET /api/hedge-fund/portfolio  — current portfolio state
GET /api/hedge-fund/positions  — list open positions
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger("metadron.api.hedge_fund")

try:
    from fastapi import APIRouter, Depends, HTTPException
    from sqlalchemy.orm import Session
except ImportError:
    raise RuntimeError("FastAPI and SQLAlchemy are required.")

from app.backend.models.database import get_db
from app.backend.models.tables import Portfolio, Trade
from app.backend.schemas.hedge_fund import (
    MacroResponse,
    PipelineResponse,
    SignalResponse,
    VoteResponse,
)

router = APIRouter()


@router.get("/pipeline", response_model=PipelineResponse)
def run_pipeline():
    """Execute the full signal pipeline and return aggregated results.

    In production this delegates to the engine's pipeline orchestrator.
    The stub returns an empty pipeline result for development purposes.
    """
    now = datetime.now(timezone.utc)
    return PipelineResponse(
        run_id=str(uuid.uuid4()),
        signals=[],
        votes=[],
        timestamp=now,
    )


@router.get("/macro", response_model=MacroResponse)
def get_macro_snapshot():
    """Return the latest macro-economic environment snapshot.

    In production this pulls from the macro data service / external feeds.
    """
    now = datetime.now(timezone.utc)
    return MacroResponse(
        gdp_growth=None,
        inflation_rate=None,
        interest_rate=None,
        vix=None,
        credit_spread=None,
        summary="Macro data unavailable — stub response.",
        timestamp=now,
    )


@router.get("/signals", response_model=List[SignalResponse])
def get_latest_signals():
    """Return the most recent trading signals from the pipeline.

    Stub: returns an empty list until the signal engine is wired in.
    """
    return []


@router.get("/portfolio", response_model=Dict[str, Any])
def get_portfolio_state(db: Session = Depends(get_db)):
    """Return the latest portfolio snapshot from the database."""
    snapshot = (
        db.query(Portfolio)
        .order_by(Portfolio.timestamp.desc())
        .first()
    )
    if snapshot is None:
        return {
            "nav": 0.0,
            "cash": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "No portfolio snapshot available.",
        }
    return {
        "nav": snapshot.nav,
        "cash": snapshot.cash,
        "gross_exposure": snapshot.gross_exposure,
        "net_exposure": snapshot.net_exposure,
        "timestamp": snapshot.timestamp.isoformat(),
    }


@router.get("/positions", response_model=List[Dict[str, Any]])
def list_positions(db: Session = Depends(get_db)):
    """Derive open positions from the most recent trades.

    A simple aggregation: group trades by ticker, net the quantities.
    """
    trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(500).all()
    positions: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        pos = positions.setdefault(
            t.ticker, {"ticker": t.ticker, "net_qty": 0.0, "avg_price": 0.0, "trades": 0}
        )
        sign = 1.0 if t.side == "buy" else -1.0
        pos["net_qty"] += sign * t.quantity
        pos["avg_price"] = t.price  # simplified — last price
        pos["trades"] += 1

    return [p for p in positions.values() if abs(p["net_qty"]) > 1e-9]
