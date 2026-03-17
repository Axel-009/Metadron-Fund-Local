"""
Metadron Capital — Pydantic schemas for hedge-fund domain objects.

Covers signal pipeline outputs, macro snapshots, cube analytics,
and alpha generation responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise RuntimeError("Pydantic is required. Install it with: pip install pydantic")


class SignalResponse(BaseModel):
    """A single trading signal emitted by the pipeline."""

    ticker: str
    direction: str  # "long" | "short" | "flat"
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str
    timestamp: datetime


class VoteResponse(BaseModel):
    """Aggregated vote from multiple agents for a ticker."""

    ticker: str
    bullish: int
    bearish: int
    neutral: int
    consensus: str
    details: Optional[List[Dict[str, Any]]] = None


class PipelineResponse(BaseModel):
    """Full output of the signal pipeline run."""

    run_id: str
    signals: List[SignalResponse]
    votes: List[VoteResponse]
    timestamp: datetime


class MacroResponse(BaseModel):
    """Macro-economic environment snapshot."""

    gdp_growth: Optional[float] = None
    inflation_rate: Optional[float] = None
    interest_rate: Optional[float] = None
    vix: Optional[float] = None
    credit_spread: Optional[float] = None
    summary: Optional[str] = None
    timestamp: datetime


class CubeResponse(BaseModel):
    """Multi-dimensional analytics cube result."""

    dimensions: List[str]
    metrics: Dict[str, float]
    slices: Optional[List[Dict[str, Any]]] = None


class AlphaResponse(BaseModel):
    """Alpha generation result for a given strategy or ticker universe."""

    strategy: str
    alpha_bps: float
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: Optional[float] = None
    tickers: List[str]
    period_start: datetime
    period_end: datetime
