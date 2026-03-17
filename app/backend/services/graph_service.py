"""
Metadron Capital — Graph / pipeline DAG service.

Provides a structured view of the signal pipeline as a directed
acyclic graph (DAG), suitable for rendering in a UI or inspecting
the flow of data through the system.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger("metadron.services.graph")


class GraphService:
    """Service for querying the pipeline graph topology.

    When the engine modules are available, the graph is derived from
    the actual pipeline configuration; otherwise a canonical stub
    DAG is returned.
    """

    def __init__(self):
        self._pipeline = None
        try:
            from engine.pipeline import PipelineGraph  # type: ignore[import]

            self._pipeline = PipelineGraph()
            logger.info("Engine pipeline graph loaded.")
        except ImportError:
            logger.warning(
                "engine.pipeline not available — graph queries will return stubs."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pipeline_graph(self) -> Dict[str, Any]:
        """Return the full pipeline DAG structure.

        Returns
        -------
        dict
            ``nodes`` — list of processing nodes
            ``edges`` — list of directed edges between nodes
        """
        if self._pipeline is not None:
            try:
                return self._pipeline.to_dict()
            except Exception as exc:
                logger.error("Failed to serialise pipeline graph: %s", exc)

        return self._stub_graph()

    def get_signal_flow(self, ticker: str) -> Dict[str, Any]:
        """Return the signal flow path for a specific ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol to trace through the pipeline.

        Returns
        -------
        dict
            Subgraph showing which nodes processed this ticker.
        """
        if self._pipeline is not None:
            try:
                return self._pipeline.trace(ticker)
            except Exception as exc:
                logger.error("Failed to trace signal flow for %s: %s", ticker, exc)

        return self._stub_signal_flow(ticker)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_graph() -> Dict[str, Any]:
        """Return a canonical stub pipeline DAG."""
        nodes: List[Dict[str, Any]] = [
            {"id": "data_ingest", "label": "Data Ingestion", "type": "source"},
            {"id": "macro_analysis", "label": "Macro Analysis", "type": "analysis"},
            {"id": "quant_signals", "label": "Quant Signals", "type": "analysis"},
            {"id": "agent_votes", "label": "Agent Voting", "type": "aggregation"},
            {"id": "risk_check", "label": "Risk Check", "type": "validation"},
            {"id": "order_gen", "label": "Order Generation", "type": "execution"},
        ]
        edges: List[Dict[str, str]] = [
            {"from": "data_ingest", "to": "macro_analysis"},
            {"from": "data_ingest", "to": "quant_signals"},
            {"from": "macro_analysis", "to": "agent_votes"},
            {"from": "quant_signals", "to": "agent_votes"},
            {"from": "agent_votes", "to": "risk_check"},
            {"from": "risk_check", "to": "order_gen"},
        ]
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _stub_signal_flow(ticker: str) -> Dict[str, Any]:
        """Return a stub signal-flow trace for the given ticker."""
        return {
            "ticker": ticker,
            "path": [
                "data_ingest",
                "quant_signals",
                "agent_votes",
                "risk_check",
                "order_gen",
            ],
            "status": "stub",
            "message": f"No live pipeline data for {ticker}.",
        }
