"""
Metadron Capital — Agent service.

Wraps the engine's agent subsystem (engine/agents/) to expose
agent metadata and performance metrics to API consumers.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

logger = logging.getLogger("metadron.services.agents")


class AgentService:
    """Service layer for querying hedge-fund agent state and performance.

    Falls back to stub data when the engine agent modules are not
    importable (e.g. during early development or in isolated tests).
    """

    def __init__(self):
        self._agent_registry = None
        try:
            from engine.agents import registry  # type: ignore[import]

            self._agent_registry = registry
            logger.info("Engine agent registry loaded.")
        except ImportError:
            logger.warning(
                "engine.agents not available — agent queries will return stubs."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_agents(self) -> List[Dict[str, Any]]:
        """Return a list of all registered agents with basic metadata.

        Each entry contains at minimum ``id``, ``name``, ``type``, and
        ``status``.
        """
        if self._agent_registry is not None:
            try:
                return self._agent_registry.list_agents()
            except Exception as exc:
                logger.error("Failed to list agents: %s", exc)

        return self._stub_agents()

    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Return performance metrics for a specific agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier of the agent.

        Returns
        -------
        dict
            Performance summary including hit-rate, PnL contribution, etc.
        """
        if self._agent_registry is not None:
            try:
                return self._agent_registry.get_performance(agent_id)
            except Exception as exc:
                logger.error("Failed to get performance for agent %s: %s", agent_id, exc)

        return self._stub_performance(agent_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_agents() -> List[Dict[str, Any]]:
        """Return placeholder agent data."""
        return [
            {"id": "macro-analyst", "name": "Macro Analyst", "type": "analyst", "status": "idle"},
            {"id": "quant-researcher", "name": "Quant Researcher", "type": "researcher", "status": "idle"},
            {"id": "risk-manager", "name": "Risk Manager", "type": "risk", "status": "idle"},
        ]

    @staticmethod
    def _stub_performance(agent_id: str) -> Dict[str, Any]:
        """Return placeholder performance metrics."""
        return {
            "agent_id": agent_id,
            "hit_rate": None,
            "pnl_contribution": None,
            "signals_generated": 0,
            "status": "stub",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
