"""
Bridge for graphifyy codebase knowledge graph.
Agents use this to query the Metadron system graph without reading raw files.
Install: pip install graphifyy
Build graph: graphify . (from repo root)
"""
from __future__ import annotations
import logging
import subprocess
import json
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GRAPH_JSON = _REPO_ROOT / "graphify-out" / "graph.json"
_GRAPH_REPORT = _REPO_ROOT / "graphify-out" / "GRAPH_REPORT.md"

try:
    import graphifyy  # noqa: F401
    GRAPHIFY_AVAILABLE = True
except ImportError:
    GRAPHIFY_AVAILABLE = False
    logger.info("graphifyy not installed — run: pip install graphifyy")


class GraphifyBridge:
    """Query the Metadron codebase knowledge graph."""

    def is_available(self) -> bool:
        return GRAPHIFY_AVAILABLE and _GRAPH_JSON.exists()

    def get_report(self) -> str:
        """Returns GRAPH_REPORT.md contents (god nodes, surprising connections)."""
        if not _GRAPH_REPORT.exists():
            return "No graph report found. Run: graphify . from repo root."
        try:
            return _GRAPH_REPORT.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("GraphifyBridge.get_report error: %s", e)
            return ""

    def query(self, question: str) -> str:
        """Query the knowledge graph. Returns text answer."""
        if not self.is_available():
            return "Graph not available. Run: graphify . from repo root."
        try:
            result = subprocess.run(
                ["graphify", "query", question, "--graph", str(_GRAPH_JSON)],
                capture_output=True, text=True, timeout=30, cwd=str(_REPO_ROOT),
            )
            return result.stdout.strip() or result.stderr.strip()
        except Exception as e:
            logger.warning("GraphifyBridge.query error: %s", e)
            return f"Query failed: {e}"

    def get_god_nodes(self) -> list:
        """Returns the highest-degree concept nodes from graph.json."""
        if not _GRAPH_JSON.exists():
            return []
        try:
            data = json.loads(_GRAPH_JSON.read_text(encoding="utf-8"))
            nodes = data.get("nodes", [])
            sorted_nodes = sorted(nodes, key=lambda n: n.get("degree", 0), reverse=True)
            return [
                {"id": n.get("id"), "label": n.get("label"), "degree": n.get("degree")}
                for n in sorted_nodes[:20]
            ]
        except Exception as e:
            logger.warning("GraphifyBridge.get_god_nodes error: %s", e)
            return []
