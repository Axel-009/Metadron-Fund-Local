"""Graphify Knowledge Graph — API Router.

Endpoints for codebase knowledge graph status, god-node analysis,
natural-language queries, graph regeneration trigger, and report access.

Build graph:  graphify .  (from repo root)
Install:      pip install graphifyy
"""

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Query

logger = logging.getLogger("metadron-api.graphify")
router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_graphify = None


def _get_bridge():
    global _graphify
    if _graphify is None:
        from engine.agents.graphify_bridge import GraphifyBridge
        _graphify = GraphifyBridge()
    return _graphify


@router.get("/status")
async def graphify_status():
    """Knowledge graph status, availability, and god-node summary."""
    try:
        bridge = _get_bridge()
        available = bridge.is_available()
        god_nodes = bridge.get_god_nodes() if available else []
        graph_path = _REPO_ROOT / "graphify-out" / "graph.json"
        report_path = _REPO_ROOT / "graphify-out" / "GRAPH_REPORT.md"
        return {
            "available": available,
            "graph_exists": graph_path.exists(),
            "report_exists": report_path.exists(),
            "graph_size_kb": round(graph_path.stat().st_size / 1024, 1) if graph_path.exists() else 0,
            "god_nodes": god_nodes,
            "god_node_count": len(god_nodes),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/status error: {e}")
        return {"available": False, "error": str(e)}


@router.get("/report")
async def graphify_report():
    """Return the full GRAPH_REPORT.md analysis."""
    try:
        bridge = _get_bridge()
        report = bridge.get_report()
        return {
            "report": report,
            "has_report": bool(report and "No graph report" not in report),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/report error: {e}")
        return {"report": "", "has_report": False, "error": str(e)}


@router.get("/query")
async def graphify_query(
    q: str = Query(..., description="Natural language question about the codebase"),
):
    """Query the knowledge graph with a natural language question."""
    try:
        bridge = _get_bridge()
        answer = bridge.query(q)
        return {
            "question": q,
            "answer": answer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/query error: {e}")
        return {"question": q, "answer": "", "error": str(e)}


@router.post("/generate")
async def graphify_generate():
    """Trigger knowledge graph regeneration.

    Runs `graphify .` in the repo root. This may take 1-5 minutes
    depending on codebase size. Returns immediately with status.
    """
    try:
        # Check if graphify is installed
        try:
            import graphifyy  # noqa: F401
        except ImportError:
            return {
                "status": "error",
                "message": "graphifyy not installed. Run: pip install graphifyy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Run graphify in background (non-blocking)
        process = subprocess.Popen(
            ["graphify", "."],
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        return {
            "status": "generating",
            "pid": process.pid,
            "message": "Graph generation started. Check /graphify/status for completion.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/generate error: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/god-nodes")
async def graphify_god_nodes(limit: int = Query(20, ge=1, le=100)):
    """Return the highest-degree concept nodes from the knowledge graph."""
    try:
        bridge = _get_bridge()
        nodes = bridge.get_god_nodes()
        return {
            "nodes": nodes[:limit],
            "total": len(nodes),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"graphify/god-nodes error: {e}")
        return {"nodes": [], "total": 0, "error": str(e)}
