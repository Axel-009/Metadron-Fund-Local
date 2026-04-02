"""
Monitoring router — REPORTS, TECH, ARCHIVE tabs
Wraps: PlatinumReport, DailyReport, PortfolioReport, AnomalyDetector,
       SectorTracker, LearningLoop, MemoryMonitor
"""
from fastapi import APIRouter, Query
from datetime import datetime
from pathlib import Path
import logging
import os

logger = logging.getLogger("metadron-api.monitoring")
router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_anomaly = None
_sector_tracker = None
_learning = None


def _get_anomaly():
    global _anomaly
    if _anomaly is None:
        from engine.monitoring.anomaly_detector import AnomalyDetector
        _anomaly = AnomalyDetector()
    return _anomaly


def _get_sector_tracker():
    global _sector_tracker
    if _sector_tracker is None:
        from engine.monitoring.sector_tracker import SectorTracker
        _sector_tracker = SectorTracker()
    return _sector_tracker


def _get_learning():
    global _learning
    if _learning is None:
        from engine.monitoring.learning_loop import LearningLoop
        _learning = LearningLoop()
    return _learning


# ─── REPORTS tab ───────────────────────────────────────────

@router.get("/reports/list")
async def reports_list():
    """Available reports and their last generation timestamps."""
    try:
        logs_dir = PROJECT_ROOT / "logs"
        reports = []

        report_types = [
            ("platinum", "Platinum Report", "Executive macro state"),
            ("portfolio", "Portfolio Report", "Performance deep-dive"),
            ("daily", "Daily Report", "EOD reconciliation"),
            ("sector", "Sector Report", "Sector performance"),
        ]

        for folder, name, desc in report_types:
            report_dir = logs_dir / folder
            last_file = None
            last_time = None

            if report_dir.exists():
                files = sorted(report_dir.glob("*.txt"), key=os.path.getmtime, reverse=True)
                if not files:
                    files = sorted(report_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
                if files:
                    last_file = files[0].name
                    last_time = datetime.fromtimestamp(os.path.getmtime(files[0])).isoformat()

            reports.append({
                "id": folder,
                "name": name,
                "description": desc,
                "last_generated": last_time,
                "last_file": last_file,
                "status": "ready" if last_file else "not_generated",
            })

        return {"reports": reports, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/reports/list error: {e}")
        return {"reports": [], "error": str(e)}


@router.get("/reports/generate")
async def generate_report(report_type: str = Query("platinum")):
    """Generate a report on demand."""
    try:
        if report_type == "platinum":
            from engine.monitoring.platinum_report import PlatinumReport
            pr = PlatinumReport()
            result = pr.generate() if hasattr(pr, "generate") else "Not available"
        elif report_type == "portfolio":
            from engine.monitoring.portfolio_report import PortfolioReport
            pr = PortfolioReport()
            result = pr.generate() if hasattr(pr, "generate") else "Not available"
        elif report_type == "daily":
            from engine.monitoring.daily_report import DailyReport
            dr = DailyReport()
            result = dr.generate() if hasattr(dr, "generate") else "Not available"
        else:
            return {"error": f"Unknown report type: {report_type}"}

        return {
            "report_type": report_type,
            "content": result if isinstance(result, str) else str(result),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"monitoring/reports/generate error: {e}")
        return {"error": str(e)}


# ─── TECH tab ──────────────────────────────────────────────

@router.get("/engines/health")
async def engines_health():
    """All engine health: status, latency, memory, errors."""
    try:
        import importlib
        import time

        engines = [
            ("L1", "UniverseEngine", "engine.data.universe_engine"),
            ("L2", "MacroEngine", "engine.signals.macro_engine"),
            ("L2", "MetadronCube", "engine.signals.metadron_cube"),
            ("L2", "StatArbEngine", "engine.signals.stat_arb_engine"),
            ("L3", "AlphaOptimizer", "engine.ml.alpha_optimizer"),
            ("L4", "BetaCorridor", "engine.portfolio.beta_corridor"),
            ("L5", "ExecutionEngine", "engine.execution.execution_engine"),
            ("L5", "OptionsEngine", "engine.execution.options_engine"),
            ("L6", "ResearchBots", "engine.agents.research_bots"),
            ("L7", "L7Execution", "engine.execution.l7_unified_execution_surface"),
            ("MON", "AnomalyDetector", "engine.monitoring.anomaly_detector"),
        ]

        results = []
        for layer, name, module_path in engines:
            start = time.time()
            try:
                importlib.import_module(module_path)
                latency = (time.time() - start) * 1000  # ms
                results.append({
                    "id": layer, "name": name, "level": layer,
                    "status": "online", "latency": round(latency, 1),
                    "errors": 0,
                })
            except Exception as ex:
                results.append({
                    "id": layer, "name": name, "level": layer,
                    "status": "offline", "latency": 0,
                    "errors": 1, "error_msg": str(ex)[:100],
                })

        return {"engines": results, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/engines/health error: {e}")
        return {"engines": [], "error": str(e)}


@router.get("/anomalies")
async def anomalies():
    """Statistical anomaly scanner results."""
    try:
        ad = _get_anomaly()
        if hasattr(ad, "scan"):
            results = ad.scan()
        elif hasattr(ad, "detect"):
            results = ad.detect()
        else:
            results = []

        if isinstance(results, list):
            return {"anomalies": results, "timestamp": datetime.utcnow().isoformat()}
        return {"anomalies": results if isinstance(results, dict) else str(results), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/anomalies error: {e}")
        return {"anomalies": [], "error": str(e)}


@router.get("/sectors")
async def sector_tracking():
    """Sector performance + missed opportunities."""
    try:
        st = _get_sector_tracker()
        if hasattr(st, "get_tracking"):
            data = st.get_tracking()
        elif hasattr(st, "track"):
            data = st.track()
        else:
            data = {}
        return {**(data if isinstance(data, dict) else {"data": str(data)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/sectors error: {e}")
        return {"error": str(e)}


@router.get("/learning-loop")
async def learning_loop():
    """Closed-loop feedback: signal accuracy → tier weights."""
    try:
        ll = _get_learning()
        if hasattr(ll, "get_state"):
            state = ll.get_state()
        elif hasattr(ll, "summary"):
            state = ll.summary()
        else:
            state = {}
        return {**(state if isinstance(state, dict) else {"data": str(state)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/learning-loop error: {e}")
        return {"error": str(e)}


# ─── ARCHIVE tab ───────────────────────────────────────────

@router.get("/archive/files")
async def archive_files(
    folder: str = Query("logs", description="Subfolder to list"),
    limit: int = Query(50, ge=1, le=200),
):
    """List archived files: TX logs, reconciliation, learning loop."""
    try:
        base = PROJECT_ROOT / folder
        if not base.exists() or not str(base.resolve()).startswith(str(PROJECT_ROOT)):
            return {"files": [], "error": "Invalid folder"}

        files = []
        for f in sorted(base.rglob("*"), key=os.path.getmtime, reverse=True):
            if f.is_file() and not f.name.startswith("."):
                files.append({
                    "name": f.name,
                    "path": str(f.relative_to(PROJECT_ROOT)),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
                if len(files) >= limit:
                    break

        return {"files": files, "folder": folder, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"monitoring/archive/files error: {e}")
        return {"files": [], "error": str(e)}


@router.get("/archive/read")
async def archive_read(path: str = Query(..., description="Relative file path")):
    """Read an archived file's content."""
    try:
        file_path = PROJECT_ROOT / path
        if not file_path.exists() or not str(file_path.resolve()).startswith(str(PROJECT_ROOT)):
            return {"error": "File not found or access denied"}

        content = file_path.read_text(errors="replace")[:50000]  # Cap at 50KB
        return {
            "path": path,
            "content": content,
            "size": file_path.stat().st_size,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"monitoring/archive/read error: {e}")
        return {"error": str(e)}
