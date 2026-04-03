"""
Metadron Capital — PM2 Process Integration Layer

Provides programmatic control over all platform services via PM2.
Used by the platform orchestrator and live loop to manage service health,
restart failed processes, and coordinate startup/shutdown sequences.
"""

import subprocess
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PLATFORM_ROOT = Path(__file__).resolve().parent.parent
ECOSYSTEM_CONFIG = PLATFORM_ROOT / "ecosystem.config.cjs"

SERVICE_GROUPS = {
    "core": ["engine-api", "express-frontend"],
    "mirofish": ["mirofish-backend", "mirofish-frontend"],
    "qwen": ["qwen-model-server"],
    "news": ["news-engine"],
    "trading": ["live-loop", "market-open", "market-close", "hourly-tasks", "platform-orchestrator"],
    "all": [
        "engine-api", "express-frontend",
        "mirofish-backend", "mirofish-frontend",
        "qwen-model-server", "news-engine",
        "live-loop", "market-open", "market-close",
        "hourly-tasks", "platform-orchestrator",
    ],
}

SERVICE_PORTS = {
    "engine-api": 8001,
    "express-frontend": 5000,
    "mirofish-backend": 5001,
    "mirofish-frontend": 5174,
    "qwen-model-server": 7860,
}


def _run_pm2(args: list[str], timeout: int = 30) -> dict:
    cmd = ["pm2"] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(PLATFORM_ROOT))
        return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        logger.error(f"PM2 command timed out: {' '.join(cmd)}")
        return {"success": False, "stdout": "", "stderr": "Command timed out", "returncode": -1}
    except FileNotFoundError:
        logger.error("PM2 not found. Install with: npm install -g pm2")
        return {"success": False, "stdout": "", "stderr": "PM2 not installed", "returncode": -1}


def get_process_list() -> list[dict]:
    result = _run_pm2(["jlist"])
    if not result["success"]:
        return []
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return []


def get_service_status(service_name: str) -> Optional[dict]:
    processes = get_process_list()
    for proc in processes:
        if proc.get("name") == service_name:
            return {
                "name": proc["name"],
                "status": proc.get("pm2_env", {}).get("status", "unknown"),
                "pid": proc.get("pid"),
                "memory": proc.get("monit", {}).get("memory", 0),
                "cpu": proc.get("monit", {}).get("cpu", 0),
                "uptime": proc.get("pm2_env", {}).get("pm_uptime", 0),
                "restarts": proc.get("pm2_env", {}).get("restart_time", 0),
            }
    return None


def start_services(group: str = "all") -> dict:
    services = SERVICE_GROUPS.get(group)
    if not services:
        return {"success": False, "error": f"Unknown service group: {group}"}
    if group == "all":
        return _run_pm2(["start", str(ECOSYSTEM_CONFIG)])
    return _run_pm2(["start", str(ECOSYSTEM_CONFIG), "--only", ",".join(services)])


def stop_services(group: str = "all") -> dict:
    services = SERVICE_GROUPS.get(group)
    if not services:
        return {"success": False, "error": f"Unknown service group: {group}"}
    results = [_run_pm2(["stop", s]) for s in services]
    return {"success": all(r["success"] for r in results), "results": results}


def restart_services(group: str = "all") -> dict:
    services = SERVICE_GROUPS.get(group)
    if not services:
        return {"success": False, "error": f"Unknown service group: {group}"}
    results = [_run_pm2(["restart", s]) for s in services]
    return {"success": all(r["success"] for r in results), "results": results}


def health_check() -> dict:
    processes = get_process_list()
    health = {"timestamp": time.time(), "healthy": True, "services": {}, "summary": {"online": 0, "stopped": 0, "errored": 0, "total": 0}}
    for proc in processes:
        name = proc.get("name", "unknown")
        status = proc.get("pm2_env", {}).get("status", "unknown")
        restarts = proc.get("pm2_env", {}).get("restart_time", 0)
        memory_mb = proc.get("monit", {}).get("memory", 0) / (1024 * 1024)
        service_health = {"status": status, "pid": proc.get("pid"), "memory_mb": round(memory_mb, 1), "cpu": proc.get("monit", {}).get("cpu", 0), "restarts": restarts, "healthy": status == "online" and restarts < 20}
        health["services"][name] = service_health
        health["summary"]["total"] += 1
        if status == "online": health["summary"]["online"] += 1
        elif status == "stopped": health["summary"]["stopped"] += 1
        else: health["summary"]["errored"] += 1; health["healthy"] = False
    return health


def graceful_shutdown() -> dict:
    shutdown_order = ["market-open", "market-close", "hourly-tasks", "live-loop", "platform-orchestrator", "qwen-model-server", "news-engine", "mirofish-frontend", "mirofish-backend", "express-frontend", "engine-api"]
    results = {}
    for service in shutdown_order:
        result = _run_pm2(["stop", service])
        results[service] = result["success"]
    return {"success": all(results.values()), "results": results}


def startup_sequence() -> dict:
    startup_order = [("engine-api", 5), ("express-frontend", 3), ("news-engine", 2), ("mirofish-backend", 3), ("mirofish-frontend", 2), ("platform-orchestrator", 2), ("live-loop", 2)]
    results = {}
    for service, wait_time in startup_order:
        result = _run_pm2(["start", str(ECOSYSTEM_CONFIG), "--only", service])
        results[service] = result["success"]
        if result["success"]: time.sleep(wait_time)
    return {"success": all(results.values()), "results": results}
