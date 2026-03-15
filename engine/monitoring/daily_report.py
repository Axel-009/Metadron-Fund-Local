"""Daily reporting — open/close reports with sector heatmap.

Generates:
    - Morning open report (pre-market macro scan)
    - Evening close report (reconciliation)
    - GICS sector heatmap (5-bucket ANSI)
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from ..data.yahoo_data import get_adj_close, get_returns
from ..data.universe_engine import SECTOR_ETFS
from ..signals.macro_engine import MacroSnapshot


# ---------------------------------------------------------------------------
# Heatmap buckets
# ---------------------------------------------------------------------------
HEATMAP_BUCKETS = {
    "STRONG_UP": {"min": 0.02, "color": "\033[92m", "symbol": "██"},
    "UP":        {"min": 0.005, "color": "\033[32m", "symbol": "▓▓"},
    "FLAT":      {"min": -0.005, "color": "\033[33m", "symbol": "░░"},
    "DOWN":      {"min": -0.02, "color": "\033[31m", "symbol": "▒▒"},
    "STRONG_DOWN": {"min": -999, "color": "\033[91m", "symbol": "██"},
}
RESET = "\033[0m"


def get_bucket(change: float) -> tuple[str, str, str]:
    """Classify a return into a heatmap bucket."""
    for name, cfg in HEATMAP_BUCKETS.items():
        if change >= cfg["min"]:
            return name, cfg["color"], cfg["symbol"]
    return "STRONG_DOWN", "\033[91m", "██"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_sector_heatmap(date: Optional[str] = None) -> str:
    """ANSI sector heatmap for today."""
    start = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    prices = get_adj_close(list(SECTOR_ETFS.values()), start=start)
    inv = {v: k for k, v in SECTOR_ETFS.items()}

    lines = []
    lines.append("=" * 60)
    lines.append("METADRON CAPITAL — SECTOR HEATMAP")
    lines.append("=" * 60)

    if prices.empty:
        lines.append("  No data available")
        return "\n".join(lines)

    for col in prices.columns:
        sector = inv.get(col, col)
        r = prices[col].dropna()
        if len(r) < 2:
            continue
        change = float(r.iloc[-1] / r.iloc[-2] - 1)
        bucket, color, symbol = get_bucket(change)
        lines.append(f"  {color}{symbol}{RESET} {sector:<30} {change:>+7.2%}  [{bucket}]")

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_open_report(macro: Optional[MacroSnapshot] = None) -> dict:
    """Morning pre-market report."""
    report = {
        "type": "OPEN",
        "timestamp": datetime.now().isoformat(),
        "regime": macro.regime.value if macro else "UNKNOWN",
        "vix": macro.vix if macro else 0,
        "spy_1m": macro.spy_return_1m if macro else 0,
        "sector_rankings": macro.sector_rankings if macro else {},
        "heatmap": generate_sector_heatmap(),
    }
    return report


def generate_close_report(
    portfolio_summary: dict,
    trades: list[dict] = None,
    macro: Optional[MacroSnapshot] = None,
) -> dict:
    """Evening reconciliation report."""
    report = {
        "type": "CLOSE",
        "timestamp": datetime.now().isoformat(),
        "regime": macro.regime.value if macro else "UNKNOWN",
        "portfolio": portfolio_summary,
        "trades_today": trades or [],
        "heatmap": generate_sector_heatmap(),
    }
    return report


def save_report(report: dict, log_dir: Path = Path("logs/reports")):
    """Save report to JSON."""
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    rtype = report.get("type", "unknown").lower()
    path = log_dir / f"{date_str}_{rtype}.json"
    # Strip ANSI codes from heatmap for JSON
    clean = dict(report)
    if "heatmap" in clean:
        import re
        clean["heatmap"] = re.sub(r'\033\[[0-9;]*m', '', clean["heatmap"])
    path.write_text(json.dumps(clean, indent=2, default=str))
    return path
