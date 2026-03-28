#!/usr/bin/env python3
"""Metadron Capital — Hourly Scan.

Runs every hour during market hours (09:30–16:00 ET).
Performs: position drift check → signal refresh → risk monitoring → hourly recap.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from engine.execution.execution_engine import ExecutionEngine
from engine.monitoring.daily_report import generate_sector_heatmap


def main():
    now = datetime.now()
    hour = now.strftime("%H:%M")

    print("=" * 70)
    print(f"METADRON CAPITAL — HOURLY SCAN [{hour}]")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Initialize engine (in production, load persisted state)
    # Initialize engine (NAV resolved dynamically from Alpaca)
    engine = ExecutionEngine()  # NAV auto-resolved from broker

    # Run signal refresh (lighter than full pipeline)
    print("Running signal refresh...")
    result = engine.run_pipeline()

    stages = result.get("stages", {})

    # Macro update
    macro_data = stages.get("macro", {})
    print(f"REGIME: {macro_data.get('regime', 'UNKNOWN')}")
    print(f"  VIX: {macro_data.get('vix', 0):.1f}")
    print()

    # Cube update
    cube = stages.get("cube", {})
    print(f"CUBE: Regime={cube.get('regime', 'RANGE')} | Beta={cube.get('target_beta', 0):.3f}")
    print()

    # Active signals
    alpha = stages.get("alpha", {})
    top = alpha.get("top_signals", [])
    if top:
        print(f"TOP SIGNALS ({len(top)}):")
        for s in top[:5]:
            print(f"  {s['ticker']:<8} Tier={s['tier']} Alpha={s['alpha']:.4f}")
        print()

    # Execution summary
    exec_data = stages.get("execution", {})
    trades = exec_data.get("trades", [])
    portfolio = exec_data.get("portfolio", {})

    print(f"TRADES THIS SCAN: {len(trades)}")
    print(f"PORTFOLIO: NAV=${portfolio.get('nav', 0):,.2f} | "
          f"Positions={portfolio.get('positions', 0)} | "
          f"PnL=${portfolio.get('total_pnl', 0):,.2f}")
    print()

    # Heatmap
    try:
        print(generate_sector_heatmap())
    except Exception:
        print("  [Heatmap unavailable]")
    print()

    print("=" * 70)
    print(f"HOURLY SCAN COMPLETE [{hour}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
