#!/usr/bin/env python3
"""Metadron Capital — Evening Close (16:00 ET).

Runs: EOD reconciliation → platinum report → agent scorecard update.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from engine.monitoring.daily_report import generate_close_report, save_report, generate_sector_heatmap


def main():
    print("=" * 70)
    print("METADRON CAPITAL — EVENING CLOSE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # In production, this would load the running engine state
    # For now, generate a close report template
    portfolio_summary = {
        "nav": 1_000_000,
        "cash": 800_000,
        "total_pnl": 0,
        "positions": 0,
    }

    report = generate_close_report(portfolio_summary)
    path = save_report(report)

    print(generate_sector_heatmap())
    print()
    print(f"Close report saved to: {path}")
    print()
    print("=" * 70)
    print("EVENING CLOSE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
