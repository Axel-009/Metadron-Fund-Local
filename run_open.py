#!/usr/bin/env python3
"""Metadron Capital — Morning Open (09:30 ET).

Runs: macro scan → universe ranking → morning signals → open report.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from engine.execution.execution_engine import ExecutionEngine
from engine.monitoring.daily_report import generate_open_report, generate_sector_heatmap, save_report


def main():
    print("=" * 70)
    print("METADRON CAPITAL — MORNING OPEN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Initialize engine
    print("Initializing execution engine...")
    engine = ExecutionEngine(initial_nav=1_000_000.0)

    # Run full pipeline
    print("Running signal pipeline: Universe → Macro → Cube → Alpha → Execute")
    print()
    result = engine.run_pipeline()

    # Display results
    stages = result.get("stages", {})

    # Macro
    macro_data = stages.get("macro", {})
    print(f"REGIME: {macro_data.get('regime', 'UNKNOWN')}")
    print(f"  VIX: {macro_data.get('vix', 0):.1f}")
    print(f"  SPY 1M: {macro_data.get('spy_1m', 0):.2%}")
    print(f"  SPY 3M: {macro_data.get('spy_3m', 0):.2%}")
    print()

    # Sector rankings
    rankings = macro_data.get("sector_rankings", {})
    if rankings:
        print("SECTOR RANKINGS (by macro favourability):")
        for i, (sector, score) in enumerate(rankings.items()):
            print(f"  {i+1:>2}. {sector:<30} {score:>+8.4f}")
        print()

    # Cube
    cube = stages.get("cube", {})
    print(f"METADRON CUBE:")
    print(f"  Regime: {cube.get('regime', 'RANGE')}")
    print(f"  Target Beta: {cube.get('target_beta', 0):.3f}")
    print(f"  Beta Cap: {cube.get('beta_cap', 0):.2f}")
    print(f"  Max Leverage: {cube.get('max_leverage', 0):.1f}x")
    print(f"  Risk Budget: {cube.get('risk_budget', 0):.1%}")
    sleeves = cube.get("sleeves", {})
    if sleeves:
        print("  Sleeves:")
        for k, v in sleeves.items():
            print(f"    {k}: {v:.1%}")
    print()

    # Alpha
    alpha = stages.get("alpha", {})
    print(f"ALPHA OPTIMIZER:")
    print(f"  Expected Return: {alpha.get('expected_return', 0):.2%}")
    print(f"  Volatility: {alpha.get('volatility', 0):.2%}")
    print(f"  Sharpe: {alpha.get('sharpe', 0):.2f}")
    top = alpha.get("top_signals", [])
    if top:
        print("  Top signals:")
        for s in top[:5]:
            print(f"    {s['ticker']:<8} Tier={s['tier']} Alpha={s['alpha']:.4f}")
    print()

    # Execution
    exec_data = stages.get("execution", {})
    trades = exec_data.get("trades", [])
    print(f"TRADES: {len(trades)} executed")
    for t in trades[:10]:
        print(f"  {t['side']:<4} {t.get('qty', 0):>6} {t['ticker']:<8} @ vote={t.get('vote_score', 0):.1f}")
    print()

    # Portfolio
    portfolio = exec_data.get("portfolio", {})
    print(f"PORTFOLIO:")
    print(f"  NAV: ${portfolio.get('nav', 0):,.2f}")
    print(f"  Cash: ${portfolio.get('cash', 0):,.2f}")
    print(f"  PnL: ${portfolio.get('total_pnl', 0):,.2f}")
    print(f"  Positions: {portfolio.get('positions', 0)}")
    print(f"  Gross Exp: {portfolio.get('gross_exposure', 0):.1%}")
    print(f"  Net Exp: {portfolio.get('net_exposure', 0):.1%}")
    print()

    # Heatmap
    print(generate_sector_heatmap())
    print()

    # Generate and save open report
    from engine.signals.macro_engine import MacroEngine
    macro_engine = MacroEngine()
    macro_snap = macro_engine.analyze()
    report = generate_open_report(macro_snap)
    path = save_report(report)
    print(f"Report saved to: {path}")
    print()
    print("=" * 70)
    print("MORNING OPEN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
