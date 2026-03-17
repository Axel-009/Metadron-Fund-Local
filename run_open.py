#!/usr/bin/env python3
"""Metadron Capital — Morning Open (09:30 ET).

Runs: macro scan → universe ranking → morning signals → full pipeline → reports.

Generates:
    - Full signal pipeline execution
    - Platinum Report (30-section executive macro state)
    - Portfolio Analytics Report (scenario engine + performance)
    - Research Bot intelligence (11 GICS sector bots)
    - Sector heatmap
    - Memory/session status
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

    # Execution report (ASCII)
    print(engine.format_execution_report())
    print()

    # Heatmap
    print(generate_sector_heatmap())
    print()

    # Research Bots
    try:
        from engine.agents.research_bots import ResearchBotManager
        print("Running 11 GICS sector research bots...")
        research_mgr = ResearchBotManager()
        # Run research with universe data
        universe_data = {}
        for sector in engine.universe.get_sectors():
            secs = engine.universe.get_by_sector(sector)
            universe_data[sector] = [s.ticker for s in secs[:15]]
        research_mgr.run_daily_research(universe_data)
        print(research_mgr.print_dna_report())
        print()
        research_mgr.save_state()
    except Exception as e:
        print(f"  Research bots: {e}")

    # Contagion Engine
    try:
        from engine.signals.contagion_engine import ContagionEngine
        contagion = ContagionEngine()
        print(contagion.format_contagion_report())
        print()
    except Exception as e:
        print(f"  Contagion engine: {e}")

    # Statistical Arbitrage Engine
    try:
        from engine.signals.stat_arb_engine import StatArbEngine
        stat_arb = StatArbEngine()
        signals = stat_arb.get_trading_signals()
        print(stat_arb.format_stat_arb_report())
        print()
    except Exception as e:
        print(f"  Stat arb engine: {e}")

    # Distressed Asset Engine
    try:
        from engine.signals.distressed_asset_engine import DistressedAssetEngine
        dae = DistressedAssetEngine()
        dae.analyze()
        print(dae.format_distress_report())
        print()
    except Exception as e:
        print(f"  Distressed asset engine: {e}")

    # CVR Engine
    try:
        from engine.signals.cvr_engine import CVREngine
        cvr = CVREngine()
        cvr.analyze()
        print(cvr.format_cvr_report())
        print()
    except Exception as e:
        print(f"  CVR engine: {e}")

    # Event-Driven Engine
    try:
        from engine.signals.event_driven_engine import EventDrivenEngine
        ede = EventDrivenEngine()
        ede.analyze()
        print(ede.format_event_report())
        print()
    except Exception as e:
        print(f"  Event-driven engine: {e}")

    # Options Theta-Gamma Optimizer
    try:
        from engine.execution.options_engine import OptionsStrategyBuilder
        opt_builder = OptionsStrategyBuilder(spot=450.0, vol=0.18, rate=0.05)
        tg_result = opt_builder.theta_gamma_optimize()
        metrics = tg_result.get("metrics", {})
        print("OPTIONS θ+Γ OPTIMIZER:")
        print(f"  Formula: {tg_result.get('formula', 'max(Θ+Γ) - c·V')}")
        print(f"  Best OTM: {metrics.get('otm_pct', 0):.0%}")
        print(f"  Theta: {metrics.get('theta', 0):.4f}")
        print(f"  Gamma: {metrics.get('gamma', 0):.6f}")
        print(f"  Score: {metrics.get('score', 0):.4f}")
        print()
    except Exception as e:
        print(f"  Options optimizer: {e}")

    # Sector Tracker
    try:
        from engine.monitoring.sector_tracker import SectorTrackingEngine
        tracker = SectorTrackingEngine()
        print(tracker.format_sector_dashboard())
        print()
    except Exception as e:
        print(f"  Sector tracker: {e}")

    # Platinum Report
    try:
        from engine.monitoring.platinum_report import PlatinumReportGenerator
        plat_gen = PlatinumReportGenerator()
        plat_report = plat_gen.generate_open_report()
        print(plat_report)
        # Save to logs
        log_dir = Path("logs/platinum")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"open_{datetime.now().strftime('%Y%m%d')}.txt").write_text(plat_report)
    except Exception as e:
        print(f"  Platinum report: {e}")

    # Portfolio Analytics Report
    try:
        from engine.monitoring.portfolio_report import PortfolioReportGenerator
        port_gen = PortfolioReportGenerator()
        port_report = port_gen.generate_open_report()
        print(port_report)
        log_dir = Path("logs/portfolio")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"open_{datetime.now().strftime('%Y%m%d')}.txt").write_text(port_report)
    except Exception as e:
        print(f"  Portfolio report: {e}")

    # Memory/Session status
    try:
        from engine.monitoring.memory_monitor import MemoryMonitor
        mem = MemoryMonitor()
        print(mem.print_report())
    except Exception as e:
        print(f"  Memory monitor: {e}")

    # Credit Quality Scoring
    try:
        from engine.ml.universe_classifier import UniverseClassifier
        classifier = UniverseClassifier()
        print("CREDIT QUALITY SCORING:")
        print(f"  Classifier loaded: {classifier is not None}")
        print()
    except Exception as e:
        print(f"  Credit quality scoring: {e}")

    # Macro Intelligence (GMTF)
    try:
        from engine.signals.macro_engine import MacroEngine as ME2
        me2 = ME2()
        if hasattr(me2, 'generate_macro_intelligence'):
            print(me2.generate_macro_intelligence())
            print()
    except Exception as e:
        print(f"  Macro intelligence: {e}")

    # Generate and save open report
    from engine.signals.macro_engine import MacroEngine
    macro_engine = MacroEngine()
    macro_snap = macro_engine.analyze()
    report = generate_open_report(macro_snap)
    path = save_report(report)
    print(f"\nReport saved to: {path}")
    print()
    print("=" * 70)
    print("MORNING OPEN COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
