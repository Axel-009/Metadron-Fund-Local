#!/usr/bin/env python3
"""Metadron Capital — Evening Close (16:00 ET).

Runs: EOD reconciliation → full reports → agent scorecard → missed opportunities.

Generates:
    - Platinum Report (close variant with EOD stats)
    - Portfolio Analytics Report (full performance deep dive)
    - Agent scorecard and hierarchy update
    - Missed opportunities analysis (>20% movers not captured)
    - Statistical anomalies log
    - Market wrap narrative
    - Sector heatmap
    - Memory/session EOD summary
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

    # Pull portfolio state from Alpaca (dynamic NAV)
    try:
        from engine.execution.alpaca_broker import AlpacaBroker
        broker = AlpacaBroker(initial_cash=0, paper=True)
        portfolio_summary = broker.get_portfolio_summary()
        print(f"Portfolio: NAV=${portfolio_summary.get('nav', 0):,.2f}, "
              f"Positions={portfolio_summary.get('positions', 0)}")
    except Exception as e:
        # Fallback: load from persistent state file
        import json as _json
        _state_path = Path("state/portfolio.json")
        if _state_path.exists():
            try:
                portfolio_summary = _json.loads(_state_path.read_text())
            except Exception:
                portfolio_summary = {"nav": 0, "cash": 0, "total_pnl": 0, "positions": 0}
        else:
            portfolio_summary = {"nav": 0, "cash": 0, "total_pnl": 0, "positions": 0}
        print(f"Warning: Alpaca unavailable ({e}), using fallback state")

    # Heatmap
    print(generate_sector_heatmap())
    print()

    # Market Wrap
    try:
        from engine.monitoring.market_wrap import MarketWrapGenerator
        wrap_gen = MarketWrapGenerator()
        print(wrap_gen.generate_ascii())
        print()
    except Exception as e:
        print(f"  Market wrap: {e}")

    # Sector Tracker — missed opportunities
    try:
        from engine.monitoring.sector_tracker import SectorTrackingEngine
        tracker = SectorTrackingEngine()

        # Missed opportunities (>20% movers)
        missed = tracker.get_missed_opportunities([])
        if missed:
            print("=" * 70)
            print("MISSED OPPORTUNITIES (>20% movers not captured)")
            print("=" * 70)
            for m in missed[:10]:
                ticker = m.get("ticker", "???")
                change = m.get("change_pct", 0)
                reason = m.get("reason", "Unknown")
                print(f"  {ticker:<8} {change:>+7.1f}% | {reason}")
            print()

        # Full dashboard
        print(tracker.format_sector_dashboard())
        print()

        # Error summary
        error_summary = tracker.get_error_summary()
        if error_summary:
            print("ERROR LOG:")
            for cat, count in error_summary.items():
                print(f"  {cat}: {count}")
            print()
    except Exception as e:
        print(f"  Sector tracker: {e}")

    # Contagion Engine — EOD systemic risk
    try:
        from engine.signals.contagion_engine import ContagionEngine
        contagion = ContagionEngine()
        scenarios = contagion.run_all_scenarios()
        print("=" * 70)
        print("CONTAGION ENGINE — SYSTEMIC RISK")
        print("=" * 70)
        for name, result in list(scenarios.items())[:5]:
            risk = result.get("systemic_risk", 0)
            print(f"  {name:<25} Systemic Risk: {risk:.3f}")
        print()
    except Exception as e:
        print(f"  Contagion engine: {e}")

    # Stat Arb — EOD pair status
    try:
        from engine.signals.stat_arb_engine import StatArbEngine
        stat_arb = StatArbEngine()
        print(stat_arb.format_stat_arb_report())
        print()
    except Exception as e:
        print(f"  Stat arb engine: {e}")

    # Anomaly Detection
    try:
        from engine.monitoring.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector()
        anomalies = detector.scan()
        if anomalies:
            print("=" * 70)
            print(f"STATISTICAL ANOMALIES: {len(anomalies)} detected")
            print("=" * 70)
            for a in anomalies[:10]:
                print(f"  [{a.get('severity', '?')}] {a.get('type', '?')}: {a.get('description', '')}")
            print()
    except Exception as e:
        print(f"  Anomaly detector: {e}")

    # Research Bots — weekly scorecard (if Friday)
    try:
        from engine.agents.research_bots import ResearchBotManager
        research_mgr = ResearchBotManager()
        research_mgr.load_state()

        # Always print DNA report at close
        print(research_mgr.print_dna_report())
        print()

        # Weekly scorecard update (every day, rank weekly on Friday)
        today = datetime.now()
        if today.weekday() == 4:  # Friday
            print("WEEKLY SCORECARD UPDATE (Friday)")
            research_mgr.update_weekly_scores()
            research_mgr.save_state()
            print("  Scores updated and saved.")
            print()
    except Exception as e:
        print(f"  Research bots: {e}")

    # Agent Scorecard (existing agent system)
    try:
        from engine.agents.agent_scorecard import AgentScorecardManager
        scorecard_mgr = AgentScorecardManager()
        print(scorecard_mgr.print_scorecard())
        print()
    except Exception as e:
        print(f"  Agent scorecard: {e}")

    # Conviction Override audit
    try:
        from engine.execution.conviction_override import ConvictionOverrideManager
        override_mgr = ConvictionOverrideManager()
        print(override_mgr.print_audit_trail())
        print()
    except Exception as e:
        print(f"  Conviction override: {e}")

    # ─── REPORT GENERATION (all 8 report types) ──────────────
    datestamp = datetime.now().strftime('%Y%m%d')

    def _save_report(name: str, log_subdir: str, content: str):
        """Persist report to logs/<subdir>/ and print."""
        d = Path(f"logs/{log_subdir}")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"close_{datestamp}.txt"
        path.write_text(content)
        print(f"  [{name}] saved -> {path}")

    # 1. Platinum Report
    try:
        from engine.monitoring.platinum_report import PlatinumReportGenerator
        plat_gen = PlatinumReportGenerator()
        plat_report = plat_gen.generate_close_report()
        print(plat_report)
        _save_report("Platinum Report", "platinum", plat_report)
    except Exception as e:
        print(f"  Platinum report: {e}")

    # 2. Daily P&L Report — handled by generate_close_report() call below

    # 3. Portfolio Analytics Report
    try:
        from engine.monitoring.portfolio_report import PortfolioReportGenerator
        port_gen = PortfolioReportGenerator()
        port_report = port_gen.generate_close_report()
        print(port_report)
        _save_report("Portfolio Analytics", "portfolio", port_report)
    except Exception as e:
        print(f"  Portfolio report: {e}")

    # 4. Risk Dashboard Report
    try:
        from engine.monitoring.risk_report import RiskReportGenerator
        risk_gen = RiskReportGenerator()
        risk_report = risk_gen.generate()
        print(risk_report)
        _save_report("Risk Dashboard", "risk", risk_report)
    except Exception as e:
        print(f"  Risk report: {e}")

    # 5. Execution Quality Report
    try:
        from engine.execution.execution_engine import ExecutionEngine
        exec_eng = ExecutionEngine()
        exec_report = exec_eng.format_execution_report() if hasattr(exec_eng, "format_execution_report") else "Execution report unavailable"
        print(exec_report)
        _save_report("Execution Quality", "execution", exec_report)
    except Exception as e:
        print(f"  Execution report: {e}")

    # 6. Monthly Investor Letter
    try:
        from engine.monitoring.investor_report import InvestorReportGenerator
        inv_gen = InvestorReportGenerator()
        inv_report = inv_gen.generate()
        print(inv_report)
        _save_report("Monthly Investor", "investor", inv_report)
    except Exception as e:
        print(f"  Investor report: {e}")

    # 7. Compliance Report
    try:
        from engine.monitoring.compliance_report import ComplianceReportGenerator
        comp_gen = ComplianceReportGenerator()
        comp_report = comp_gen.generate()
        print(comp_report)
        _save_report("Compliance Report", "compliance", comp_report)
    except Exception as e:
        print(f"  Compliance report: {e}")

    # 8. ML Model Report
    try:
        from engine.monitoring.learning_loop import LearningLoop
        ll = LearningLoop()
        ml_report = ll.format_learning_report() if hasattr(ll, "format_learning_report") else "ML model report unavailable"
        print(ml_report)
        _save_report("ML Model Report", "ml-model", ml_report)
    except Exception as e:
        print(f"  ML model report: {e}")

    # Memory/Session EOD summary
    try:
        from engine.monitoring.memory_monitor import MemoryMonitor
        mem = MemoryMonitor()
        print(mem.get_eod_report())
    except Exception as e:
        print(f"  Memory monitor: {e}")

    # Close report
    report = generate_close_report(portfolio_summary)
    path = save_report(report)
    print(f"\nClose report saved to: {path}")

    # ─── Post-session file generation ──────────────────
    # Generates dated files for ARCHIVE tab: TX logs, recon, learning, ML, errors
    print()
    print("─" * 50)
    print("POST-SESSION FILE GENERATION")
    print("─" * 50)
    try:
        from engine.ops.session_close import generate_session_files
        files = generate_session_files()
        for name, fpath in files.items():
            print(f"  {name}: {fpath}")
    except Exception as e:
        print(f"  Session file generation failed: {e}")

    print()
    print("=" * 70)
    print("EVENING CLOSE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
