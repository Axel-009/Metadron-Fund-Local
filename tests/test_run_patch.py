from engine.execution.execution_engine import ExecutionEngine
from engine.execution.run_patch_report import build_run_patch


def test_patch_renders_offline(monkeypatch):
    monkeypatch.setenv("SCHWAB_ACCOUNT_ROTH", "9565"); monkeypatch.setenv("SCHWAB_ACCOUNT_LLC", "0514"); monkeypatch.setenv("SCHWAB_ACCOUNT_INDIVIDUAL", "4806")
    e = ExecutionEngine(initial_nav=60_000, connect_broker=False)
    e.l7_submit(ticker="AAPL", side="BUY", quantity=3, signal_type="QUALITY_BUY", regime="NORMAL", sector="IG_EQUITY", reason="t")
    md = build_run_patch(exec_engine=e, save_dir=None)
    for must in ("Options scan (1–7 DTE)", "Account mandates", "ROTH", "INDIVIDUAL", "Rotation recommendation", "DRY_RUN"):
        assert must in md
