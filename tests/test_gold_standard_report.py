"""Gold-standard report renderer + rotation exits (VIEW 1 / VIEW 2 / TX LOG / VIEW 3)."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from engine.execution.broker_types import Position
from engine.execution.gold_standard_report import (
    collect_context, render, render_view1, render_view2, render_transaction_log, render_view3,
    classify_bucket, norm_bucket, fill_status, GoldStandardReporter,
)
from engine.execution.rotation_exits import compute_rotation_exits, RotationExitEngine
from engine.execution.universe_tranche_scan import UniverseTrancheScanner, TrancheConfig
from tests.test_universe_tranche_scan import FakeSchwab


def _pos(t, qty, cost, px, sector="EQUITY"):
    return Position(ticker=t, quantity=qty, avg_cost=cost, current_price=px,
                    unrealized_pnl=(px - cost) * qty, sector=sector)


class FakeBroker:
    is_connected = True
    live_orders = False
    account_display = "…9565 / …0514 / …4806"

    def __init__(self, positions):
        self.state = SimpleNamespace(nav=150_000.0, cash=6_000.0, positions=positions)

    def sync_account(self):
        return {"buying_power": 120_000.0}

    def get_daily_pnl(self):
        return 412.5

    def portfolio_snapshot(self):
        return {"drawdown": {"drawdown": 0.03, "level": "OK", "adds_allowed": True, "add_scale": 1.0},
                "accounts": {"ROTH": {"account": "XXXX9565", "nav": 50_000, "options_notional": 4_000, "options_cap": 12_500,
                                      "equities_mv": 30_000, "equities_cap": 37_500,
                                      "drawdown": {"drawdown": 0.02, "level": "OK"},
                                      "mandate": {"options_pct": 0.25, "equities_pct": 0.75, "account_last4": "9565"}}}}

    def get_price_history(self, t, days=120, frequency="daily"):
        return FakeSchwab([t, "SPY"]).get_price_history(t)


class FakeL7:
    def get_summary(self):
        return {"kill_switch": False, "total_orders_today": 5, "total_fills_today": 0, "total_dry_run_today": 5,
                "risk_level": "LOW", "avg_tca_cost_bps": 3.2, "tca_trend": "flat", "var_95_1d": 1_850.0}


@pytest.fixture
def book():
    return {
        "AAPL": _pos("AAPL", 40, 180.0, 195.0),          # IG (SP500)
        "MSFT": _pos("MSFT", 20, 400.0, 300.0),          # IG, -25% → DRAWDOWN_EXIT
        "TLTW": _pos("TLTW", 300, 25.0, 26.0),           # TLTW_CASHFLOW
        "AGG": _pos("AGG", 50, 98.0, 99.0),              # FI_MACRO
        "HYG": _pos("HYG", 60, 77.0, 78.0),              # DISTRESSED
        "XLE": _pos("XLE", -30, 90.0, 92.0),             # short hedge
        "SPY   260905C00560000": _pos("SPY   260905C00560000", 2, 310.0, 350.0, sector="OPTIONS"),
    }


@pytest.fixture
def tranche():
    tickers = [f"L{i}" for i in range(40)] + ["SPY", "TLT", "HYG", "AGG", "TLTW", "AAPL", "MSFT"]
    fb = FakeSchwab(tickers)
    cfg = TrancheConfig(shortlist_per_tranche=12, top_per_tranche=5, final_max_names=8, min_per_tranche=1)
    sc = UniverseTrancheScanner(fb, cfg=cfg, options_engine=None,
                                tranches=[("SCAN_1_SP500", [f"L{i}" for i in range(40)] + ["AAPL", "MSFT"]),
                                          ("SCAN_4_ETF_FI", ["SPY", "TLT", "HYG", "AGG", "TLTW"])])
    return sc.run(corridor_bias=0.0)


def test_bucket_normalisation_and_classification():
    assert norm_bucket("TLTW") == "TLTW_CASHFLOW" and norm_bucket("FIXED_INCOME") == "FI_MACRO"
    assert norm_bucket("DISTRESSED_EQUITY") == "DISTRESSED" and norm_bucket("EVENT_DRIVEN_CVR") == "EVENT_CVR"
    assert classify_bucket("SPY   260905C00560000") == "OPTIONS"
    assert classify_bucket("ZZZ", "OPTIONS") == "OPTIONS"
    assert classify_bucket("TLTW") == "TLTW_CASHFLOW" and classify_bucket("AGG") == "FI_MACRO"
    assert classify_bucket("HYG") == "DISTRESSED" and classify_bucket("AAPL") == "IG_EQUITY"
    assert classify_bucket("FOO", hints={"FOO": "HY_EQUITY"}) == "HY_EQUITY"
    assert fill_status(10) == "○ LIGHT" and fill_status(50) == "◐ PARTIAL" and fill_status(100) == "✓ FILLED" and fill_status(130) == "⚠ OVER"


def test_rotation_exits_drawdown_and_alpha_floor(book, tranche):
    ex = compute_rotation_exits(book, tranche, extra_sell_tickers=["AAPL"])
    names = {e.ticker: e for e in ex}
    assert "MSFT" in names and names["MSFT"].reason_short == "20% drawdown rule" and names["MSFT"].side == "SELL"
    assert "AAPL" in names and names["AAPL"].reason_short == "alpha floor breach"
    assert all(" " not in e.ticker for e in ex)                    # options never rotated here
    assert ex[0].pnl_pct <= ex[-1].pnl_pct                          # worst first
    eng = RotationExitEngine()
    assert len(eng.evaluate(book, tranche)) >= 1 and eng.last


def test_full_report_renders_all_views(book, tranche):
    broker = FakeBroker(book)
    exec_engine = SimpleNamespace(broker=broker, l7=FakeL7(), options_engine_short_dte=None)
    orders = [
        {"ticker": "MSFT", "side": "SELL", "quantity": 20, "fill_price": 300.0, "fill_quantity": 20, "status": "DRY_RUN", "sector": "IG_EQUITY", "product_type": "EQUITY", "reason": "drawdown_exit (-25.0%)"},
        {"ticker": "AAPL", "side": "BUY", "quantity": 10, "fill_price": 195.0, "fill_quantity": 10, "status": "DRY_RUN", "sector": "IG_EQUITY", "product_type": "EQUITY", "alpha": 1.23, "signal_type": "QUALITY_BUY"},
        {"ticker": "SPY", "side": "BUY", "quantity": 2, "fill_price": 3.5, "fill_quantity": 2, "status": "DRY_RUN", "product_type": "OPTION", "contract_symbol": "SPY   260905C00560000", "composite": 0.71},
    ]
    exits = compute_rotation_exits(book, tranche)
    options_report = {"universe": [["SPY", "OPTIONS_IG"]], "committed_by_bucket": {"OPTIONS_IG": 700.0},
                      "per_ticker": {"SPY": {"status": "INTENT", "reasons": []}, "TLT": {"status": "NO_EDGE", "reasons": ["edge 4bp < floor"]}},
                      "intents": [{"ticker": "SPY", "direction": "LONG", "put_call": "CALL", "strike": 560, "expiry": "2026-09-05", "dte": 2,
                                   "contracts": 2, "limit_price": 3.5, "notional": 700.0, "composite": 0.71, "edge_bps": 38, "bucket": "OPTIONS_IG",
                                   "greeks": {"delta_exposure_usd": 51_000}, "rationale": ["RSI 61 breakout", "β corridor long bias"]}]}
    flag = SimpleNamespace(to_dict=lambda: {"level": "WATCH", "score": 1.0, "triggers": ["VIX +9%"], "options_add_scale": 0.85, "equities_add_scale": 1.0})
    council = SimpleNamespace(execution_grade="B+", next_day_allocation={"IG_EQUITY": 0.38, "HY_EQUITY": 0.11}, summary=["IG slippage 4bp"])
    ctx = collect_context(exec_engine, tranche_result=tranche, options_report=options_report, orders=orders, exits=exits,
                          macro_flag=flag, drawdown=broker.portfolio_snapshot()["drawdown"], council=council, vix=17.2, cycle_time_s=42.0,
                          as_of=datetime(2026, 9, 3, 10, 30))
    assert ctx.mode == "DRY_RUN" and ctx.nav == 150_000.0 and ctx.buying_power == 120_000.0
    assert {p.bucket for p in ctx.positions} >= {"IG_EQUITY", "TLTW_CASHFLOW", "FI_MACRO", "DISTRESSED", "OPTIONS"}
    assert len(ctx.shorts()) == 1 and ctx.shorts()[0].symbol == "XLE"
    assert any(k == "DRAWDOWN_EXIT" and s == "MSFT" for k, s, _ in ctx.monitoring["anomalies"])
    assert ctx.macro["cube"] and ctx.macro["max_leverage"] is not None      # MetadronCube ran off Schwab-fed snapshot
    assert ctx.macro["gates"] and set(ctx.macro["gates"]) == {"G1_flow", "G2_macro", "G3_fund", "G4_mom"}
    assert len(ctx.deployments) == 2 and {d["bucket"] for d in ctx.deployments} == {"IG_EQUITY", "OPTIONS"}

    txt = render(ctx)
    for must in ["VIEW 1 — THINKING OUTPUT", "PHASE 2: MACRO ENGINE + METADRON CUBE", "RUN 1: SP500", "RUN 2: ETF_FI",
                 "Top 5 BUY signals", "WHY:", "SHORT-DTE OPTIONS ENGINE", "INTENT SPY", "PHASE 5: EXECUTION DECISIONS",
                 "EXIT: MSFT", "LEARNING LOOP (in-chat patch", "PHASE 7: MONITORING", "DRAWDOWN_EXIT: MSFT",
                 "VIEW 2 — SCORECARD + ALLOCATION", "SCAN SCORECARDS", "ALLOCATION FILL RATES", "EQUITY SUBTOTAL", "OPTIONS (notional)",
                 "MONEY MARKET (2%)", "TRANSACTION LOG", "ROTATION EXITS (", "NEW DEPLOYMENTS (2)", "Capital freed", "Net flow",
                 "VIEW 3 — LIVE POSITION PANEL", "┌─ IG_EQUITY", "┌─ TLTW_CASHFLOW", "SHORT HEDGE BOOK", "TABLE 2 — DERIVATIVES OVERLAY",
                 "┌─ OPTIONS — All Contracts", "PORTFOLIO GRAND TOTAL", "ALLOCATION CAPACITY SUMMARY", "ACCOUNT MANDATES", "EOD EQUITIES/ETF COUNCIL",
                 "←NEW", "Futures:  retired"]:
        assert must in txt, must
    assert "FUTURES —" not in txt                                          # futures rows omitted by design
    assert "═" * 114 in render_view1(ctx) and "═" * 135 in render_view3(ctx)
    assert f"Trades: {len(exits) + 2}" in render_transaction_log(ctx)
    assert "⚠ BELOW FLOOR" in render_view2(ctx) or "✓ FLOOR HELD" in render_view2(ctx)

    recap = render(ctx, recap=True)
    assert recap.startswith("═") and "END-OF-DAY RECAP" in recap.splitlines()[1]
    rep = GoldStandardReporter()
    assert rep.report(exec_engine, orders=orders, as_of=datetime(2026, 9, 3, 16, 0)).count("VIEW") >= 3


def test_offline_engine_renders_without_data():
    exec_engine = SimpleNamespace(broker=None, l7=None, options_engine_short_dte=None)
    txt = render(collect_context(exec_engine, mode="OFFLINE", as_of=datetime(2026, 9, 3, 9, 30)))
    assert "Mode: OFFLINE" in txt and "(no universe runs this cycle" in txt and "PORTFOLIO GRAND TOTAL" in txt
