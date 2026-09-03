"""SchwabAccountRouter — mandate routing + 20% drawdown rule (offline, nothing sent)."""
import os
import pytest

from engine.execution.account_mandates import (
    DEFAULT_MANDATES, DrawdownGuard, load_mandates, scale_sleeves_for_mandate,
)
from engine.execution.broker_types import OrderSide
from engine.execution.schwab_account_router import SchwabAccountRouter, build_schwab_broker


@pytest.fixture
def router():
    return SchwabAccountRouter(mandates=DEFAULT_MANDATES, connect=False, initial_cash=20_000, live_orders=False)


class TestMandates:
    def test_defaults(self):
        assert DEFAULT_MANDATES["ROTH"].options_pct == 0.25 and DEFAULT_MANDATES["ROTH"].equities_pct == 0.75
        assert DEFAULT_MANDATES["LLC"].options_pct == 0.0 and DEFAULT_MANDATES["LLC"].allow_options is False
        assert DEFAULT_MANDATES["INDIVIDUAL"].options_pct == 1.0 and DEFAULT_MANDATES["INDIVIDUAL"].allow_equities is False
        for m in DEFAULT_MANDATES.values():
            assert m.drawdown_rotate_pct == 0.20

    def test_load_from_env(self):
        env = {"SCHWAB_ACCOUNT_ROTH": "0514", "SCHWAB_ACCOUNT_LLC": "9565", "SCHWAB_ACCOUNT_INDIVIDUAL": "4806"}
        ms = load_mandates(env)
        assert {m.account_last4 for m in ms.values()} == {"9565", "0514", "4806"}

    def test_sleeve_scaling_keeps_hard_rules(self):
        from engine.allocation.allocation_engine import AllocationRules
        caps = scale_sleeves_for_mandate(AllocationRules(), DEFAULT_MANDATES["INDIVIDUAL"])
        assert caps["MARGIN"] == pytest.approx(0.08) and caps["MONEY_MARKET"] == pytest.approx(0.02)
        assert caps["OPTIONS_IG"] > caps.get("IG_EQUITY", 0.0)


class TestRouting:
    def test_options_go_to_individual(self, router):
        o = router.place_option_order("SPY   260904C00650000", "BUY_TO_OPEN", 1, limit_price=1.25, underlying="SPY")
        assert router.get_routing_log()[-1]["account"] == "INDIVIDUAL"
        assert str(o.status).endswith("DRY_RUN")

    def test_equities_go_to_llc(self, router):
        router.place_order("AAPL", OrderSide.BUY, 3, reason="t", sector="IG_EQUITY")
        assert router.get_routing_log()[-1]["account"] == "LLC"

    def test_llc_never_gets_options(self, router):
        for _ in range(3):
            router.place_option_order("QQQ   260904P00450000", "BUY_TO_OPEN", 1, limit_price=1.0, underlying="QQQ")
        assert all(r["account"] != "LLC" for r in router.get_routing_log())

    def test_aggregate_state(self, router):
        assert router.state.nav == pytest.approx(20_000)   # initial_cash is the portfolio total, split across mandates
        snap = router.portfolio_snapshot()
        assert set(snap["accounts"]) == {"ROTH", "LLC", "INDIVIDUAL"}
        assert snap["drawdown"]["adds_allowed"] is True


class TestDrawdownRule:
    def test_guard_levels(self):
        g = DrawdownGuard()
        g.seed_peak("X", 100.0)
        assert g.check("X", 95.0).level == "OK"
        assert g.check("X", 84.0).level == "WARN" and g.check("X", 84.0).add_scale == 0.5
        st = g.check("X", 79.0)
        assert st.level == "ROTATE_OR_CLOSE" and st.adds_allowed is False and st.add_scale == 0.0

    def test_account_in_drawdown_is_skipped(self, router):
        llc = router.brokers["LLC"]
        router.guard.seed_peak("LLC", llc.state.nav)
        llc.state.cash = llc.state.nav * 0.75          # -25 %
        llc.state.nav = llc.state.cash
        router.place_order("MSFT", OrderSide.BUY, 2, reason="t", sector="IG_EQUITY")
        assert router.get_routing_log()[-1]["account"] == "ROTH"

    def test_rotation_plan(self):
        g = DrawdownGuard(); g.seed_peak("P", 100.0); g.check("P", 70.0)
        plan = g.rotation_plan("P", {"XYZ": {"quantity": 10, "unrealized_pnl": -500.0, "realized_pnl": 0.0, "sector": "HY_EQUITY"}})
        assert plan and plan[0]["symbol"] == "XYZ"


def test_factory_single_vs_router(monkeypatch):
    for k in ("SCHWAB_ACCOUNT_ROTH", "SCHWAB_ACCOUNT_LLC", "SCHWAB_ACCOUNT_INDIVIDUAL", "SCHWAB_ACCOUNT_MANDATES"):
        monkeypatch.delenv(k, raising=False)
    assert type(build_schwab_broker(connect=False)).__name__ == "SchwabBroker"
    monkeypatch.setenv("SCHWAB_ACCOUNT_ROTH", "0514")
    monkeypatch.setenv("SCHWAB_ACCOUNT_LLC", "9565")
    monkeypatch.setenv("SCHWAB_ACCOUNT_INDIVIDUAL", "4806")
    assert type(build_schwab_broker(connect=False)).__name__ == "SchwabAccountRouter"
