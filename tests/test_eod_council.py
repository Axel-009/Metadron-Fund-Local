from engine.allocation.allocation_engine import AllocationRules
from engine.execution.eod_allocation_council import EODAllocationCouncil


class _O:
    def __init__(self, ticker, side, qty, fill, arrival, sector, status="FILLED", slip=2.0):
        self.ticker, self.side, self.quantity, self.fill_price, self.arrival_price = ticker, side, qty, fill, arrival
        self.sector, self.status, self.slippage_bps, self.product_type = sector, status, slip, "EQUITY"
        self.implementation_shortfall, self.fill_quantity = (fill - arrival) / arrival * 1e4, qty


def test_council_verdict_shape(tmp_path):
    rules = AllocationRules()
    orders = [_O("AAPL", "BUY", 10, 230.1, 230.0, "IG_EQUITY"), _O("XLE", "BUY", 5, 88.2, 88.0, "HY_EQUITY", slip=6.0)]
    c = EODAllocationCouncil(rules, log_dir=str(tmp_path))
    v = c.convene(l7_orders=orders, nav=60_000, positions={}, close_prices={"AAPL": 231.0, "XLE": 87.5},
                  drawdown={"level": "OK", "drawdown": 0.0}, macro_flag=None, momentum={"AAPL": 0.4}, regime="NORMAL", vix=15.0)
    nd = v.next_day_allocation
    assert abs(sum(nd.values()) - 1.0) < 1e-6
    assert nd["MARGIN"] == 0.08 and nd["MONEY_MARKET"] == 0.02          # hard rules
    assert len(v.votes) == 5 and v.markdown()
    # ±5 pt clip vs the allocation file
    base = {"IG_EQUITY": rules.ig_equity_pct, "HY_EQUITY": rules.hy_equity_pct}
    for k, b in base.items():
        assert abs(nd[k] - b) <= 0.05 + 1e-9
    assert list(tmp_path.glob("eod_*.json")) and list(tmp_path.glob("eod_*.md"))


def test_apply_to_rules_preserves_hard_rules():
    rules = AllocationRules()
    c = EODAllocationCouncil(rules, log_dir=None)
    v = c.convene(l7_orders=[], nav=10_000, positions={}, close_prices={}, drawdown=None)
    EODAllocationCouncil.apply_to_rules(rules, v)
    assert rules.margin_pct == 0.08 and rules.money_market_pct == 0.02
