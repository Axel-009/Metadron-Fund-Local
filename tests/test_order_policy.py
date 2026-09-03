"""House order policy: MARKET / regular session / DAY for equities, single-leg options and spreads."""
from engine.execution.schwab_broker import _apply_order_policy


def test_market_day_normal_for_all_products():
    for p in ({"orderType": "LIMIT", "price": "91.20", "session": "NORMAL", "duration": "GOOD_TILL_CANCEL"},
              {"orderType": "LIMIT", "price": "1.25", "session": "AM", "duration": "DAY"},
              {"orderType": "NET_DEBIT", "price": "1.10", "session": "NORMAL", "duration": "DAY"}):
        out = _apply_order_policy(dict(p))
        assert out["orderType"] == "MARKET" and out["duration"] == "DAY" and out["session"] == "NORMAL"
        assert "price" not in out
