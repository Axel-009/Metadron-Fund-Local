"""Three separate tranche scans → concurrence (never one full-universe pass)."""
import numpy as np
import pytest

from engine.execution.universe_tranche_scan import (
    UniverseTrancheScanner, TrancheConfig, default_tranches, ConcurrenceResult,
)


class FakeSchwab:
    """Schwab-shaped data broker: deterministic quotes + candles per ticker."""

    def __init__(self, tickers):
        self.tickers = tickers
        self.quote_calls = []
        self.history_calls = []
        self._rng = np.random.default_rng(7)
        self._drift = {t: self._rng.normal(0.0, 0.002) for t in tickers}
        self._drift["SPY"] = 0.0004

    def is_connected(self):
        return True

    def get_quotes(self, tickers):
        tickers = list(tickers)
        self.quote_calls.append(len(tickers))
        out = {}
        for i, t in enumerate(tickers):
            px = 20.0 + (hash(t) % 400)
            out[t] = {"last": px, "mark": px, "volume": 500_000 + (hash(t) % 5_000_000),
                      "net_pct_change": float(self._drift.get(t, 0.0) * 300), "high_52": px * 1.3, "low_52": px * 0.7}
        return out

    def get_price_history(self, ticker, days=120, frequency="daily"):
        self.history_calls.append(ticker)
        n = 100
        steps = self._rng.normal(self._drift.get(ticker, 0.0), 0.015, n)
        close = 100.0 * np.exp(np.cumsum(steps))
        return {"close": close, "high": close * 1.01, "low": close * 0.99, "open": close, "volume": np.full(n, 1e6), "dates": []}


@pytest.fixture
def tranches():
    t1 = [f"L{i}" for i in range(60)]
    t2 = [f"S{i}" for i in range(50)]
    t3 = [f"M{i}" for i in range(40)]
    t4 = ["SPY", "TLT", "HYG", "QQQ", "IWM", "LQD", "TLTW", "XLE", "XLF", "GLD", "AGG", "IEF"]
    return [("SCAN_1_SP500", t1), ("SCAN_2_SP400", t3), ("SCAN_3_SP600", t2), ("SCAN_4_ETF_FI", t4)]


def test_default_tranches_follow_allocation_guide_order_and_are_disjoint():
    tr = default_tranches()
    assert [n for n, _ in tr] == ["SCAN_1_SP500", "SCAN_2_SP400", "SCAN_3_SP600", "SCAN_4_ETF_FI"]
    sets = [set(t) for _, t in tr]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (sets[i] & sets[j]), (i, j)
    assert all(len(s) > 50 for s in sets[:3]) and len(sets[3]) >= 30


def test_separate_scans_per_run_then_concur(tranches):
    all_t = [t for _, ts in tranches for t in ts]
    fb = FakeSchwab(all_t)
    cfg = TrancheConfig(shortlist_per_tranche=15, top_per_tranche=6, final_max_names=10, min_dollar_volume=1.0, cvr_event_scan=False)
    sc = UniverseTrancheScanner(fb, cfg=cfg, tranches=tranches, options_engine=None)
    res = sc.run(corridor_bias=0.2)
    assert isinstance(res, ConcurrenceResult)
    # three tranche results, in order, each scored separately
    assert [t.name for t in res.tranches] == [n for n, _ in tranches]
    # quotes were requested per tranche — never the whole universe in one call
    assert max(fb.quote_calls) < len(all_t)
    assert len(fb.quote_calls) >= 3
    for t in res.tranches:
        assert t.universe_size == len(dict(tranches)[t.name])
        assert 0 < len(t.top) <= cfg.top_per_tranche
        assert all(c.tranche == t.name for c in t.top)
        # z-scores computed within tranche
        zs = [c.z_score for c in t.top]
        assert zs == sorted(zs, reverse=True)
    # concurrence: bounded, sector cap, min per tranche when eligible
    scanned = [c for c in res.final if c.tranche != "LOCKED_SLEEVE"]
    assert 0 < len(scanned) <= cfg.final_max_names + len(res.LOCKED_SLEEVES)
    # LOCKED ALLOCATION: every equity sleeve is represented (scan candidate or placeholder)
    assert {c.sleeve for c in res.final} >= set(res.LOCKED_SLEEVES) - {"CVR"}   # CVR is event-based (scan off here)
    assert isinstance(res.backfill_notes, list)
    per_tr = {}
    for c in res.final:
        per_tr[c.tranche] = per_tr.get(c.tranche, 0) + 1
    per_tr.pop("LOCKED_SLEEVE", None)
    assert len(per_tr) == 4 and all(v >= 1 for v in per_tr.values())
    sectors = {}
    for c in res.final:
        if c.sector != "Unknown":
            sectors[c.sector] = sectors.get(c.sector, 0) + 1
    assert not sectors or max(sectors.values()) <= max(1, int(cfg.final_max_names * cfg.sector_cap))
    tickers_final = [c.ticker for c in res.final]
    assert len(tickers_final) == len(set(tickers_final))
    for c in res.final:
        if c.tranche == "LOCKED_SLEEVE":
            continue
        assert sum(c.votes.values()) >= 4 and len(c.votes) == 6
    # slate + options universe + markdown
    slate = res.equity_slate()
    assert slate and all(r["decision"]["source"] == "TRANCHE_CONCURRENCE" for r in slate)
    assert all(b.startswith("OPTIONS_") for _, b in res.options_universe())
    md = res.markdown()
    assert "SCAN_1_SP500" in md and "SCAN_4_ETF_FI" in md and "Concurrence" in md
    # per-run gold-standard stats are populated
    for t in res.tranches:
        assert t.buy_n + t.sell_n + t.hold_n == t.screened
        assert t.label in ("SP500", "SP400", "SP600", "ETF_FI")
    assert any(c.why() for t in res.tranches for c in t.top)


def test_history_only_for_shortlist(tranches):
    all_t = [t for _, ts in tranches for t in ts]
    fb = FakeSchwab(all_t)
    cfg = TrancheConfig(shortlist_per_tranche=5, top_per_tranche=3, min_dollar_volume=1.0, cvr_event_scan=False)
    UniverseTrancheScanner(fb, cfg=cfg, tranches=tranches, options_engine=None).run(corridor_bias=0.0)
    # SPY reference + 5 per tranche max
    assert len(fb.history_calls) <= 1 + 5 * 4  # SPY reference + 5 per run max


def test_screen_rejects_illiquid_and_cheap(tranches):
    class Cheap(FakeSchwab):
        def get_quotes(self, tickers):
            return {t: {"last": 1.0, "mark": 1.0, "volume": 10, "net_pct_change": 0, "high_52": 2, "low_52": 0.5} for t in tickers}
    fb = Cheap([t for _, ts in tranches for t in ts])
    res = UniverseTrancheScanner(fb, cfg=TrancheConfig(cvr_event_scan=False), tranches=tranches, options_engine=None).run(corridor_bias=0.0)
    assert all(t.rejected.get("price<min", 0) == t.universe_size for t in res.tranches)
    # nothing scanned survives, but the locked sleeves are still filled with placeholders
    assert all(c.tranche == "LOCKED_SLEEVE" for c in res.final)
    assert {c.sleeve for c in res.final} == set(res.LOCKED_SLEEVES) - {"CVR"}   # CVR never gets an ETF placeholder


def test_cvr_sleeve_is_event_based():
    """CVR sleeve: filled from live cash+CVR merger targets, never an ETF proxy."""
    from engine.execution.universe_tranche_scan import ConcurrenceResult
    from engine.execution.cvr_event_scan import CVREvent
    assert "CVR" not in ConcurrenceResult.SLEEVE_FALLBACK
    res = ConcurrenceResult(tranches=[], final=[], dropped=[], corridor_bias=0.0, spy_mom_63d=0.0, as_of="t")
    ev = CVREvent("LNTH", "Lantheus", "Curium", 102.5, 12.0, "milestones", "PENDING", "2026-08-04", "H1 2027", "u",
                  price=100.85, spread_pct=0.016, cvr_upside_pct=0.119, score=0.075, why="w")
    assert res.lock_cvr_from_events([ev], ["ok"]) is True
    c = [c for c in res.final if c.sleeve == "CVR"][0]
    assert c.ticker == "LNTH" and c.tranche == "CVR_EVENT" and c.signal == "BUY"
    res2 = ConcurrenceResult(tranches=[], final=[], dropped=[], corridor_bias=0.0, spy_mom_63d=0.0, as_of="t")
    notes = res2.lock_sleeves()
    assert not any(c.sleeve == "CVR" for c in res2.final)
    assert any(n.startswith("CVR: no live cash+CVR") for n in notes)
