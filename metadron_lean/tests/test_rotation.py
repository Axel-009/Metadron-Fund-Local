import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "algorithm"))
import rotation as rot  # noqa: E402
import risk_budget as rb  # noqa: E402
import corridor as c  # noqa: E402


def path(start, daily, n=260):
    out, p = [], start
    for _ in range(n):
        out.append(p)
        p *= 1 + daily
    return out


def crash_path(start=100.0, n=260, crash_days=40, crash_total=-0.55, noise=0.04):
    """Up-trend, then a crash over the last `crash_days`, with ±noise daily swings (realised vol)."""
    up = path(start, 0.002, n - crash_days)
    d = (1 + crash_total) ** (1 / crash_days) - 1
    out, p = list(up), up[-1]
    for i in range(crash_days):
        p *= (1 + d) * (1 + (noise if i % 2 else -noise))
        out.append(p)
    return out


@pytest.fixture
def paradigm():
    spy = path(100, 0.0006)                                   # ~16%/yr — ROTATION zone
    prices = {
        "CRWV": crash_path(), "NBIS": crash_path(crash_total=-0.50),
        "IREN": crash_path(crash_total=-0.45), "HIVE": crash_path(crash_total=-0.60),
        "LMT": path(100, 0.001), "NOC": path(100, 0.001), "RTX": path(100, 0.001), "GD": path(100, 0.001),
        "LLY": path(100, 0.0008), "JNJ": path(100, 0.0004), "MRK": path(100, 0.0004), "ABBV": path(100, 0.0004),
        "SPY": spy,
    }
    return spy, prices


def test_double_sizing_removed_above_12pct():
    # market strong (Rm 16%) but regime reads CRASH → only Kelly + 4.5x hard cap
    assert rb.beta_cap("CRASH", 0.20, rm=0.16) == pytest.approx(4.5)
    lev = c.target_leverage(0.16, 0.20, cap=rb.beta_cap("CRASH", 0.20, rm=0.16)).leverage
    assert lev == pytest.approx(3.0, abs=0.01)                # Kelly at σ 20%, not 0.3x
    # below 12% the regime ceiling still applies
    assert rb.beta_cap("CRASH", 0.20, rm=0.10) < 0.5
    assert rb.beta_cap("CRASH", 0.20) < 0.5                   # no Rm given → conservative


def test_theme_crash_detected_while_market_strong(paradigm):
    spy, prices = paradigm
    st = rot.theme_state("AI_CREDIT_HUBS", ["CRWV", "NBIS"], prices, spy)
    assert st.state == "CRASH" and st.drawdown > 0.30 and st.rel_spy_21d < -0.15
    calm = rot.theme_state("DEF", ["LMT", "NOC"], prices, spy)
    assert calm.state == "NORMAL"


def test_plan_cuts_rotates_and_hedges(paradigm):
    spy, prices = paradigm
    holdings = {"CRWV": 0.06, "NBIS": 0.04, "IREN": 0.03, "LMT": 0.01}
    oil = path(70, 0.006, 60)                                  # oil +~43% in 60d
    esc = rot.escalation_score(oil=oil, vix=27.0)
    assert esc >= rot.ESCALATION_THRESHOLD
    plan = rot.build_plan("ROTATION", prices, spy, holdings, escalation=esc)
    kinds = {a.kind for a in plan.actions}
    assert {"CUT", "ROTATE", "ESCALATION_CALLS", "ESCALATION_EQUITY"} <= kinds
    cuts = {a.ticker: a.target_weight for a in plan.actions if a.kind == "CUT"}
    assert cuts == {"CRWV": 0.0, "NBIS": 0.0, "IREN": 0.0}
    rot_w = {a.ticker: a.target_weight for a in plan.actions if a.kind == "ROTATE"}
    freed = 0.13
    added = sum(rot_w.values()) - 0.01                         # LMT already held 1%
    assert added == pytest.approx(freed)
    defense = sum(rot_w[t] for t in ("LMT", "NOC", "RTX", "GD")) - 0.01
    assert defense == pytest.approx(freed * rot.DEFENSE_SHARE_ESCALATION)
    assert any(a.ticker == "VLO" for a in plan.actions)


def test_calm_rotation_tilts_to_healthcare(paradigm):
    spy, prices = paradigm
    plan = rot.build_plan("ROTATION", prices, spy, {"CRWV": 0.10}, escalation=0.0)
    rot_w = {a.ticker: a.target_weight for a in plan.actions if a.kind == "ROTATE"}
    assert sum(rot_w[t] for t in ("LLY", "JNJ", "MRK", "ABBV")) == pytest.approx(0.10 * 0.75)
    assert not any(a.kind.startswith("ESCALATION") for a in plan.actions)


def test_leaps_only_on_solvent_crashed_names(paradigm):
    spy, prices = paradigm
    cap = {"CRWV": {"merton_pd": 0.02, "interest_coverage": 3.5},      # solvent
           "NBIS": {"merton_pd": 0.12, "interest_coverage": 1.1}}      # not solvent
    plan = rot.build_plan("ROTATION", prices, spy, {}, cap_structure=cap,
                          implied_vol={"CRWV": 0.70})
    leaps = {a.ticker: a.kind for a in plan.actions if a.kind.startswith("LEAPS")}
    assert "CRWV" in leaps and "NBIS" not in leaps and "IREN" not in leaps   # unknown → no LEAPS
    rich = rot.build_plan("ROTATION", prices, spy, {}, cap_structure=cap,
                          implied_vol={"CRWV": 3.0})
    assert any(a.kind == "LEAPS_SPREAD" and a.ticker == "CRWV" for a in rich.actions)


def test_straddles_when_vol_cheap_or_event_near(paradigm):
    spy, prices = paradigm
    plan = rot.build_plan("ROTATION", prices, spy, {},
                          implied_vol={"HIVE": 0.10, "CRWV": 5.0}, days_to_event={"NBIS": 5})
    s = {a.ticker for a in plan.actions if a.kind == "STRADDLE"}
    assert "HIVE" in s and "NBIS" in s and "CRWV" not in s
    assert len(s) <= rot.STRADDLE_MAX_NAMES


def test_no_actions_when_no_theme_stress():
    spy = path(100, 0.0006)
    prices = {t: path(100, 0.0007) for t in ("CRWV", "NBIS", "IREN", "HIVE", "LMT", "LLY")}
    plan = rot.build_plan("ROTATION", prices, spy, {"CRWV": 0.05})
    assert not plan.active and plan.actions == []


def test_agent_flag_overrides_escalation():
    assert rot.escalation_score(agent_flag=True) == 1.0
    assert rot.escalation_score(agent_flag=False) == 0.0


def test_plan_is_idempotent(paradigm):
    """Re-running the plan on the post-trade book must not compound (owner bug report)."""
    spy, prices = paradigm
    holdings = {"CRWV": 0.06, "NBIS": 0.04}
    plan1 = rot.build_plan("ROTATION", prices, spy, holdings, escalation=1.0)
    after = dict(holdings)
    for a in plan1.actions:
        if a.kind in ("CUT", "ROTATE", "ESCALATION_EQUITY"):
            after[a.ticker] = a.target_weight
    plan2 = rot.build_plan("ROTATION", prices, spy, after, escalation=1.0)
    eq2 = {a.ticker: a.target_weight for a in plan2.actions if a.kind in ("CUT", "ROTATE", "ESCALATION_EQUITY")}
    for tkr, w in eq2.items():
        assert w == pytest.approx(after.get(tkr, 0.0)), tkr


def test_stress_caps_instead_of_halving():
    spy = path(100, 0.0006)
    stressed = path(100, 0.002, 240) + path(100 * 1.002 ** 240 * 0.99, -0.01, 20)   # ~18% drop
    prices = {"CRWV": stressed, "NBIS": stressed}
    plan = rot.build_plan("ROTATION", prices, spy, {"CRWV": 0.08, "NBIS": 0.01})
    cuts = {a.ticker: a.target_weight for a in plan.actions if a.kind == "CUT"}
    assert plan.themes[0].state == "STRESS"
    assert cuts == {"CRWV": rot.STRESS_NAME_CAP}           # NBIS already under the cap
