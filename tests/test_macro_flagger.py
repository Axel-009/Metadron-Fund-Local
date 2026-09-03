from engine.execution.macro_event_flagger import MacroEventFlagger


def test_quiet_tape_no_flag():
    f = MacroEventFlagger(broker=None).evaluate(headlines=[], regime="NORMAL", vix_level=14.0)
    assert f.level == "NONE" and f.options_add_scale == 1.0 and f.equities_add_scale == 1.0


def test_fomc_headline_is_important():
    f = MacroEventFlagger(broker=None).evaluate(headlines=["FOMC decision: Fed holds rates"], regime="NORMAL", vix_level=15.0)
    assert f.level in ("IMPORTANT", "CRITICAL")
    assert f.options_add_scale < 1.0 and f.force_rotation_review is True
    assert "FOMC" in " ".join(f.events + f.triggers).upper()
    assert f.markdown()


def test_vix_spike_and_basket_move():
    f = MacroEventFlagger(broker=None).evaluate(
        basket_moves={"SPY": -2.6, "^VIX": 28.0, "HYG": -1.4}, headlines=[], regime="STRESSED", vix_level=31.0)
    assert f.level in ("IMPORTANT", "CRITICAL") and f.score > 0.5
