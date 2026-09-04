

def test_cta_read_and_backfill_config():
    """WonderTrader CTA core is blended into the momentum read; SP400/SP600 back-fill config exists."""
    import numpy as np
    from engine.execution.short_dte_options_engine import ShortDTEOptionsEngine, ShortDTEConfig

    class _B:  # minimal broker stub
        def get_price_history(self, *a, **k): return {"close": np.array([])}
        def get_quote(self, *a, **k): return None
    eng = ShortDTEOptionsEngine(_B(), nav=100_000)
    # accelerating up-trend with a fresh 20-bar breakout leg (CTA z-scored ROC rewards acceleration)
    t = np.arange(120)
    up = 100 * np.exp(0.001 * t) * np.where(t >= 100, np.exp(0.012 * (t - 99)), 1.0)
    mr = eng.momentum_read(up, up * 1.01, up * 0.99)
    assert mr.cta is not None and mr.cta.direction == 1 and mr.cta.strength > 0
    assert any(n.startswith("CTA ") for n in mr.notes)
    assert mr.direction_score > 0
    down = up[::-1].copy()
    mr2 = eng.momentum_read(down, down * 1.01, down * 0.99)
    assert mr2.cta.direction == -1 and mr2.direction_score < 0
    cfg = ShortDTEConfig()
    assert cfg.backfill_max_tries >= 1 and cfg.backfill_names_per_bucket >= 1
    assert eng.backfill_candidates == {} and eng.last_backfill == {}


def test_tenor_preference_and_window():
    from engine.execution.short_dte_options_engine import ShortDTEConfig
    cfg = ShortDTEConfig()
    assert cfg.dte_min == 1 and cfg.dte_max == 30 and cfg.tenor_pref_days == 7
    # factor: 1.0 inside 7 DTE, floor at 30 DTE
    def factor(dte):
        if dte <= cfg.tenor_pref_days:
            return 1.0
        frac = min(1.0, (dte - cfg.tenor_pref_days) / (cfg.dte_max - cfg.tenor_pref_days))
        return 1.0 - (1.0 - cfg.tenor_pref_floor) * frac
    assert factor(1) == factor(7) == 1.0
    assert 0.85 < factor(14) < 1.0
    assert abs(factor(30) - cfg.tenor_pref_floor) < 1e-9
    assert cfg.g9_options_delta == 0.20
