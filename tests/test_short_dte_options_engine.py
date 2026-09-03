

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
