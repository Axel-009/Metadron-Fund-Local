import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import corridor as c  # noqa: E402


def test_endpoints_and_midpoint():
    assert c.corridor_beta(0.07) == pytest.approx(-0.029)
    assert c.corridor_beta(0.12) == pytest.approx(0.425)
    assert c.corridor_beta(0.095) == pytest.approx(-0.029 + 0.454 / 2)   # 0.198
    assert c.corridor_beta(0.03) == pytest.approx(-0.029)                 # flat below
    assert c.corridor_beta(0.20) == pytest.approx(0.425)                  # flat above


def test_merton_parameters():
    assert c.LAMBDA_SIGMA2 == pytest.approx(0.1101, abs=1e-4)
    assert c.HURDLE == pytest.approx(0.0732, abs=1e-4)
    assert c.risk_aversion(0.15) == pytest.approx(4.89, abs=0.01)
    assert c.risk_aversion(0.20) == pytest.approx(2.75, abs=0.01)


def test_gamma_is_flattened_bell():
    assert c.corridor_gamma(0.07) == 0.0 and c.corridor_gamma(0.12) == 0.0
    assert c.corridor_gamma(0.05) == 0.0 and c.corridor_gamma(0.15) == 0.0
    peak = c.corridor_gamma(0.095)
    assert peak * 0.01 == pytest.approx(0.136, abs=1e-3)                  # per 1% of Rm
    assert c.corridor_gamma(0.08) < peak and c.corridor_gamma(0.11) < peak
    assert c.corridor_gamma(0.08) == pytest.approx(c.corridor_gamma(0.11))  # symmetric


def test_gamma_integrates_to_span_and_is_derivative():
    n = 20000
    h = (c.R_HIGH - c.R_LOW) / n
    area = sum(c.corridor_gamma(c.R_LOW + (i + 0.5) * h) * h for i in range(n))
    assert area == pytest.approx(c.SPAN, rel=1e-6)
    x, e = 0.0912, 1e-7
    numeric = (c.corridor_beta(x + e) - c.corridor_beta(x - e)) / (2 * e)
    assert numeric == pytest.approx(c.corridor_gamma(x), rel=1e-5)


def test_leverage_mapping_and_caps():
    s = c.target_leverage(0.12, 0.15, cap=5.0)
    assert s.leverage == pytest.approx(0.425 * 4.7)                       # 2.0
    s = c.target_leverage(0.05, 0.15)
    assert s.leverage == pytest.approx(-0.029 * 4.7)                      # −0.136 floor
    assert s.position == "BELOW"
    s = c.target_leverage(0.12, 0.15, cap=0.65)
    assert s.leverage == pytest.approx(0.65)                              # regime cap binds
    hi_vol = c.target_leverage(0.11, 0.30, cap=5.0)
    assert hi_vol.leverage < c.target_leverage(0.11, 0.15, cap=5.0).leverage


def test_vol_scale_modes():
    assert c.vol_scale(0.15) == pytest.approx(1.0)
    assert c.vol_scale(0.30, "linear") == pytest.approx(0.5)
    assert c.vol_scale(0.30, "squared") == pytest.approx(0.25)
    assert c.vol_scale(0.01) == 2.0                                       # clipped


def test_throttle_and_estimators():
    assert not c.should_rebalance(1.00, 1.04)
    assert c.should_rebalance(1.00, 1.06)
    closes = [100 * math.exp(0.10 / 252 * i) for i in range(253)]
    assert c.annualised_drift(closes) == pytest.approx(math.exp(0.10) - 1, rel=1e-6)
    assert c.realised_vol(closes) == pytest.approx(0.0, abs=1e-9)
