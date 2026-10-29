import math

import pytest

from optlab.market_data.spec import breakeven_at_expiry, payoff_at_expiry, premium_krw, round_to_tick
from optlab.pricing import black76

F, K, T, R = 400.0, 402.5, 3 / 365, 0.03


def test_put_call_parity():
    c = black76.price(F, K, T, R, 0.2, True)
    p = black76.price(F, K, T, R, 0.2, False)
    assert c - p == pytest.approx(math.exp(-R * T) * (F - K), abs=1e-9)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("sigma", [0.08, 0.2, 0.6])
def test_implied_vol_roundtrip(is_call, sigma):
    px = black76.price(F, K, T, R, sigma, is_call)
    assert black76.implied_vol(px, F, K, T, R, is_call) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_below_intrinsic():
    assert black76.implied_vol(0.5, 410.0, 400.0, T, R, True) is None


def test_greeks_match_finite_difference():
    g = black76.greeks(F, K, T, R, 0.2, True)
    h = 0.01
    up = black76.price(F + h, K, T, R, 0.2, True)
    dn = black76.price(F - h, K, T, R, 0.2, True)
    assert g.delta == pytest.approx((up - dn) / (2 * h), rel=1e-4)
    vega_fd = (black76.price(F, K, T, R, 0.21, True) - black76.price(F, K, T, R, 0.19, True)) / 2
    assert g.vega == pytest.approx(vega_fd, rel=1e-3)
    theta_fd = black76.price(F, K, T - 1 / 365 / 100, R, 0.2, True) - black76.price(F, K, T, R, 0.2, True)
    assert g.theta == pytest.approx(theta_fd * 100, rel=1e-2)


def test_spec_helpers():
    assert premium_krw(15.25) == 3_812_500
    assert breakeven_at_expiry(947.5, 15.25, True) == pytest.approx(962.75)
    assert breakeven_at_expiry(947.5, 7.0, False) == pytest.approx(940.5)
    assert payoff_at_expiry(940.0, 947.5, 7.0, False) == 125_000
    assert round_to_tick(12.33, "up") == 12.35
    assert round_to_tick(12.33, "down") == 12.30
    assert round_to_tick(3.456) == 3.46
