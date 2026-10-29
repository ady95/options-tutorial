from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from optlab.models.forecast import Forecast
from optlab.pricing import black76
from optlab.strategy import clock
from optlab.strategy.scenarios import forecast_scenarios, implied_scenarios
from optlab.strategy.signal import StrategyConfig, decide


def test_variance_clock_session_and_weekend():
    assert clock.variance_minutes(datetime(2026, 10, 29, 10, 0), datetime(2026, 10, 29, 15, 30)) == 330
    # 금 15:00 → 월 09:30: 금 30분 + 월 30분 + 밤(주말) 1번
    fri, mon = datetime(2026, 10, 30, 15, 0), datetime(2026, 11, 2, 9, 30)
    assert clock.variance_minutes(fri, mon) == 60 + clock.OVERNIGHT_MINUTES


def test_forecast_scenarios_match_quantiles():
    fc = Forecast(horizon=15, p_up=0.6, absret_q50=0.003, absret_q90=0.008)
    r, w = forecast_scenarios(fc, n=2000, direction_weight=1.0)
    assert w.sum() == pytest.approx(1.0)
    mag = np.abs(r[:2000])
    assert np.quantile(mag, 0.5) == pytest.approx(0.003, rel=0.01)
    assert np.quantile(mag, 0.9) == pytest.approx(0.008, rel=0.01)
    assert w[r > 0].sum() == pytest.approx(0.6)
    _, w0 = forecast_scenarios(fc, n=10, direction_weight=0.0)
    assert np.allclose(w0, w0[0])


def test_market_iv_scenarios_are_fair():
    # 시장 IV대로 움직이면 h분 뒤 기대 가치 = 지금 가치(시간가치 감소와 감마 이익이 상쇄)
    F, K, iv, h = 940.0, 940.0, 0.5, 15
    T0 = 300 / clock.ANNUAL_UNITS
    T1 = (300 - h) / clock.ANNUAL_UNITS
    v0 = black76.price(F, K, T0, 0.0, iv, True)
    r, w = implied_scenarios(iv, h, n=2000)
    v1 = np.sum(w * black76.price_array(F * np.exp(r), K, T1, 0.0, np.full(len(r), iv), True))
    assert v1 == pytest.approx(v0, rel=0.002)


def _chain(spread: float) -> pd.DataFrame:
    rows = []
    for k in (935.0, 937.5, 940.0, 942.5, 945.0):
        for cp in ("C", "P"):
            v = black76.price(940.0, k, 300 / clock.ANNUAL_UNITS, 0.03, 0.5, cp == "C")
            rows.append({"code": f"{cp}{k}", "cp": cp, "strike": k, "bid1": round(v - spread / 2, 2),
                         "ask1": round(v + spread / 2, 2), "age": 0.0})
    return pd.DataFrame(rows)


def test_decide_buy_when_forecast_vol_high_and_hold_when_costly():
    now, expiry = datetime(2026, 10, 29, 10, 0), datetime(2026, 10, 29, 15, 30)
    big = Forecast(horizon=15, p_up=0.5, absret_q50=0.008, absret_q90=0.018)
    cfg = StrategyConfig(direction_weight=0.0)
    assert decide(_chain(0.1), 940.0, now, expiry, big, cfg).action in ("CALL_BUY", "PUT_BUY")
    small = Forecast(horizon=15, p_up=0.5, absret_q50=0.002, absret_q90=0.005)
    assert decide(_chain(0.1), 940.0, now, expiry, small, cfg).reason == "NO_EDGE"
    late = datetime(2026, 10, 29, 15, 5)
    assert decide(_chain(0.1), 940.0, late, expiry, big, cfg).reason == "NEAR_CLOSE"
