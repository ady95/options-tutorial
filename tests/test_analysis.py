"""12장 분석 모듈 단위 테스트."""
import numpy as np
import pandas as pd

from optlab.analysis.drift import VolCalibrator, psi
from optlab.analysis.performance import bootstrap_mean_ci, trade_stats, trades_needed
from optlab.analysis.regimes import label_regimes


def test_trade_stats_payoff_profit_factor_and_streak():
    s = trade_stats([100, -50, -50, 200, -100])
    assert s["trades"] == 5 and s["net_krw"] == 100
    assert s["win_rate"] == 0.4
    assert s["payoff"] == 150 / (200 / 3)            # 평균 이익 150, 평균 손실 66.7
    assert s["profit_factor"] == 300 / 200
    assert s["max_losing_streak"] == 2
    assert s["mdd_krw"] == -100                       # 누적 100 → 0, 200 → 100: 고점 대비 가장 크게 빠진 폭


def test_bootstrap_by_day_needs_two_days_and_widens_interval():
    rng = np.random.default_rng(1)
    day_effect = np.repeat(rng.normal(0, 300, 10), 20)          # 같은 날 거래는 같은 시장 움직임을 공유
    pnl = day_effect + rng.normal(50, 100, 200)
    days = np.repeat(np.arange(10), 20)
    lo, hi = bootstrap_mean_ci(pnl)
    dlo, dhi = bootstrap_mean_ci(pnl, groups=days)
    assert dhi - dlo > 2 * (hi - lo)
    assert np.isnan(bootstrap_mean_ci(pnl[:20], groups=days[:20])[0])
    assert trades_needed(50, 100) == 16


def test_psi_zero_for_same_distribution_large_for_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 20000))
    assert psi(a, pd.Series(rng.normal(0, 1, 20000))) < 0.01
    assert psi(a, pd.Series(rng.normal(0, 2, 20000))) > 0.25


def test_vol_calibrator_uses_only_past_days():
    dates = np.repeat(pd.date_range("2026-01-01", periods=30).date, 10)
    p = pd.DataFrame({"date": dates, "q90": 1.0, "y_abs": 0.5})
    p.loc[p["date"] >= dates[200], "y_abs"] = 3.0              # 21번째 날부터 갑자기 거칠어짐
    k = VolCalibrator(window=20).factors(p)
    assert (k.iloc[:20] == 1.0).all()                          # 준비 기간
    assert k.iloc[20] == 0.5                                   # 그날의 결과는 아직 쓰지 않는다
    assert k.iloc[21] == 0.5                                   # 거친 날이 20일 중 1일(5%)이면 90% 분위수는 그대로
    assert k.iloc[23] == 3.0                                   # 3일(15%)이 되면 배율이 따라 올라간다


def test_label_regimes_time_buckets_and_previous_day_vol():
    days = pd.date_range("2026-10-26", periods=4, freq="D").date
    p = pd.DataFrame({"date": np.repeat(days, 2), "is_thu": 0, "is_mon": 0,
                      "r1": [0, 1, 0, 2, 0, 3, 0, 4], "minute_of_day": [30, 360, 400, 100] * 2})
    r = label_regimes(p)
    assert list(r["time_bucket"].astype(str)[:4]) == ["09:15~09:59", "14:45~15:14", "15:15~", "10:00~14:44"]
    assert r["prev_day_vol"].isna()[:2].all()                  # 첫날은 전날이 없다
    assert r["prev_day_vol"].iloc[2] == p.loc[:1, "r1"].std()  # 둘째 날에는 첫날의 변동성
