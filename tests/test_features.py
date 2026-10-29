import numpy as np
import pandas as pd

from optlab.features.build import add_labels, build_features


def _bars():
    # 하루 08:45~09:04 20분, 09:00 분봉이 빠진 데이터. 종가 = 100 + 분
    times = [t for t in pd.date_range("2026-10-28 08:45", periods=20, freq="1min") if t.minute != 0]
    times += list(pd.date_range("2026-10-29 08:45", periods=5, freq="1min"))
    close = [100 + i for i in range(len(times))]
    return pd.DataFrame({"time": times, "open": close, "high": close, "low": close, "close": close,
                         "volume": 1.0, "contract": "A01612"})


def test_labels_use_clock_time_and_stay_within_day():
    df = add_labels(build_features(_bars()), horizons=(5,))
    row = df[df["time"] == pd.Timestamp("2026-10-28 08:56")].iloc[0]
    # 08:56 + 5분 = 09:01. 09:00이 빠져 있어 '5개 뒤 분봉'은 09:02지만 레이블은 09:01 종가여야 한다
    fut = df[df["time"] == pd.Timestamp("2026-10-28 09:01")]["close"].iloc[0]
    assert np.isclose(row["y_ret_5"], np.log(fut / row["close"]))
    # 그날 마지막 5분 안의 행은 레이블 없음(다음 날로 넘어가지 않음)
    last = df[df["time"] == pd.Timestamp("2026-10-28 09:02")].iloc[0]
    assert np.isnan(last["y_ret_5"])


def test_features_do_not_cross_days():
    df = build_features(_bars())
    first_next_day = df[df["time"] == pd.Timestamp("2026-10-29 08:45")].iloc[0]
    assert np.isnan(first_next_day["r1"]) and np.isnan(first_next_day["ret_5"])
