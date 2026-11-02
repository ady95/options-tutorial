"""13장 B형: 변동성 관점 — IV와 실현변동성(RV)을 같은 구간으로 맞춰 비교한다.

IV는 '지금부터 만기까지'의 평균 변동성이다(05장, 변동성 시계). 비교할 예측도 같은 구간이어야 한다.
- 15분 예측과 비교: 장중 시간대별 변동성 패턴으로 IV를 그 15분 구간의 값으로 바꾼다(iv_for_window)
- 만기까지와 비교: 일별 실현분산을 HAR 회귀로 예측해 만기까지 쌓는다(HarForecaster)
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from optlab.strategy import clock


def _session(bars: pd.DataFrame) -> pd.DataFrame:
    b = bars.sort_values("time").copy()
    t = b["time"].dt.time
    b = b[(t >= clock.SESSION_START) & (t <= clock.SESSION_END)]
    b["d"] = b["time"].dt.date
    b["r"] = np.log(b["close"] / b.groupby("d")["close"].shift(1))
    return b


def intraday_profile(bars: pd.DataFrame) -> pd.Series:
    """장중 분(09:01=1 … 15:30=390)별 1분 수익률 제곱의 평균, 전체 평균을 1로 맞춘 값."""
    b = _session(bars)
    m = b["time"].dt.hour * 60 + b["time"].dt.minute - 540
    prof = (b["r"] ** 2).groupby(m).mean().reindex(range(1, clock.SESSION_MINUTES + 1)).fillna(1.0)
    return prof / prof.mean()


def _minute(ts: datetime) -> int:
    return ts.hour * 60 + ts.minute - 540


def iv_for_window(iv: float, ts: datetime, expiry: datetime, profile: pd.Series, h_min: int = 15) -> float:
    """만기까지의 평균 IV를 ts부터 h_min분 구간의 IV로 바꾼다.

    IV가 맞다면 만기까지의 분산 IV²·T가 시간대 패턴대로 나뉜다고 본다. 밤과 다른 날의 장중은 평균(1)으로 둔다.
    """
    m0 = _minute(ts)
    win = profile.reindex(range(m0 + 1, m0 + h_min + 1)).fillna(1.0).sum() / h_min
    T_min = clock.variance_minutes(ts, expiry)
    today = profile.reindex(range(m0 + 1, clock.SESSION_MINUTES + 1)).fillna(1.0)
    total = today.sum() + (T_min - len(today))
    return float(iv * np.sqrt(win / (total / T_min)))


def daily_variance(bars: pd.DataFrame) -> pd.DataFrame:
    """거래일별 실현분산: 장중(1분 수익률 제곱합) + 밤(전날 종가 → 09:00 갭의 제곱)."""
    b = _session(bars)
    day = b.groupby("d").agg(intra=("r", lambda x: float(np.nansum(x ** 2))), first=("close", "first"),
                             last=("close", "last"), n=("r", "size"))
    day["night"] = np.log(day["first"] / day["last"].shift(1)) ** 2
    day = day.dropna()
    day = day[day["n"] > 300]                       # 반쪽 거래일(데이터 누락) 제외
    day["D"] = day["intra"] + day["night"]
    return day[["intra", "night", "D"]]


class HarForecaster:
    """HAR 회귀: log(다음 날 분산) ~ log(오늘) + log(최근 5일 평균) + log(최근 22일 평균)."""

    def fit(self, day: pd.DataFrame) -> "HarForecaster":
        X, y = self._X(day), np.log(day["D"]).shift(-1)
        ok = X.notna().all(axis=1) & y.notna()
        self.lr = LinearRegression().fit(X[ok], y[ok])
        resid = y[ok] - self.lr.predict(X[ok])
        self.adj = float(np.exp(0.5 * resid.var()))     # 로그에서 예측했으므로 평균 분산으로 되돌린다
        self.intra_share = float(day["intra"].sum() / day["D"].sum())
        self.until = day.index.max()
        return self

    @staticmethod
    def _X(day: pd.DataFrame) -> pd.DataFrame:
        D = day["D"]
        return pd.DataFrame({"d1": np.log(D), "w5": np.log(D.rolling(5).mean()), "m22": np.log(D.rolling(22).mean())})

    def next_day(self, day: pd.DataFrame) -> float:
        """day의 마지막 거래일까지 보고 다음 거래일 하루의 분산."""
        return float(np.exp(self.lr.predict(self._X(day).tail(1))[0]) * self.adj)

    def vol_to_expiry(self, D: float, ts: datetime, expiry: datetime, profile: pd.Series) -> float:
        """하루 분산 예측 D를 ts~만기 구간에 쌓아 연율 변동성으로(IV와 같은 시계)."""
        m0 = _minute(ts)
        today = profile.reindex(range(m0 + 1, clock.SESSION_MINUTES + 1)).fillna(1.0).sum()
        today_var = D * self.intra_share * today / clock.SESSION_MINUTES
        T_min = clock.variance_minutes(ts, expiry)
        rest_days = max(T_min - max(0, clock.SESSION_MINUTES - m0), 0) / (clock.SESSION_MINUTES + clock.OVERNIGHT_MINUTES)
        T = T_min / clock.ANNUAL_UNITS
        return float(np.sqrt((today_var + D * rest_days) / T))


def realized_vol(bars: pd.DataFrame, start: datetime, end: datetime) -> float:
    """start~end 장중 1분 수익률(밤사이 갭은 다음 날 첫 분에 들어감)로 잰 연율 변동성."""
    b = bars[(bars["time"] > start) & (bars["time"] <= end)].sort_values("time")
    t = b["time"].dt.time
    b = b[(t >= clock.SESSION_START) & (t <= clock.SESSION_END)]
    r = np.log(b["close"]).diff().dropna()
    return float(np.sqrt((r ** 2).sum() / clock.years(start, end)))
