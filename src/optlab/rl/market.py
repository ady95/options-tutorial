"""합성 옵션 시장 — 선물 1분봉과 하루 IV로 ATM 옵션 호가를 만든다.

실제 옵션 호가 기록은 하루치 세 개뿐이라 강화학습을 학습시킬 수 없다. 그래서 10개월치 선물 1분봉 위에
옵션 가격을 Black-76으로 만들어 쓴다. **실제 옵션 시장 데이터가 아니다.**
- F: 선물 1분봉 종가
- IV: 그날 장 시작 시점의 HAR 만기까지 변동성 예측(13-1절 방법 2). 하루 동안 고정
- 만기: 그날 이후 가장 가까운 월·목 위클리 만기(만기 당일은 쓰지 않는다)
- 호가: 중간가 ± 중간가 × rel_spread / 2를 호가단위 바깥쪽으로 반올림

이 시장에서는 IV가 시계(05장 변동성 시계)대로 고르게 줄어들고, 실제 지수는 오전에 더 많이 움직인다.
13-1절에서 본 실제 시장의 모습과 같은 방향이다. 변동성 위험 프리미엄(IV가 평균적으로 RV보다 높은 것)은
넣지 않았다. 실제 시장에서는 옵션 매수가 이 환경보다 불리할 수 있다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from optlab.market_data.spec import round_to_tick, tick_size
from optlab.pricing import black76
from optlab.strategy import clock
from optlab.strategy.signal import expiry_datetime
from optlab.strategy.volview import HarForecaster, daily_variance, intraday_profile

STRIKE_STEP = 2.5
RISK_FREE = 0.03
DECIDE_FROM, DECIDE_UNTIL = time(9, 15), time(15, 15)   # 30분 특징이 생긴 뒤부터, 15:15에 모두 청산


def next_weekly_expiry(d: date) -> date:
    """d 다음 날 이후 가장 가까운 월요일 또는 목요일."""
    for k in range(1, 8):
        x = d + timedelta(days=k)
        if x.weekday() in (0, 3):
            return x
    raise AssertionError


@dataclass
class OptionDay:
    """하루치 합성 시장. 배열은 모두 결정 시각(1분) 순서."""
    day: date
    times: np.ndarray        # datetime64
    F: np.ndarray            # 선물 종가
    T: np.ndarray            # 잔존만기(연, 변동성 시계)
    iv: float
    expiry: datetime
    rows: pd.DataFrame       # 같은 시각의 특징과 07장 예측(p_up, q50, q90)

    def __len__(self) -> int:
        return len(self.F)

    def mid(self, i: int, strike: float, is_call: bool) -> float:
        return black76.price(float(self.F[i]), strike, float(self.T[i]), RISK_FREE, self.iv, is_call)

    def quote(self, i: int, strike: float, is_call: bool, rel_spread: float) -> tuple[float, float]:
        """(매수호가, 매도호가)."""
        m = self.mid(i, strike, is_call)
        half = max(m * rel_spread / 2, tick_size(m))
        return max(round_to_tick(m - half, "down"), 0.0), round_to_tick(m + half, "up")

    def atm(self, i: int) -> float:
        return round(float(self.F[i]) / STRIKE_STEP) * STRIKE_STEP


def har_iv_by_day(bars: pd.DataFrame, days, profile_until: str) -> dict:
    """그날 09:00에 알 수 있었던 HAR 만기까지 변동성. HAR은 그 달 시작 전까지로 매달 다시 학습한다."""
    profile = intraday_profile(bars[bars["time"] < profile_until])
    dv = daily_variance(bars)
    out, har, fitted_month = {}, None, None
    for d in sorted(days):
        hist = dv[dv.index < d]
        month = pd.Period(d, "M")
        if month != fitted_month:
            har, fitted_month = HarForecaster().fit(dv[dv.index < month.start_time.date()]), month
        open_ = datetime.combine(d, clock.SESSION_START)
        out[d] = har.vol_to_expiry(har.next_day(hist), open_, expiry_datetime(next_weekly_expiry(d)), profile)
    return out


def build_days(pred: pd.DataFrame, bars: pd.DataFrame, iv: dict) -> list[OptionDay]:
    """pred: 12장 워크포워드 예측(time, date, 특징, p_up, q50, q90). bars: 선물 1분봉(close)."""
    px = bars.drop_duplicates("time", keep="last").set_index("time")["close"]
    out = []
    for d, g in pred.groupby("date", sort=True):
        t = g["time"].dt.time
        g = g[(t >= DECIDE_FROM) & (t <= DECIDE_UNTIL)].sort_values("time").reset_index(drop=True)
        if len(g) < 300 or d not in iv:
            continue
        expiry = expiry_datetime(next_weekly_expiry(d))
        T = np.array([clock.years(ts.to_pydatetime(), expiry) for ts in g["time"]])
        out.append(OptionDay(d, g["time"].to_numpy(), px.reindex(g["time"]).to_numpy(float), T,
                             float(iv[d]), expiry, g))
    return out
