"""1분봉 → 특징(feature)과 레이블(label).

원칙
- 특징은 시각 t까지의 정보만 쓴다(t 분봉의 종가 포함, 그 이후 금지).
- 모든 계산은 거래일 안에서만 한다. 전일 정보는 '전일 종가 대비 갭' 하나만 쓴다.
  → 월물 교체(롤오버)로 생기는 가격 점프가 특징·레이블에 섞이지 않는다.
- 레이블은 같은 거래일 안의 h분 뒤 종가로 만든다. 장 마감 전 h분 이내는 레이블이 없다(NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 15, 30)
WINDOWS = (5, 15, 30)
SESSION_MINUTES = 420  # 08:45~15:45


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.sort_values("time").reset_index(drop=True).copy()
    df["date"] = df["time"].dt.date
    g = df.groupby("date", sort=False)

    logc = np.log(df["close"])
    df["r1"] = logc - g["close"].shift(1).pipe(np.log)
    for w in WINDOWS:
        df[f"ret_{w}"] = logc - np.log(g["close"].shift(w))
        r2 = df["r1"] ** 2
        df[f"rv_{w}"] = np.sqrt(r2.groupby(df["date"]).rolling(w, min_periods=w).sum().reset_index(level=0, drop=True))
        hi = g["high"].rolling(w, min_periods=w).max().reset_index(level=0, drop=True)
        lo = g["low"].rolling(w, min_periods=w).min().reset_index(level=0, drop=True)
        df[f"range_{w}"] = (hi - lo) / df["close"]

    # 당일 누적 VWAP 대비 위치(분봉 대표가격 = (고+저+종)/3)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = (typical * df["volume"]).groupby(df["date"]).cumsum()
    vv = df["volume"].groupby(df["date"]).cumsum()
    df["vwap_dev"] = (df["close"] - pv / vv.replace(0, np.nan)) / df["close"]

    vol_mean = g["volume"].rolling(30, min_periods=10).mean().reset_index(level=0, drop=True)
    df["vol_ratio"] = df["volume"] / vol_mean.replace(0, np.nan)

    # 시가 대비, 전일 종가 대비 갭(같은 월물일 때만)
    day_open = g["open"].transform("first")
    df["ret_from_open"] = logc - np.log(day_open)
    daily = df.groupby("date").agg(last_close=("close", "last"), contract=("contract", "last"))
    prev = daily.shift(1)
    same = prev["contract"] == daily["contract"]
    gap = pd.Series(np.where(same, np.log(df.groupby("date")["open"].first() / prev["last_close"]), np.nan),
                    index=daily.index)
    df["gap_open"] = df["date"].map(gap)

    # 시간 특징
    minute = (df["time"].dt.hour * 60 + df["time"].dt.minute) - (8 * 60 + 45)
    df["minute_of_day"] = minute
    df["minutes_to_close"] = SESSION_MINUTES - minute
    df["weekday"] = df["time"].dt.weekday
    df["is_thu"] = (df["weekday"] == 3).astype(int)   # 목요일 위클리 만기일
    df["is_mon"] = (df["weekday"] == 0).astype(int)   # 월요일 위클리 만기일
    return df


def add_labels(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """h분 뒤 레이블. 분봉 개수가 아니라 시각으로 찾는다.

    15:35~15:44 종가 단일가 구간처럼 분봉이 없는 시간이 있으므로, 'h개 뒤 분봉'은 'h분 뒤'가 아니다.
    t+h분 시점 이하의 마지막 종가를 쓰고, t+h분이 그날 마지막 분봉보다 뒤면 레이블을 비운다.
    """
    out = df.sort_values("time").reset_index(drop=True)
    last_time = out.groupby("date")["time"].transform("max")
    ref = out[["time", "close"]].rename(columns={"time": "target", "close": "fut_close"})
    for h in horizons:
        target = out["time"] + pd.Timedelta(minutes=h)
        m = pd.merge_asof(pd.DataFrame({"target": target}), ref, on="target", direction="backward")
        fut = m["fut_close"].where(target <= last_time)
        out[f"y_ret_{h}"] = np.log(fut / out["close"])
        out[f"y_up_{h}"] = (out[f"y_ret_{h}"] > 0).astype(float).where(fut.notna())
        out[f"y_absret_{h}"] = out[f"y_ret_{h}"].abs()
    return out


FEATURES = ([f"ret_{w}" for w in WINDOWS] + [f"rv_{w}" for w in WINDOWS] + [f"range_{w}" for w in WINDOWS]
            + ["r1", "vwap_dev", "vol_ratio", "ret_from_open", "gap_open",
               "minute_of_day", "minutes_to_close", "weekday", "is_thu", "is_mon"])


def time_split(df: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """거래일 단위 시간순 분할. 레이블이 하루를 넘지 않으므로 날짜 경계가 곧 퍼지(purge) 역할을 한다."""
    d = pd.to_datetime(df["date"])
    return (df[d <= train_end], df[(d > train_end) & (d <= valid_end)], df[d > valid_end])
