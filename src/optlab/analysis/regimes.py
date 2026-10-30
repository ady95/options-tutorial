"""국면 나누기. 국면은 그 시점에 이미 알 수 있는 정보로만 정한다(전날 변동성, 요일, 시각)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TIME_BUCKETS = ([-1, 74, 359, 389, 10_000], ["09:15~09:59", "10:00~14:44", "14:45~15:14", "15:15~"])


def label_regimes(p: pd.DataFrame) -> pd.DataFrame:
    """walk_forward() 결과(특징 열 포함)에 국면 열을 붙인다."""
    d = pd.to_datetime(p["date"])
    out = p.copy()
    out["day_type"] = np.select([p["is_thu"] == 1, p["is_mon"] == 1], ["목(위클리 만기)", "월(위클리 만기)"], "화·수·금")
    out["monthly_expiry"] = (p["is_thu"] == 1) & d.dt.day.between(8, 14)      # 둘째 목요일
    day_vol = p.groupby("date")["r1"].std()
    out["prev_day_vol"] = p["date"].map(day_vol.shift(1))                     # 전날 1분 수익률 표준편차
    out["vol_regime"] = pd.qcut(out["prev_day_vol"], 3, labels=["저변동", "중간", "고변동"])
    edges, labels = TIME_BUCKETS
    out["time_bucket"] = pd.cut(p["minute_of_day"], edges, labels=labels)    # minute_of_day: 08:45부터 분
    return out


def by_day_bootstrap(p: pd.DataFrame, stat, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """거래일을 통째로 다시 뽑는 부트스트랩 95% 구간. 15분 레이블은 겹치므로 분봉 하나하나는 독립이 아니다."""
    rng = np.random.default_rng(seed)
    groups = dict(tuple(p.groupby("date")))
    keys = list(groups)
    vals = [stat(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])) for _ in range(n_boot)]
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)
