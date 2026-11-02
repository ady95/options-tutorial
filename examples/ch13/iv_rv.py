"""13-1 실습: IV와 RV를 같은 구간으로 맞춰 비교하고, 스큐를 본다.

    python examples/ch13/iv_rv.py

1) 장중 시간대별 변동성 패턴(선물 1분봉, 분석 기록과 겹치지 않게 2026-09까지)
2) 기록 세 개의 매분: ATM IV, 15분 구간으로 바꾼 IV, 07장 15분 예측 변동성, 실제 다음 15분 RV
3) 일별 실현분산 HAR 예측(2026-04~ 월별 워크포워드) — 만기까지 쌓는 B형 예측
4) 기록별 만기까지: IV vs HAR 예측 vs 실제 RV
5) 스큐 기울기와 지수-IV 상관
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from common import all_bars, sessions

from optlab.features.build import build_features
from optlab.models.forecast import Forecaster
from optlab.strategy import clock
from optlab.strategy.scenarios import annualized_vol, forecast_scenarios
from optlab.strategy.signal import implied_forward
from optlab.strategy.spreads import skew_slope, smile
from optlab.strategy.volview import HarForecaster, daily_variance, intraday_profile, iv_for_window, realized_vol

MODEL = Forecaster.load(__import__("common").ROOT / "models_store" / "forecaster.joblib")
R = 0.03


def per_minute(name, src, bars, profile) -> pd.DataFrame:
    lo, hi = src.span()
    day = pd.Timestamp(lo.date())
    rows = []
    for ts in pd.date_range(max(lo, day + pd.Timedelta("9h5min")), min(hi, day + pd.Timedelta("15h5min")), freq="1min"):
        ts = ts + pd.Timedelta(seconds=2)
        ch = src.chain(ts)
        T = clock.years(ts.to_pydatetime(), src.expiry)
        F = implied_forward(ch, T, R)
        if F is None:
            continue
        sm = smile(ch, F, T, R)
        fit = skew_slope(sm)
        if fit is None:
            continue
        atm_iv, slope = fit
        b = bars[bars["time"] <= ts.floor("min") - pd.Timedelta(minutes=1)]
        b = b[b["time"].dt.date.isin(sorted(b["time"].dt.date.unique())[-2:])]
        fc = next(f for f in MODEL.predict(build_features(b)) if f.horizon == 15)
        fvol = annualized_vol(*forecast_scenarios(fc, 100, 0.5), 15)
        nb = bars[(bars["time"] >= ts.floor("min")) & (bars["time"] < ts.floor("min") + pd.Timedelta(minutes=15))]
        r = np.log(nb["close"]).diff().dropna()
        rv = float(np.sqrt((r ** 2).sum() * 15 / len(r) / (15 / clock.ANNUAL_UNITS))) if len(r) >= 10 else np.nan
        rows.append({"session": name, "ts": ts, "F": F, "atm_iv": atm_iv, "skew": slope,
                     "iv_15m": iv_for_window(atm_iv, ts.to_pydatetime(), src.expiry, profile), "fvol_15m": fvol, "rv_15m": rv})
    return pd.DataFrame(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)
    bars = all_bars()
    profile = intraday_profile(bars[bars["time"] < "2026-10-01"])
    print("== 1. 장중 변동성 패턴(30분 평균, 전체 평균 = 1)")
    print({f"{9 + (i * 30) // 60:02d}:{(i * 30) % 60:02d}": round(v, 2)
           for i, v in profile.groupby((profile.index - 1) // 30).mean().items()}, "\n")

    df = pd.concat([per_minute(n, s, bars, profile) for n, _, s in sessions()])
    print("== 2. 기록별 평균(연율). iv_15m = ATM IV를 다음 15분 구간으로 바꾼 값")
    g = df.groupby("session")[["atm_iv", "iv_15m", "fvol_15m", "rv_15m"]].mean().round(3)
    ok = df.dropna(subset=["rv_15m"])
    g["rv>iv_15m"] = ok.groupby("session").apply(lambda x: f"{(x.rv_15m > x.iv_15m).mean():.0%}")
    g["minutes"] = df.groupby("session").size()
    print(g.to_string(), "\n")

    print("== 3. 다음 거래일 실현분산 예측(HAR vs 직전 5일 평균), 2026-04~ 월별 워크포워드")
    day = daily_variance(bars)
    print(f"거래일 {len(day)}, 밤사이 갭이 하루 분산에서 차지하는 비율 {day['night'].sum() / day['D'].sum():.0%}")
    rows = []
    for m in pd.period_range("2026-04", "2026-10", freq="M"):
        idx = pd.to_datetime(day.index).to_period("M")
        tr = day[idx < m]
        har = HarForecaster().fit(tr)
        for d in day.index[idx == m]:
            hist = day[day.index < d]
            rows.append({"d": d, "y": day.loc[d, "D"], "har": har.next_day(hist), "naive5": hist["D"].tail(5).mean()})
    p = pd.DataFrame(rows)
    qlike = lambda y, f: float(np.mean(y / f - np.log(y / f) - 1))   # noqa: E731  변동성 예측의 표준 손실(작을수록 좋음)
    for k in ("har", "naive5"):
        print(f"  {k:7s} QLIKE {qlike(p.y, p[k]):.3f}, 로그 상관 {np.corrcoef(np.log(p.y), np.log(p[k]))[0, 1]:.3f}  ({len(p)}일)")
    print()

    print("== 4. 만기까지: ATM IV vs HAR 예측 vs 실제 RV(만기가 지난 기록만)")
    from common import settlement
    for name, desc, src in sessions():
        x = df[df["session"] == name]
        d0 = x["ts"].min().date()
        hist = day[day.index < d0]
        har = HarForecaster().fit(hist)
        D = har.next_day(hist)
        hv = np.mean([har.vol_to_expiry(D, t.to_pydatetime(), src.expiry, profile) for t in x["ts"]])
        S = settlement(src.expiry.date())
        rv = realized_vol(bars, x["ts"].median().to_pydatetime(), src.expiry) if S else np.nan
        print(f"  {desc}: IV {x['atm_iv'].mean():.3f}, HAR {hv:.3f}, 실제 {rv:.3f}" + (f", 결제가 {S:.2f}" if S else ", 만기 전"))
    print()

    print("== 5. 스큐 기울기(d IV / d log(K/F), 음수 = 풋 쪽 IV가 높음)와 5분 지수-IV 상관")
    for name, x in df.groupby("session"):
        x = x.set_index("ts").sort_index()
        d = pd.DataFrame({"dF": np.log(x["F"]).diff(5), "dIV": x["atm_iv"].diff(5)}).dropna()
        print(f"  {name:14s} 기울기 중앙값 {x['skew'].median():+.2f} (25~75% {x['skew'].quantile(.25):+.2f}~{x['skew'].quantile(.75):+.2f}), "
              f"지수-IV 상관 {d['dF'].corr(d['dIV']):+.2f} ({len(d)}개)")


if __name__ == "__main__":
    main()
