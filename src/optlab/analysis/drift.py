"""워크포워드 재학습 정책 비교와 모델 드리프트 감지.

07장 예측기와 같은 모델(방향: 로지스틱 회귀, 변동폭: LightGBM 분위수 회귀)을 h분 하나에 대해 학습한다.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from optlab.features.build import FEATURES
from optlab.models.forecast import LGB_QUANTILE_PARAMS, Forecaster

# 정규화된 수익률·변동성 특징(달력·시각 특징은 분포가 늘 같으므로 PSI에서 뺀다)
PSI_FEATURES = [f for f in FEATURES if f not in ("weekday", "is_thu", "is_mon", "minute_of_day", "minutes_to_close")]


def _X(d: pd.DataFrame) -> pd.DataFrame:
    return d[FEATURES].fillna(0.0)


def fit_h(train: pd.DataFrame, h: int = 15, seed: int = 42) -> tuple:
    d = train.dropna(subset=[f"y_up_{h}"])
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)).fit(_X(d), d[f"y_up_{h}"])
    q = {a: lgb.LGBMRegressor(objective="quantile", alpha=a, random_state=seed, **LGB_QUANTILE_PARAMS)
         .fit(_X(d), d[f"y_absret_{h}"]) for a in (0.5, 0.9)}
    return lr, q


def from_forecaster(fc: Forecaster, h: int = 15) -> tuple:
    return fc.direction[h], {0.5: fc.vol[(h, 0.5)], 0.9: fc.vol[(h, 0.9)]}


def predict_h(model: tuple, rows: pd.DataFrame) -> pd.DataFrame:
    lr, q = model
    return pd.DataFrame({"p_up": lr.predict_proba(_X(rows))[:, 1],
                         "q50": np.maximum(q[0.5].predict(_X(rows)), 0.0),
                         "q90": q[0.9].predict(_X(rows))}, index=rows.index)


def walk_forward(df: pd.DataFrame, months, policy: str, h: int = 15, fixed_end: str | None = None,
                 window: int = 3) -> pd.DataFrame:
    """월마다 예측을 만든다. policy:
    fixed     fixed_end까지로 한 번만 학습
    expanding 그 달 이전 전체로 매달 재학습(07-4절)
    rolling   그 달 이전 window개월로 매달 재학습
    """
    ym = pd.to_datetime(df["date"]).dt.to_period("M")
    fixed = fit_h(df[pd.to_datetime(df["date"]) <= pd.Timestamp(fixed_end)], h) if policy == "fixed" else None
    out = []
    for m in months:
        m = pd.Period(m, "M")
        if policy == "fixed":
            model = fixed
        elif policy == "expanding":
            model = fit_h(df[ym < m], h)
        else:
            model = fit_h(df[(ym < m) & (ym >= m - window)], h)
        te = df[ym == m]
        out.append(te[["time", "date", f"y_up_{h}", f"y_absret_{h}"] + FEATURES].join(predict_h(model, te)))
    return pd.concat(out).rename(columns={f"y_up_{h}": "y_up", f"y_absret_{h}": "y_abs"})


def pinball(y, q, a: float) -> float:
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def forecast_metrics(p: pd.DataFrame) -> pd.Series:
    """방향 AUC와 변동폭 분위수가 맞은 비율. cov50은 0.5, cov90은 0.9 근처여야 보정이 맞다."""
    auc = roc_auc_score(p["y_up"], p["p_up"]) if p["y_up"].nunique() > 1 else np.nan
    return pd.Series({"n": len(p), "days": p["date"].nunique(), "auc": auc,
                      "cov50": (p["y_abs"] <= p["q50"]).mean(), "cov90": (p["y_abs"] <= p["q90"]).mean(),
                      "mean_abs_bp": p["y_abs"].mean() * 1e4})


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """모집단 안정성 지수(PSI). 기준 분포의 10분위 구간에 현재 값이 얼마나 다르게 들어가는가.

    관례상 0.1 미만 안정, 0.25 이상 큰 변화로 본다. 하루에 값이 하나인 특징(gap_open)은 분봉마다 같은 값이
    수백 번 반복되므로 실제 표본(거래일 수)보다 훨씬 크게 나온다.
    """
    r, c = ref.dropna().to_numpy(), cur.dropna().to_numpy()
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3 or not len(c):
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(r, edges)[0] / len(r), 1e-4, None)
    b = np.clip(np.histogram(c, edges)[0] / len(c), 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def rolling_monitor(p: pd.DataFrame, days: int = 20) -> pd.DataFrame:
    """거래일마다 직전 days일(그날 포함)의 cov90과 방향 AUC."""
    dates = sorted(p["date"].unique())
    rows = []
    for i in range(days - 1, len(dates)):
        w = p[p["date"].isin(dates[i - days + 1: i + 1])]
        rows.append({"date": dates[i], "cov90": (w["y_abs"] <= w["q90"]).mean(),
                     "auc": roc_auc_score(w["y_up"], w["p_up"])})
    return pd.DataFrame(rows)


class VolCalibrator:
    """90% 분위수 예측에 곱할 배율 k를 직전 window 거래일의 실제 결과로 정한다.

    k = 직전 기간 (실제 |수익률| / 예측 q90)의 90% 분위수. 시장이 거칠어져 예측이 좁아지면 k가 1보다 커진다.
    하루 단위로만 갱신하므로 장중에는 그날의 결과를 쓰지 않는다(15분 뒤 레이블이 아직 없음).
    """

    def __init__(self, window: int = 20, level: float = 0.9):
        self.window, self.level = window, level

    def factors(self, p: pd.DataFrame) -> pd.Series:
        ratio = (p["y_abs"] / p["q90"].clip(lower=1e-6)).groupby(p["date"]).apply(np.asarray)
        dates = list(ratio.index)
        k = {}
        for i, d in enumerate(dates):
            past = dates[max(0, i - self.window): i]
            k[d] = float(np.quantile(np.concatenate([ratio[x] for x in past]), self.level)) \
                if len(past) == self.window else 1.0
        return pd.Series(k, name="k")

    def apply(self, p: pd.DataFrame) -> pd.DataFrame:
        k = self.factors(p)
        return p.assign(k=p["date"].map(k), q90=p["q90"] * p["date"].map(k))
