"""단기 예측기 — h분 뒤 방향 확률과 변동폭 분포.

07장 실측 결론에 맞춘 구성:
- 방향: 로지스틱 회귀(LightGBM보다 일관되게 나았고, 확률이 0.5 근처로 겸손하게 나온다). 약한 보조 신호로만 쓴다.
- 변동폭: LightGBM 분위수 회귀로 |h분 로그수익률|의 50%·90% 분위수. 08장 옵션 재평가의 핵심 입력.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from optlab.features.build import FEATURES, HORIZONS

QUANTILES = (0.5, 0.9)
LGB_QUANTILE_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


@dataclass
class Forecast:
    horizon: int
    p_up: float
    absret_q50: float   # |로그수익률| 중앙값
    absret_q90: float   # |로그수익률| 90% 분위수


class Forecaster:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.direction: dict[int, object] = {}
        self.vol: dict[tuple[int, float], lgb.LGBMRegressor] = {}
        self.trained_until: str | None = None

    @staticmethod
    def _X(df: pd.DataFrame) -> pd.DataFrame:
        return df[FEATURES].fillna(0.0)

    def fit(self, df: pd.DataFrame) -> "Forecaster":
        df = df.dropna(subset=["ret_30"])
        for h in HORIZONS:
            d = df.dropna(subset=[f"y_up_{h}"])
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
            self.direction[h] = clf.fit(self._X(d), d[f"y_up_{h}"])
            for q in QUANTILES:
                reg = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=self.seed, **LGB_QUANTILE_PARAMS)
                self.vol[(h, q)] = reg.fit(self._X(d), d[f"y_absret_{h}"])
        self.trained_until = str(df["date"].max())
        return self

    def predict(self, rows: pd.DataFrame) -> list[Forecast]:
        """rows의 마지막 행(가장 최근 분봉) 기준 예측."""
        x = self._X(rows.tail(1))
        out = []
        for h in HORIZONS:
            q50 = float(self.vol[(h, 0.5)].predict(x)[0])
            q90 = float(self.vol[(h, 0.9)].predict(x)[0])
            out.append(Forecast(h, float(self.direction[h].predict_proba(x)[0, 1]), max(q50, 0.0), max(q90, q50)))
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "Forecaster":
        return joblib.load(path)


def interval_coverage(y_abs: np.ndarray, q90: np.ndarray) -> float:
    """실제 |수익률|이 예측 90% 분위수 이하인 비율. 잘 맞으면 0.9 근처."""
    return float((y_abs <= q90).mean())
