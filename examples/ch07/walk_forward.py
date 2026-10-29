"""7장 실습: 월별 워크포워드 검증.

    python examples/ch06/make_dataset.py      # data/dataset.parquet
    python examples/ch07/walk_forward.py

매달: 그 달 이전 전체 데이터로 학습 → 그 달 평가. 방향 AUC(로지스틱·LightGBM)와 15분 변동폭 상관.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.features.build import FEATURES  # noqa: E402

P = dict(n_estimators=150, learning_rate=0.03, num_leaves=15, min_child_samples=200, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=42)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore")
    df = pd.read_parquet(ROOT / "data" / "dataset.parquet").dropna(subset=["ret_30"])
    df["ym"] = pd.to_datetime(df["date"]).dt.to_period("M")
    X = lambda d: d[FEATURES].fillna(0.0)    # noqa: E731
    rows = []
    for m in pd.period_range("2026-04", "2026-10", freq="M"):
        tr, te = df[df["ym"] < m], df[df["ym"] == m]
        r = {"month": str(m), "days": te["date"].nunique()}
        for h in (5, 15, 30):
            a, c = tr.dropna(subset=[f"y_up_{h}"]), te.dropna(subset=[f"y_up_{h}"])
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)).fit(X(a), a[f"y_up_{h}"])
            gb = lgb.LGBMClassifier(**P).fit(X(a), a[f"y_up_{h}"])
            r[f"lr{h}"] = roc_auc_score(c[f"y_up_{h}"], lr.predict_proba(X(c))[:, 1])
            r[f"lgb{h}"] = roc_auc_score(c[f"y_up_{h}"], gb.predict_proba(X(c))[:, 1])
        a, c = tr.dropna(subset=["y_absret_15"]), te.dropna(subset=["y_absret_15"])
        q = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **P).fit(X(a), a["y_absret_15"])
        r["vol15_corr"] = np.corrcoef(c["y_absret_15"], q.predict(X(c)))[0, 1]
        rows.append(r)
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    print("평균", out.drop(columns=["month", "days"]).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
