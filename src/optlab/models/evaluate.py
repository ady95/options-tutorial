"""방향 확률·변동폭 예측 평가 지표."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def classification_report(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": len(y), "accuracy": float(((p > 0.5) == y).mean()), "auc": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p))}


def calibration_table(y: np.ndarray, p: np.ndarray, bins=(0, .45, .48, .5, .52, .55, 1)) -> pd.DataFrame:
    """예측 확률 구간별 실제 상승 비율. 잘 보정된 모델은 두 값이 비슷하다."""
    cut = pd.cut(p, bins, include_lowest=True)
    return (pd.DataFrame({"bin": cut, "p": p, "y": y}).groupby("bin", observed=True)
            .agg(n=("y", "size"), mean_pred=("p", "mean"), actual_up=("y", "mean")))


def regression_report(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"mae": float(np.mean(np.abs(y - pred))), "corr": float(np.corrcoef(y, pred)[0, 1])}
