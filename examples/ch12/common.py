"""12장 예제 공통: 선물 분봉 → 특징·레이블, 워크포워드 예측 캐시."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.analysis.drift import from_forecaster, predict_h, walk_forward  # noqa: E402
from optlab.features.build import FEATURES, add_labels, build_features  # noqa: E402
from optlab.models.forecast import Forecaster  # noqa: E402

DATA = ROOT / "data"
MODEL_PATH = ROOT / "models_store" / "forecaster.joblib"
MONTHS = [str(m) for m in pd.period_range("2026-04", "2026-10", freq="M")]


def load_dataset() -> pd.DataFrame:
    """06장 연속 분봉 + 그 뒤에 받은 날짜별 분봉 파일(futures_1min_YYYYMMDD.parquet)."""
    parts = [pd.read_parquet(DATA / "futures_1min_continuous.parquet")]
    parts += [pd.read_parquet(p) for p in sorted(DATA.glob("futures_1min_2*.parquet"))]
    bars = pd.concat(parts).drop_duplicates("time", keep="last")
    return add_labels(build_features(bars)).dropna(subset=["ret_30", "y_up_15", "y_absret_15"])


def predictions(df: pd.DataFrame, policy: str, refresh: bool = False) -> pd.DataFrame:
    """정책별 워크포워드 15분 예측. 학습에 몇십 초 걸리므로 data/ch12/에 저장해 두고 다시 쓴다.

    deployed는 07장에서 저장한 예측기(models_store/forecaster.joblib)를 그대로 쓴다.
    """
    cache = DATA / "ch12" / f"wf_{policy}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    if policy == "deployed":
        fc = Forecaster.load(MODEL_PATH)
        te = df[pd.to_datetime(df["date"]) > pd.Timestamp(fc.trained_until)]
        p = te[["time", "date", "y_up_15", "y_absret_15"] + FEATURES]
        p = p.join(predict_h(from_forecaster(fc), te)).rename(columns={"y_up_15": "y_up", "y_absret_15": "y_abs"})
    elif policy == "fixed":
        p = walk_forward(df, MONTHS, "fixed", fixed_end="2026-03-31")
    else:
        p = walk_forward(df, MONTHS, policy)
    cache.parent.mkdir(exist_ok=True)
    p.to_parquet(cache)
    return p

