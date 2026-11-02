"""14장 예제 공통: 합성 옵션 시장(하루 = 에피소드)과 학습·검증·테스트 기간.

12장 워크포워드 예측(expanding, 매달 재학습)을 관측으로 쓰므로 모든 날의 예측은 그날 이전 데이터로만 학습됐다.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# 12장 예제의 common.py(데이터셋·워크포워드 예측 캐시). 이 파일과 이름이 같아 경로로 불러온다
_spec = importlib.util.spec_from_file_location("ch12_common", ROOT / "examples" / "ch12" / "common.py")
ch12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch12)

from optlab.rl.market import OptionDay, build_days, har_iv_by_day  # noqa: E402

OUT = ROOT / "runs" / "ch14"
CACHE = ROOT / "data" / "ch14" / "days.pkl"
# 학습 4개월, 검증 1개월(모델 고르기), 테스트 2개월(한 번만 본다)
SPLITS = {"train": ("2026-04", "2026-07"), "valid": ("2026-08", "2026-08"), "test": ("2026-09", "2026-10")}


def load_days(refresh: bool = False) -> dict[str, list[OptionDay]]:
    if CACHE.exists() and not refresh:
        days = pickle.loads(CACHE.read_bytes())
    else:
        df = ch12.load_dataset()
        pred = ch12.predictions(df, "expanding")
        bars = df[["time", "close"]]
        iv = har_iv_by_day(ch12_bars(), sorted(pred["date"].unique()), profile_until="2026-04-01")
        days = build_days(pred, bars, iv)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(days))
    out = {}
    for name, (a, b) in SPLITS.items():
        out[name] = [d for d in days if a <= f"{d.day:%Y-%m}" <= b]
    return out


def ch12_bars():
    """HAR·시간대 패턴용 선물 1분봉 전체(13장과 같은 파일)."""
    parts = [pd.read_parquet(ch12.DATA / "futures_1min_continuous.parquet")]
    parts += [pd.read_parquet(p) for p in sorted(ch12.DATA.glob("futures_1min_2*.parquet"))]
    return pd.concat(parts).drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
