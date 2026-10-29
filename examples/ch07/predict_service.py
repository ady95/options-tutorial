"""7장 실습: 실시간 입력을 받는 예측 서비스.

    python examples/ch07/predict_service.py --train --train-end 2026-08-31   # 학습(기간 지정)
    python examples/ch07/predict_service.py --train                  # 전체 이력으로 학습(Live용)
    python examples/ch07/predict_service.py --mode replay --date 2026-10-28 --at 10:30
    python examples/ch07/predict_service.py --mode live --minutes 3  # 장중: 1분마다 당일 분봉을 받아 예측

출력: h분(5·15·30) 뒤 상승확률, |수익률| 50%·90% 분위수(bp와 지수 포인트)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.features.build import add_labels, build_features  # noqa: E402
from optlab.models.forecast import Forecaster  # noqa: E402

MODEL_PATH = ROOT / "models_store" / "forecaster.joblib"
BARS_PATH = ROOT / "data" / "futures_1min_continuous.parquet"


def show(fc: Forecaster, bars: pd.DataFrame, label: str) -> None:
    feats = build_features(bars)
    last = feats.iloc[-1]
    print(f"[{label}] 기준 분봉 {last['time']:%Y-%m-%d %H:%M}, 선물 {last['close']:.2f}  (모델 학습 ~{fc.trained_until})")
    print(f"{'h':>4}{'상승확률':>9}{'|r| 50%':>10}{'|r| 90%':>10}{'50% pt':>9}{'90% pt':>9}")
    for f in fc.predict(feats):
        print(f"{f.horizon:>3}분{f.p_up:>9.3f}{f.absret_q50 * 1e4:>8.1f}bp{f.absret_q90 * 1e4:>8.1f}bp"
              f"{f.absret_q50 * last['close']:>9.2f}{f.absret_q90 * last['close']:>9.2f}")


def live_bars(kis, code: str) -> pd.DataFrame:
    from optlab.market_data.history import fetch_contract
    today = date.today()
    return fetch_contract(kis, code, today, today)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--train-end", help="이 날짜까지만 학습(Replay 시연은 학습에 안 쓴 날짜로)")
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--date", default="2026-10-28")
    p.add_argument("--at", default="10:30")
    p.add_argument("--minutes", type=int, default=3)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    if args.train:
        t0 = time.time()
        df = add_labels(build_features(pd.read_parquet(BARS_PATH)))
        if args.train_end:
            df = df[pd.to_datetime(df["date"]) <= args.train_end]
        Forecaster().fit(df).save(MODEL_PATH)
        print(f"학습 완료 {time.time() - t0:.0f}초 -> {MODEL_PATH}")
        return

    fc = Forecaster.load(MODEL_PATH)
    if args.mode == "replay":
        bars = pd.read_parquet(BARS_PATH)
        day = bars[bars["time"].dt.date == date.fromisoformat(args.date)]
        cut = datetime.combine(date.fromisoformat(args.date), datetime.strptime(args.at, "%H:%M").time())
        show(fc, day[day["time"] <= cut], "Replay")
        return

    from optlab.broker.kis import KisClient, KisConfig, load_dotenv_if_present
    from optlab.market_data import master
    load_dotenv_if_present(ROOT / ".env")
    kis = KisClient(KisConfig.from_env("real"))
    mpath = ROOT / ".cache" / "fo_idx_code_mts.mst"
    if not mpath.exists():
        master.download_master(mpath)
    code = master.nearest_futures(master.load_master(mpath)).code
    for i in range(args.minutes):
        show(fc, live_bars(kis, code), f"Live {code}")
        if i < args.minutes - 1:
            time.sleep(60 - datetime.now().second + 2)   # 다음 분봉이 확정된 직후
            print()


if __name__ == "__main__":
    main()
