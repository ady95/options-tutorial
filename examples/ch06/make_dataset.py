"""6장 실습: 1분봉 → 특징·레이블 데이터셋.

    python examples/ch06/make_dataset.py
결과: data/dataset.parquet (특징 + 5·15·30분 레이블)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.features.build import FEATURES, HORIZONS, add_labels, build_features, time_split  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=Path, default=ROOT / "data" / "futures_1min_continuous.parquet")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "dataset.parquet")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    pd.set_option("display.width", 140)

    bars = pd.read_parquet(args.bars)
    df = add_labels(build_features(bars))
    print(f"분봉 {len(bars):,}개, {df['date'].nunique()}거래일, 특징 {len(FEATURES)}개")

    print("\n== 특징 결측 비율(장 초반 창 길이만큼 비는 것이 정상)")
    print(df[FEATURES].isna().mean().sort_values(ascending=False).head(6).round(3).to_string())

    print("\n== 레이블 요약")
    rows = []
    for h in HORIZONS:
        r = df[f"y_ret_{h}"]
        rows.append({"h": h, "n": int(r.notna().sum()), "상승비율": df[f"y_up_{h}"].mean(),
                     "변화없음": float((r == 0).mean()), "|수익률| 중앙값(bp)": r.abs().median() * 1e4,
                     "|수익률| 90%(bp)": r.abs().quantile(.9) * 1e4})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    tr, va, te = time_split(df, "2026-06-30", "2026-08-31")
    print("\n== 시간순 분할")
    for name, d in (("학습", tr), ("검증", va), ("테스트", te)):
        print(f"{name}: {d['date'].min()} ~ {d['date'].max()}, {d['date'].nunique()}일, "
              f"15분 상승비율 {d['y_up_15'].mean():.3f}")

    df.to_parquet(args.out, index=False)
    print("\n저장:", args.out)


if __name__ == "__main__":
    main()
