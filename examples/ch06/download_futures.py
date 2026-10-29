"""6장 실습: 코스피200 선물 최근월 연속 1분봉 내려받기.

    python examples/ch06/download_futures.py --start 2025-12-12 --end 2026-10-28

KIS 실전 키 필요(시세 조회만). 결과: data/futures_1min_continuous.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.broker.kis import KisClient, KisConfig, load_dotenv_if_present  # noqa: E402
from optlab.market_data.history import fetch_continuous, front_month_schedule  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, default=date(2025, 12, 12))
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "futures_1min_continuous.parquet")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    load_dotenv_if_present(ROOT / ".env")
    kis = KisClient(KisConfig.from_env("real"))
    for code, s, e in front_month_schedule(args.start, args.end):
        print(f"  {code}: {s} ~ {e}")

    last = [0.0]

    def progress(code, oldest, n):
        if time.time() - last[0] > 10:
            last[0] = time.time()
            print(f"{code} {oldest:%Y-%m-%d %H:%M}까지 {n:,}개")

    t0 = time.time()
    df = fetch_continuous(kis, args.start, args.end, progress)
    args.out.parent.mkdir(exist_ok=True)
    df.to_parquet(args.out, index=False)
    days = df["time"].dt.date.nunique()
    print(f"\n완료 {time.time() - t0:.0f}초: {len(df):,}개 분봉, {days}거래일, "
          f"{df['time'].min()} ~ {df['time'].max()} -> {args.out}")
    print(df.groupby("contract")["time"].agg(["min", "max", "count"]).to_string())


if __name__ == "__main__":
    main()
