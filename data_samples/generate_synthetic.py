"""Replay Mode용 합성 샘플 데이터 생성기.

※ 생성 결과는 실제 시장 데이터가 아니다. 실습 재현용이며 전략 수익성의 근거로 사용하지 않는다.

생성 파일 (data_samples/synthetic/):
- futures_1min.csv : 코스피200 선물 1분봉 (정규장 08:45~15:45)
- option_quotes_1min.csv : ATM ±10 행사가 콜·풋의 1분 간격 최우선 호가(Black-76 + 스프레드)

사용법:
    python data_samples/generate_synthetic.py --days 5 --seed 42
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlab.market_data.spec import round_to_tick  # noqa: E402
from optlab.pricing import black76  # noqa: E402

STRIKE_STEP = 2.5
RISK_FREE = 0.03


def trading_days(start: date, n: int) -> list[date]:
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def next_thursday(d: date) -> date:
    return d + timedelta(days=(3 - d.weekday()) % 7 or 7)


def generate(days: int, seed: int, start: date, f0: float, vol: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    minutes_per_day = 7 * 60  # 08:45~15:45
    dt_year = 1.0 / (252 * minutes_per_day)

    fut_rows, opt_rows = [], []
    f = f0
    for day in trading_days(start, days):
        expiry = datetime.combine(next_thursday(day), time(15, 20))
        for m in range(minutes_per_day):
            ts = datetime.combine(day, time(8, 45)) + timedelta(minutes=m)
            ret = rng.normal(0.0, vol * math.sqrt(dt_year))
            o = f
            path = o * np.exp(np.cumsum(rng.normal(0.0, vol * math.sqrt(dt_year / 4), 4)))
            c = o * math.exp(ret)
            h = max(o, c, path.max())
            lo = min(o, c, path.min())
            v = int(rng.integers(200, 2000))
            fut_rows.append((ts, round(o, 2), round(h, 2), round(lo, 2), round(c, 2), v))
            f = c

            T = max((expiry - ts).total_seconds() / (365 * 24 * 3600), 1e-6)
            atm = round(f / STRIKE_STEP) * STRIKE_STEP
            for k in range(-10, 11):
                K = atm + k * STRIKE_STEP
                skew = vol * (1.0 + 0.15 * abs(k) / 10)
                for is_call in (True, False):
                    theo = black76.price(f, K, T, RISK_FREE, skew, is_call)
                    if theo < 0.01:
                        continue
                    half = max(0.01, theo * 0.01) * (1 + abs(k) / 5)
                    bid = round_to_tick(max(theo - half, 0.01), "down")
                    ask = round_to_tick(theo + half, "up")
                    if ask <= bid:
                        ask = round(bid + 0.01 if bid < 10 else bid + 0.05, 2)
                    opt_rows.append((
                        ts, expiry.date().isoformat(), "C" if is_call else "P", K,
                        bid, ask, int(rng.integers(1, 300)), int(rng.integers(1, 300)),
                    ))

    fut = pd.DataFrame(fut_rows, columns=["time", "open", "high", "low", "close", "volume"])
    opt = pd.DataFrame(opt_rows, columns=["time", "expiry", "cp", "strike", "bid", "ask", "bid_size", "ask_size"])
    return fut, opt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 5))
    p.add_argument("--f0", type=float, default=965.0, help="시작 선물가격(포인트)")
    p.add_argument("--vol", type=float, default=0.18, help="연 변동성")
    p.add_argument("--out", type=Path, default=ROOT / "data_samples" / "synthetic")
    args = p.parse_args()

    fut, opt = generate(args.days, args.seed, args.start, args.f0, args.vol)
    args.out.mkdir(parents=True, exist_ok=True)
    fut.to_csv(args.out / "futures_1min.csv", index=False)
    opt.to_csv(args.out / "option_quotes_1min.csv", index=False)
    print(f"[합성 데이터 - 실제 시장 데이터 아님] 선물 {len(fut):,}행, 옵션 호가 {len(opt):,}행 -> {args.out}")


if __name__ == "__main__":
    main()
