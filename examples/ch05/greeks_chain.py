"""5장 실습: 옵션 체인 IV·Greeks 분석기 (Replay Mode).

    python data_samples/generate_synthetic.py      # 최초 1회
    python examples/ch05/greeks_chain.py --time "2026-01-05 10:00"

합성 샘플의 특정 시각 옵션 체인에서 중간가(mid)로 IV를 역산하고 Greeks를 계산한다.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.pricing import black76  # noqa: E402

RISK_FREE = 0.03


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--time", default="2026-01-05 10:00")
    p.add_argument("--data", type=Path, default=ROOT / "data_samples" / "synthetic")
    args = p.parse_args()

    fut = pd.read_csv(args.data / "futures_1min.csv", parse_dates=["time"])
    opt = pd.read_csv(args.data / "option_quotes_1min.csv", parse_dates=["time"])
    ts = pd.Timestamp(args.time)

    F = float(fut.loc[fut["time"] == ts, "close"].iloc[0])
    chain = opt[opt["time"] == ts].copy()
    expiry = datetime.fromisoformat(chain["expiry"].iloc[0]).replace(hour=15, minute=20)
    T = (expiry - ts.to_pydatetime()).total_seconds() / (365 * 24 * 3600)

    rows = []
    for r in chain.itertuples():
        is_call = r.cp == "C"
        mid = (r.bid + r.ask) / 2
        iv = black76.implied_vol(mid, F, r.strike, T, RISK_FREE, is_call)
        if iv is None:
            continue
        g = black76.greeks(F, r.strike, T, RISK_FREE, iv, is_call)
        rows.append((r.cp, r.strike, r.bid, r.ask, iv * 100, g.delta, g.gamma, g.theta, g.vega))

    out = pd.DataFrame(rows, columns=["cp", "strike", "bid", "ask", "iv%", "delta", "gamma", "theta/day", "vega/1%"])
    print(f"[합성 데이터] {ts}  선물 F={F:.2f}  잔존만기 {T * 365:.2f}일")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(out.sort_values(["cp", "strike"]).to_string(index=False))


if __name__ == "__main__":
    main()
