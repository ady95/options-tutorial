"""5장 실습: 기록한 실제 호가로 IV·Greeks 계산하고 KIS 제공 IV와 비교.

    python examples/ch05/live_chain_iv.py data/ticks_YYYYMMDD_HHMM.db --time 10:30:00

1. 지정 시각의 as-of 호가 스냅숏(04장 quality.snapshots)
2. 콜·풋 중간가 패리티로 만기까지의 내재 선도가격 F 추정(스프레드가 좁은 행사가들의 중앙값)
3. 중간가로 IV 역산, Greeks 계산
4. 같은 시점 KIS 체결 메시지의 IV(hts_ints_vltl)와 나란히 출력
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.market_data import quality, store  # noqa: E402
from optlab.market_data.master import weekly_expiry  # noqa: E402
from optlab.pricing import black76  # noqa: E402

RISK_FREE = 0.03
SETTLE_TIME = time(15, 30)   # 최종결제가격 = 만기일 코스피200 종가(15:30)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("db", type=Path)
    p.add_argument("--time", required=True, help="HH:MM:SS")
    p.add_argument("--max-spread", type=float, default=0.10, help="선도가격 추정에 쓸 최대 스프레드 비율")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    d = store.load(args.db)
    inst = d["instruments"]
    opts = inst.dropna(subset=["strike"])
    product, series = opts["product"].iloc[0], opts["series"].iloc[0]
    day = d["quotes"]["received_time"].iloc[0].date()
    ts = pd.Timestamp(datetime.combine(day, time.fromisoformat(args.time)))
    expiry = datetime.combine(weekly_expiry(product, series), SETTLE_TIME)
    T = (expiry - ts.to_pydatetime()).total_seconds() / (365 * 24 * 3600)

    q = d["quotes"][d["quotes"]["received_time"] <= ts]
    snap = quality.snapshots(q, inst, "1s")
    snap = snap[snap["time"] == snap["time"].max()]
    snap = snap[(snap["bid1"] > 0) & (snap["ask1"] > 0)].assign(mid=lambda x: (x.bid1 + x.ask1) / 2)
    snap["spread_ratio"] = (snap["ask1"] - snap["bid1"]) / snap["mid"]

    pair = snap.pivot(index="strike", columns="cp", values=["mid", "spread_ratio"]).dropna()
    tight = pair[(pair[("spread_ratio", "C")] <= args.max_spread) & (pair[("spread_ratio", "P")] <= args.max_spread)]
    use = tight if len(tight) else pair
    fwds = [black76.implied_forward(k, r[("mid", "C")], r[("mid", "P")], T, RISK_FREE) for k, r in use.iterrows()]
    F = float(pd.Series(fwds).median())

    fut = d["trades"][(d["trades"]["code"].str.startswith("A")) & (d["trades"]["received_time"] <= ts)]
    fut_px = fut["price"].iloc[-1] if len(fut) else float("nan")
    kis_iv = (d["trades"][(d["trades"]["received_time"] <= ts) & d["trades"]["iv"].notna()]
              .sort_values("received_time").groupby("code")["iv"].last())

    print(f"[실제 시세 기록] {ts:%Y-%m-%d %H:%M:%S}  {series} 만기 {expiry:%m-%d %H:%M}, 잔존 {T * 365 * 24:.2f}시간")
    print(f"내재 선도가격 F={F:.2f} (행사가 {len(use)}개 중앙값)  /  12월물 선물 체결가 {fut_px:.2f}")
    rows = []
    for r in snap.sort_values(["cp", "strike"]).itertuples():
        is_call = r.cp == "C"
        iv = black76.implied_vol(r.mid, F, r.strike, T, RISK_FREE, is_call)
        g = black76.greeks(F, r.strike, T, RISK_FREE, iv, is_call) if iv else None
        rows.append({"cp": r.cp, "strike": r.strike, "bid": r.bid1, "ask": r.ask1, "mid": r.mid,
                     "iv%": iv * 100 if iv else None, "kis_iv%": kis_iv.get(r.code),
                     "delta": g.delta if g else None, "gamma": g.gamma if g else None,
                     "theta/h": g.theta / 24 if g else None, "vega/1%": g.vega if g else None})
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
