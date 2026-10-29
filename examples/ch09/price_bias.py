"""09-1: 같은 매매를 '마지막 체결가'·'중간가'·'호가(Ask 매수, Bid 매도)'로 계산하면 얼마나 다른가.

매분, ATM ± N 행사가의 옵션을 하나씩 사서 h분 뒤 판다고 보고 세 가지 가격으로 손익을 잰다.
신호와 무관하게 모든 옵션·시점을 보므로 '가격을 무엇으로 쓰느냐'의 차이만 드러난다.

    python examples/ch09/price_bias.py --db data/ticks_20261029_live.db
    python examples/ch09/price_bias.py --db data/ticks_20261029_mon.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch08"))
from market import LiveDB, market_state  # noqa: E402

from optlab.market_data.spec import MULTIPLIER_KRW  # noqa: E402


def last_price_at(ticks: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """ts까지 알 수 있었던 종목별 마지막 체결가와 그 체결이 몇 초 전이었는지."""
    t = ticks[ticks["received_time"] <= ts].sort_values("received_time").groupby("code").tail(1)
    return t.assign(last_age=(ts - t["received_time"]).dt.total_seconds())[["code", "price", "last_age"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--atm-range", type=int, default=3)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    src = LiveDB(Path(args.db))
    lo, hi = src.span()
    h = pd.Timedelta(minutes=args.horizon)
    rows = []
    for ts in pd.date_range(lo, hi - h, freq="1min"):
        ch, F, _ = market_state(src, ts)
        if F is None:
            continue
        step = ch["strike"].drop_duplicates().sort_values().diff().median()
        ch = ch[(ch["strike"] - F).abs() <= (args.atm_range + 0.5) * step]
        later = src.chain(ts + h)
        a = ch.merge(last_price_at(src.trades, ts), on="code", how="left")
        b = later[["code", "bid1", "ask1"]].merge(last_price_at(src.trades, ts + h), on="code", how="left")
        m = a.merge(b, on="code", suffixes=("", "_1"))
        m = m[(m["bid1"] > 0) & (m["ask1"] > 0) & (m["bid1_1"] > 0) & (m["ask1_1"] > 0)]
        rows.append(m.assign(time=ts))
    d = pd.concat(rows, ignore_index=True)
    d["mid"], d["mid_1"] = (d.bid1 + d.ask1) / 2, (d.bid1_1 + d.ask1_1) / 2
    d["quote_pnl"] = d.bid1_1 - d.ask1
    d["mid_pnl"] = d.mid_1 - d.mid
    d["last_pnl"] = d.price_1 - d.price
    d["outside"] = (d.price < d.bid1) | (d.price > d.ask1)

    print(f"[{src.label}] 보유 {args.horizon}분, ATM ±{args.atm_range}, 왕복 {len(d):,}건(옵션×시점, 1계약)")
    print("\n== 1. 마지막 체결가는 얼마나 오래된 가격인가")
    print(f"체결 기록이 없는 경우 {d['price'].isna().mean():.1%}")
    age = d["last_age"].dropna()
    print(f"마지막 체결 후 경과: 중앙값 {age.median():.0f}초, 90% {age.quantile(0.9):.0f}초, 최대 {age.max():.0f}초")
    has = d.dropna(subset=["price"])
    print(f"마지막 체결가가 지금의 매수~매도호가 밖에 있는 비율 {has['outside'].mean():.1%}")

    v = d.dropna(subset=["price", "price_1"])
    print(f"\n== 2. 같은 왕복을 세 가지 가격으로(두 시점 모두 체결 기록이 있는 {len(v):,}건, 포인트, 수수료 제외)")
    res = pd.DataFrame({
        "평균 손익": [v.last_pnl.mean(), v.mid_pnl.mean(), v.quote_pnl.mean()],
        "원화": [x * MULTIPLIER_KRW for x in (v.last_pnl.mean(), v.mid_pnl.mean(), v.quote_pnl.mean())],
        "이긴 비율": [(v.last_pnl > 0).mean(), (v.mid_pnl > 0).mean(), (v.quote_pnl > 0).mean()],
        "0인 비율": [(v.last_pnl == 0).mean(), (v.mid_pnl == 0).mean(), (v.quote_pnl == 0).mean()],
    }, index=["마지막 체결가", "중간가", "호가(Ask→Bid)"])
    print(res.round(3).to_string())
    gap = v.last_pnl - v.quote_pnl
    print(f"\n체결가 - 호가 차이: 평균 {gap.mean():+.3f}pt ({gap.mean() * MULTIPLIER_KRW:+,.0f}원), "
          f"표준편차 {gap.std():.3f}pt, 체결가 쪽이 더 좋게 나온 비율 {(gap > 0).mean():.0%}")
    spread = (v.ask1 - v.bid1 + v.ask1_1 - v.bid1_1) / 2
    print(f"참고: 진입·청산 시점 스프레드 평균 {spread.mean():.3f}pt")


if __name__ == "__main__":
    main()
