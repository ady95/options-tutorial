"""4장 실습: 데이터 품질 리포트.

    python examples/ch04/quality_report.py data/ticks_YYYYMMDD_HHMM.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.market_data import quality, store  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("db", type=Path)
    p.add_argument("--freq", default="1s", help="스냅숏 간격")
    p.add_argument("--wide", type=float, default=0.2, help="넓은 스프레드 기준(중간가 대비)")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    pd.set_option("display.width", 140)

    d = store.load(args.db)
    q, t, inst = d["quotes"], d["trades"], d["instruments"]
    opt_codes = set(inst.dropna(subset=["strike"])["code"])
    span = q["received_time"].max() - q["received_time"].min()
    print(f"== 개요: 호가 {len(q):,}건, 체결 {len(t):,}건, 종목 {q['code'].nunique()}개, 기간 {span}")
    if len(d["events"]):
        print("운영 이벤트:", d["events"]["type"].value_counts().to_dict())

    print("\n== 1. 수신 지연(초) — 거래소 시각이 초 단위라 0~1초 과대 추정")
    print(quality.latency_summary(q, t).round(2).to_string(index=False))

    flags = quality.quote_flags(q[q["code"].isin(opt_codes)], args.wide)
    print("\n== 2·3. 옵션 호가 이상")
    print(f"한쪽 호가 없음 {flags['one_sided'].mean():.1%}, 교차(매수>=매도) {int(flags['crossed'].sum())}건, "
          f"넓은 스프레드(>{args.wide:.0%}) {flags['wide'].mean():.1%}")
    by = (flags.dropna(subset=["spread"]).merge(inst, on="code")
          .groupby(["cp", "strike"])["spread"].median().unstack("cp"))
    print("행사가별 스프레드 중앙값(포인트):")
    print(by.round(2).to_string())

    snap = quality.snapshots(q, inst, args.freq)
    inv = quality.strike_inversions(snap)
    n_snap = snap["time"].nunique() if len(snap) else 0
    print(f"\n== 4. 행사가 역전 ({args.freq} 스냅숏 {n_snap:,}개)")
    if inv.empty:
        print("없음")
    else:
        print(inv["kind"].value_counts().to_dict(), f"/ 역전이 있었던 시점 {inv['time'].nunique():,}개 "
              f"({inv['time'].nunique() / max(n_snap, 1):.1%})")
        print(inv.drop_duplicates(["cp", "rich_strike", "kind"]).head(6).to_string(index=False))
        # 오래된 호가가 만든 역전을 빼면 얼마나 남나
        for lim in (60, 10):
            fresh = quality.strike_inversions(snap[snap["age"] <= lim])
            share = fresh["time"].nunique() / max(n_snap, 1) if len(fresh) else 0.0
            print(f"  호가 age <= {lim}초만 비교: 역전 시점 {share:.1%}")

    st = quality.stale_trades(t, snap)
    print("\n== 5. 마지막 체결가가 현재 호가 범위 밖")
    if st.empty:
        print("없음")
    else:
        print(f"{len(st):,}건 / 스냅숏 행 {len(snap):,}개 ({len(st) / len(snap):.1%}), 체결 후 경과 중앙값 {st['trade_age'].median():.0f}초")
        print(st.sort_values("trade_age", ascending=False).drop_duplicates("code").head(5).round(2).to_string(index=False))

    print("\n== 6. 무소식 구간(종목별 최대 메시지 간격, 초)")
    sil = quality.silence(q, t).merge(inst, left_index=True, right_on="code")
    print(sil[["code", "cp", "strike", "messages", "median_gap", "max_gap"]].head(8).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
