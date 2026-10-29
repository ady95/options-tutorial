"""4장 실습: 실시간 시세를 SQLite에 기록하기.

    python examples/ch04/record.py --minutes 30                 # Live (장중, KIS 실전 키)
    python examples/ch04/record.py --mode replay --minutes 60   # 합성 데이터

구독(Live): 선물 체결·호가 2건 + ATM ± width 콜·풋 호가와 체결 (기본 width 4 → 36건) = 38건
저장 위치: data/ticks_YYYYMMDD_HHMM.db  (주문 없음)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.market_data.store import EventStore  # noqa: E402

MASTER_PATH = ROOT / ".cache" / "fo_idx_code_mts.mst"


async def record_live(args, store: EventStore) -> None:
    from optlab.broker.kis import KisClient, KisConfig, load_dotenv_if_present
    from optlab.market_data import master, realtime

    load_dotenv_if_present(ROOT / ".env")
    kis = KisClient(KisConfig.from_env("real"))
    if not MASTER_PATH.exists():
        master.download_master(MASTER_PATH)
    inst = master.load_master(MASTER_PATH)
    fut = master.nearest_futures(inst)
    atm = float(kis.index_price("2001")["bstp_nmix_prpr"])
    series = args.series or master.series_list(inst, args.product)[0]
    chain = master.option_chain(inst, args.product, series, atm, width=args.width)

    store.add_instrument(fut.code, fut.product, None, fut.series, None)
    subs = [("H0IFCNT0", fut.code), ("H0IFASP0", fut.code)]
    for k, c, p in chain:
        for o in (c, p):
            store.add_instrument(o.code, o.product, o.cp, o.series, k)
            subs += [("H0IOASP0", o.code), ("H0IOCNT0", o.code)]
    print(f"{args.product} {series}, ATM≈{atm:.2f}, 구독 {len(subs)}건, {args.minutes}분 기록")

    n = 0

    async def consume() -> None:
        nonlocal n
        async for ev in realtime.stream(kis.approval_key(), subs):
            store.add(ev)
            n += 1
            if ev["type"] in ("error", "disconnected"):
                print("!!", ev["msg"])
            elif n % 5000 == 0:
                print(f"{datetime.now():%H:%M:%S} {n:,}건")

    try:
        await asyncio.wait_for(consume(), timeout=args.minutes * 60)
    except asyncio.TimeoutError:
        pass
    print(f"기록 종료: {n:,}건")


async def record_replay(args, store: EventStore) -> None:
    from optlab.broker.replay import ReplaySource, synthetic_code

    src = ReplaySource(ROOT / "data_samples" / "synthetic")
    for q in src.opt[["cp", "strike"]].drop_duplicates().itertuples():
        store.add_instrument(synthetic_code(q.cp, q.strike), "SYNTHETIC", q.cp, "SYN", q.strike)
    n = 0
    async for ev in src.stream(speed=0):
        store.add(ev)
        if ev["tr_id"] == "REPLAY_FUT":
            n += 1
            if n >= args.minutes:
                break
    print(f"[합성 데이터] {n}분 기록")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["live", "replay"], default="live")
    p.add_argument("--minutes", type=int, default=30)
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--product", default="KOSPI200_WEEKLY_THU", choices=["KOSPI200_WEEKLY_THU", "KOSPI200_WEEKLY_MON"])
    p.add_argument("--series")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    out = args.out or ROOT / "data" / f"{'ticks' if args.mode == 'live' else 'replay'}_{datetime.now():%Y%m%d_%H%M}.db"
    store = EventStore(out)
    try:
        asyncio.run(record_live(args, store) if args.mode == "live" else record_replay(args, store))
    finally:
        store.close()
        print("저장:", out)


if __name__ == "__main__":
    main()
