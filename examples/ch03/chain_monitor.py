"""3장 실습: 실시간 지수·옵션 체인 모니터.

    python examples/ch03/chain_monitor.py --mode replay              # 합성 데이터 재생
    python examples/ch03/chain_monitor.py --mode live --seconds 60   # KIS 실시간(장중)

live 모드는 ATM ± width 행사가의 콜·풋 호가와 선물 체결을 WebSocket으로 구독한다.
구독 수 = (2*width+1) * 2 + 1  ≤ 40.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.broker.replay import ReplaySource, synthetic_code  # noqa: E402

MASTER_PATH = ROOT / ".cache" / "fo_idx_code_mts.mst"


class Board:
    """종목별 최신 호가와 지연을 들고 있다가 표로 그린다."""

    def __init__(self, rows: list[tuple[float, str, str]]):
        self.rows = rows                     # (행사가, 콜 코드, 풋 코드)
        self.book: dict[str, tuple[float, float, int, int]] = {}
        self.fut = None
        self.lags: list[float] = []
        self.count = 0

    def on_event(self, ev: dict) -> None:
        self.count += 1
        if ev["type"] == "quote":
            self.book[ev["code"]] = (ev["bids"][0], ev["asks"][0], ev["bid_sizes"][0], ev["ask_sizes"][0])
        elif ev["type"] == "trade" and ev["tr_id"] in ("H0IFCNT0", "REPLAY_FUT"):
            self.fut = ev["price"]
        if ev.get("exchange_time") and ev["tr_id"].startswith("H0"):
            self.lags.append((ev["received_time"] - ev["exchange_time"]).total_seconds())

    def render(self, clock: str) -> str:
        fut = f"{self.fut:.2f}" if self.fut else "-"
        lines = [f"[{clock}] 선물 {fut}  이벤트 {self.count:,}건",
                 f"{'콜 잔량':>7}{'콜 매수':>8}{'콜 매도':>8} | {'행사가':^7} | {'풋 매수':>8}{'풋 매도':>8}{'풋 잔량':>8}"]
        for k, c, p in self.rows:
            cb, ca, cbs, cas = self.book.get(c, (0, 0, 0, 0))
            pb, pa, pbs, pas = self.book.get(p, (0, 0, 0, 0))
            lines.append(f"{cbs:>3}/{cas:<3}{cb:>8.2f}{ca:>8.2f} | {k:^7.1f} | {pb:>8.2f}{pa:>8.2f}{pbs:>4}/{pas:<3}")
        if self.lags:
            s = sorted(self.lags)
            lines.append(f"수신 지연(초) p50={s[len(s) // 2]:.2f} max={s[-1]:.2f}  (거래소 시각이 초 단위라 0~1초 오차 포함)")
        return "\n".join(lines)


async def run_replay(args) -> None:
    data_dir = ROOT / "data_samples" / "synthetic"
    if not data_dir.exists():
        sys.exit("합성 데이터가 없습니다. 먼저 python data_samples/generate_synthetic.py 를 실행하세요.")
    src = ReplaySource(data_dir)
    first = src.times()[0]
    atm = round(float(src.fut.iloc[0]["close"]) / 2.5) * 2.5
    strikes = [atm + 2.5 * i for i in range(-args.width, args.width + 1)]
    board = Board([(k, synthetic_code("C", k), synthetic_code("P", k)) for k in strikes])
    print("[합성 데이터 - 실제 시장 데이터 아님]")
    n = 0
    async for ev in src.stream(start=str(first), speed=args.speed, strikes=set(strikes)):
        board.on_event(ev)
        if ev["tr_id"] == "REPLAY_FUT":
            n += 1
            if n % args.every == 0:
                print(board.render(ev["exchange_time"].strftime("%m-%d %H:%M")) + "\n")
            if n >= args.minutes:
                break


async def run_live(args) -> None:
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
    subs = [("H0IFCNT0", fut.code)]
    for _k, c, p in chain:
        subs += [("H0IOASP0", c.code), ("H0IOASP0", p.code)]
    print(f"{args.product} {series}, ATM≈{atm:.2f}, 구독 {len(subs)}건")

    board = Board([(k, c.code, p.code) for k, c, p in chain])

    async def consume() -> None:
        last = 0.0
        async for ev in realtime.stream(kis.approval_key(), subs):
            if ev["type"] in ("error", "disconnected"):
                print("!!", ev["msg"])
                continue
            board.on_event(ev)
            if time.time() - last >= args.every:
                last = time.time()
                print(board.render(ev["received_time"].strftime("%H:%M:%S")) + "\n")

    # 이벤트가 끊겨도 정해진 시간에 끝나도록 바깥에서 시간 제한을 건다
    try:
        await asyncio.wait_for(consume(), timeout=args.seconds)
    except asyncio.TimeoutError:
        pass
    print(board.render("종료"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["replay", "live"], default="replay")
    p.add_argument("--width", type=int, default=4, help="ATM 위아래 행사가 수")
    p.add_argument("--every", type=int, default=10, help="replay: N분마다 / live: N초마다 출력")
    p.add_argument("--minutes", type=int, default=30, help="replay: 재생할 분 수")
    p.add_argument("--speed", type=float, default=600.0, help="replay 배속")
    p.add_argument("--seconds", type=int, default=60, help="live: 수신 시간")
    p.add_argument("--product", default="KOSPI200_WEEKLY_THU", choices=["KOSPI200_WEEKLY_THU", "KOSPI200_WEEKLY_MON"])
    p.add_argument("--series", help="live: 만기 시리즈(예: 2611W1). 생략하면 가장 가까운 만기")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # 파이프로 출력할 때도 줄 단위로 바로 보이게
    asyncio.run(run_replay(args) if args.mode == "replay" else run_live(args))


if __name__ == "__main__":
    main()
