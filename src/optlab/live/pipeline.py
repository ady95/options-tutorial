"""실시간 자동 모의매매 파이프라인 — 수신 → 시장 상태 → 결정 → 위험관리 → 주문 상태기계 → 가상 브로커.

asyncio 작업은 두 개다.
- 수신(consume): 메시지가 올 때마다 sched.advance(수신 시각) → 시장 상태 반영 → sched.on_quote()
- 시계(timer): Live에서만. 메시지가 끊겨도 0.2초마다 sched.advance(지금)을 불러 결정·점검이 멈추지 않게 한다.
  선물 시세가 끊긴 것을 알아채는 것도 이 시계다(메시지가 없으면 수신 쪽은 깨어나지 않는다).
상태를 바꾸는 코드(advance, on_quote, on_trade)에는 await가 없다. 그래서 두 작업이 섞여 실행돼도
한 번의 처리 도중에 다른 작업이 끼어들지 않는다.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Callable, Iterator

import pandas as pd

from optlab.execution.scheduler import Scheduler
from optlab.features.build import build_features
from optlab.live.market import MarketBook, Quote, Trade, quote_from_event, trade_from_event
from optlab.strategy import clock
from optlab.strategy.signal import decide, implied_forward


def make_decider(book: MarketBook, model, cfg, expiry, r: float = 0.03) -> Callable:
    """결정 시각 ts에 알 수 있었던 체인·분봉만으로 8장 시그널 엔진을 돌린다."""
    def decide_fn(ts: pd.Timestamp):
        ch = book.chain(ts)
        if ch.empty:
            return None
        F = implied_forward(ch, clock.years(ts.to_pydatetime(), expiry), r)
        if F is None:
            return None
        fc = next(f for f in model.predict(build_features(book.bars_until(ts))) if f.horizon == cfg.horizon_min)
        return decide(ch, F, ts.to_pydatetime(), expiry, fc, cfg)
    return decide_fn


class ChainTracker:
    """지수를 따라 구독 체인을 옮긴다(03-1·04-3·08-3절에서 미룬 일).

    가운데 행사가에서 recenter_steps칸 이상 멀어지면 새 ATM ± width로 바꾼다. 경계에서 오가며 구독을
    자주 바꾸지 않도록 한 칸이 아니라 두 칸을 기다린다. 해제를 먼저 보내 한 연결의 구독 한도를 넘지 않게 한다.
    """

    def __init__(self, universe: pd.DataFrame, width: int, center: float, recenter_steps: int = 2):
        self.universe = universe              # 그 만기의 전체 행사가(code, cp, strike)
        self.strikes = sorted(universe["strike"].unique())
        self.width, self.steps, self.center = width, recenter_steps, center

    def window(self, center: float) -> list[float]:
        i = self.strikes.index(center)
        return self.strikes[max(0, i - self.width): i + self.width + 1]

    def codes(self, strikes) -> list[str]:
        return list(self.universe.loc[self.universe["strike"].isin(strikes), "code"])

    def check(self, F: float | None) -> tuple[list[str], list[str], tuple, tuple] | None:
        """옮겨야 하면 (추가 코드, 해제 코드, 이전 범위, 새 범위)."""
        if F is None:
            return None
        near = min(self.strikes, key=lambda k: abs(k - F))
        if abs(self.strikes.index(near) - self.strikes.index(self.center)) < self.steps:
            return None
        old, new = self.window(self.center), self.window(near)
        self.center = near
        return (self.codes(sorted(set(new) - set(old))), self.codes(sorted(set(old) - set(new))),
                (old[0], old[-1]), (new[0], new[-1]))


def resubscriber(tracker, book: MarketBook, get_trader, control=None, subs=None, limit: int = 40):
    """분마다 ATM을 확인해 구독 체인을 옮긴다. 보유 중인 종목은 해제하지 않는다(청산할 호가가 필요하다).
    control이 없으면(Replay) 옮겨야 할 시점만 기록한다. 기록 파일에는 새 행사가의 호가가 없기 때문이다."""
    pinned: set[str] = set()

    def on_minute(t: pd.Timestamp) -> None:
        trader = get_trader()
        held = {trader.pos["code"]} if trader.pos else set()
        r = tracker.check(book.futures_mid())
        window = set(tracker.codes(tracker.window(tracker.center)))
        stale_pins = [c for c in pinned if c not in held and c not in window]   # 청산이 끝난 고정 종목
        if r is None and not stale_pins:
            return
        add, remove, old, new = r if r else ([], [], None, None)
        keep = [c for c in remove if c in held]
        remove = [c for c in remove if c not in held] + stale_pins
        pinned.difference_update(stale_pins)
        pinned.update(keep)
        if control is None:
            if r:
                trader._log("RESUBSCRIBE_NEEDED", f"{old[0]:g}~{old[1]:g} → {new[0]:g}~{new[1]:g} "
                                                  f"(기록 파일이라 실제 구독은 바꾸지 않음)")
            return
        room = limit - (len(subs) - len(remove))
        k = tracker.universe.set_index("code")["strike"]
        add = sorted(add, key=lambda c: abs(k[c] - tracker.center))[:max(room, 0)]   # 한도가 모자라면 ATM에 가까운 것부터
        for c in remove:
            control.put_nowait(("H0IOASP0", c, False))     # 해제를 먼저(구독 한도)
            subs.remove(("H0IOASP0", c))
        for c in add:
            control.put_nowait(("H0IOASP0", c, True))
            subs.append(("H0IOASP0", c))
        book.drop(remove)
        what = f"{old[0]:g}~{old[1]:g} → {new[0]:g}~{new[1]:g}, " if r else ""
        trader._log("RESUBSCRIBE", f"{what}해제 {len(remove)}, 추가 {len(add)}, 보유 종목 유지 {len(keep)}")

    return on_minute


def minute_marks(start: pd.Timestamp, end: pd.Timestamp, delay_sec: float) -> Iterator[pd.Timestamp]:
    """매분 hh:mm:00 + delay. delay는 직전 분봉의 마지막 체결이 도착할 시간이다."""
    t = start.ceil("min")
    while t <= end:
        yield t + pd.Timedelta(seconds=delay_sec)
        t += pd.Timedelta(minutes=1)


class SimClock:
    """기록 시각을 speed배로 흐르게 하는 시계. 배속 재생에서 Live의 벽시계 자리를 대신한다."""

    def __init__(self, t0: pd.Timestamp, speed: float):
        self.t0, self.speed, self.wall0 = t0, speed, time.perf_counter()

    def __call__(self) -> pd.Timestamp:
        return self.t0 + pd.Timedelta(seconds=(time.perf_counter() - self.wall0) * self.speed)


async def replay_feed(quotes: pd.DataFrame, trades: pd.DataFrame, after: pd.Timestamp | None = None,
                      sim: SimClock | None = None) -> AsyncIterator[tuple]:
    """기록 파일을 수신 시각 순서로 내보낸다. sim이 없으면 기다리지 않고, 있으면 그 시계에 맞춰 내보낸다."""
    parts = [df.assign(src_kind=k) for k, df in ((0, quotes), (1, trades)) if len(df)]
    allv = pd.concat(parts, ignore_index=True).sort_values(["received_time", "src_kind"], kind="stable")
    if after is not None:
        allv = allv[allv["received_time"] > after]
    for i, row in enumerate(allv.itertuples(index=False)):
        if sim is not None:
            lag = (row.received_time - sim()).total_seconds() / sim.speed
            if lag > 0.005:
                await asyncio.sleep(lag)
        elif i % 2000 == 0:
            await asyncio.sleep(0)
        if row.src_kind == 0:
            yield "quote", Quote(row.code, row.exchange_time, row.received_time, row.bid1, row.ask1,
                                 row.bid_size1, row.ask_size1, row.depth)
        else:
            yield "trade", Trade(row.code, row.exchange_time, row.received_time, row.price, row.acml_vol)


async def live_feed(stream: AsyncIterator[dict], store=None) -> AsyncIterator[tuple]:
    """realtime.stream()의 dict 이벤트 → (종류, 객체). store가 있으면 받은 그대로 기록도 한다(12장 분석용)."""
    async for ev in stream:
        if store is not None:
            store.add(ev)
        if ev["type"] == "quote":
            yield "quote", quote_from_event(ev)
        elif ev["type"] == "trade":
            yield "trade", trade_from_event(ev)
        else:
            yield ev["type"], ev


async def run(feed: AsyncIterator[tuple], book: MarketBook, sched: Scheduler,
              wall_clock: Callable[[], pd.Timestamp] | None = None, stop_at: pd.Timestamp | None = None,
              tick_sec: float = 0.2, on_minute: Callable[[pd.Timestamp], None] | None = None) -> str:
    """feed가 끝나거나 stop_at에 닿을 때까지 돈다. 끝난 이유를 돌려준다.
    on_minute: 분이 바뀔 때마다 한 번 부른다(구독 체인 이동 등)."""
    trader = sched.trader
    minute = [None]

    def step(t: pd.Timestamp) -> None:
        sched.advance(t)
        if on_minute is not None and t.floor("min") != minute[0]:
            minute[0] = t.floor("min")
            on_minute(t)

    async def consume() -> str:
        async for kind, ev in feed:
            if kind in ("quote", "trade"):
                t = ev.received_time
                if stop_at is not None and t >= stop_at:
                    return "STOP_AT"
                step(t)
                if kind == "quote":
                    book.on_quote(ev)
                    sched.on_quote(ev)
                else:
                    book.on_trade(ev)
            else:                              # 수신기의 오류·재접속 알림도 감사 기록에 남긴다
                trader._log("FEED_" + kind.upper(), ev.get("msg", ""))
        return "FEED_END"

    async def timer() -> str:
        while True:
            await asyncio.sleep(tick_sec)
            now = wall_clock()
            if stop_at is not None and now >= stop_at:
                return "STOP_AT"
            step(now)

    if wall_clock is None:
        return await consume()
    tasks = [asyncio.create_task(consume()), asyncio.create_task(timer())]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for p in pending:
        p.cancel()
    return next(iter(done)).result()
