"""시각 순서 스케줄러 — 10장 Replay와 11장 실시간 파이프라인이 함께 쓰는 핵심 규칙.

메시지 하나(시각 t)를 반영하기 전에, t 이전에 일어났어야 할 일을 시각 순서대로 모두 처리한다.
    결정 시각 · 장애 주입 · 브로커의 주문 도착/만료 · 청산 예정 시각 · 1초마다 점검(tick)
Replay에서는 메시지의 수신 시각이 시계이고, Live에서는 벽시계로 advance()를 주기적으로 부른다.
advance()와 on_quote()에는 await가 없으므로 asyncio에서 한 번에 끝까지 실행된다(락이 필요 없다).
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import pandas as pd

from optlab.execution.paper import PaperBroker
from optlab.execution.trader import Trader


class Scheduler:
    def __init__(self, trader: Trader, broker: PaperBroker, decide_fn: Callable,
                 decisions: Iterable[pd.Timestamp],
                 faults: list[tuple[pd.Timestamp, str, Callable]] | None = None,
                 start: pd.Timestamp | None = None):
        self.trader, self.broker, self.decide_fn = trader, broker, decide_fn
        self._dec = iter(decisions)
        self.next_dec = next(self._dec, None)
        self.faults = sorted(faults or [], key=lambda f: f[0])
        self.j = 0
        self.next_tick: pd.Timestamp | None = None
        self.clock: pd.Timestamp | None = start  # 처리한 마지막 시각. 시계는 거꾸로 가지 않는다
        self.decide_sec: list[float] = []        # 결정 한 번에 걸린 계산 시간(초)

    def deliver(self) -> None:
        for ev in self.broker.drain():
            self.trader.on_broker(ev)

    def advance(self, t: pd.Timestamp) -> None:
        trader, broker = self.trader, self.broker
        if self.next_tick is None:
            self.next_tick = t.ceil("s")
        while True:
            cand = []
            if self.next_dec is not None and self.next_dec < t:
                cand.append((self.next_dec, 0, "decide"))
            if self.j < len(self.faults) and self.faults[self.j][0] <= t:
                cand.append((self.faults[self.j][0], 1, "fault"))
            b = broker.next_due()
            if b is not None and b <= t:
                cand.append((b, 2, "broker"))
            d = trader.next_due()
            if d is not None and d <= t:
                cand.append((d, 3, "trader"))
            if self.next_tick < t:
                cand.append((self.next_tick, 4, "tick"))
            if not cand:
                break
            when, _, what = min(cand)
            if self.clock is not None and when < self.clock:
                when = self.clock                   # 재시작 복구로 이미 지난 청산 예정 시각 등
            self.clock = when
            if what == "decide":
                self.next_dec = next(self._dec, None)
                trader.tick(when)
                if trader.wants_signal():
                    t0 = time.perf_counter()
                    sig = self.decide_fn(when)
                    self.decide_sec.append(time.perf_counter() - t0)
                    trader.on_signal(when, sig)
                elif trader.phase.value == "FLAT":
                    trader._log("DECISION_SKIPPED", trader.mode.value)
            elif what == "fault":
                _, name, fn = self.faults[self.j]
                self.j += 1
                trader.now = when
                trader._log("FAULT", name)
                fn(trader, broker)
            elif what == "broker":
                broker.process_due(when)
            elif what == "trader":
                trader.tick(when)
            else:
                trader.tick(when)
                self.next_tick = when + pd.Timedelta(seconds=1)
            self.deliver()
        self.clock = max(self.clock, t) if self.clock is not None else t

    def on_quote(self, ev) -> None:
        self.broker.on_quote(ev)
        self.trader.on_market(ev)
        self.deliver()
