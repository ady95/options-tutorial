"""기록된 호가로 Trader + PaperBroker를 돌리는 이벤트 스케줄러(Replay Mode).

메시지 하나(시각 t)를 반영하기 전에, t 이전에 일어났어야 할 일을 시각 순서대로 모두 처리한다.
    결정 시각(매분) · 브로커의 주문 도착/만료 · 청산 예정 시각 · 1초마다 점검(tick) · 장애 주입
9장 백테스트 엔진과 같은 순서 규칙이므로, 장애와 위험 한도가 없으면 같은 거래가 나와야 한다.
11장 실시간 모의매매는 같은 Trader·PaperBroker를 실시간 수신기에 연결한다.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from optlab.execution.paper import PaperBroker
from optlab.execution.trader import Trader


def run(quotes: pd.DataFrame, decide_fn: Callable, trader: Trader, broker: PaperBroker,
        start: pd.Timestamp, end: pd.Timestamp,
        faults: list[tuple[pd.Timestamp, str, Callable]] | None = None) -> Trader:
    q = quotes.sort_values("received_time", kind="stable")
    data_end = q["received_time"].iloc[-1]
    hold = pd.Timedelta(minutes=trader.cfg.horizon_min)
    minutes = pd.Series(q["received_time"].dt.ceil("min").unique())
    dec = [t for t in minutes if start <= t <= end and t + hold + broker.lat <= data_end]
    faults = sorted(faults or [], key=lambda f: f[0])
    k = j = 0
    next_tick = q["received_time"].iloc[0].ceil("s")

    def deliver():
        for ev in broker.drain():
            trader.on_broker(ev)

    for ev in q.itertuples(index=False):
        t = ev.received_time
        while True:
            cand = []
            if k < len(dec) and dec[k] < t:
                cand.append((dec[k], 0, "decide"))
            if j < len(faults) and faults[j][0] <= t:
                cand.append((faults[j][0], 1, "fault"))
            b = broker.next_due()
            if b is not None and b <= t:
                cand.append((b, 2, "broker"))
            d = trader.next_due()
            if d is not None and d <= t:
                cand.append((d, 3, "trader"))
            if next_tick < t:
                cand.append((next_tick, 4, "tick"))
            if not cand:
                break
            when, _, what = min(cand)
            if what == "decide":
                k += 1
                trader.tick(when)
                if trader.wants_signal():
                    trader.on_signal(when, decide_fn(when))
                elif trader.phase.value == "FLAT":
                    trader._log("DECISION_SKIPPED", trader.mode.value)
            elif what == "fault":
                _, name, fn = faults[j]
                j += 1
                trader.now = when
                trader._log("FAULT", name)
                fn(trader, broker)
            elif what == "broker":
                broker.process_due(when)
            elif what == "trader":
                trader.tick(when)
            else:
                trader.tick(when)
                next_tick = when + pd.Timedelta(seconds=1)
            deliver()
        broker.on_quote(ev)
        trader.on_market(ev)
        deliver()
    trader.finish(data_end)
    return trader
