"""기록된 호가로 Trader + PaperBroker를 돌린다(Replay Mode).

순서 규칙은 Scheduler(scheduler.py)에 있다. 9장 백테스트 엔진과 같은 규칙이므로,
장애와 위험 한도가 없으면 같은 거래가 나와야 한다.
11장 실시간 모의매매는 같은 Scheduler를 asyncio 수신 루프에 연결한다(optlab.live.pipeline).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from optlab.execution.paper import PaperBroker
from optlab.execution.scheduler import Scheduler
from optlab.execution.trader import Trader


def run(quotes: pd.DataFrame, decide_fn: Callable, trader: Trader, broker: PaperBroker,
        start: pd.Timestamp, end: pd.Timestamp,
        faults: list[tuple[pd.Timestamp, str, Callable]] | None = None) -> Trader:
    q = quotes.sort_values("received_time", kind="stable")
    data_end = q["received_time"].iloc[-1]
    hold = pd.Timedelta(minutes=trader.cfg.horizon_min)
    minutes = pd.Series(q["received_time"].dt.ceil("min").unique())
    dec = [t for t in minutes if start <= t <= end and t + hold + broker.lat <= data_end]
    sched = Scheduler(trader, broker, decide_fn, dec, faults)
    for ev in q.itertuples(index=False):
        sched.advance(ev.received_time)
        sched.on_quote(ev)
    trader.finish(data_end)
    return trader
