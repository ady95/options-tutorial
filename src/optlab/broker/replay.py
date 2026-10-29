"""Replay Mode — 합성 샘플을 실시간 스트림처럼 재생한다.

data_samples/generate_synthetic.py가 만든 CSV를 읽어, KIS 실시간 수신기(market_data.realtime.stream)와
같은 형태의 이벤트 dict를 시간 순서대로 내보낸다. 합성 데이터이며 실제 시장 데이터가 아니다.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import pandas as pd


def synthetic_code(cp: str, strike: float) -> str:
    """합성 데이터용 종목코드. 실제 KIS 코드와 겹치지 않게 'SYN' 접두어를 붙인다."""
    return f"SYN{cp}{strike:.1f}"


class ReplaySource:
    def __init__(self, data_dir: Path):
        self.fut = pd.read_csv(data_dir / "futures_1min.csv", parse_dates=["time"])
        self.opt = pd.read_csv(data_dir / "option_quotes_1min.csv", parse_dates=["time"])

    def times(self) -> list[pd.Timestamp]:
        return list(self.fut["time"])

    def chain_at(self, ts: pd.Timestamp) -> pd.DataFrame:
        return self.opt[self.opt["time"] == ts]

    async def stream(self, start: str | None = None, speed: float = 60.0,
                     strikes: set[float] | None = None) -> AsyncIterator[dict]:
        """1분 데이터를 speed배속으로 재생한다(speed=60이면 1분이 1초)."""
        fut = self.fut if start is None else self.fut[self.fut["time"] >= pd.Timestamp(start)]
        opt_by_time = dict(tuple(self.opt.groupby("time")))
        for row in fut.itertuples():
            now = datetime.now()
            yield {"type": "trade", "tr_id": "REPLAY_FUT", "code": "SYNFUT", "price": row.close,
                   "exchange_time": row.time.to_pydatetime(), "received_time": now}
            chain = opt_by_time.get(row.time)
            if chain is not None:
                for q in chain.itertuples():
                    if strikes is not None and q.strike not in strikes:
                        continue
                    yield {"type": "quote", "tr_id": "REPLAY_OPT", "code": synthetic_code(q.cp, q.strike),
                           "cp": q.cp, "strike": q.strike,
                           "asks": [q.ask], "bids": [q.bid], "ask_sizes": [q.ask_size], "bid_sizes": [q.bid_size],
                           "exchange_time": row.time.to_pydatetime(), "received_time": now}
            if speed > 0:
                await asyncio.sleep(60.0 / speed)
