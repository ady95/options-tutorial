"""실시간 시장 상태 — 수신한 호가·체결로 '지금 알 수 있는' 체인과 선물 분봉을 만든다.

- 체인: 종목별 마지막 호가. 8장 chain_at()과 같은 열(code, bid1, ask1, age, cp, strike)을 돌려준다.
- 분봉: 선물 체결로 직접 만든다. 거래량은 체결 수량의 합이 아니라 누적거래량(acml_vol)의 차이로 구한다.
  2026-10-29 기록에서 체결 수량 합은 REST 분봉과 39개 중 28개만 일치했고, 누적거래량 차이는 39개 모두 일치했다.
- 첫 분(수신을 시작한 분)은 앞부분이 빠져 있으므로 쓰지 않고, 그 분까지는 prefix_fn(REST 또는 기록 파일)으로 채운다.
"""
from __future__ import annotations

import json
from collections import namedtuple
from typing import Callable

import pandas as pd

Quote = namedtuple("Quote", "code exchange_time received_time bid1 ask1 bid_size1 ask_size1 depth")
Trade = namedtuple("Trade", "code exchange_time received_time price acml_vol")


def _ts(t):
    return None if t is None else pd.Timestamp(t)


def quote_from_event(ev: dict) -> Quote:
    """realtime.parse_frame()의 호가 dict → Replay 기록과 같은 모양의 Quote."""
    return Quote(ev["code"], _ts(ev["exchange_time"]), _ts(ev["received_time"]), ev["bids"][0], ev["asks"][0],
                 ev["bid_sizes"][0], ev["ask_sizes"][0],
                 json.dumps([ev["bids"], ev["asks"], ev["bid_sizes"], ev["ask_sizes"]]))


def trade_from_event(ev: dict) -> Trade:
    return Trade(ev["code"], _ts(ev["exchange_time"]), _ts(ev["received_time"]), ev["price"], ev["acml_vol"])


class BarBuilder:
    """선물 체결 → 1분봉(시작 시각 표시, REST 분봉과 같은 규칙)."""

    def __init__(self, contract: str):
        self.contract = contract
        self.done: list[dict] = []          # 완성된 분봉
        self.cur: dict | None = None        # 만드는 중인 분봉
        self.first_minute: pd.Timestamp | None = None
        self.prev_acml: float | None = None  # 직전 분봉 끝의 누적거래량

    def on_trade(self, tr: Trade) -> None:
        if tr.exchange_time is None or not tr.price:
            return
        m = tr.exchange_time.floor("min")
        if self.first_minute is None:
            self.first_minute = m
        if self.cur is not None and m > self.cur["time"]:
            self._close()
        if self.cur is None:
            self.cur = {"time": m, "open": tr.price, "high": tr.price, "low": tr.price, "close": tr.price,
                        "acml": tr.acml_vol}
        else:
            c = self.cur
            c["high"], c["low"], c["close"], c["acml"] = max(c["high"], tr.price), min(c["low"], tr.price), tr.price, \
                tr.acml_vol

    def _close(self) -> None:
        c = self.cur
        if c["time"] > self.first_minute:   # 첫 분은 앞부분이 빠져 있다
            self.done.append({"time": c["time"], "open": c["open"], "high": c["high"], "low": c["low"],
                              "close": c["close"], "volume": c["acml"] - self.prev_acml, "contract": self.contract})
        self.prev_acml = c["acml"]
        self.cur = None

    def complete_until(self, last_minute: pd.Timestamp) -> bool:
        """last_minute 분봉이 끝났는가. 다음 분의 체결이 아직 안 왔으면 지금까지의 체결로 닫는다(False)."""
        if self.cur is not None and self.cur["time"] <= last_minute:
            self._close()
            return False
        return True


class MarketBook:
    def __init__(self, instruments: pd.DataFrame, futures_code: str, prefix_fn: Callable[[pd.Timestamp], pd.DataFrame],
                 keep_days: int = 2):
        opts = instruments.dropna(subset=["strike"])
        self.opts = opts.set_index("code")[["cp", "strike"]]
        self.fut = futures_code
        self.quotes: dict[str, Quote] = {}
        self.fut_quote = None                # 선물 최근 호가(ATM 추적용)
        self.bars = BarBuilder(futures_code)
        self.prefix_fn = prefix_fn
        self.prefix: pd.DataFrame | None = None
        self.keep_days = keep_days
        self.late_bars = 0                   # 결정 시각에 직전 분봉이 아직 안 닫혀 있던 횟수

    def on_quote(self, q) -> None:
        if q.code in self.opts.index:
            self.quotes[q.code] = q
        elif q.code == self.fut:
            self.fut_quote = q

    def futures_mid(self) -> float | None:
        q = self.fut_quote
        return (q.bid1 + q.ask1) / 2 if q is not None and q.bid1 > 0 and q.ask1 > 0 else None

    def drop(self, codes) -> None:
        """구독을 해제한 종목의 호가를 버린다(더는 갱신되지 않으므로)."""
        for c in codes:
            self.quotes.pop(c, None)

    def on_trade(self, tr) -> None:
        if tr.code == self.fut:
            self.bars.on_trade(tr)

    def chain(self, ts: pd.Timestamp) -> pd.DataFrame:
        rows = [(q.code, q.bid1, q.ask1, (ts - q.received_time).total_seconds())
                for q in self.quotes.values() if q.received_time <= ts]
        ch = pd.DataFrame(rows, columns=["code", "bid1", "ask1", "age"])
        return ch.merge(self.opts.reset_index(), on="code")

    def bars_until(self, ts: pd.Timestamp) -> pd.DataFrame:
        """ts에 완성된 분봉(마지막 = ts가 속한 분의 직전 분). 특징 계산에 필요한 최근 keep_days 거래일만."""
        last = ts.floor("min") - pd.Timedelta(minutes=1)
        if not self.bars.complete_until(last):
            self.late_bars += 1
        first = self.bars.first_minute
        if self.prefix is None:              # 첫 분봉까지는 REST(또는 기록 파일)로 채운다
            p = self.prefix_fn(last if first is None else min(first, last))
            days = sorted(p["time"].dt.date.unique())[-self.keep_days:]
            p = p[p["time"].dt.date.isin(days)]
            if first is None or first > last:   # 첫 분봉이 아직 끝나지 않았다. 다음에 다시 받는다
                return p.reset_index(drop=True)
            self.prefix = p
        built = pd.DataFrame(self.bars.done)
        out = pd.concat([self.prefix, built]) if len(built) else self.prefix
        return out[out["time"] <= last].reset_index(drop=True)
