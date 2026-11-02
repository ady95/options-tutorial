"""여러 다리(leg) 주문의 가상체결 — 13장 스프레드.

09장 엔진은 포지션 하나·종목 하나를 다룬다. 스프레드는 두 종목을 함께 사고팔아야 하므로, 다리마다
09장과 같은 체결 규칙(fills.match: 지연 뒤 도착, 표시 잔량만큼, 대기열 체결 없음)으로 따로 맞춰 본다.

다리를 내는 방식:
- sequential  첫 다리(매수)가 다 체결된 뒤, 그 순간의 호가로 둘째 다리(매도)를 낸다
- together    두 다리를 같은 시각에 낸다
둘째 다리가 유효시간 안에 체결되지 않으면 이미 체결된 다리를 되판다(UNWIND). 체결된 다리 하나만
들고 있는 것은 스프레드가 아니라 맨 옵션 한 개(naked)이기 때문이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from optlab.backtest.engine import BacktestConfig
from optlab.backtest.fills import Order, match


class QuoteBook:
    """종목별 호가 시계열. at(code, t) = t 이전 마지막 호가."""

    def __init__(self, quotes: pd.DataFrame):
        q = quotes.sort_values("received_time", kind="stable")
        self.by = {c: g.reset_index(drop=True) for c, g in q.groupby("code")}
        self.ts = {c: g["received_time"].to_numpy() for c, g in self.by.items()}
        self.end = q["received_time"].iloc[-1]

    def at(self, code: str, t: pd.Timestamp):
        i = np.searchsorted(self.ts[code], np.datetime64(t), side="right") - 1
        return None if i < 0 else self.by[code].iloc[i]

    def after(self, code: str, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
        a = np.searchsorted(self.ts[code], np.datetime64(t0), side="right")
        b = np.searchsorted(self.ts[code], np.datetime64(t1), side="right")
        return self.by[code].iloc[a:b]


def fill(book: QuoteBook, side: str, code: str, qty: int, limit: float, sent: pd.Timestamp,
         latency: float, timeout: float, tag: str = "") -> Order:
    """지정가 주문 하나를 기록된 호가로 체결해 본다(09장 engine과 같은 규칙)."""
    lat = pd.Timedelta(seconds=latency)
    o = Order(side, code, qty, limit, sent, sent + lat, sent + lat + pd.Timedelta(seconds=timeout), tag)
    q = book.at(code, o.active)
    if q is not None:
        match(o, q, o.active)
    if o.remaining:
        for ev in book.after(code, o.active, o.expire).itertuples(index=False):
            match(o, ev, ev.received_time)
            if not o.remaining:
                break
    o.arrived = True
    return o


def done_time(o: Order) -> pd.Timestamp:
    return o.fills[-1][0] if o.fills and not o.remaining else o.expire


def close_out(book: QuoteBook, side: str, code: str, qty: int, t: pd.Timestamp, latency: float,
              timeout: float, max_tries: int = 20) -> tuple[float, pd.Timestamp, int]:
    """그 시각의 반대편 호가로 내고, 남으면 새 호가로 다시 낸다. (평균가, 끝난 시각, 주문 수)."""
    got, cost, tries = 0, 0.0, 0
    while got < qty and tries < max_tries and t < book.end:
        q = book.at(code, t)
        px = q.bid1 if side == "SELL" else q.ask1
        o = fill(book, side, code, qty - got, px, t, latency, timeout)
        got += o.filled
        cost += sum(x * p for _, x, p in o.fills)
        tries += 1
        t = done_time(o)
    if got < qty:                                   # 데이터 끝: 마지막 호가로 평가
        q = book.at(code, book.end)
        px = q.bid1 if side == "SELL" else q.ask1
        cost += (qty - got) * px
    return cost / qty, t, tries


def vertical(book: QuoteBook, long_code: str, short_code: str, t: pd.Timestamp, cfg: BacktestConfig,
             mode: str = "sequential") -> dict:
    """수직 스프레드 한 번: 진입 → horizon 뒤 청산. 단일 옵션은 short_code=None."""
    ql, qs = book.at(long_code, t), book.at(short_code, t) if short_code else None
    rec = {"time": t, "long": long_code, "short": short_code, "sig_long_ask": ql.ask1,
           "sig_short_bid": qs.bid1 if qs is not None else np.nan,
           "sig_debit_mid": (ql.bid1 + ql.ask1) / 2 - ((qs.bid1 + qs.ask1) / 2 if qs is not None else 0)}
    leg1 = fill(book, "BUY", long_code, cfg.qty, ql.ask1, t, cfg.latency_sec, cfg.entry_timeout_sec)
    if short_code is None:
        if leg1.remaining:
            return {**rec, "status": "NO_FILL"}
        leg2 = None
    else:
        if mode == "together":
            leg2 = fill(book, "SELL", short_code, cfg.qty, qs.bid1, t, cfg.latency_sec, cfg.entry_timeout_sec)
        elif leg1.remaining:
            return {**rec, "status": "NO_FILL"}
        else:
            t1 = done_time(leg1)
            b = book.at(short_code, t1)                 # 첫 다리 체결 순간의 매수호가
            rec["short_bid_at_leg1"] = b.bid1
            leg2 = fill(book, "SELL", short_code, cfg.qty, b.bid1, t1, cfg.latency_sec, cfg.entry_timeout_sec)
        if leg1.remaining or leg2.remaining:            # 한쪽만 체결: 체결된 쪽을 되돌린다
            pnl = 0.0
            for o, back in ((leg1, "SELL"), (leg2, "BUY")):
                if o.filled:
                    px, _, _ = close_out(book, back, o.code, o.filled, done_time(o), cfg.latency_sec, cfg.exit_timeout_sec)
                    sign = 1 if back == "SELL" else -1
                    pnl += sign * (px - o.avg_price) * o.filled - cfg.fee_rate * (px + o.avg_price) * o.filled
            status = "NO_FILL" if not (leg1.filled or leg2.filled) else "UNWIND"
            return {**rec, "status": status, "pnl_pt": pnl, "pnl_krw": pnl * cfg.multiplier}
    entry_done = max(done_time(leg1), done_time(leg2)) if leg2 else done_time(leg1)
    exit_at = entry_done + pd.Timedelta(minutes=cfg.horizon_min)
    if exit_at >= book.end:
        return {**rec, "status": "DATA_END"}
    lx, _, n1 = close_out(book, "SELL", long_code, cfg.qty, exit_at, cfg.latency_sec, cfg.exit_timeout_sec)
    entry_debit = leg1.avg_price - (leg2.avg_price if leg2 else 0.0)
    fees = cfg.fee_rate * (leg1.avg_price + lx)
    exit_credit = lx
    if leg2:
        sx, _, n2 = close_out(book, "BUY", short_code, cfg.qty, exit_at, cfg.latency_sec, cfg.exit_timeout_sec)
        exit_credit -= sx
        fees += cfg.fee_rate * (leg2.avg_price + sx)
    pnl = (exit_credit - entry_debit) - fees
    return {**rec, "status": "OK", "entry_debit": entry_debit, "exit_credit": exit_credit, "fees": fees,
            "leg_gap_sec": (done_time(leg2) - done_time(leg1)).total_seconds() if leg2 else 0.0,
            "pnl_pt": pnl, "pnl_krw": pnl * cfg.multiplier * cfg.qty}
