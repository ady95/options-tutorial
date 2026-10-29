"""호가 기반 이벤트 백테스트.

기록된 호가 메시지를 수신 시각 순서대로 한 건씩 재생한다. 정해진 결정 시각(1분마다)이 오면
그때까지 도착한 호가만으로 시그널을 계산하고, 매수 신호면 지정가 주문을 낸다. 주문은 지연 뒤에
거래소에 도착하고, 그 뒤의 호가와 잔량에 따라 체결·부분체결·미체결이 갈린다(fills.py).

포지션은 한 번에 하나다. 진입이 체결되면 horizon_min 뒤에 그때의 매수호가로 매도 주문을 내고,
exit_timeout_sec 안에 다 팔리지 않으면 새 매수호가로 다시 낸다(따라가기). 데이터가 끝날 때까지
못 팔면 마지막 매수호가로 평가하고 DATA_END로 표시한다.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from optlab.backtest.fills import Order, match
from optlab.market_data.spec import MULTIPLIER_KRW, round_to_tick, tick_size


@dataclass
class BacktestConfig:
    qty: int = 1                    # 신호 하나당 계약 수
    latency_sec: float = 0.5        # 결정 → 거래소 도착(가정치). 4장 시세 지연 실측 0.4~0.9초 참고
    entry_timeout_sec: float = 5.0  # 진입 주문이 이 시간 안에 다 체결되지 않으면 남은 수량 취소
    entry_ticks: int = 0            # 매수 지정가 = 결정 시각 매도호가 + N틱(0이면 매도호가 그대로)
    exit_timeout_sec: float = 5.0   # 청산 주문 재호가 간격
    horizon_min: int = 15           # 보유 시간(분). 8장 시그널의 horizon과 같게 둔다
    fee_rate: float = 0.001         # 수수료(거래대금 대비, 한쪽) — 가정치
    multiplier: int = MULTIPLIER_KRW

    @classmethod
    def from_yaml(cls, path: Path) -> "BacktestConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in (raw.get("backtest") or {}).items() if k in names}
        st = raw.get("strategy") or {}
        for k in ("horizon_min", "fee_rate"):      # 시그널 엔진과 같은 값을 쓴다
            if k in st:
                kw.setdefault(k, st[k])
        return cls(**kw)


@dataclass
class Result:
    trades: pd.DataFrame        # 진입~청산 한 건이 한 행
    orders: pd.DataFrame        # 낸 주문 전부(미체결 포함)
    decisions: pd.DataFrame     # 결정 시각마다 무엇을 했나


def run(quotes: pd.DataFrame, decide_fn: Callable, cfg: BacktestConfig,
        start: pd.Timestamp, end: pd.Timestamp) -> Result:
    """quotes: code, received_time, bid1, ask1, bid_size1, ask_size1, (depth)
    decide_fn(ts) -> Signal(action, code, entry, ...) | None
    start~end: 결정 시각 범위. 청산이 데이터 끝을 넘을 진입은 하지 않는다."""
    q = quotes.sort_values("received_time", kind="stable")
    data_end = q["received_time"].iloc[-1]
    hold = pd.Timedelta(minutes=cfg.horizon_min)
    lat = pd.Timedelta(seconds=cfg.latency_sec)
    # 결정 시각: 호가가 들어온 분의 경계(장이 닫힌 밤·주말은 건너뛴다)
    minutes = pd.Series(q["received_time"].dt.ceil("min").unique())
    dec_times = [t for t in minutes if start <= t <= end and t + hold + lat <= data_end]
    book: dict[str, object] = {}
    orders: list[Order] = []
    decisions, trades = [], []
    order: Order | None = None
    pos: dict | None = None
    k = 0

    def send(side, code, qty, limit, at, timeout, tag):
        o = Order(side, code, qty, limit, at, at + lat, at + lat + pd.Timedelta(seconds=timeout), tag)
        orders.append(o)
        return o

    def on_done(o: Order, now):
        """주문이 끝났을 때(전량 체결 또는 만료) 포지션 상태를 바꾼다."""
        nonlocal pos
        if o.tag == "ENTRY":
            if o.filled:
                pos = {**pos, "qty": o.filled, "entry_px": o.avg_price, "entry_fill": o.fills[-1][0],
                       "exit_due": o.fills[-1][0] + hold, "exit_fills": []}
            else:
                pos = None
            return None
        pos["exit_fills"] += o.fills
        left = pos["qty"] - sum(x for _, x, _ in pos["exit_fills"])
        if left > 0:                                # 남은 수량은 새 매수호가로 다시 낸다
            bid = book[pos["code"]].bid1
            pos["n_exit_orders"] += 1
            return send("SELL", pos["code"], left, bid, now, cfg.exit_timeout_sec, "EXIT")
        trades.append(_close(pos, cfg, "OK"))
        pos = None
        return None

    for ev in q.itertuples(index=False):
        t = ev.received_time
        # 1) 이 메시지 전에 와 있어야 할 결정들(그때까지 받은 호가로만 판단)
        while k < len(dec_times) and dec_times[k] < t:
            ts = dec_times[k]
            k += 1
            if pos is not None or order is not None:
                decisions.append({"time": ts, "action": "BUSY"})
                continue
            sig = decide_fn(ts)
            act = getattr(sig, "action", "SKIP") if sig is not None else "SKIP"
            decisions.append({"time": ts, "action": act, "reason": getattr(sig, "reason", "")})
            if act in ("CALL_BUY", "PUT_BUY"):
                limit = round_to_tick(sig.entry + cfg.entry_ticks * tick_size(sig.entry), "up")
                b = book.get(sig.code)
                pos = {"signal_time": ts, "code": sig.code, "cp": sig.cp, "strike": sig.strike,
                       "sig_ask": sig.entry, "sig_mid": (b.bid1 + b.ask1) / 2 if b is not None else None,
                       "exp_pnl": sig.exp_pnl, "n_exit_orders": 1}
                order = send("BUY", sig.code, cfg.qty, limit, ts, cfg.entry_timeout_sec, "ENTRY")
        # 2) 이 메시지 전에 일어났어야 할 일을 모두 처리한다. 이때 book은 아직 이 메시지 이전 상태다
        while True:
            if order is not None and t >= order.active and not order.arrived:
                # 주문이 거래소에 도착하는 순간, 그때 서 있던 호가와 먼저 맞춰 본다
                order.arrived = True
                if order.code in book:
                    match(order, book[order.code], order.active)
                if order.remaining == 0:
                    order = on_done(order, order.active)
            elif order is not None and t > order.expire:
                order = on_done(order, order.expire)       # 만료: 남은 수량 취소(청산이면 재호가)
            elif pos is not None and order is None and "exit_due" in pos and t >= pos["exit_due"]:
                b = book[pos["code"]]                      # 청산 시각의 매수호가로 매도
                pos.update(exit_bid=b.bid1, exit_mid=(b.bid1 + b.ask1) / 2)
                order = send("SELL", pos["code"], pos["qty"], b.bid1, pos["exit_due"],
                             cfg.exit_timeout_sec, "EXIT")
            else:
                break
        # 3) 호가 갱신, 그리고 대기 중인 주문과 맞춰 보기
        book[ev.code] = ev
        if order is not None and order.arrived and ev.code == order.code:
            match(order, ev, t)
            if order.remaining == 0:
                order = on_done(order, t)

    if pos is not None and "exit_due" in pos:          # 데이터가 끝날 때까지 못 판 수량
        if order is not None and order.tag == "EXIT":
            pos["exit_fills"] += order.fills
        left = pos["qty"] - sum(x for _, x, _ in pos["exit_fills"])
        if left > 0:
            pos["exit_fills"].append((data_end, left, book[pos["code"]].bid1))
        pos.setdefault("exit_bid", book[pos["code"]].bid1)
        pos.setdefault("exit_mid", pos["exit_bid"])
        trades.append(_close(pos, cfg, "DATA_END"))

    return Result(pd.DataFrame(trades), _orders_frame(orders), pd.DataFrame(decisions))


def _close(pos: dict, cfg: BacktestConfig, status: str) -> dict:
    fills = pos["exit_fills"]
    qty = pos["qty"]
    exit_px = sum(x * p for _, x, p in fills) / qty
    fees = cfg.fee_rate * (pos["entry_px"] + exit_px) * qty
    pnl_pt = (exit_px - pos["entry_px"]) * qty - fees
    return {
        "signal_time": pos["signal_time"], "code": pos["code"], "cp": pos["cp"], "strike": pos["strike"],
        "qty": qty, "sig_mid": pos["sig_mid"], "sig_ask": pos["sig_ask"], "entry_px": pos["entry_px"],
        "entry_fill": pos["entry_fill"], "exit_due": pos["exit_due"], "exit_mid": pos["exit_mid"],
        "exit_bid": pos["exit_bid"], "exit_px": exit_px, "exit_done": fills[-1][0],
        "n_exit_orders": pos["n_exit_orders"], "fees": fees, "pnl_pt": pnl_pt,
        "pnl_krw": pnl_pt * cfg.multiplier, "exp_pnl": pos["exp_pnl"], "status": status,
    }


def _orders_frame(orders: list[Order]) -> pd.DataFrame:
    return pd.DataFrame([{
        "tag": o.tag, "side": o.side, "code": o.code, "qty": o.qty, "limit": o.limit, "sent": o.sent,
        "filled": o.filled, "avg_price": o.avg_price, "n_fills": len(o.fills),
        "first_fill_sec": (o.fills[0][0] - o.sent).total_seconds() if o.fills else None,
    } for o in orders])
