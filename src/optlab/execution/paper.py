"""가상 브로커 — 실계좌 대신 주문을 받아 기록된(또는 실시간) 호가로 가상체결한다.

체결 규칙은 9장 가상체결 모델(backtest/fills.py)과 같다. 여기에 '증권사처럼 행동하는' 부분을 더했다.
- 주문마다 브로커 주문번호를 붙이고, 클라이언트 주문 ID(client_id)로도 찾을 수 있다.
- 유효시간(tif_sec)이 지나면 남은 수량을 스스로 취소한다(실제 증권사라면 취소 주문을 보내야 한다).
- 체결·종료 통지는 events 큐에 쌓이고, 연결이 끊긴 동안의 통지는 사라진다. 그래서 재연결 후
  open_orders()·positions()로 대조(reconcile)해야 한다.
- 장애 주입: disconnect(), fail_next("reject" | "lost_ack").
- state_path를 주면 상태가 바뀔 때마다 파일에 저장한다(11장). 증권사 서버가 주문 기록을 잃지 않는 것을
  흉내 낸 것으로, 매매 프로그램이 죽었다 다시 켜질 때 load()로 되살려 대조의 기준으로 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import json
import os
from pathlib import Path

import pandas as pd

from optlab.backtest.fills import Order, match


class BrokerDisconnected(ConnectionError):
    pass


class OrderRejected(RuntimeError):
    pass


class AckTimeout(TimeoutError):
    """주문은 접수됐을 수도, 안 됐을 수도 있다. 같은 주문을 다시 보내면 안 된다."""


@dataclass
class BrokerOrder:
    broker_id: str
    client_id: str
    order: Order
    status: str = "WORKING"      # WORKING | FILLED | CANCELLED
    reported: int = 0            # 통지한 체결 수


class PaperBroker:
    def __init__(self, latency_sec: float = 0.5, state_path: Path | None = None):
        self.state_path = state_path
        self.saved_at: pd.Timestamp | None = None
        self.lat = pd.Timedelta(seconds=latency_sec)
        self.book: dict[str, object] = {}
        self.orders: dict[str, BrokerOrder] = {}
        self.by_client: dict[str, BrokerOrder] = {}
        self.position: dict[str, int] = {}
        self.events: list[tuple] = []
        self.connected = True
        self._fail: str | None = None
        self._seq = 0

    # ── 장애 주입 ─────────────────────────────────────────
    def disconnect(self) -> None:
        self.connected = False

    def reconnect(self) -> None:
        self.connected = True

    def fail_next(self, kind: str) -> None:
        self._fail = kind

    # ── 주문 API ─────────────────────────────────────────
    def submit(self, client_id: str, side: str, code: str, qty: int, limit: float,
               now: pd.Timestamp, tif_sec: float, tag: str = "") -> str:
        if not self.connected:
            raise BrokerDisconnected("submit")
        if self._fail == "reject":
            self._fail = None
            raise OrderRejected(client_id)
        self._seq += 1
        bid = f"P{self._seq:05d}"
        o = Order(side, code, qty, limit, now, now + self.lat, now + self.lat + pd.Timedelta(seconds=tif_sec), tag)
        bo = BrokerOrder(bid, client_id, o)
        self.orders[bid] = bo
        self.by_client[client_id] = bo
        self._save(now)
        if self._fail == "lost_ack":      # 주문은 들어갔지만 응답이 오지 않음
            self._fail = None
            raise AckTimeout(client_id)
        return bid

    def cancel(self, broker_id: str, now: pd.Timestamp) -> None:
        if not self.connected:
            raise BrokerDisconnected("cancel")
        bo = self.orders[broker_id]
        if bo.status == "WORKING":
            self._finish(bo, "CANCELLED", now)

    def find(self, client_id: str) -> BrokerOrder | None:
        if not self.connected:
            raise BrokerDisconnected("find")
        return self.by_client.get(client_id)

    def open_orders(self) -> list[BrokerOrder]:
        if not self.connected:
            raise BrokerDisconnected("open_orders")
        return [bo for bo in self.orders.values() if bo.status == "WORKING"]

    def positions(self) -> dict[str, int]:
        if not self.connected:
            raise BrokerDisconnected("positions")
        return {k: v for k, v in self.position.items() if v}

    # ── 시간 진행 ─────────────────────────────────────────
    def next_due(self) -> pd.Timestamp | None:
        due = []
        for bo in self.orders.values():
            if bo.status == "WORKING":
                due.append(bo.order.expire if bo.order.arrived else bo.order.active)
        return min(due) if due else None

    def process_due(self, now: pd.Timestamp) -> None:
        """now에 도래한 도착·만료를 처리한다. 도착하면 그때 서 있던 호가와 먼저 맞춰 본다."""
        for bo in list(self.orders.values()):
            o = bo.order
            if bo.status != "WORKING":
                continue
            if not o.arrived and o.active <= now:
                o.arrived = True
                if o.code in self.book:
                    match(o, self.book[o.code], o.active)
                self._report(bo, o.active)
            elif o.arrived and o.expire <= now:
                self._finish(bo, "CANCELLED", o.expire)

    def on_quote(self, ev) -> None:
        self.book[ev.code] = ev
        for bo in list(self.orders.values()):
            if bo.status == "WORKING" and bo.order.arrived and bo.order.code == ev.code:
                match(bo.order, ev, ev.received_time)
                self._report(bo, ev.received_time)

    # ── 내부 ─────────────────────────────────────────────
    def _report(self, bo: BrokerOrder, now) -> None:
        o = bo.order
        for ts, q, px in o.fills[bo.reported:]:
            self.position[o.code] = self.position.get(o.code, 0) + (q if o.side == "BUY" else -q)
            self._emit(("FILL", bo.client_id, q, px, ts))
        changed = bo.reported < len(o.fills) or o.active == now
        bo.reported = len(o.fills)
        if o.remaining == 0:
            self._finish(bo, "FILLED", now)
        elif changed:
            self._save(now)

    def _finish(self, bo: BrokerOrder, status: str, now) -> None:
        bo.status = status
        self._emit(("DONE", bo.client_id, status, bo.order.filled, now))
        self._save(now)

    def _emit(self, ev: tuple) -> None:
        if self.connected:              # 끊긴 동안의 통지는 사라진다
            self.events.append(ev)

    def drain(self) -> list[tuple]:
        out, self.events = self.events, []
        return out

    # ── 상태 저장·복원(11장) ──────────────────────────────
    def _save(self, now) -> None:
        self.saved_at = now
        if self.state_path is None:
            return
        ts = lambda t: None if t is None else pd.Timestamp(t).isoformat()  # noqa: E731
        orders = [{"broker_id": bo.broker_id, "client_id": bo.client_id, "status": bo.status, "reported": bo.reported,
                   "side": o.side, "code": o.code, "qty": o.qty, "limit": o.limit, "tag": o.tag, "arrived": o.arrived,
                   "sent": ts(o.sent), "active": ts(o.active), "expire": ts(o.expire),
                   "fills": [[ts(t), q, px] for t, q, px in o.fills]}
                  for bo in self.orders.values() for o in [bo.order]]
        state = {"saved_at": ts(now), "seq": self._seq, "latency_sec": self.lat.total_seconds(),
                 "position": self.position, "orders": orders}
        tmp = Path(str(self.state_path) + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.state_path)          # 쓰다가 죽어도 이전 파일이 온전히 남는다

    @classmethod
    def load(cls, state_path: Path) -> "PaperBroker":
        st = json.loads(Path(state_path).read_text(encoding="utf-8"))
        b = cls(st["latency_sec"], state_path)
        b._seq, b.position = st["seq"], {k: int(v) for k, v in st["position"].items()}
        b.saved_at = pd.Timestamp(st["saved_at"]) if st["saved_at"] else None
        for d in st["orders"]:
            o = Order(d["side"], d["code"], d["qty"], d["limit"], pd.Timestamp(d["sent"]), pd.Timestamp(d["active"]),
                      pd.Timestamp(d["expire"]), d["tag"], d["arrived"],
                      [(pd.Timestamp(t), int(q), float(px)) for t, q, px in d["fills"]])
            bo = BrokerOrder(d["broker_id"], d["client_id"], o, d["status"], d["reported"])
            b.orders[bo.broker_id] = bo
            b.by_client[bo.client_id] = bo
        return b
