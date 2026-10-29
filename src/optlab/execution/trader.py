"""주문 상태기계 — 신호를 받아 위험 검사 → 주문 → 보유 → 청산까지 한 포지션의 생애를 관리한다.

상태는 두 층으로 나눈다.
- 운영 상태(Mode): 지금 새 주문을 내도 되는가. KILLED > RISK_STOP > BROKER_DISCONNECTED > DATA_STALE > DATA_READY
- 포지션 단계(Phase): FLAT → SIGNAL_READY → RISK_APPROVED → ORDER_SUBMITTED → POSITION_MONITORED → EXIT_ORDER → FLAT
주문 하나하나의 상태(WORKING, FILLED, CANCELLED, REJECTED, UNKNOWN)는 LocalOrder가 가진다.

신규 진입은 DATA_READY에서만 한다. 청산은 어떤 상태에서도 시도하되, 시세를 믿을 수 없거나(DATA_STALE)
브로커와 끊겼으면 기다린다. 결과를 모르는 주문(UNKNOWN)이 있으면 확인될 때까지 어떤 주문도 새로 내지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd

from optlab.execution.paper import AckTimeout, BrokerDisconnected, OrderRejected
from optlab.market_data.spec import MULTIPLIER_KRW, round_to_tick, tick_size
from optlab.risk.health import HealthMonitor, KillSwitch
from optlab.risk.limits import Exposure, Intent, RiskConfig, check_entry, loss_breached


class Mode(str, Enum):
    DATA_READY = "DATA_READY"
    DATA_STALE = "DATA_STALE"
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    RISK_STOP = "RISK_STOP"
    KILLED = "KILLED"


class Phase(str, Enum):
    FLAT = "FLAT"
    SIGNAL_READY = "SIGNAL_READY"
    RISK_APPROVED = "RISK_APPROVED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    POSITION_MONITORED = "POSITION_MONITORED"
    EXIT_ORDER = "EXIT_ORDER"


ALLOWED = {
    Phase.FLAT: {Phase.SIGNAL_READY},
    Phase.SIGNAL_READY: {Phase.RISK_APPROVED, Phase.FLAT},          # 위험 검사 거절이면 FLAT
    Phase.RISK_APPROVED: {Phase.ORDER_SUBMITTED, Phase.FLAT},       # 주문 거부·연결 끊김이면 FLAT
    Phase.ORDER_SUBMITTED: {Phase.POSITION_MONITORED, Phase.FLAT},  # 하나도 체결 안 되면 FLAT
    Phase.POSITION_MONITORED: {Phase.EXIT_ORDER},
    Phase.EXIT_ORDER: {Phase.FLAT, Phase.POSITION_MONITORED},       # 다 못 팔면 다시 보유 감시
}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class TraderConfig:
    qty: int = 1                 # 신호당 계약 수
    horizon_min: int = 15
    entry_timeout_sec: float = 5.0
    exit_timeout_sec: float = 5.0
    entry_ticks: int = 0
    fee_rate: float = 0.001


@dataclass
class LocalOrder:
    client_id: str
    tag: str                     # ENTRY | EXIT
    side: str
    code: str
    qty: int
    limit: float
    sent: pd.Timestamp
    state: str = "SUBMITTING"    # SUBMITTING | WORKING | FILLED | CANCELLED | REJECTED | UNKNOWN
    fills: list[tuple] = field(default_factory=list)   # (시각, 수량, 가격)

    @property
    def filled(self) -> int:
        return sum(q for _, q, _ in self.fills)


class Trader:
    def __init__(self, risk: RiskConfig, tcfg: TraderConfig, broker, heartbeat_code: str,
                 expiry: date | None = None, kill: KillSwitch | None = None):
        self.risk, self.cfg, self.broker = risk, tcfg, broker
        self.health = HealthMonitor(risk, heartbeat_code)
        self.kill = kill or KillSwitch(None)
        self.expiry = expiry
        self.phase = Phase.FLAT
        self.flags = {"stale": True, "broker_down": False, "risk_stop": False}
        self.mode = Mode.DATA_STALE
        self.book: dict[str, object] = {}
        self.orders: dict[str, LocalOrder] = {}
        self.pos: dict | None = None
        self.trades: list[dict] = []
        self.log: list[dict] = []
        self.realized_krw = 0.0
        self.rejects = 0
        self.order_times: list[pd.Timestamp] = []
        self.now: pd.Timestamp | None = None

    # ── 기록 ─────────────────────────────────────────────
    def _log(self, kind: str, detail: str = "") -> None:
        self.log.append({"time": self.now, "kind": kind, "mode": self.mode.value,
                         "phase": self.phase.value, "detail": detail})

    def _go(self, to: Phase, why: str = "") -> None:
        if to not in ALLOWED[self.phase]:
            raise IllegalTransition(f"{self.phase.value} -> {to.value}")
        self.phase = to
        self._log("PHASE", why)

    def _update_mode(self) -> None:
        m = (Mode.KILLED if self.kill.engaged else Mode.RISK_STOP if self.flags["risk_stop"]
             else Mode.BROKER_DISCONNECTED if self.flags["broker_down"]
             else Mode.DATA_STALE if self.flags["stale"] else Mode.DATA_READY)
        if m != self.mode:
            self.mode = m
            self._log("MODE", self.health.reason if m == Mode.DATA_STALE else self.kill.reason if m == Mode.KILLED else "")

    # ── 상태 조회 ─────────────────────────────────────────
    def unknown_orders(self) -> list[LocalOrder]:
        return [o for o in self.orders.values() if o.state in ("UNKNOWN", "SUBMITTING")]

    def working(self, tag: str | None = None) -> list[LocalOrder]:
        return [o for o in self.orders.values() if o.state == "WORKING" and (tag is None or o.tag == tag)]

    def exposure(self) -> Exposure:
        qty = self.pos["qty"] if self.pos and "entry_px" in self.pos else 0
        pending = sum(o.qty - o.filled for o in self.working("ENTRY"))
        unreal = 0.0
        if qty:
            b = self.book.get(self.pos["code"])
            bid = b.bid1 if b is not None else self.pos["entry_px"]
            unreal = (bid - self.pos["entry_px"]) * qty * MULTIPLIER_KRW
        delta = (self.pos.get("delta") or 0.0) * qty if self.pos else 0.0
        return Exposure(open_qty=qty, pending_qty=pending, net_delta=delta,
                        day_pnl_krw=self.realized_krw + unreal, recent_orders=list(self.order_times))

    def wants_signal(self) -> bool:
        return self.phase == Phase.FLAT and self.mode == Mode.DATA_READY and not self.unknown_orders()

    def next_due(self) -> pd.Timestamp | None:
        if self.phase == Phase.POSITION_MONITORED and self.pos and not self.pos.get("exit_waiting"):
            return self.pos["exit_due"]          # 청산을 기다리는 중이면 1초 점검(tick)이 다시 시도한다
        return None

    # ── 입력 ─────────────────────────────────────────────
    def on_market(self, ev) -> None:
        self.now = ev.received_time
        self.book[ev.code] = ev
        self.health.on_message(ev.code, getattr(ev, "exchange_time", None), ev.received_time)

    def on_signal(self, ts: pd.Timestamp, sig) -> None:
        """결정 시각에 시그널을 받는다. 매수 신호면 위험 검사 후 주문."""
        self.now = ts
        if sig is None or sig.action not in ("CALL_BUY", "PUT_BUY"):
            return
        if not self.wants_signal():
            self._log("SIGNAL_IGNORED", f"{sig.action} {sig.code} ({self.mode.value})")
            return
        self._go(Phase.SIGNAL_READY, f"{sig.action} {sig.cp}{sig.strike:g} @{sig.entry}")
        limit = round_to_tick(sig.entry + self.cfg.entry_ticks * tick_size(sig.entry), "up")
        intent = Intent(now=ts.to_pydatetime(), code=sig.code, qty=self.cfg.qty, limit=limit, delta=sig.delta,
                        expiry_day=self.expiry is not None and ts.date() == self.expiry)
        reasons = check_entry(intent, self.exposure(), self.risk)
        if reasons:
            self._go(Phase.FLAT, "RISK_REJECT " + ",".join(reasons))
            return
        self._go(Phase.RISK_APPROVED)
        self.pos = {"signal_time": ts, "code": sig.code, "cp": sig.cp, "strike": sig.strike, "sig_ask": sig.entry,
                    "delta": sig.delta, "exp_pnl": sig.exp_pnl, "exit_fills": [], "n_exit_orders": 0}
        cid = f"{ts:%Y%m%d-%H%M%S}-{sig.code}-B"
        if self._send(cid, "ENTRY", "BUY", sig.code, self.cfg.qty, limit, self.cfg.entry_timeout_sec):
            self._go(Phase.ORDER_SUBMITTED, cid)
        else:
            self.pos = None
            self._go(Phase.FLAT, "SEND_FAILED")

    def on_broker(self, ev: tuple) -> None:
        kind, cid = ev[0], ev[1]
        self.now = max(self.now, ev[-1]) if self.now is not None else ev[-1]   # 통지가 생긴 시각
        lo = self.orders.get(cid)
        if lo is None:
            self._log("UNKNOWN_EVENT", f"{kind} {cid}")
            return
        if lo.state in ("UNKNOWN", "SUBMITTING"):
            lo.state = "WORKING"                    # 응답은 못 받았지만 통지로 존재가 확인됨
            self._log("ADOPTED", cid)
        if kind == "FILL":
            _, _, q, px, ts = ev
            lo.fills.append((ts, q, px))
        elif kind == "DONE":
            _, _, status, filled, ts = ev
            if lo.filled != filled:                 # 통지를 놓친 체결이 있다 → 브로커 기록으로 맞춘다
                self._sync_fills(lo)
            self._done(lo, status, ts)

    # ── 주기 점검 ─────────────────────────────────────────
    def tick(self, now: pd.Timestamp) -> None:
        self.now = now
        if self.kill.poll() and self.mode != Mode.KILLED:
            self._update_mode()
            self._stop_all("KILL_SWITCH")
        self.flags["stale"] = not self.health.check(now.to_pydatetime())
        if not self.broker.connected and not self.flags["broker_down"]:
            self.flags["broker_down"] = True        # 체결 통지 연결이 끊김. 그사이 통지는 잃는다
            self._update_mode()
            self._log("BROKER_DOWN", "connection lost")
        if self.flags["broker_down"]:
            self._reconcile()
        for lo in self.unknown_orders():
            self._resolve_unknown(lo)
        if not self.flags["risk_stop"] and loss_breached(self.exposure(), self.risk):
            self.flags["risk_stop"] = True
            self._update_mode()
            self._stop_all(f"DAILY_LOSS {self.exposure().day_pnl_krw:,.0f}")
        prev = self.mode
        self._update_mode()
        if self.mode == Mode.DATA_STALE and prev == Mode.DATA_READY:
            for lo in self.working("ENTRY"):        # 시세를 못 믿으면 진입 대기 주문부터 거둔다
                self._cancel(lo)
        self._check_exit(now)

    def _check_exit(self, now) -> None:
        if self.phase != Phase.POSITION_MONITORED or not self.pos:
            return
        why = ("FORCE_" + self.mode.value if self.mode in (Mode.KILLED, Mode.RISK_STOP)
               else "FLATTEN_BY" if now.time() >= self.risk.flatten_by
               else "HORIZON" if now >= self.pos["exit_due"] else None)
        if why is None:
            return
        if self.flags["stale"] or self.flags["broker_down"] or self.unknown_orders():
            if not self.pos.get("exit_waiting"):
                self.pos["exit_waiting"] = True
                self._log("EXIT_WAIT", f"{why}, {self.mode.value}")
            return
        self.pos["exit_waiting"] = False
        self.pos.setdefault("exit_reason", why)
        self._send_exit(now)

    def _send_exit(self, now) -> None:
        left = self.pos["qty"] - sum(q for _, q, _ in self.pos["exit_fills"])
        b = self.book[self.pos["code"]]
        if "exit_bid" not in self.pos:
            self.pos.update(exit_bid=b.bid1, exit_mid=(b.bid1 + b.ask1) / 2, exit_sent=now)
        self.pos["n_exit_orders"] += 1
        cid = f"{self.pos['signal_time']:%Y%m%d-%H%M%S}-{self.pos['code']}-S{self.pos['n_exit_orders']}"
        self._go(Phase.EXIT_ORDER, f"{self.pos['exit_reason']} {left}계약 @{b.bid1}")
        if not self._send(cid, "EXIT", "SELL", self.pos["code"], left, b.bid1, self.cfg.exit_timeout_sec):
            self._go(Phase.POSITION_MONITORED, "EXIT_SEND_FAILED")
            self.pos["exit_waiting"] = True

    # ── 주문 송신·취소·대조 ───────────────────────────────
    def _send(self, cid, tag, side, code, qty, limit, tif) -> bool:
        if cid in self.orders:                       # 멱등성: 같은 의도는 한 번만 주문
            self._log("DUPLICATE_BLOCKED", cid)
            return False
        lo = LocalOrder(cid, tag, side, code, qty, limit, self.now)
        self.orders[cid] = lo
        self.order_times.append(self.now)
        try:
            self.broker.submit(cid, side, code, qty, limit, self.now, tif, tag)
            lo.state = "WORKING"
            self.rejects = 0
            self._log("ORDER", f"{cid} {side} {qty}@{limit}")
            return True
        except AckTimeout:
            lo.state = "UNKNOWN"                     # 다시 보내지 않는다. tick에서 확인한다
            self._log("ACK_TIMEOUT", cid)
            return True
        except BrokerDisconnected:
            lo.state = "REJECTED"
            self.flags["broker_down"] = True
            self._update_mode()
            self._log("BROKER_DOWN", f"submit {cid}")
            return False
        except OrderRejected:
            lo.state = "REJECTED"
            self.rejects += 1
            self._log("REJECTED", f"{cid} 연속 {self.rejects}")
            if self.rejects >= self.risk.max_rejects:
                self.flags["risk_stop"] = True
                self._update_mode()
            return False

    def _cancel(self, lo: LocalOrder) -> None:
        try:
            bo = self.broker.find(lo.client_id)
            if bo is not None:
                self.broker.cancel(bo.broker_id, self.now)
                self._log("CANCEL", lo.client_id)
        except BrokerDisconnected:
            self.flags["broker_down"] = True
            self._update_mode()

    def _resolve_unknown(self, lo: LocalOrder) -> None:
        try:
            bo = self.broker.find(lo.client_id)
        except BrokerDisconnected:
            return
        if bo is None:
            self._log("NOT_PLACED", lo.client_id)
            self._done(lo, "REJECTED", self.now)
        else:
            lo.state = "WORKING"
            self._log("FOUND", f"{lo.client_id} -> {bo.broker_id}")
            self._sync_fills(lo)
            if bo.status != "WORKING":
                self._done(lo, bo.status, self.now)

    def _sync_fills(self, lo: LocalOrder) -> None:
        bo = self.broker.find(lo.client_id)
        if bo is not None and len(bo.order.fills) > len(lo.fills):
            missed = bo.order.fills[len(lo.fills):]
            lo.fills.extend(missed)
            self._log("MISSED_FILLS", f"{lo.client_id} {sum(q for _, q, _ in missed)}계약")

    def _reconcile(self) -> None:
        """재연결 시도. 성공하면 브로커의 주문·포지션 기록을 진실로 삼아 로컬 상태를 맞춘다."""
        try:
            broker_pos = self.broker.positions()
        except BrokerDisconnected:
            return
        self.flags["broker_down"] = False
        self._update_mode()
        self._log("RECONNECTED")
        local = self.pos["qty"] - sum(q for _, q, _ in self.pos["exit_fills"]) if self.pos and "qty" in self.pos else 0
        remote = sum(broker_pos.values())
        self._log("RECONCILE", f"브로커 {broker_pos or '{}'} / 로컬 {local}계약" + ("" if local == remote else " 불일치"))
        for lo in list(self.orders.values()):         # 끊긴 동안 놓친 체결·종료 통지를 브로커 기록으로 채운다
            if lo.state in ("UNKNOWN", "SUBMITTING"):
                self._resolve_unknown(lo)
            elif lo.state == "WORKING":
                self._sync_and_close(lo)

    def _sync_and_close(self, lo: LocalOrder) -> None:
        self._sync_fills(lo)
        bo = self.broker.find(lo.client_id)
        if bo is not None and bo.status != "WORKING":
            self._done(lo, bo.status, self.now)

    def _stop_all(self, why: str) -> None:
        self._log("STOP", why)
        for lo in self.working("ENTRY"):
            self._cancel(lo)

    # ── 주문 종료 처리 ────────────────────────────────────
    def _done(self, lo: LocalOrder, status: str, ts) -> None:
        if lo.state in ("FILLED", "CANCELLED", "REJECTED"):
            return
        lo.state = status
        if lo.tag == "ENTRY":
            if lo.filled:
                last = lo.fills[-1][0]
                self.pos.update(qty=lo.filled, entry_px=sum(q * p for _, q, p in lo.fills) / lo.filled,
                                entry_fill=last, exit_due=last + pd.Timedelta(minutes=self.cfg.horizon_min))
                self._go(Phase.POSITION_MONITORED, f"{lo.filled}계약 @{self.pos['entry_px']:.2f}")
            else:
                self.pos = None
                self._go(Phase.FLAT, f"ENTRY_{status}")
            return
        self.pos["exit_fills"] += lo.fills
        left = self.pos["qty"] - sum(q for _, q, _ in self.pos["exit_fills"])
        if left > 0:
            self._go(Phase.POSITION_MONITORED, f"{left}계약 남음")
            self._check_exit(ts)                     # 새 매수호가로 다시 낸다(따라가기)
        else:
            self._close(ts, "OK")

    def _close(self, ts, status: str) -> None:
        p = self.pos
        qty = p["qty"]
        exit_px = sum(q * px for _, q, px in p["exit_fills"]) / qty
        fees = self.cfg.fee_rate * (p["entry_px"] + exit_px) * qty
        pnl = (exit_px - p["entry_px"]) * qty - fees
        self.realized_krw += pnl * MULTIPLIER_KRW
        self.trades.append({
            "signal_time": p["signal_time"], "code": p["code"], "cp": p["cp"], "strike": p["strike"], "qty": qty,
            "sig_ask": p["sig_ask"], "entry_px": p["entry_px"], "exit_bid": p.get("exit_bid"), "exit_px": exit_px,
            "n_exit_orders": p["n_exit_orders"], "exit_reason": p.get("exit_reason", status), "fees": fees,
            "pnl_pt": pnl, "pnl_krw": pnl * MULTIPLIER_KRW, "status": status})
        self.pos = None
        if self.phase == Phase.EXIT_ORDER:
            self._go(Phase.FLAT, f"{pnl * MULTIPLIER_KRW:+,.0f}원")

    def finish(self, now: pd.Timestamp) -> None:
        """데이터 끝: 못 판 수량은 마지막 매수호가로 평가한다."""
        self.now = now
        if self.pos and "qty" in self.pos:
            for lo in self.working("EXIT"):
                self.pos["exit_fills"] += lo.fills
                lo.state = "CANCELLED"
            left = self.pos["qty"] - sum(q for _, q, _ in self.pos["exit_fills"])
            if left > 0:
                self.pos["exit_fills"].append((now, left, self.book[self.pos["code"]].bid1))
            self.pos.setdefault("exit_reason", "DATA_END")
            self._close(now, "DATA_END")
            self.phase = Phase.FLAT                  # 실행 종료. 상태기계 밖의 정리
