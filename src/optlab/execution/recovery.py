"""재시작 복구 — 프로그램이 죽었다 다시 켜졌을 때 Trader의 상태를 되살린다.

원칙: 수량과 주문 결과는 브로커가 진실이고, 감사 기록은 '왜 그 주문을 냈는가'(신호 시각·진입 의도)를 알려 준다.
1. 당일 정지 상태(RISK_STOP, KILLED)와 확정 손익은 감사 기록에서 되살린다(같은 날 풀리지 않게).
2. 주문은 브로커의 주문 기록으로 다시 만든다. 같은 client_id를 다시 보내지 않게 된다(멱등성).
3. 포지션은 브로커 체결로 계산하고, 마지막 진입 의도(INTENT)에서 신호 정보를 붙인다.
   의도 기록이 없는 보유분(ORPHAN)은 출처를 모르므로 바로 청산 대상으로 둔다.
복구는 상태기계 밖의 일이므로 단계(Phase)를 직접 정한다(10-2절의 전이 표를 거치지 않는다).
"""
from __future__ import annotations

import pandas as pd

from optlab.execution.paper import PaperBroker
from optlab.execution.trader import LocalOrder, Mode, Phase, Trader


def _time(v):
    return pd.Timestamp(v) if isinstance(v, str) else v


def restore(trader: Trader, broker: PaperBroker, records: list[dict], now: pd.Timestamp) -> list[str]:
    """records: 감사 기록(같은 날). 무엇을 되살렸는지 설명 목록을 돌려준다."""
    trader.now = now
    notes = []
    # 1. 당일 정지 상태와 확정 손익
    modes = {r.get("mode") for r in records if r["kind"] == "MODE"}
    if Mode.RISK_STOP.value in modes:
        trader.flags["risk_stop"] = True
        notes.append("RISK_STOP 유지")
    if Mode.KILLED.value in modes:
        trader.kill.engage("RESTORED")
        notes.append("KILLED 유지")
    for r in records:
        if r["kind"] == "TRADE":
            t = dict(r["trade"])
            t["signal_time"] = _time(t["signal_time"])
            trader.trades.append(t)
            trader.realized_krw += t["pnl_krw"]
    closed = {t["signal_time"] for t in trader.trades}
    if trader.trades:
        notes.append(f"확정 거래 {len(trader.trades)}건 {trader.realized_krw:+,.0f}원")

    # 2. 주문: 브로커 기록이 진실
    for bo in broker.orders.values():
        o = bo.order
        lo = LocalOrder(bo.client_id, o.tag, o.side, o.code, o.qty, o.limit, o.sent, bo.status, list(o.fills))
        trader.orders[bo.client_id] = lo
        trader.order_times.append(o.sent)
    working = [lo for lo in trader.orders.values() if lo.state == "WORKING"]

    # 3. 포지션: 수량은 브로커, 설명은 마지막 진입 의도
    intents = [r for r in records if r["kind"] == "INTENT"]
    open_intent = next((r for r in reversed(intents) if _time(r["pos"]["signal_time"]) not in closed), None)
    held = broker.positions()
    trader.pos, trader.phase = None, Phase.FLAT
    if open_intent and open_intent["client_id"] in trader.orders:
        cid = open_intent["client_id"]
        entry = trader.orders[cid]
        pos = dict(open_intent["pos"])
        pos["signal_time"] = _time(pos["signal_time"])
        exits = [lo for k, lo in trader.orders.items() if k.startswith(cid[:-1] + "S")]
        pos.update(exit_fills=[f for lo in exits if lo.state != "WORKING" for f in lo.fills],
                   n_exit_orders=len(exits))
        trader.pos = pos
        if entry.state == "WORKING":
            trader.phase = Phase.ORDER_SUBMITTED
            notes.append(f"진입 주문 대기 중 {cid}")
        elif entry.filled == 0:
            trader.pos = None
            notes.append(f"진입 미체결 {cid}")
        else:
            last = entry.fills[-1][0]
            pos.update(qty=entry.filled, entry_px=sum(q * p for _, q, p in entry.fills) / entry.filled,
                       entry_fill=last, exit_due=last + pd.Timedelta(minutes=trader.cfg.horizon_min))
            if exits:                            # 청산 사유는 감사 기록의 마지막 EXIT_ORDER 전이에서
                after = records[records.index(open_intent):]
                why = [r["detail"].split()[0] for r in after
                       if r["kind"] == "PHASE" and r.get("phase") == "EXIT_ORDER" and r.get("detail")]
                pos.setdefault("exit_reason", why[-1] if why else "RESTORED")
                pos.setdefault("exit_bid", exits[0].limit)
            left = pos["qty"] - sum(q for _, q, _ in pos["exit_fills"])
            if any(lo.state == "WORKING" for lo in exits):
                trader.phase = Phase.EXIT_ORDER
                notes.append(f"보유 {pos['qty']}계약, 청산 주문 대기 중")
            elif left > 0:
                trader.phase = Phase.POSITION_MONITORED
                notes.append(f"보유 {left}계약 @{pos['entry_px']:.2f}, 청산 예정 {pos['exit_due']:%H:%M:%S}")
            else:                                # 꺼져 있던 사이 청산까지 끝났다
                trader.phase = Phase.EXIT_ORDER
                trader._close(now, "OK")
                notes.append("꺼져 있던 사이 청산 완료 → 거래 확정")
    # 의도 없이 브로커에만 있는 보유분
    mine = trader.pos["code"] if trader.pos else None
    for code, q in held.items():
        if code != mine and q > 0:
            buys = [f for lo in trader.orders.values() if lo.code == code and lo.side == "BUY" for f in lo.fills]
            px = sum(qq * p for _, qq, p in buys) / sum(qq for _, qq, _ in buys) if buys else 0.0
            trader.pos = {"signal_time": now, "code": code, "cp": "?", "strike": float("nan"), "sig_ask": px,
                          "delta": None, "exp_pnl": None, "exit_fills": [], "n_exit_orders": 0,
                          "qty": q, "entry_px": px, "entry_fill": now, "exit_due": now}
            trader.phase = Phase.POSITION_MONITORED
            notes.append(f"ORPHAN {code} {q}계약 → 즉시 청산 대상")
    local = trader.pos["qty"] - sum(q for _, q, _ in trader.pos["exit_fills"]) if trader.pos and "qty" in trader.pos else 0
    trader._log("RESTORE", f"브로커 {held or '{}'} / 복원 {local}계약, 대기 주문 {len(working)}건; " + "; ".join(notes))
    return notes


def catch_up(broker: PaperBroker, quotes: pd.DataFrame, until: pd.Timestamp) -> int:
    """꺼져 있던 동안에도 거래소는 돈다: 마지막 저장 이후 ~ 재시작까지의 호가를 브로커에만 흘린다.
    그사이 체결·종료 통지는 받을 프로그램이 없었으므로 버린다. 버린 통지 수를 돌려준다.
    Replay에서만 할 수 있다. Live의 가상 브로커는 프로그램과 함께 멈추므로 꺼져 있던 동안의 체결을 모른다
    (실제 증권사라면 그동안에도 체결된다. 그래서 재시작 후 대조가 필수다)."""
    since = broker.saved_at
    if since is None:                                # 주문을 낸 적이 없다
        return 0
    before = quotes[quotes["received_time"] <= since]
    for q in before.sort_values("received_time").groupby("code").tail(1).itertuples(index=False):
        broker.book[q.code] = q                      # 저장 시점에 서 있던 호가
    gap = quotes[(quotes["received_time"] > since) & (quotes["received_time"] <= until)]
    for q in gap.sort_values("received_time", kind="stable").itertuples(index=False):
        while (b := broker.next_due()) is not None and b <= q.received_time:
            broker.process_due(b)
        broker.on_quote(q)
    while (b := broker.next_due()) is not None and b <= until:
        broker.process_due(b)
    return len(broker.drain())
