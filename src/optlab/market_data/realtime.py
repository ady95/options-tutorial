"""KIS WebSocket 실시간 체결·호가 수신.

- 한 연결의 구독은 40건 이하로 제한한다(2026-10 실측: 42번째에서 OPSP0008 MAX SUBSCRIBE OVER).
- 연결이 끊기면 재접속 후 같은 목록을 다시 구독한다.
- 모든 이벤트에 거래소 시각(exchange_time)과 수신 시각(received_time)을 함께 기록한다.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from typing import AsyncIterator

import websockets

WS_URL = {"real": "ws://ops.koreainvestment.com:21000/tryitout",
          "paper": "ws://ops.koreainvestment.com:31000/tryitout"}
MAX_SUBSCRIPTIONS = 40

# 공식 예제(koreainvestment/open-trading-api)의 컬럼 정의 중 이 책에서 쓰는 위치만 뽑았다.
_ASP = {"askp": slice(2, 7), "bidp": slice(7, 12), "askp_rsqn": slice(22, 27), "bidp_rsqn": slice(27, 32)}
TRADE_FIELDS = {
    "H0IOCNT0": {"price": 2, "last_qty": 9, "acml_vol": 10, "iv": 33, "ask1": 41, "bid1": 42},  # 58필드
    "H0IFCNT0": {"price": 5, "last_qty": 9, "acml_vol": 10, "ask1": 34, "bid1": 35},           # 50필드
}
QUOTE_TRS = {"H0IOASP0", "H0IFASP0"}  # 38필드, 같은 배치


def _exchange_time(hhmmss: str, today: date) -> datetime | None:
    if len(hhmmss) != 6 or not hhmmss.isdigit():
        return None
    return datetime(today.year, today.month, today.day, int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:]))


def _num(s: str) -> float:
    return float(s) if s else 0.0


def parse_frame(msg: str, received_time: datetime | None = None) -> list[dict]:
    """'0|TR_ID|건수|f1^f2^...' 형식의 데이터 프레임을 이벤트 dict 목록으로 바꾼다."""
    received_time = received_time or datetime.now()
    _enc, tr_id, count, payload = msg.split("|", 3)
    fields = payload.split("^")
    n = int(count)
    per = len(fields) // n
    events = []
    for i in range(n):
        f = fields[i * per:(i + 1) * per]
        base = {"tr_id": tr_id, "code": f[0],
                "exchange_time": _exchange_time(f[1], received_time.date()),
                "received_time": received_time}
        if tr_id in QUOTE_TRS:
            base.update(type="quote",
                        asks=[_num(x) for x in f[_ASP["askp"]]], bids=[_num(x) for x in f[_ASP["bidp"]]],
                        ask_sizes=[int(_num(x)) for x in f[_ASP["askp_rsqn"]]],
                        bid_sizes=[int(_num(x)) for x in f[_ASP["bidp_rsqn"]]])
        elif tr_id in TRADE_FIELDS:
            base["type"] = "trade"
            for name, idx in TRADE_FIELDS[tr_id].items():
                base[name] = _num(f[idx])
        else:
            base.update(type="raw", fields=f)
        events.append(base)
    return events


def subscribe_message(approval_key: str, tr_id: str, tr_key: str, subscribe: bool = True) -> str:
    return json.dumps({
        "header": {"approval_key": approval_key, "custtype": "P",
                   "tr_type": "1" if subscribe else "2", "content-type": "utf-8"},
        "body": {"input": {"tr_id": tr_id, "tr_key": tr_key}},
    })


async def stream(approval_key: str, subscriptions: list[tuple[str, str]], mode: str = "real",
                 max_retries: int = 5) -> AsyncIterator[dict]:
    """(tr_id, 종목코드) 목록을 구독하고 이벤트를 차례로 내보낸다."""
    if len(subscriptions) > MAX_SUBSCRIPTIONS:
        raise ValueError(f"구독은 {MAX_SUBSCRIPTIONS}건 이하여야 합니다: {len(subscriptions)}건")
    retries = 0
    while True:
        try:
            async with websockets.connect(WS_URL[mode], ping_interval=None, open_timeout=10) as ws:
                for tr_id, key in subscriptions:
                    await ws.send(subscribe_message(approval_key, tr_id, key))
                    await asyncio.sleep(0.05)
                retries = 0
                async for msg in ws:
                    if msg[0] in "01":
                        for ev in parse_frame(msg):
                            yield ev
                        continue
                    j = json.loads(msg)
                    header, body = j.get("header", {}), j.get("body", {})
                    if header.get("tr_id") == "PINGPONG":
                        await ws.send(msg)               # 받은 그대로 돌려줘야 연결이 유지된다
                    elif body.get("rt_cd") not in (None, "0"):
                        yield {"type": "error", "tr_id": header.get("tr_id"), "code": header.get("tr_key"),
                               "msg": f"{body.get('msg_cd')} {body.get('msg1')}",
                               "received_time": datetime.now()}
        except (OSError, websockets.exceptions.WebSocketException) as e:
            retries += 1
            if retries > max_retries:
                raise
            yield {"type": "disconnected", "msg": f"{type(e).__name__}, 재접속 {retries}/{max_retries}",
                   "received_time": datetime.now()}
            await asyncio.sleep(min(2 ** retries, 30))
