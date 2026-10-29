"""가상체결 모델 — 지정가 주문을 기록된 호가에 맞춰 본다.

규칙(보수적으로 잡았다):
- 주문은 보낸 시각 + 지연(latency_sec) 뒤에 거래소에 도착한다. 그 전의 호가로는 체결되지 않는다.
- 매수 지정가 L은 그 순간의 매도호가 중 L 이하인 단계만, 표시된 잔량만큼 체결된다(매도는 반대).
  남은 수량은 대기하다가 이후 호가가 다시 L 이하로 들어오면 체결된다.
- 대기 중인 주문이 줄을 서서 체결되는 경우(상대방이 내 가격에 와서 체결)는 가정하지 않는다.
  대기열에서 내 순서를 알 수 없기 때문이다.
- 새 호가 메시지가 오면 그 잔량을 새로 본다. 내가 가져간 잔량이 다시 채워졌다고 보는 셈이라,
  2계약 이상에서는 약간 낙관적이다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Order:
    side: str                   # BUY | SELL
    code: str
    qty: int
    limit: float
    sent: pd.Timestamp          # 주문을 낸 시각(결정 시각)
    active: pd.Timestamp        # 거래소 도착 시각 = sent + 지연
    expire: pd.Timestamp        # 이때까지 다 체결되지 않으면 남은 수량 취소
    tag: str = ""               # ENTRY | EXIT
    arrived: bool = False       # 거래소에 도착해 서 있던 호가와 이미 맞춰 봤는가
    fills: list[tuple[pd.Timestamp, int, float]] = field(default_factory=list)

    @property
    def filled(self) -> int:
        return sum(q for _, q, _ in self.fills)

    @property
    def remaining(self) -> int:
        return self.qty - self.filled

    @property
    def avg_price(self) -> float | None:
        n = self.filled
        return sum(q * p for _, q, p in self.fills) / n if n else None


def levels(quote, side: str) -> list[tuple[float, int]]:
    """호가 한 건에서 주문이 가져갈 상대편 단계 [(가격, 잔량), ...]. depth가 없으면 1단계만."""
    depth = getattr(quote, "depth", None)
    if isinstance(depth, str) and depth:
        bids, asks, bid_sizes, ask_sizes = json.loads(depth)
        px, sz = (asks, ask_sizes) if side == "BUY" else (bids, bid_sizes)
    elif side == "BUY":
        px, sz = [quote.ask1], [quote.ask_size1]
    else:
        px, sz = [quote.bid1], [quote.bid_size1]
    return [(float(p), int(s)) for p, s in zip(px, sz) if p and p > 0 and s and s > 0]


def match(order: Order, quote, ts: pd.Timestamp) -> int:
    """호가 한 건에 주문을 맞춰 보고 체결된 수량을 돌려준다. 체결가는 각 단계의 호가(지정가보다 유리할 수 있음)."""
    got = 0
    for price, size in levels(quote, order.side):
        if order.remaining <= 0:
            break
        ok = price <= order.limit + 1e-9 if order.side == "BUY" else price >= order.limit - 1e-9
        if not ok:
            break                      # 호가는 가격순이므로 이후 단계도 안 된다
        q = min(size, order.remaining)
        order.fills.append((ts, q, price))
        got += q
    return got
