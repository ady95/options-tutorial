"""브로커 어댑터 인터페이스.

한국투자증권(KIS), 재생(Replay) 데이터, (추후) 키움증권 등 시세 공급원을 같은 인터페이스로 다룬다.
주문은 이 책의 기본 범위에서 브로커로 보내지 않고 execution 패키지의 가상체결 엔진이 처리한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator


@dataclass(frozen=True)
class Quote:
    """최우선 호가 스냅샷. 거래소 시각과 수신 시각을 모두 기록한다."""
    symbol: str
    exchange_time: datetime
    received_time: datetime
    bid: float | None
    ask: float | None
    bid_size: int
    ask_size: int
    last: float | None = None


class MarketDataSource(ABC):
    @abstractmethod
    def option_chain(self, expiry: str) -> list[dict]:
        """만기의 옵션 체인(종목코드·행사가·콜/풋)을 반환한다."""

    @abstractmethod
    def stream_quotes(self, symbols: list[str]) -> AsyncIterator[Quote]:
        """실시간(또는 재생) 호가 스트림."""
