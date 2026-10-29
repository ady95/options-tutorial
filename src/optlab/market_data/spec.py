"""코스피200 (위클리)옵션 상품명세 상수와 보조 함수.

주의: 거래승수·호가단위·거래시간은 변경될 수 있다. 실제 사용 전 한국거래소 최신 상품명세로 재확인한다.
"""
from __future__ import annotations

import math
from datetime import time

MULTIPLIER_KRW = 250_000          # 옵션 1포인트당 원화 금액
TICK_BELOW_10 = 0.01              # 프리미엄 10 미만 호가단위
TICK_ABOVE_10 = 0.05              # 프리미엄 10 이상 호가단위
REGULAR_OPEN = time(8, 45)
REGULAR_CLOSE = time(15, 45)
LAST_TRADING_DAY_CLOSE = time(15, 20)


def tick_size(premium: float) -> float:
    return TICK_BELOW_10 if premium < 10.0 else TICK_ABOVE_10


def round_to_tick(premium: float, side: str = "nearest") -> float:
    """프리미엄을 호가단위에 맞춘다. side: 'nearest' | 'up'(매수 지정가) | 'down'(매도 지정가)."""
    tick = tick_size(premium)
    n = premium / tick
    if side == "up":
        n = math.ceil(n - 1e-9)
    elif side == "down":
        n = math.floor(n + 1e-9)
    else:
        n = round(n)
    return round(n * tick, 2)


def premium_krw(premium: float, contracts: int = 1) -> int:
    """프리미엄(포인트) × 거래승수 × 계약수 = 원화 금액."""
    return int(round(premium * MULTIPLIER_KRW * contracts))


def breakeven_at_expiry(strike: float, premium: float, is_call: bool) -> float:
    """수수료를 제외한 단순 만기 손익분기 기초지수."""
    return strike + premium if is_call else strike - premium


def payoff_at_expiry(underlying: float, strike: float, premium: float, is_call: bool, contracts: int = 1) -> int:
    """옵션 매수 포지션의 만기 손익(원). 거래비용 미반영."""
    intrinsic = max(underlying - strike, 0.0) if is_call else max(strike - underlying, 0.0)
    return premium_krw(intrinsic - premium, contracts)
