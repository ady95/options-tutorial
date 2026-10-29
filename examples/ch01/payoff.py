"""1장 실습: 콜·풋 만기 손익과 손익분기점 계산하기.

    python examples/ch01/payoff.py --strike 947.5 --call 15.25 --put 7.00

설명용 계산이다. 실제 주문 가능한 호가·수수료·슬리피지는 반영하지 않는다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from optlab.market_data.spec import breakeven_at_expiry, payoff_at_expiry, premium_krw  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strike", type=float, default=947.5)
    p.add_argument("--call", type=float, default=15.25, help="콜 프리미엄(포인트)")
    p.add_argument("--put", type=float, default=7.00, help="풋 프리미엄(포인트)")
    args = p.parse_args()

    for name, prem, is_call in (("콜", args.call, True), ("풋", args.put, False)):
        be = breakeven_at_expiry(args.strike, prem, is_call)
        print(f"{name} 매수 1계약 프리미엄: {premium_krw(prem):>12,}원 | 만기 손익분기: {be:.2f}")

    print("\n만기 기초지수별 손익(원)")
    print(f"{'지수':>8} {'콜 매수':>14} {'풋 매수':>14}")
    for step in range(-20, 25, 5):
        s = args.strike + step
        c = payoff_at_expiry(s, args.strike, args.call, True)
        q = payoff_at_expiry(s, args.strike, args.put, False)
        print(f"{s:>8.1f} {c:>14,} {q:>14,}")


if __name__ == "__main__":
    main()
