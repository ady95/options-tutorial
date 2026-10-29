"""1장 보충: 지수가 올라도 콜옵션이 손실 나는 경우 (Black-76 계산).

    python examples/ch01/time_value.py

선물 965, 행사가 965 ATM 콜을 IV 20%·잔존 3일에 샀다고 가정하고,
하루 뒤 지수 상승폭과 IV 변화에 따라 옵션 가격이 어떻게 달라지는지 계산한다.
호가 스프레드·수수료는 반영하지 않은 이론가격 비교다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from optlab.market_data.spec import premium_krw  # noqa: E402
from optlab.pricing import black76  # noqa: E402

F0, K, R = 965.0, 965.0, 0.03


def main() -> None:
    entry = black76.price(F0, K, 3 / 365, R, 0.20, True)
    print(f"진입: 선물 {F0:.1f}, 잔존 3일, IV 20% -> 콜 이론가 {entry:.2f}")

    print("\n[시간가치 감소] 지수·IV가 그대로일 때")
    for days in (7, 5, 3, 1, 0.25):
        p = black76.price(F0, K, days / 365, R, 0.20, True)
        print(f"  잔존 {days:>4}일: {p:6.2f}")

    print("\n[하루 뒤] 지수 상승폭 x IV 변화별 손익 (1계약, 원)")
    print(f"{'상승폭':>6} {'IV 22%':>12} {'IV 20%':>12} {'IV 18%':>12}")
    for up in (0, 3, 6, 10):
        cells = []
        for iv in (0.22, 0.20, 0.18):
            p = black76.price(F0 + up, K, 2 / 365, R, iv, True)
            cells.append(premium_krw(p - entry))
        print(f"{up:>6} " + " ".join(f"{c:>12,}" for c in cells))


if __name__ == "__main__":
    main()
