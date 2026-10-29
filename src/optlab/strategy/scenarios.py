"""예측 → 지수 수익률 시나리오.

07장 예측기는 h분 뒤 |로그수익률|의 50%·90% 분위수(q50, q90)와 상승확률 p_up을 준다.
- 크기: q50과 q90을 정확히 지나는 로그정규분포로 |r|의 분포를 만든다.
- 방향: 같은 크기에 +는 p, -는 1-p의 확률을 준다. p = 0.5 + direction_weight × (p_up - 0.5).
  07장에서 방향 예측력이 약했으므로 direction_weight로 줄여 반영한다(0이면 방향 무시).
결과는 (수익률 배열, 확률 배열) 쌍이다. 확률의 합은 1이다.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

from optlab.models.forecast import Forecast
from optlab.strategy.clock import ANNUAL_UNITS

Z90 = float(ndtri(0.9))   # 1.2816


def forecast_scenarios(fc: Forecast, n: int = 100, direction_weight: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    q50 = max(fc.absret_q50, 1e-6)
    q90 = max(fc.absret_q90, q50 * 1.01)
    mu, s = math.log(q50), (math.log(q90) - math.log(q50)) / Z90
    u = (np.arange(n) + 0.5) / n
    mag = np.exp(mu + s * ndtri(u))
    p = 0.5 + direction_weight * (fc.p_up - 0.5)
    r = np.concatenate([mag, -mag])
    w = np.concatenate([np.full(n, p / n), np.full(n, (1 - p) / n)])
    return r, w


def implied_scenarios(iv: float, h_minutes: float, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """시장 IV가 맞다고 할 때의 h분 수익률 분포(정규, 평균은 선도가격이 마틴게일이 되게 -σ²/2)."""
    sd = iv * math.sqrt(h_minutes / ANNUAL_UNITS)
    u = (np.arange(2 * n) + 0.5) / (2 * n)
    return -0.5 * sd * sd + sd * ndtri(u), np.full(2 * n, 1 / (2 * n))


def annualized_vol(r: np.ndarray, w: np.ndarray, h_minutes: float) -> float:
    """시나리오 분포의 변동성을 연율로. 같은 시계의 IV와 바로 비교할 수 있다."""
    return math.sqrt(float(np.sum(w * r * r)) / (h_minutes / ANNUAL_UNITS))
