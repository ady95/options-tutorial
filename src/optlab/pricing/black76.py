"""Black-76 옵션 가격평가 — 이론가, Greeks, 내재변동성(IV).

코스피200 옵션은 선물(또는 선물 환산 기초가격) F 기준의 Black-76 모형으로 평가한다.
- F: 선물가격(포인트), K: 행사가, T: 잔존만기(연 단위), r: 무위험이자율, sigma: 변동성
- 결과 가격 단위는 옵션 프리미엄 포인트다. 원화 금액은 market_data.spec.premium_krw()로 환산한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(F: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-76 이론가격. T 또는 sigma가 0 이하이면 할인된 내재가치를 반환한다."""
    df = math.exp(-r * T) if T > 0 else 1.0
    if T <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return df * intrinsic
    d1, d2 = _d1_d2(F, K, T, sigma)
    if is_call:
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


@dataclass(frozen=True)
class Greeks:
    delta: float  # F 1포인트 변화당 가격 변화
    gamma: float  # F 1포인트 변화당 delta 변화
    theta: float  # 1일(달력일) 경과당 가격 변화
    vega: float   # 변동성 1%p(0.01) 변화당 가격 변화


def greeks(F: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> Greeks:
    if T <= 0 or sigma <= 0:
        itm = (F > K) if is_call else (F < K)
        return Greeks(delta=(1.0 if is_call else -1.0) if itm else 0.0, gamma=0.0, theta=0.0, vega=0.0)
    df = math.exp(-r * T)
    d1, d2 = _d1_d2(F, K, T, sigma)
    sqrt_t = math.sqrt(T)
    pdf_d1 = _norm_pdf(d1)

    delta = df * _norm_cdf(d1) if is_call else -df * _norm_cdf(-d1)
    gamma = df * pdf_d1 / (F * sigma * sqrt_t)
    vega = df * F * pdf_d1 * sqrt_t
    # dPrice/dT(연 단위)의 부호를 바꿔 "시간 경과에 따른 변화"로 만든다.
    theta_year = -df * F * pdf_d1 * sigma / (2.0 * sqrt_t) + r * price(F, K, T, r, sigma, is_call)
    return Greeks(delta=delta, gamma=gamma, theta=theta_year / 365.0, vega=vega / 100.0)


def implied_forward(K: float, call_mid: float, put_mid: float, T: float, r: float) -> float:
    """콜·풋 패리티 C - P = e^{-rT}(F - K)로 만기까지의 내재 선도가격 F를 구한다."""
    return K + math.exp(r * T) * (call_mid - put_mid)


def implied_vol(
    market_price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float | None:
    """이분법으로 내재변동성을 역산한다. 가격이 무차익 범위를 벗어나면 None."""
    if T <= 0 or market_price <= 0:
        return None
    df = math.exp(-r * T)
    intrinsic = df * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    upper = df * (F if is_call else K)
    if not (intrinsic - tol <= market_price < upper):
        return None
    p_lo = price(F, K, T, r, lo, is_call) - market_price
    p_hi = price(F, K, T, r, hi, is_call) - market_price
    if p_lo > 0 or p_hi < 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = price(F, K, T, r, mid, is_call) - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
