"""13장: 변동성 스큐와 수직 스프레드(Bull Call·Bear Put)의 손익 구조."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optlab.pricing import black76


def smile(chain: pd.DataFrame, F: float, T: float, r: float, max_age: float = 60.0) -> pd.DataFrame:
    """행사가별 외가격(OTM) 옵션의 IV. F 위는 콜, 아래는 풋을 쓴다(내가격은 거래가 적고 스프레드가 넓다)."""
    c = chain[(chain["bid1"] > 0) & (chain["ask1"] > 0) & (chain["age"] <= max_age)].copy()
    c = c[((c["cp"] == "C") & (c["strike"] >= F)) | ((c["cp"] == "P") & (c["strike"] < F))]
    c["mid"] = (c["bid1"] + c["ask1"]) / 2
    c["iv"] = [black76.implied_vol(m, F, k, T, r, cp == "C") for m, k, cp in zip(c["mid"], c["strike"], c["cp"])]
    c["log_moneyness"] = np.log(c["strike"] / F)
    return c.dropna(subset=["iv"]).sort_values("strike")


def skew_slope(sm: pd.DataFrame, max_dist: float = 12.5) -> tuple[float, float] | None:
    """F ± max_dist 안의 OTM IV로 (ATM IV, d IV / d log(K/F)). 기울기가 음수면 풋 쪽 IV가 더 높다."""
    F = float(np.median(sm["strike"] / np.exp(sm["log_moneyness"])))
    near = sm[(sm["strike"] - F).abs() <= max_dist]
    if len(near) < 3:
        return None
    slope, atm = np.polyfit(near["log_moneyness"], near["iv"], 1)
    return float(atm), float(slope)


@dataclass
class Vertical:
    """수직 스프레드. cp='C'면 K1 < K2(Bull Call), cp='P'면 K1 > K2(Bear Put). K1을 사고 K2를 판다."""
    cp: str
    k1: float
    k2: float
    debit: float          # 순지불액(포인트). 호가로 들어가면 K1 매도호가 − K2 매수호가

    @property
    def width(self) -> float:
        return abs(self.k2 - self.k1)

    @property
    def max_loss(self) -> float:
        return self.debit

    @property
    def max_profit(self) -> float:
        return self.width - self.debit

    @property
    def breakeven(self) -> float:
        return self.k1 + self.debit if self.cp == "C" else self.k1 - self.debit

    def payoff(self, S) -> np.ndarray:
        """만기 지수 S에서의 손익(포인트, 수수료 전)."""
        S = np.asarray(S, dtype=float)
        if self.cp == "C":
            v = np.maximum(S - self.k1, 0) - np.maximum(S - self.k2, 0)
        else:
            v = np.maximum(self.k1 - S, 0) - np.maximum(self.k2 - S, 0)
        return v - self.debit
