"""기대순손익 기반 시그널 엔진 — CALL BUY / PUT BUY / HOLD.

한 시점의 옵션 체인 호가와 07장 예측(h분 뒤 수익률 분포)을 받아 옵션마다 다음을 계산한다.

    진입가   = 지금의 매도호가(Ask)            ← 사려면 Ask에 사야 한다
    청산가   = h분 뒤 모델 중간가 - 반 스프레드  ← 팔려면 Bid에 팔아야 한다
    손익     = 청산가 - 진입가 - 수수료(양쪽)
    순우위   = E[손익] - 최소 우위(min_edge_points)

필터(유동성·신선도·ATM 범위)를 통과한 후보 중 순우위가 가장 큰 옵션이 0보다 크면 매수 신호,
아니면 HOLD다. 주문은 내지 않는다. 신호를 받아 가상체결하는 것은 09장 이후의 일이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from optlab.models.forecast import Forecast
from optlab.pricing import black76
from optlab.strategy import clock
from optlab.strategy.scenarios import annualized_vol, forecast_scenarios

SETTLE_TIME = time(15, 30)   # 최종결제가격 = 만기일 코스피200 종가


@dataclass
class StrategyConfig:
    horizon_min: int = 15                # 보유 기간(분). 예측기의 5·15·30 중 하나
    atm_range: int = 3                   # ATM ± N 행사가만 후보
    max_age_sec: float = 10.0            # 마지막 호가 이후 경과 시간 상한
    max_spread_ratio: float = 0.2        # (Ask-Bid)/중간가 상한
    min_premium: float = 0.2             # 중간가 하한(포인트)
    fee_rate: float = 0.001              # 수수료(거래대금 대비, 한쪽). 자신의 계좌 조건으로 바꾼다
    min_edge_points: float = 0.05        # 위험완충: 이만큼은 남아야 진입
    iv_shocks: tuple = (-0.1, 0.0, 0.1)  # 청산 시점 IV 변화(상대)
    iv_weights: tuple = (0.25, 0.5, 0.25)
    direction_weight: float = 0.5        # 상승확률 반영 비율(0이면 방향 무시). 07장: 방향 예측력은 약하다
    exit_cutoff: time = time(15, 15)     # 청산 예정 시각이 이보다 늦으면 진입하지 않음
    risk_free: float = 0.03
    overnight_minutes: float = clock.OVERNIGHT_MINUTES
    n_scenarios: int = 100

    @classmethod
    def from_yaml(cls, path: Path) -> "StrategyConfig":
        raw = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("strategy", {})
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in raw.items() if k in names}
        for k in ("iv_shocks", "iv_weights"):
            if k in kw:
                kw[k] = tuple(kw[k])
        if "exit_cutoff" in kw:
            kw["exit_cutoff"] = time.fromisoformat(str(kw["exit_cutoff"]))
        return cls(**kw)


@dataclass
class Signal:
    time: datetime
    action: str                  # CALL_BUY | PUT_BUY | HOLD
    reason: str                  # 진입 근거 또는 HOLD 사유
    code: str | None = None
    cp: str | None = None
    strike: float | None = None
    entry: float | None = None   # 진입 지정가(Ask)
    exp_pnl: float | None = None
    net_edge: float | None = None
    p_win: float | None = None
    forecast_vol: float | None = None
    atm_iv: float | None = None
    table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


def implied_forward(chain: pd.DataFrame, T: float, r: float, max_spread: float = 0.1) -> float | None:
    """같은 행사가의 콜·풋 중간가 패리티로 선도가격 F를 추정한다(스프레드가 좁은 행사가들의 중앙값)."""
    c = chain[(chain["bid1"] > 0) & (chain["ask1"] > 0)].assign(mid=lambda x: (x.bid1 + x.ask1) / 2)
    c = c.assign(sr=(c.ask1 - c.bid1) / c.mid)
    pair = c.pivot_table(index="strike", columns="cp", values=["mid", "sr"]).dropna()
    if pair.empty:
        return None
    tight = pair[(pair[("sr", "C")] <= max_spread) & (pair[("sr", "P")] <= max_spread)]
    use = tight if len(tight) else pair
    fwd = [black76.implied_forward(k, row[("mid", "C")], row[("mid", "P")], T, r) for k, row in use.iterrows()]
    return float(np.median(fwd))


def evaluate(chain: pd.DataFrame, F: float, now: datetime, expiry: datetime,
             fc: Forecast, cfg: StrategyConfig) -> pd.DataFrame:
    """옵션마다 필터 결과와 기대손익 구성요소를 계산한다.

    chain 열: code, cp, strike, bid1, ask1, age(초)
    반환 열: ... reject(빈 문자열이면 후보), iv, model_edge, entry_cost, exit_cost, fees, exp_pnl, net, p_win, pnl_q10
    """
    h = cfg.horizon_min
    T0 = clock.years(now, expiry, cfg.overnight_minutes)
    T1 = clock.years(now + timedelta(minutes=h), expiry, cfg.overnight_minutes)
    r_s, w_s = forecast_scenarios(fc, cfg.n_scenarios, cfg.direction_weight)
    shocks, iv_w = np.array(cfg.iv_shocks), np.array(cfg.iv_weights)
    w = (w_s[:, None] * iv_w[None, :]).ravel()          # 지수 시나리오 × IV 시나리오
    F1 = np.repeat(F * np.exp(r_s), len(shocks))

    strikes = np.sort(chain["strike"].unique())
    step = np.median(np.diff(strikes)) if len(strikes) > 1 else 2.5
    max_dist = (cfg.atm_range + 0.5) * step   # F에서 행사가 N칸 반 이내. 구독 체인이 F를 벗어나면 후보가 줄어든다

    rows = []
    for o in chain.itertuples():
        bid, ask = float(o.bid1), float(o.ask1)
        mid = (bid + ask) / 2
        row = {"code": o.code, "cp": o.cp, "strike": o.strike, "bid": bid, "ask": ask, "mid": mid,
               "age": o.age, "spread_ratio": (ask - bid) / mid if mid > 0 else np.nan, "reject": ""}
        is_call = o.cp == "C"
        iv = black76.implied_vol(mid, F, o.strike, T0, cfg.risk_free, is_call) if bid > 0 and ask > 0 else None
        row["iv"] = iv
        if bid <= 0 or ask <= 0:
            row["reject"] = "ONE_SIDED"
        elif abs(o.strike - F) > max_dist:
            row["reject"] = "OUT_OF_RANGE"
        elif o.age > cfg.max_age_sec:
            row["reject"] = "STALE_QUOTE"
        elif row["spread_ratio"] > cfg.max_spread_ratio:
            row["reject"] = "WIDE_SPREAD"
        elif mid < cfg.min_premium:
            row["reject"] = "LOW_PREMIUM"
        elif iv is None:
            row["reject"] = "NO_IV"
        if not row["reject"]:
            sig = np.tile(iv * (1 + shocks), len(r_s))
            mid1 = black76.price_array(F1, o.strike, T1, cfg.risk_free, sig, is_call)
            half = (ask - bid) / 2                          # 청산 때도 지금과 같은 스프레드를 가정
            exit_bid = np.maximum(mid1 - half, 0.0)
            fees = cfg.fee_rate * (ask + exit_bid)
            pnl = exit_bid - ask - fees
            order = np.argsort(pnl)
            cum = np.cumsum(w[order])
            row.update(
                model_edge=float(np.sum(w * mid1)) - mid,     # 예측분포로 본 중간가 변화
                entry_cost=ask - mid,                         # Ask에 사서 내는 비용
                exit_cost=float(np.sum(w * (mid1 - exit_bid))),  # Bid에 팔아 내는 비용
                fees=float(np.sum(w * fees)),
                exp_pnl=float(np.sum(w * pnl)),
                p_win=float(np.sum(w[pnl > 0])),
                pnl_q10=float(pnl[order][np.searchsorted(cum, 0.1)]),
            )
            row["net"] = row["exp_pnl"] - cfg.min_edge_points
        rows.append(row)
    return pd.DataFrame(rows)


def decide(chain: pd.DataFrame, F: float, now: datetime, expiry: datetime,
           fc: Forecast, cfg: StrategyConfig) -> Signal:
    """한 시점의 매매신호. 우위가 비용과 위험완충을 넘지 못하면 HOLD."""
    exit_at = now + timedelta(minutes=cfg.horizon_min)
    if exit_at.time() > cfg.exit_cutoff or exit_at > expiry:
        return Signal(now, "HOLD", "NEAR_CLOSE")
    tab = evaluate(chain, F, now, expiry, fc, cfg)
    T0 = clock.years(now, expiry, cfg.overnight_minutes)
    r_s, w_s = forecast_scenarios(fc, cfg.n_scenarios, cfg.direction_weight)
    fvol = annualized_vol(r_s, w_s, cfg.horizon_min)
    near = tab.dropna(subset=["iv"])
    atm_iv = float(near.loc[(near["strike"] - F).abs().nsmallest(2).index, "iv"].mean()) if len(near) else None
    base = dict(table=tab, forecast_vol=fvol, atm_iv=atm_iv)

    cand = tab[tab["reject"] == ""]
    if cand.empty:
        return Signal(now, "HOLD", "NO_CANDIDATE", **base)
    best = cand.loc[cand["net"].idxmax()]
    if best["net"] <= 0:
        return Signal(now, "HOLD", "NO_EDGE", exp_pnl=best["exp_pnl"], net_edge=best["net"], **base)
    action = "CALL_BUY" if best["cp"] == "C" else "PUT_BUY"
    reason = f"EDGE {best['net']:.2f}pt (예측 변동성 {fvol:.1%} vs IV {best['iv']:.1%})"
    return Signal(now, action, reason, code=best["code"], cp=best["cp"], strike=float(best["strike"]),
                  entry=float(best["ask"]), exp_pnl=float(best["exp_pnl"]), net_edge=float(best["net"]),
                  p_win=float(best["p_win"]), **base)


def chain_at(quotes: pd.DataFrame, instruments: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """ts에 '알 수 있었던' 종목별 최신 호가(수신 시각 기준 as-of)와 경과 시간."""
    opts = instruments.dropna(subset=["strike"])
    q = quotes[(quotes["received_time"] <= ts) & quotes["code"].isin(opts["code"])]
    last = q.sort_values("received_time").groupby("code").tail(1)
    last = last.assign(age=(ts - last["received_time"]).dt.total_seconds())
    return last[["code", "bid1", "ask1", "age"]].merge(opts[["code", "cp", "strike"]], on="code")


def expiry_datetime(expiry_date) -> datetime:
    return datetime.combine(expiry_date, SETTLE_TIME)
