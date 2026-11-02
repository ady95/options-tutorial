"""비교용 정책과 같은 조건의 평가.

모든 정책은 같은 환경(합성 시장, 같은 호가·수수료·체결 가정)에서 같은 행동 공간으로 움직인다.
정책 = env를 보고 행동을 돌려주는 함수. 규칙·LightGBM은 15분 보유 후 청산(08·09장과 같음).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from optlab.market_data.spec import MULTIPLIER_KRW
from optlab.models.forecast import Forecast
from optlab.rl.env import OptionTradingEnv
from optlab.strategy.signal import StrategyConfig, decide

HOLD_MIN = 15


def hold(env: OptionTradingEnv):
    """아무것도 하지 않는다. 손익 0이 기준선이다."""
    return 0


def _held(env: OptionTradingEnv) -> bool:
    return bool(env.lots) and env.i - env.opened_at >= HOLD_MIN


def momentum(threshold: float = 1.5):
    """규칙: 최근 15분 수익률이 IV가 말하는 15분 표준편차의 threshold배를 넘으면 그 방향으로 산다."""
    def act(env: OptionTradingEnv):
        if env.lots:
            return 3 if _held(env) else 0
        z = env._market(env.d)[env.i, 4]
        return 1 if z > threshold else 2 if z < -threshold else 0
    return act


def lgbm_signal(cfg: StrategyConfig | None = None):
    """08장 시그널 엔진을 합성 체인(ATM 콜·풋)에 그대로 적용한다. 입력 예측은 12장 워크포워드 예측."""
    cfg = cfg or StrategyConfig()

    def act(env: OptionTradingEnv):
        if env.lots:
            return 3 if _held(env) else 0
        d, i = env.d, env.i
        now = pd.Timestamp(d.times[i]).to_pydatetime()
        if now + timedelta(minutes=HOLD_MIN) > now.replace(hour=15, minute=15):
            return 0
        k = d.atm(i)
        rows = []
        for cp in ("C", "P"):
            bid, ask = d.quote(i, k, cp == "C", env.cfg.rel_spread)
            rows.append({"code": f"{cp}{k}", "cp": cp, "strike": k, "bid1": bid, "ask1": ask, "age": 0.0})
        r = d.rows.iloc[i]
        fc = Forecast(HOLD_MIN, float(r["p_up"]), max(float(r["q50"]), 0.0), max(float(r["q90"]), float(r["q50"])))
        sig = decide(pd.DataFrame(rows), float(d.F[i]), now, d.expiry, fc, cfg)
        return {"CALL_BUY": 1, "PUT_BUY": 2}.get(sig.action, 0)
    return act


def sb3(model, deterministic: bool = True):
    """Stable-Baselines3 모델을 정책 함수로."""
    def act(env: OptionTradingEnv):
        a, _ = model.predict(env._obs(), deterministic=deterministic)
        return a
    return act


def round_trips(fills: list, fee_rate: float) -> list[dict]:
    """체결 목록 → 포지션이 0에서 시작해 0으로 돌아오기까지 한 건씩(손익은 포인트)."""
    out, pos, cash, start, peak, cp0 = [], 0, 0.0, None, 0, None
    for t, cp, _, q, px in fills:
        if pos == 0:
            start, peak, cp0 = t, 0, cp
        pos += q if cp == "C" else -q
        peak = max(peak, abs(pos))
        cash += -q * px - fee_rate * px * abs(q)
        if pos == 0:
            out.append({"open": start, "close": t, "cp": cp0, "max_qty": peak, "pnl_pt": cash,
                        "hold_min": (t - start) / np.timedelta64(1, "m")})
            cash = 0.0
    return out


def evaluate(env: OptionTradingEnv, policy, days: range | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(날짜별 표, 거래별 표). 손익은 원(1포인트 = 250,000원)."""
    per_day, trades = [], []
    for k in days if days is not None else range(len(env.days)):
        env.reset(options={"day": k})
        done = False
        while not done:
            _, _, done, _, _ = env.step(policy(env))
        per_day.append({"day": env.d.day, "pnl_krw": env.pnl * MULTIPLIER_KRW, "cost_krw": env.cost * MULTIPLIER_KRW,
                        "entries": env.entries, "turnover": env.turnover})
        trades += [{"day": env.d.day, **x, "pnl_krw": x["pnl_pt"] * MULTIPLIER_KRW}
                   for x in round_trips(env.fills, env.cfg.fee_rate)]
    cols = ["day", "open", "close", "cp", "max_qty", "pnl_pt", "hold_min", "pnl_krw"]
    return pd.DataFrame(per_day), pd.DataFrame(trades, columns=cols)


def summarize(per_day: pd.DataFrame, trades: pd.DataFrame) -> dict:
    from optlab.analysis.performance import bootstrap_mean_ci
    p = per_day["pnl_krw"].to_numpy()
    cum = np.cumsum(p)
    lo, hi = bootstrap_mean_ci(p) if p.std() > 0 else (0.0, 0.0)
    return {"days": len(p), "net_krw": float(p.sum()), "day_avg_krw": float(p.mean()),
            "day_ci_lo": lo, "day_ci_hi": hi,
            "trades": len(trades), "trades_per_day": len(trades) / max(len(p), 1),
            "win_rate": float((trades["pnl_krw"] > 0).mean()) if len(trades) else np.nan,
            "cost_krw": float(per_day["cost_krw"].sum()),
            "mdd_krw": float((cum - np.maximum.accumulate(np.maximum(cum, 0))).min()) if len(p) else 0.0}
