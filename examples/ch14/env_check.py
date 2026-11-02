"""14-1절: 환경 점검 — 관측·행동, 비용과 움직임의 크기, 아무 생각 없는 정책들의 손익.

    python examples/ch14/env_check.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from common import load_days

from optlab.market_data.spec import MULTIPLIER_KRW
from optlab.rl.env import EnvConfig, OptionTradingEnv
from optlab.rl.policies import evaluate, summarize

OBS_NAMES = ["p_up(방향)", "q50/IV15", "q90/IV15", "ret_5", "ret_15", "ret_30", "rv_15/IV15", "ret_from_open",
             "minutes_to_close", "is_mon", "is_thu", "남은 거래일", "포지션", "평가손익률", "보유 시간", "IV"]


def cost_vs_move(days, rel_spread: float, fee: float, h: int = 15) -> dict:
    """ATM 콜을 매도호가에 사서 h분 뒤 매수호가에 팔 때: 비용(중간가 대비)과 중간가 변화의 크기."""
    cost, move = [], []
    for d in days:
        for i in range(0, len(d) - h, 5):
            k = d.atm(i)
            m0, m1 = d.mid(i, k, True), d.mid(i + h, k, True)
            b1 = d.quote(i + h, k, True, rel_spread)[0]
            a0 = d.quote(i, k, True, rel_spread)[1]
            cost.append((a0 - m0) + (m1 - b1) + fee * (a0 + b1))
            move.append(m1 - m0)
    cost, move = np.array(cost), np.array(move)
    return {"중간가": np.median([d.mid(0, d.atm(0), True) for d in days]),
            "왕복 비용": np.median(cost), "|15분 변화|": np.median(np.abs(move)),
            "평균 변화": move.mean(), "|변화|>비용": (np.abs(move) > cost).mean(), "변화>비용": (move > cost).mean()}


def main() -> None:
    days = load_days()
    pd.set_option("display.width", 160)
    print("== 에피소드:", {k: f"{len(v)}일 ({v[0].day}~{v[-1].day})" for k, v in days.items()})
    env = OptionTradingEnv(days["train"], EnvConfig())
    obs, _ = env.reset(options={"day": 0})
    print(f"관측 {env.observation_space.shape[0]}개, 행동 {env.action_space} (0 대기, 1 콜, 2 풋, 3 청산)")
    print(pd.Series(obs, index=OBS_NAMES).round(2).to_string())

    print("\n== 1) ATM 콜 15분 보유: 왕복 비용 vs 중간가 변화(포인트, 학습 기간)")
    rows = {name: cost_vs_move(days["train"], rs, fee) for name, rs, fee in
            [("7%", 0.07, 0.001), ("12%", 0.12, 0.001), ("1틱", 0.0, 0.0)]}
    print(pd.DataFrame(rows).T.round(3).to_string())

    print("\n== 2) 생각 없는 정책들(1계약, 원)")
    out = []
    for split in ("train", "valid", "test"):
        e = OptionTradingEnv(days[split], EnvConfig())
        for name, pol in [("09:15 콜 사서 끝까지", lambda x: 1 if x.i == 0 else 0),
                          ("09:15 풋 사서 끝까지", lambda x: 2 if x.i == 0 else 0),
                          ("무작위", lambda x, r=np.random.default_rng(0): int(r.integers(4)))]:
            s = summarize(*evaluate(e, pol))
            out.append({"기간": split, "정책": name, "하루 평균": s["day_avg_krw"], "거래/일": s["trades_per_day"],
                        "비용 합계": s["cost_krw"]})
    print(pd.DataFrame(out).round(1).to_string(index=False))
    print(f"\n(1포인트 = {MULTIPLIER_KRW:,}원. 비용은 중간가 대비 호가 차이와 수수료)")


if __name__ == "__main__":
    main()
