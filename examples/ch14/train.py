"""14-2·14-3절: PPO(이산 행동)·SAC(연속 비중) 학습.

학습 기간(2026-04~07)의 날을 무작위로 골라 에피소드로 쓴다. eval_every걸음마다 검증 기간(2026-08)
20일 손익을 재고, 가장 좋았던 시점의 모델을 남긴다. 테스트 기간(2026-09~10)은 여기서 보지 않는다.

    python examples/ch14/train.py --algo ppo --seed 0
    python examples/ch14/train.py --algo sac --seed 0
    python examples/ch14/train.py --algo ppo --seed 0 --cost none     # 비용 없는 환경(비교용)
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from common import OUT, load_days
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from optlab.rl.env import EnvConfig, OptionTradingEnv
from optlab.rl.policies import evaluate, sb3, summarize

COSTS = {"real": dict(rel_spread=0.07, fee_rate=0.001),   # 실제 기록 중 가장 좁았던 스프레드(만기일 오전)
         "wide": dict(rel_spread=0.12, fee_rate=0.001),   # 월 위클리 오후
         "none": dict(rel_spread=0.0, fee_rate=0.0)}      # 호가단위 1틱만


class ValidCallback(BaseCallback):
    def __init__(self, env: OptionTradingEnv, every: int, path):
        super().__init__()
        self.env, self.every, self.path = env, every, path
        self.best, self.curve = float("-inf"), []

    def _on_step(self) -> bool:
        if self.n_calls % self.every == 0:
            s = summarize(*evaluate(self.env, sb3(self.model)))
            self.curve.append({"step": self.num_timesteps, "valid_net_krw": s["net_krw"],
                               "valid_trades": s["trades"]})
            print(f"  {self.num_timesteps:>7,}걸음  검증 {s['net_krw']:>14,.0f}원  거래 {s['trades']}", flush=True)
            if s["net_krw"] > self.best:
                self.best = s["net_krw"]
                self.model.save(self.path)
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="기본: PPO 300,000 / SAC 60,000")
    ap.add_argument("--cost", choices=list(COSTS), default="real")
    ap.add_argument("--penalty", type=float, default=0.0, help="회전율 패널티(계약당 포인트)")
    a = ap.parse_args()
    torch.set_num_threads(1)

    days = load_days()
    mode = "discrete" if a.algo == "ppo" else "continuous"
    cfg = EnvConfig(mode=mode, turnover_penalty=a.penalty, **COSTS[a.cost])
    eval_cfg = EnvConfig(mode=mode, **COSTS[a.cost])                 # 평가에는 패널티를 넣지 않는다
    train_env = Monitor(OptionTradingEnv(days["train"], cfg, seed=a.seed))
    tag = f"{a.algo}_{a.cost}_p{a.penalty:g}_s{a.seed}"
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    path = OUT / "models" / tag

    if a.algo == "ppo":
        steps = a.steps or 300_000
        model = PPO("MlpPolicy", train_env, ent_coef=0.01, seed=a.seed, verbose=0)
        every = 25_000
    else:
        steps = a.steps or 60_000
        model = SAC("MlpPolicy", train_env, seed=a.seed, verbose=0)
        every = 10_000
    cb = ValidCallback(OptionTradingEnv(days["valid"], eval_cfg), every, path)
    t0 = time.time()
    print(f"[{tag}] 학습 {steps:,}걸음 (학습 {len(days['train'])}일, 검증 {len(days['valid'])}일)", flush=True)
    model.learn(steps, callback=cb)
    model.save(OUT / "models" / f"{tag}_final")

    best = (PPO if a.algo == "ppo" else SAC).load(path)
    res = {"tag": tag, "algo": a.algo, "cost": a.cost, "penalty": a.penalty, "seed": a.seed, "steps": steps,
           "minutes": round((time.time() - t0) / 60, 1), "curve": cb.curve,
           "best_step": max(cb.curve, key=lambda c: c["valid_net_krw"])["step"]}
    for name in ("best", "final"):
        m = best if name == "best" else model
        res[name] = {split: summarize(*evaluate(OptionTradingEnv(days[split], eval_cfg), sb3(m)))
                     for split in ("train", "valid", "test")}
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / f"{tag}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for split, s in res["best"].items():
        print(f"  best@{res['best_step']:,} {split:5s} {s['net_krw']:>14,.0f}원  거래 {s['trades']}")


if __name__ == "__main__":
    main()
