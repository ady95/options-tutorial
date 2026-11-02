"""14-4절: 규칙·LightGBM·PPO·SAC를 같은 합성 시장, 같은 비용으로 비교한다.

train.py 결과(runs/ch14/results/*.json)를 모으고, 기준 정책은 여기서 직접 평가한다(runs/ch14/baselines.json에 저장).
테스트 기간(2026-09~10)은 모델을 고른 뒤 한 번만 본 성과다.

    python examples/ch14/compare.py
"""
from __future__ import annotations

import json

import pandas as pd
from common import OUT, load_days
from stable_baselines3 import PPO, SAC
from train import COSTS

from optlab.rl import policies as P
from optlab.rl.env import EnvConfig, OptionTradingEnv

SPLITS = ("train", "valid", "test")
BASELINES = {"대기": P.hold, "모멘텀 규칙": P.momentum(), "LightGBM(08장 엔진)": P.lgbm_signal()}


def baselines(days, refresh: bool = False) -> dict:
    path = OUT / "baselines.json"
    res = json.loads(path.read_text(encoding="utf-8")) if path.exists() and not refresh else {}
    for cost in ("real", "none"):
        for name, pol in BASELINES.items():
            key = f"{cost}|{name}"
            if key not in res:
                res[key] = {s: P.summarize(*P.evaluate(OptionTradingEnv(days[s], EnvConfig(**COSTS[cost])), pol))
                            for s in SPLITS}
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


def table(rows: list[dict]) -> pd.DataFrame:
    t = pd.DataFrame(rows).set_index("정책")
    for c in t.columns:
        if c.endswith("(만원)"):
            t[c] = (t[c] / 1e4).round(0).astype(int)
    return t


def behavior(policy, days, cost: str, mode: str = "discrete") -> dict:
    """테스트 기간에 이 정책이 실제로 한 일."""
    per_day, tr = P.evaluate(OptionTradingEnv(days, EnvConfig(mode=mode, **COSTS[cost])), policy)
    if tr.empty:
        return {"거래": 0}
    h = pd.to_datetime(tr["open"]).dt.hour
    return {"거래": len(tr), "거래한 날": f"{(per_day['entries'] > 0).mean():.0%}",
            "콜 비율": f"{(tr['cp'] == 'C').mean():.0%}", "보유(분, 중앙값)": round(tr["hold_min"].median()),
            "최대 계약(평균)": round(tr["max_qty"].mean(), 1), "09시대 진입": f"{(h == 9).mean():.0%}",
            "승률": f"{(tr['pnl_krw'] > 0).mean():.0%}"}


def main() -> None:
    days = load_days()
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    base = baselines(days)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "results").glob("*.json"))]

    for cost, label in [("real", "스프레드 7% + 수수료 0.1%"), ("none", "비용 없음(호가단위 1틱)")]:
        rows = []
        for name in BASELINES:
            b = base[f"{cost}|{name}"]
            rows.append({"정책": name, "학습(만원)": b["train"]["net_krw"], "검증(만원)": b["valid"]["net_krw"],
                         "테스트(만원)": b["test"]["net_krw"], "테스트 하루 CI 하한(만원)": b["test"]["day_ci_lo"],
                         "테스트 하루 CI 상한(만원)": b["test"]["day_ci_hi"], "거래/일": round(b["test"]["trades_per_day"], 1)})
        for r in runs:
            if r["cost"] != cost or r["penalty"]:
                continue
            b = r["best"]
            rows.append({"정책": f"{r['algo'].upper()} s{r['seed']} @{r['best_step'] // 1000}k",
                         "학습(만원)": b["train"]["net_krw"], "검증(만원)": b["valid"]["net_krw"],
                         "테스트(만원)": b["test"]["net_krw"], "테스트 하루 CI 하한(만원)": b["test"]["day_ci_lo"],
                         "테스트 하루 CI 상한(만원)": b["test"]["day_ci_hi"],
                         "거래/일": round(b["test"]["trades_per_day"], 1)})
        print(f"\n== {label} — 1계약 기준, 학습 83일·검증 20일·테스트 40일")
        print(table(rows).to_string())

    print("\n== 회전율 패널티(계약당 1포인트, 학습에만) — 스프레드 7%, 검증 최고점 모델과 마지막 모델")
    rows = []
    for r in runs:
        if not r["penalty"]:
            continue
        for which in ("best", "final"):
            b = r[which]
            rows.append({"정책": f"{r['algo'].upper()} s{r['seed']} {'검증 최고' if which == 'best' else '마지막'}",
                         "학습(만원)": b["train"]["net_krw"], "검증(만원)": b["valid"]["net_krw"],
                         "테스트(만원)": b["test"]["net_krw"], "테스트 하루 CI 하한(만원)": b["test"]["day_ci_lo"],
                         "테스트 하루 CI 상한(만원)": b["test"]["day_ci_hi"], "테스트 거래": b["test"]["trades"]})
    if rows:
        print(table(rows).to_string())

    print("\n== 학습 곡선: 검증 손익(만원)이 가장 좋았던 시점과 마지막 시점")
    for r in runs:
        c = pd.DataFrame(r["curve"])
        f = r["final"]
        print(f"{r['tag']:22s} 최고 {c['valid_net_krw'].max() / 1e4:>8,.0f} @{r['best_step']:>7,}  "
              f"마지막 {c['valid_net_krw'].iloc[-1] / 1e4:>8,.0f}  | 마지막 모델 학습 {f['train']['net_krw'] / 1e4:>9,.0f}"
              f" 테스트 {f['test']['net_krw'] / 1e4:>8,.0f}  ({r['minutes']}분)")

    print("\n== 테스트 기간 행동(검증으로 고른 모델)")
    rows = []
    for r in runs:
        m = (PPO if r["algo"] == "ppo" else SAC).load(OUT / "models" / r["tag"])
        mode = "discrete" if r["algo"] == "ppo" else "continuous"
        rows.append({"모델": r["tag"], **behavior(P.sb3(m), days["test"], r["cost"], mode)})
    for cost in ("real", "none"):
        rows.append({"모델": f"lgbm_{cost}", **behavior(P.lgbm_signal(), days["test"], cost)})
    print(pd.DataFrame(rows).set_index("모델").to_string())


if __name__ == "__main__":
    main()
