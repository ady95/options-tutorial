"""거래 단위 성과 지표와 표본 불확실성.

입력은 거래 한 건이 한 행인 표(09장 백테스트 --save CSV, 11장 감사 기록의 TRADE)이고 pnl_krw 열이 있어야 한다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def trade_stats(pnl: pd.Series | np.ndarray) -> dict:
    """비용 차감 손익(원) 목록의 요약. 손익 0은 패배로 센다."""
    p = np.asarray(pnl, dtype=float)
    if not len(p):
        return {"trades": 0}
    win, loss = p[p > 0], p[p <= 0]
    cum = np.cumsum(p)
    streak = longest = 0
    for x in p:
        streak = streak + 1 if x <= 0 else 0
        longest = max(longest, streak)
    return {
        "trades": len(p),
        "net_krw": float(p.sum()),
        "avg_krw": float(p.mean()),                     # 기대값(expectancy): 거래 한 건의 평균
        "win_rate": float(len(win) / len(p)),
        "avg_win_krw": float(win.mean()) if len(win) else 0.0,
        "avg_loss_krw": float(loss.mean()) if len(loss) else 0.0,
        # 손익비 = 평균 이익 / 평균 손실 크기. 승률과 함께 봐야 한다
        "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) and loss.mean() < 0 else float("nan"),
        # 이익 합계 / 손실 합계 크기. 1보다 커야 남는다
        "profit_factor": float(win.sum() / -loss.sum()) if loss.sum() < 0 else float("inf"),
        "mdd_krw": float((cum - np.maximum.accumulate(np.maximum(cum, 0))).min()),
        "max_losing_streak": longest,
    }


def bootstrap_mean_ci(pnl, groups=None, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """거래당 평균 손익의 부트스트랩 신뢰구간.

    groups(예: 거래일)를 주면 거래가 아니라 그룹을 통째로 뽑는다. 같은 날의 거래는 같은 시장 움직임을
    함께 겪으므로 서로 독립이 아니다. 그룹이 하나뿐이면 구간을 만들 수 없어 (nan, nan)을 돌려준다.
    """
    p = np.asarray(pnl, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        means = rng.choice(p, (n_boot, len(p))).mean(axis=1)
    else:
        g = pd.Series(p).groupby(np.asarray(groups))
        sums, counts = g.sum().to_numpy(), g.size().to_numpy()
        if len(sums) < 2:
            return float("nan"), float("nan")
        idx = rng.integers(0, len(sums), (n_boot, len(sums)))
        means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(lo), float(hi)


def trades_needed(avg: float, std: float, z: float = 2.0) -> float:
    """평균이 표준오차의 z배가 되려면 독립인 거래가 몇 건 필요한가: (z·σ/μ)²."""
    return float("inf") if avg == 0 else float((z * std / avg) ** 2)


def trades_from_audit(path: Path) -> pd.DataFrame:
    """11장 감사 기록(audit.jsonl)의 확정 거래(TRADE)를 표로."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break                                   # 쓰다가 죽은 마지막 줄
        if r.get("kind") == "TRADE":
            rows.append({"run": r["run"], "time": r["time"], **r["trade"]})
    return pd.DataFrame(rows)
