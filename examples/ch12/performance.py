"""12-1 실습: 거래 성과 지표와 표본 불확실성.

    # 09장 백테스트로 거래 내역을 남긴다(이름=경로)
    python examples/ch09/backtest.py --db data/ticks_20261029_live.db --each --save runs/ch12/thu_signal.csv
    python examples/ch12/performance.py thu_signal=runs/ch12/thu_signal.csv
    # 11장 실행 기록 폴더의 감사 기록도 읽는다
    python examples/ch12/performance.py live=runs/20261030-live

거래당 평균 손익의 신뢰구간을 두 가지로 낸다. 거래를 하나씩 다시 뽑는 것(iid)과 거래일을 통째로 다시 뽑는 것.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import ROOT  # noqa: F401  (src 경로 설정)

from optlab.analysis.performance import bootstrap_mean_ci, trade_stats, trades_from_audit, trades_needed


def load(path: Path) -> pd.DataFrame:
    if path.is_dir():
        t = trades_from_audit(path / "audit.jsonl")
        if len(t):
            t["day"] = pd.to_datetime(t["signal_time"]).dt.date
        return t
    t = pd.read_csv(path, parse_dates=["signal_time"])
    t["day"] = t["signal_time"].dt.date
    return t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("items", nargs="+", help="이름=경로(09장 --save CSV 또는 11장 실행 기록 폴더)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    rows = []
    for item in args.items:
        name, path = item.split("=", 1)
        t = load(Path(path))
        if not len(t):
            rows.append({"name": name, "trades": 0})
            continue
        s = trade_stats(t["pnl_krw"])
        lo, hi = bootstrap_mean_ci(t["pnl_krw"])
        dlo, dhi = bootstrap_mean_ci(t["pnl_krw"], groups=t["day"])
        need = trades_needed(s["avg_krw"], t["pnl_krw"].std(ddof=1)) if len(t) > 1 else float("nan")
        rows.append({"name": name, "days": t["day"].nunique(), **s,
                     "ci_iid": f"[{lo:,.0f}, {hi:,.0f}]",
                     "ci_by_day": "하루뿐: 계산 불가" if pd.isna(dlo) else f"[{dlo:,.0f}, {dhi:,.0f}]",
                     "need_trades": need})
    out = pd.DataFrame(rows).set_index("name").T
    fmt = {"net_krw": "{:,.0f}", "avg_krw": "{:,.0f}", "avg_win_krw": "{:,.0f}", "avg_loss_krw": "{:,.0f}",
           "mdd_krw": "{:,.0f}", "win_rate": "{:.0%}", "payoff": "{:.2f}", "profit_factor": "{:.2f}",
           "need_trades": "{:,.0f}", "days": "{:.0f}", "max_losing_streak": "{:.0f}"}
    for k, f in fmt.items():
        if k in out.index:
            out.loc[k] = [f.format(v) if pd.notna(v) else "-" for v in out.loc[k]]
    print(out.to_string())


if __name__ == "__main__":
    main()
