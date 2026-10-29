"""09-2: 주문 지연·수량·지정가를 바꾸면 체결과 손익이 어떻게 달라지나.

신호는 한 번만 계산해 두고(같은 신호), 가상체결 조건만 바꿔 신호별 백테스트를 반복한다.

    python examples/ch09/fill_sweep.py --db data/ticks_20261029_live.db
    python examples/ch09/fill_sweep.py --db data/ticks_20261029_mon.db --strategy tightest
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest import backtest, quotes_of, tightest_atm
from market import ROOT, LiveDB, open_source
from signal_engine import MODEL_PATH, run_once

from optlab.backtest import engine
from optlab.backtest.report import summarize
from optlab.models.forecast import Forecaster
from optlab.strategy.signal import StrategyConfig

CASES = [  # (지연 초, 계약 수, 진입 틱)
    (0.0, 1, 0), (0.5, 1, 0), (2.0, 1, 0), (5.0, 1, 0),
    (0.5, 3, 0), (0.5, 5, 0), (0.5, 5, 2),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--strategy", choices=["signal", "tightest"], default="signal")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    scfg = StrategyConfig.from_yaml(args.config)
    base = engine.BacktestConfig.from_yaml(args.config)
    src = open_source(args.db)
    if not isinstance(src, LiveDB):
        raise SystemExit("기록 파일(--db)이 필요합니다. 합성 데이터는 1분마다 호가가 한 번뿐이라 지연 효과가 없습니다.")
    model = Forecaster.load(MODEL_PATH) if args.strategy == "signal" else None
    cache: dict = {}

    def decide_fn(ts):   # 같은 시각의 신호는 한 번만 계산
        if ts not in cache:
            if model is None:
                cache[ts] = tightest_atm(src, scfg, ts)
            else:
                out = run_once(src, model, scfg, ts)
                cache[ts] = out[0] if out else None
        return cache[ts]

    quotes = quotes_of(src)
    lo, hi = src.span()
    print(f"[{src.label}] 전략 {args.strategy}, 신호별 백테스트, 보유 {base.horizon_min}분")
    rows = []
    for lat, qty, ticks in CASES:
        cfg = replace(base, latency_sec=lat, qty=qty, entry_ticks=ticks)
        res = backtest(quotes, decide_fn, cfg, lo, hi, each=True)
        s = summarize(res, cfg.multiplier)
        od = res.orders[res.orders["tag"] == "ENTRY"] if len(res.orders) else pd.DataFrame()
        rows.append({
            "지연(초)": lat, "계약": qty, "진입틱": ticks, "주문": s["entry_orders"],
            "전량": s["filled_full"], "부분": s["filled_partial"], "미체결": s["unfilled"],
            "체결수량비율": od["filled"].sum() / od["qty"].sum() if len(od) else None,
            "첫체결(초)": od["first_fill_sec"].median() if len(od) else None,
            "진입슬리피지": s.get("slip_entry_pt"), "청산슬리피지": s.get("slip_exit_pt"),
            "계약당손익(원)": s["net_krw"] / res.trades["qty"].sum() if s["trades"] else None,
        })
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
