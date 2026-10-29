"""8장 실습 2: CALL BUY · PUT BUY · HOLD 시그널 엔진.

    # 한 시점을 자세히(후보별 기대순손익 분해와 필터 사유)
    python examples/ch08/signal_engine.py --db data/ticks_20261029_live.db --time 10:10
    # 기록 구간 전체를 1분마다
    python examples/ch08/signal_engine.py --db data/ticks_20261029_live.db
    # 합성 데이터
    python examples/ch08/signal_engine.py --start 2026-01-08T09:30 --end 2026-01-08T11:30

설정은 configs/default.yaml의 strategy 절. --horizon, --direction-weight, --fee-rate로 덮어쓸 수 있다.
주문은 내지 않는다.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from dataclasses import replace

import pandas as pd

from market import ROOT, LiveDB, market_state, open_source

from optlab.features.build import build_features
from optlab.models.forecast import Forecaster
from optlab.strategy.signal import StrategyConfig, decide

MODEL_PATH = ROOT / "models_store" / "forecaster.joblib"
COLS = ["cp", "strike", "bid", "ask", "age", "spread_ratio", "iv", "reject",
        "model_edge", "entry_cost", "exit_cost", "fees", "exp_pnl", "p_win"]


def run_once(src, fc_model, cfg, ts):
    ch, F, expiry = market_state(src, ts, cfg.risk_free)
    if F is None or ch.empty:
        return None
    fc = next(f for f in fc_model.predict(build_features(src.bars_until(ts))) if f.horizon == cfg.horizon_min)
    return decide(ch, F, ts.to_pydatetime(), expiry, fc, cfg), F, fc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", help="04장 기록 파일. 없으면 합성 데이터")
    p.add_argument("--time", help="한 시점만 자세히: HH:MM(기록) 또는 YYYY-MM-DDTHH:MM(합성)")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--every", type=int, default=1, help="분 간격")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--horizon", type=int, choices=[5, 15, 30])
    p.add_argument("--direction-weight", type=float)
    p.add_argument("--fee-rate", type=float)
    p.add_argument("--check", action="store_true", help="(기록 파일) h분 뒤 실제 매수호가로 청산했다면? 09장 백테스트의 맛보기")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option("display.width", 160)

    cfg = StrategyConfig.from_yaml(args.config)
    over = {"horizon_min": args.horizon, "direction_weight": args.direction_weight, "fee_rate": args.fee_rate}
    cfg = replace(cfg, **{k: v for k, v in over.items() if v is not None})
    src = open_source(args.db)
    model = Forecaster.load(MODEL_PATH)
    parse = (lambda s: pd.Timestamp(f"{src.day} {s}")) if isinstance(src, LiveDB) else pd.Timestamp
    print(f"[{src.label}] 보유 {cfg.horizon_min}분, 방향 반영 {cfg.direction_weight}, 수수료 {cfg.fee_rate:.2%}, "
          f"최소 우위 {cfg.min_edge_points}pt")

    if args.time:
        ts = parse(args.time)
        sig, F, fc = run_once(src, model, cfg, ts)
        print(f"{ts:%Y-%m-%d %H:%M}  F={F:.2f}, 상승확률 {fc.p_up:.3f}, "
              f"예측 변동성 {sig.forecast_vol:.1%} / ATM IV {sig.atm_iv:.1%}")
        tab = sig.table.reindex(columns=COLS).sort_values(["reject", "cp", "strike"])
        print(tab.round(3).to_string(index=False))
        print(f"\n신호: {sig.action}  {sig.reason}")
        if sig.code:
            print(f"  {sig.cp} {sig.strike:g} 매도호가 {sig.entry:.2f}에 매수, 기대손익 {sig.exp_pnl:+.2f}pt, "
                  f"이길 확률 {sig.p_win:.0%}")
        return

    lo, hi = src.span()
    start = max(parse(args.start), lo) if args.start else lo
    end = min(parse(args.end), hi) if args.end else hi
    actions, holds, rejects, rows, checks = Counter(), Counter(), Counter(), [], []
    print(f"{'time':>5} {'F':>8} {'p_up':>5} {'fvol':>6} {'atmIV':>6}  signal")
    for ts in pd.date_range(start, end, freq=f"{args.every}min"):
        out = run_once(src, model, cfg, ts)
        if out is None:
            continue
        sig, F, fc = out
        actions[sig.action] += 1
        if sig.action == "HOLD":
            holds[sig.reason] += 1
        if len(sig.table):
            rejects.update(r or "PASS" for r in sig.table["reject"])
        what = (f"{sig.action} {sig.cp}{sig.strike:g} @{sig.entry:.2f} 기대 {sig.exp_pnl:+.2f}" if sig.code
                else f"HOLD {sig.reason}" + (f" (최선 {sig.exp_pnl:+.2f})" if sig.exp_pnl is not None else ""))
        if args.check and sig.code and isinstance(src, LiveDB):
            later = src.chain(ts + pd.Timedelta(minutes=cfg.horizon_min))
            if ts + pd.Timedelta(minutes=cfg.horizon_min) <= hi and sig.code in set(later["code"]):
                bid = float(later.loc[later["code"] == sig.code, "bid1"].iloc[0])
                pnl = bid - sig.entry - cfg.fee_rate * (sig.entry + bid)
                checks.append({"exp": sig.exp_pnl, "real": pnl})
                what += f" -> {cfg.horizon_min}분 뒤 매수호가 {bid:.2f}, 실제 {pnl:+.2f}"
        fv = f"{sig.forecast_vol:.0%}" if sig.forecast_vol else "-"
        iv = f"{sig.atm_iv:.0%}" if sig.atm_iv else "-"
        print(f"{ts:%H:%M} {F:>8.2f} {fc.p_up:>5.2f} {fv:>6} {iv:>6}  {what}")
        rows.append(sig)
    print(f"\n== 요약 {len(rows)}개 시점: {dict(actions)}")
    print("HOLD 사유:", dict(holds))
    tot = sum(rejects.values())
    print("후보 필터(옵션×시점):", {k: f"{v / tot:.0%}" for k, v in rejects.most_common()})
    if checks:
        c = pd.DataFrame(checks)
        print(f"사후 확인 {len(c)}건: 기대손익 평균 {c['exp'].mean():+.2f}pt, 실제 평균 {c['real'].mean():+.2f}pt, "
              f"실제 이긴 비율 {(c['real'] > 0).mean():.0%}, 실제 범위 {c['real'].min():+.2f} ~ {c['real'].max():+.2f}")


if __name__ == "__main__":
    main()
