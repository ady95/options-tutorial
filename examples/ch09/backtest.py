"""9장 실습: 시그널 엔진을 호가 기반 이벤트 백테스트로 검증한다.

    # 04장 기록 파일 + 8장 시그널 엔진
    python examples/ch09/backtest.py --db data/ticks_20261029_live.db
    python examples/ch09/backtest.py --db data/ticks_20261029_live.db --horizon 5 --latency 1.0
    # 비교 기준: 예측 없이 스프레드가 가장 좁은 ATM 옵션을 매번 산다
    python examples/ch09/backtest.py --db data/ticks_20261029_live.db --strategy tightest
    # 합성 데이터(Replay Mode)
    python examples/ch09/backtest.py --strategy tightest --start 2026-01-05T09:00 --end 2026-01-05T15:00

설정은 configs/default.yaml의 strategy·backtest 절. 주문은 내지 않는다(가상체결).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch08"))
from market import ROOT, LiveDB, Synthetic, market_state, open_source  # noqa: E402
from signal_engine import MODEL_PATH, run_once  # noqa: E402

from optlab.backtest import engine  # noqa: E402
from optlab.backtest.report import summarize  # noqa: E402
from optlab.broker.replay import synthetic_code  # noqa: E402
from optlab.models.forecast import Forecaster  # noqa: E402
from optlab.strategy.signal import Signal, StrategyConfig  # noqa: E402

COLS = ["signal_time", "cp", "strike", "qty", "sig_mid", "sig_ask", "entry_px", "exit_mid", "exit_bid",
        "exit_px", "n_exit_orders", "exp_pnl", "pnl_pt", "status"]


def quotes_of(src) -> pd.DataFrame:
    if isinstance(src, LiveDB):
        return src.quotes
    o = src.opt   # 합성 데이터: 1분마다 체인 전체가 한 번씩 갱신된다
    return pd.DataFrame({"code": [synthetic_code(cp, k) for cp, k in zip(o["cp"], o["strike"])],
                         "received_time": o["time"], "bid1": o["bid"], "ask1": o["ask"],
                         "bid_size1": o["bid_size"], "ask_size1": o["ask_size"]})


def tightest_atm(src, cfg: StrategyConfig, ts: pd.Timestamp) -> Signal:
    """비교 기준 전략: 예측을 쓰지 않고, F에 가장 가까운 행사가의 콜·풋 중 스프레드가 좁은 쪽을 산다."""
    if (ts + timedelta(minutes=cfg.horizon_min)).time() > cfg.exit_cutoff:
        return Signal(ts, "HOLD", "NEAR_CLOSE")
    ch, F, _ = market_state(src, ts, cfg.risk_free)
    if F is None:
        return Signal(ts, "HOLD", "NO_CANDIDATE")
    c = ch[(ch["bid1"] > 0) & (ch["ask1"] > 0) & (ch["age"] <= cfg.max_age_sec)]
    c = c[(c["strike"] - F).abs() == (c["strike"] - F).abs().min()]
    if c.empty:
        return Signal(ts, "HOLD", "NO_CANDIDATE")
    o = c.loc[(c["ask1"] - c["bid1"]).idxmin()]
    return Signal(ts, "CALL_BUY" if o["cp"] == "C" else "PUT_BUY", "BASELINE", code=o["code"], cp=o["cp"],
                  strike=float(o["strike"]), entry=float(o["ask1"]), exp_pnl=float("nan"))


def backtest(quotes: pd.DataFrame, decide_fn, cfg: engine.BacktestConfig, start, end, each: bool = False):
    """each=False: 포지션 하나로 운용(보유 중에는 새 신호를 받지 않는다).
    each=True: 매분 신호를 하나씩 따로 백테스트해 합친다(포지션이 겹친다). 신호의 질을 보는 용도."""
    if not each:
        return engine.run(quotes, decide_fn, cfg, start, end)
    t = quotes["received_time"]
    pad = pd.Timedelta(minutes=cfg.horizon_min + 5)
    parts = []
    for ts in pd.Series(t.dt.ceil("min").unique()):
        if start <= ts <= end:
            win = quotes[(t >= ts - pd.Timedelta(minutes=10)) & (t <= ts + pad)]
            parts.append(engine.run(win, decide_fn, cfg, ts, ts))
    return engine.Result(*(pd.concat([x for r in parts if len(x := getattr(r, f))] or [pd.DataFrame()],
                                     ignore_index=True) for f in ("trades", "orders", "decisions")))


def last_trade_pnl(trades: pd.DataFrame, ticks: pd.DataFrame, fee_rate: float) -> pd.Series:
    """같은 거래를 '마지막 체결가'로 사고팔았다고 계산하면? (체결가 백테스트, 09-1절)"""
    out = []
    for r in trades.itertuples():
        t = ticks[ticks["code"] == r.code]
        buy = t.loc[t["received_time"] <= r.signal_time, "price"]
        sell = t.loc[t["received_time"] <= r.exit_due, "price"]
        if buy.empty or sell.empty:
            out.append(float("nan"))
            continue
        b, s = buy.iloc[-1], sell.iloc[-1]
        out.append((s - b - fee_rate * (b + s)) * r.qty)
    return pd.Series(out, index=trades.index, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", help="04장 기록 파일. 없으면 합성 데이터")
    p.add_argument("--strategy", choices=["signal", "tightest"], default="signal")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--start", help="HH:MM(기록) 또는 YYYY-MM-DDTHH:MM(합성)")
    p.add_argument("--end")
    p.add_argument("--horizon", type=int, choices=[5, 15, 30])
    p.add_argument("--qty", type=int)
    p.add_argument("--latency", type=float, help="주문 지연(초)")
    p.add_argument("--entry-ticks", type=int, help="매수 지정가를 매도호가보다 N틱 높게")
    p.add_argument("--direction-weight", type=float)
    p.add_argument("--each", action="store_true", help="신호마다 따로 백테스트(포지션 겹침 허용)")
    p.add_argument("--save", help="거래 내역 CSV 경로")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    pd.set_option("display.width", 200)

    scfg = StrategyConfig.from_yaml(args.config)
    bcfg = engine.BacktestConfig.from_yaml(args.config)
    s_over = {"horizon_min": args.horizon, "direction_weight": args.direction_weight}
    b_over = {"horizon_min": args.horizon, "qty": args.qty, "latency_sec": args.latency,
              "entry_ticks": args.entry_ticks}
    scfg = replace(scfg, **{k: v for k, v in s_over.items() if v is not None})
    bcfg = replace(bcfg, **{k: v for k, v in b_over.items() if v is not None})

    src = open_source(args.db)
    parse = (lambda s: pd.Timestamp(f"{src.day} {s}")) if isinstance(src, LiveDB) else pd.Timestamp
    lo, hi = src.span()
    start = max(parse(args.start), lo) if args.start else lo
    end = min(parse(args.end), hi) if args.end else hi
    print(f"[{src.label}] {start:%m-%d %H:%M}~{end:%m-%d %H:%M}, 전략 {args.strategy}{' (신호별)' if args.each else ''}, 보유 {bcfg.horizon_min}분, "
          f"{bcfg.qty}계약, 지연 {bcfg.latency_sec}초, 진입 +{bcfg.entry_ticks}틱, 수수료 {bcfg.fee_rate:.2%}")

    if args.strategy == "signal":
        model = Forecaster.load(MODEL_PATH)

        def decide_fn(ts):
            out = run_once(src, model, scfg, ts)
            return out[0] if out else None
    else:
        def decide_fn(ts):
            return tightest_atm(src, scfg, ts)

    quotes = quotes_of(src)
    res = backtest(quotes, decide_fn, bcfg, start, end, args.each)
    acts = res.decisions["action"].value_counts().to_dict() if len(res.decisions) else {}
    print(f"결정 {len(res.decisions)}회: {acts}")

    s = summarize(res, bcfg.multiplier)
    print(f"진입 주문 {s['entry_orders']}건: 전량 체결 {s['filled_full']}, 부분 체결 {s['filled_partial']}, "
          f"미체결 {s['unfilled']}")
    if not s["trades"]:
        print("거래 없음")
        return
    tr = res.trades
    tr["last_pnl"] = last_trade_pnl(tr, src.trades, bcfg.fee_rate) if isinstance(src, LiveDB) else float("nan")
    print("\n== 거래 내역(가격은 1계약 포인트, pnl_pt는 수량·수수료 반영)")
    view = tr[COLS + ["last_pnl"]].assign(signal_time=tr["signal_time"].dt.strftime("%m-%d %H:%M"))
    print(view.round(3).to_string(index=False))

    m = bcfg.multiplier
    print(f"\n== 요약 {s['trades']}건 (데이터 끝까지 청산 못 함 {s['data_end']}건)")
    print(f"평균 스프레드 비용: 진입 {s['spread_entry_pt']:.3f}pt, 청산 {s['spread_exit_pt']:.3f}pt")
    print(f"평균 슬리피지(결정 시각 호가 대비, +면 불리): 진입 {s['slip_entry_pt']:+.3f}pt, "
          f"청산 {s['slip_exit_pt']:+.3f}pt")
    print(f"  중간가로 사고팔았다면(수수료 제외) {s['gross_mid_pt'] * m:>+12,.0f}원")
    if tr["last_pnl"].notna().any():
        print(f"  마지막 체결가로 계산하면          {tr['last_pnl'].sum() * m:>+12,.0f}원")
    print(f"  호가·잔량·지연 반영(수수료 포함)   {s['net_krw']:>+12,.0f}원  (수수료 {s['fees_krw']:,.0f}원)")
    print(f"승률 {s['win_rate']:.0%}, 거래당 평균 {s['avg_krw']:+,.0f}원, MDD {s['mdd_krw']:,.0f}원")
    if args.strategy == "signal":
        print(f"기대손익 평균 {s['exp_pnl_pt']:+.3f}pt vs 실제 {s['real_pnl_pt']:+.3f}pt (1계약)")
    if args.save:
        tr.to_csv(args.save, index=False)
        print(f"저장: {args.save}")


if __name__ == "__main__":
    main()
