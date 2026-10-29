"""11장 실습: 실시간 AI 옵션 자동 모의매매.

    # Replay: 04장 기록 파일을 실시간처럼 흘려 전체 파이프라인을 돌린다
    python examples/ch11/paper_trade.py --mode replay --db data/ticks_20261029_live.db --allow-expiry-day
    # 10배속으로(타이머가 도는 Live 경로를 그대로 쓴다)
    python examples/ch11/paper_trade.py --mode replay --db data/ticks_20261029_live.db --allow-expiry-day --speed 10
    # 강제 종료와 재시작
    python examples/ch11/paper_trade.py ... --run-dir runs/crash --crash-at 10:21:59
    python examples/ch11/paper_trade.py ... --run-dir runs/crash --resume --resume-at 10:30:00
    # Live: 장중 KIS 실시간 시세로(주문은 가상 브로커). 끝날 때 남은 포지션은 마지막 매수호가로 평가
    python examples/ch11/paper_trade.py --mode live --minutes 60

설정은 configs/default.yaml(strategy, backtest, risk, run 절). 실계좌 주문은 하지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch08"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch10"))
from market import BARS_PATH, ROOT, LiveDB  # noqa: E402
from signal_engine import MODEL_PATH, run_once  # noqa: E402
from fault_replay import check_invariants  # noqa: E402

from optlab.backtest.engine import BacktestConfig  # noqa: E402
from optlab.execution.paper import PaperBroker  # noqa: E402
from optlab.execution.recovery import catch_up, restore  # noqa: E402
from optlab.execution.scheduler import Scheduler  # noqa: E402
from optlab.execution.trader import Trader, TraderConfig  # noqa: E402
from optlab.live import audit, pipeline  # noqa: E402
from optlab.live.market import MarketBook  # noqa: E402
from optlab.models.forecast import Forecaster  # noqa: E402
from optlab.risk.health import KillSwitch  # noqa: E402
from optlab.risk.limits import RiskConfig  # noqa: E402
from optlab.strategy.signal import StrategyConfig, expiry_datetime  # noqa: E402

MASTER_PATH = ROOT / ".cache" / "fo_idx_code_mts.mst"


def live_setup(args, run_cfg: dict, run_dir: Path):
    """KIS 실시간 구독 목록, 종목표, 만기, 분봉 앞부분(REST)."""
    from optlab.broker.kis import KisClient, KisConfig, load_dotenv_if_present
    from optlab.market_data import master, realtime
    from optlab.market_data.history import fetch_contract
    from optlab.market_data.store import EventStore

    load_dotenv_if_present(ROOT / ".env")
    kis = KisClient(KisConfig.from_env("real"))
    if not MASTER_PATH.exists() or date.fromtimestamp(MASTER_PATH.stat().st_mtime) < date.today():
        master.download_master(MASTER_PATH)          # 종목표는 매일 바뀐다(새 위클리 만기물)
    inst = master.load_master(MASTER_PATH)
    fut = master.nearest_futures(inst)
    product = run_cfg["product"]
    series = master.series_list(inst, product)[0]
    atm = float(kis.index_price("2001")["bstp_nmix_prpr"])
    store = EventStore(run_dir / "ticks.db")
    store.add_instrument(fut.code, fut.product, None, fut.series, None)
    rows = []
    for k, c, p in master.option_chain(inst, product, series, atm, width=1000):   # 그 만기의 전체 행사가
        for o in (c, p):
            store.add_instrument(o.code, o.product, o.cp, o.series, k)
            rows.append({"code": o.code, "cp": o.cp, "strike": k})
    universe = pd.DataFrame(rows)
    center = min(universe["strike"], key=lambda k: abs(k - atm))
    tracker = pipeline.ChainTracker(universe, run_cfg["chain_width"], center, run_cfg["recenter_steps"])
    subs = [("H0IFCNT0", fut.code), ("H0IFASP0", fut.code)]
    subs += [("H0IOASP0", c) for c in tracker.codes(tracker.window(center))]   # 옵션은 호가만
    hist = pd.read_parquet(BARS_PATH)
    if (date.today() - hist["time"].max().date()).days > 4:
        print(f"!! {BARS_PATH.name}이 {hist['time'].max():%Y-%m-%d}까지입니다. 06장 download_futures.py로 갱신하세요")

    def prefix_fn(m: pd.Timestamp) -> pd.DataFrame:
        today = fetch_contract(kis, fut.code, date.today(), date.today())
        bars = pd.concat([hist, today]).drop_duplicates("time", keep="last").sort_values("time")
        return bars[bars["time"] <= m]

    expiry = expiry_datetime(master.weekly_expiry(product, series))
    control: asyncio.Queue = asyncio.Queue()
    feed = pipeline.live_feed(realtime.stream(kis.approval_key(), subs, control=control), store)
    print(f"[Live] {product} {series}(만기 {expiry:%m-%d}), ATM≈{atm:.2f}, 구독 {len(subs)}건, 선물 {fut.code}")
    return feed, universe, fut.code, expiry, prefix_fn, store, tracker, control, subs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--mode", choices=["replay", "live"])
    p.add_argument("--db", help="Replay: 04장 기록 파일")
    p.add_argument("--speed", type=float, default=0, help="Replay 배속. 0이면 기다리지 않음")
    p.add_argument("--bars", choices=["stream", "rest"], default="stream",
                   help="Replay 분봉: stream=체결로 직접 생성(Live와 같음), rest=완성된 분봉 파일(10장과 같음)")
    p.add_argument("--delay", type=float, help="결정 지연(초). 기본은 설정 파일 run.decide_delay_sec")
    p.add_argument("--minutes", type=int, default=30, help="Live 실행 시간(분)")
    p.add_argument("--run-dir")
    p.add_argument("--crash-at", help="Replay: 이 시각(HH:MM:SS)에 정리 없이 종료(강제 종료 흉내)")
    p.add_argument("--resume", action="store_true", help="run-dir의 감사 기록·브로커 상태로 재시작")
    p.add_argument("--resume-at", help="Replay 재시작 시각(HH:MM:SS)")
    p.add_argument("--no-restore", action="store_true", help="(비교용) 재시작하면서 복구를 건너뜀 — 하면 안 되는 방법")
    p.add_argument("--allow-expiry-day", action="store_true", help="만기일 신규 진입 허용(설정 기본값은 금지)")
    p.add_argument("--log", action="store_true", help="PHASE 전이까지 모두 출력")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option("display.width", 200)

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_cfg = raw["run"]
    mode = args.mode or raw.get("mode", "replay")
    delay = run_cfg["decide_delay_sec"] if args.delay is None else args.delay
    scfg, bcfg = StrategyConfig.from_yaml(args.config), BacktestConfig.from_yaml(args.config)
    risk = RiskConfig.from_yaml(args.config)
    if args.allow_expiry_day:
        risk.skip_expiry_day = False
    run_dir = Path(args.run_dir or ROOT / run_cfg["run_dir"] / f"{datetime.now():%Y%m%d}-{mode}")
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path, state_path = run_dir / "audit.jsonl", run_dir / "broker.json"
    if not args.resume and (audit_path.exists() or state_path.exists()):
        raise SystemExit(f"{run_dir}에 이전 실행 기록이 있습니다. --resume으로 이어 가거나 다른 --run-dir을 쓰세요")
    if args.resume and not state_path.exists():
        raise SystemExit(f"{state_path} 없음. 이어 갈 실행이 없습니다")
    model = Forecaster.load(MODEL_PATH)
    store = None

    if mode == "replay":
        if not args.db:
            raise SystemExit("--mode replay에는 --db가 필요합니다")
        src = LiveDB(Path(args.db))
        fut_code = src.inst.loc[src.inst["strike"].isna(), "code"].iloc[0]
        day = src.day
        lo, hi = src.span()
        data_end = src.quotes["received_time"].max()
        book = MarketBook(src.inst, fut_code, lambda m: src.bars[src.bars["time"] <= m])
        ks = sorted(src.inst["strike"].dropna().unique())           # 기록한 체인과 같은 간격으로 넓힌 행사가
        step = min(b - a for a, b in zip(ks, ks[1:]))
        grid = [ks[0] + step * i for i in range(-40, len(ks) + 40)]
        uni = pd.DataFrame({"code": [f"K{k:g}" for k in grid], "cp": "C", "strike": grid})
        tracker = pipeline.ChainTracker(uni, (len(ks) - 1) // 2, ks[len(ks) // 2], run_cfg["recenter_steps"])
        control = subs = None
        expiry = src.expiry
        start = pd.Timestamp(f"{day} {args.resume_at}") if args.resume else src.quotes["received_time"].min()
        stop_at = pd.Timestamp(f"{day} {args.crash_at}") if args.crash_at else None
        hold = pd.Timedelta(minutes=bcfg.horizon_min)
        lat = pd.Timedelta(seconds=bcfg.latency_sec)
        # 10장 재생기와 같은 결정 시각: 기록 첫 분 ~ 마지막 분 중 청산까지 데이터가 남는 시각
        decisions = [t for t in pipeline.minute_marks(lo, hi, delay) if t > start and t + hold + lat <= data_end]
        if args.bars == "rest":
            decide_fn = lambda ts: (run_once(src, model, scfg, ts) or [None])[0]  # noqa: E731
        else:
            decide_fn = pipeline.make_decider(book, model, scfg, expiry, scfg.risk_free)
        fut_trades = src.trades[src.trades["code"] == fut_code]
        sim = pipeline.SimClock(start, args.speed) if args.speed > 0 else None
        feed = pipeline.replay_feed(src.quotes, fut_trades, after=start if args.resume else None, sim=sim)
        wall = sim
        label = f"Replay {src.label}" + (f" {args.speed:g}배속" if sim else "")
    else:
        feed, inst, fut_code, expiry, prefix_fn, store, tracker, control, subs = live_setup(args, run_cfg, run_dir)
        book = MarketBook(inst, fut_code, prefix_fn)
        now = pd.Timestamp.now()
        start, stop_at = now, now + pd.Timedelta(minutes=args.minutes)
        decisions = pipeline.minute_marks(now, stop_at, delay)
        decide_fn = pipeline.make_decider(book, model, scfg, expiry, scfg.risk_free)
        wall = pd.Timestamp.now
        label = "Live"

    broker = PaperBroker.load(state_path) if args.resume else PaperBroker(bcfg.latency_sec, state_path)
    lost = 0
    if args.resume and mode == "replay":
        lost = catch_up(broker, src.quotes, start)
    tcfg = TraderConfig(qty=bcfg.qty, horizon_min=bcfg.horizon_min, entry_timeout_sec=bcfg.entry_timeout_sec,
                        exit_timeout_sec=bcfg.exit_timeout_sec, entry_ticks=bcfg.entry_ticks, fee_rate=bcfg.fee_rate)
    trader = Trader(risk, tcfg, broker, fut_code, expiry.date(), KillSwitch(Path.cwd() / risk.kill_file))
    log = audit.AuditLog(audit_path, f"{datetime.now():%H%M%S}")
    records = audit.read(audit_path)
    trader.sink = log.write
    trader.now = start
    trader._log("RUN_START", f"{label}, 결정 지연 {delay:g}초, 만기일 진입 {'허용' if not risk.skip_expiry_day else '금지'}")
    if args.resume:
        if lost:
            trader._log("LOST_NOTICES", f"꺼져 있던 동안의 통지 {lost}건")
        if args.no_restore:
            trader._log("RESTORE_SKIPPED", "감사 기록과 브로커 기록을 보지 않고 빈 상태로 시작")
        else:
            restore(trader, broker, records, start)
    print(f"[{label}] 실행 폴더 {run_dir}")

    sched = Scheduler(trader, broker, decide_fn, decisions, start=start)
    try:
        on_minute = pipeline.resubscriber(tracker, book, lambda: trader, control, subs)
        why = asyncio.run(pipeline.run(feed, book, sched, wall_clock=wall, stop_at=stop_at, on_minute=on_minute))
    except KeyboardInterrupt:
        why = "KEYBOARD"
    end = pd.Timestamp.now() if mode == "live" else (stop_at if why == "STOP_AT" else data_end)
    if mode == "replay" and why == "STOP_AT":
        trader.now = stop_at
        trader._log("CRASH", "정리 없이 종료")
        log.close()
        show(trader, broker, risk, sched, book, args.log)
        print(f"\n!! {stop_at:%H:%M:%S}에 강제 종료(finish 없음). 이어 가기: --resume --resume-at HH:MM:SS")
        sys.exit(3)
    trader.finish(end)
    trader._log("RUN_END", why)
    log.close()
    if store is not None:
        store.close()
    show(trader, broker, risk, sched, book, args.log)


def show(trader, broker, risk, sched, book, full: bool) -> None:
    log = pd.DataFrame(trader.log)
    log["time"] = pd.to_datetime(log["time"]).dt.strftime("%H:%M:%S.%f").str[:12]
    print((log if full else log[log["kind"] != "PHASE"]).to_string(index=False, max_colwidth=90))
    tr = pd.DataFrame(trader.trades)
    if len(tr):
        print()
        print(tr[["signal_time", "cp", "strike", "qty", "entry_px", "exit_px", "n_exit_orders", "exit_reason",
                  "pnl_krw", "status"]].assign(signal_time=pd.to_datetime(tr["signal_time"]).dt.strftime("%H:%M:%S"))
              .round(2).to_string(index=False))
    d = pd.Series(sched.decide_sec)
    bad = check_invariants(trader, broker, risk.max_contracts)
    print(f"\n== 요약: 거래 {len(tr)}건, 손익 {tr['pnl_krw'].sum() if len(tr) else 0:+,.0f}원, "
          f"최종 상태 {trader.mode.value}, 브로커 잔고 {sum(broker.position.values())}, "
          f"규칙 위반 {'; '.join(bad) or '없음'}")
    if len(d):
        print(f"   결정 {len(d)}번, 계산 시간 중앙값 {d.median() * 1000:.0f}ms / 최대 {d.max() * 1000:.0f}ms, "
              f"결정 시각에 직전 분봉 미완성 {book.late_bars}번")


if __name__ == "__main__":
    main()
