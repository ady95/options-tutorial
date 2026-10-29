"""10장 실습: 기록된 호가로 시그널 → 위험관리 → 주문 상태기계 → 가상 브로커를 돌리고, 장애를 주입한다.

    python examples/ch10/fault_replay.py --db data/ticks_20261029_live.db --scenario normal
    python examples/ch10/fault_replay.py --db data/ticks_20261029_live.db --scenario all
    python examples/ch10/fault_replay.py --db data/ticks_20261029_live.db --scenario loss_limit --log

시나리오: default, normal, no_limits, data_stale, latency, disconnect, lost_ack, reject, loss_limit, kill, all
기록 파일이 위클리 만기일이므로 default를 뺀 시나리오는 만기일 신규 진입을 허용한다(설정 기본값은 금지).
주문은 내지 않는다(가상 브로커).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch08"))
from market import ROOT, LiveDB  # noqa: E402
from signal_engine import MODEL_PATH, run_once  # noqa: E402

from optlab.backtest.engine import BacktestConfig  # noqa: E402
from optlab.execution import replay  # noqa: E402
from optlab.execution.paper import PaperBroker  # noqa: E402
from optlab.execution.trader import Mode, Trader, TraderConfig  # noqa: E402
from optlab.models.forecast import Forecaster  # noqa: E402
from optlab.risk.health import KillSwitch  # noqa: E402
from optlab.risk.limits import RiskConfig  # noqa: E402
from optlab.strategy.signal import StrategyConfig  # noqa: E402

HEARTBEAT = "A01612"   # 선물 최근월물(04장 기록에 함께 구독)


def at(day, hhmmss: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmmss}")


def scenario(name: str, day, quotes: pd.DataFrame, kill_path: Path):
    """(호가, 위험 설정 변경, 장애 목록)."""
    q, risk, faults = quotes, {"skip_expiry_day": False}, []
    if name == "default":             # 설정 파일 그대로(만기일 진입 금지)
        risk = {}
    elif name == "no_limits":           # 9장과 같은 조건(한도 해제) → 9장과 같은 거래가 나와야 한다
        risk.update(max_premium_krw=10**9, max_net_delta=10.0, daily_loss_limit_krw=10**12)
    elif name == "data_stale":        # 청산 직전 선물 시세가 90초 끊김
        t0, t1 = at(day, "10:20:30"), at(day, "10:22:00")
        q = q[~((q["code"] == HEARTBEAT) & q["received_time"].between(t0, t1))]
    elif name == "latency":           # 1분 동안 모든 메시지가 4초 늦게 도착(거래소 시각을 4초 당겨 흉내)
        t0, t1 = at(day, "10:25:30"), at(day, "10:26:30")
        m = q["received_time"].between(t0, t1)
        q = q.copy()
        q.loc[m, "exchange_time"] = q.loc[m, "exchange_time"] - pd.Timedelta(seconds=4)
    elif name == "disconnect":        # 재호가한 청산 주문이 체결되기 직전 브로커 연결이 끊김(체결 통지 유실)
        faults = [(at(day, "10:21:06.200"), "BROKER_DISCONNECT", lambda t, b: b.disconnect()),
                  (at(day, "10:21:30"), "BROKER_RECONNECT", lambda t, b: b.reconnect())]
    elif name == "lost_ack":          # 첫 진입 주문의 응답이 사라짐(주문은 접수됨)
        faults = [(at(day, "10:05:59"), "LOST_ACK_NEXT", lambda t, b: b.fail_next("lost_ack"))]
    elif name == "reject":            # 첫 진입 주문이 거부됨
        faults = [(at(day, "10:05:59"), "REJECT_NEXT", lambda t, b: b.fail_next("reject"))]
    elif name == "loss_limit":        # 일손실 한도를 50만 원으로 낮춤
        risk["daily_loss_limit_krw"] = 500_000
    elif name == "kill":              # 보유 중 Kill Switch 파일 생성
        faults = [(at(day, "10:12:00"), "KILL_FILE", lambda t, b: kill_path.touch())]
    return q, risk, faults


def check_invariants(trader: Trader, broker: PaperBroker, max_contracts: int) -> list[str]:
    """어떤 장애에서도 지켜져야 하는 규칙."""
    bad = []
    if len(broker.by_client) != len(broker.orders):
        bad.append("같은 client_id로 주문이 두 번 접수됨")
    log = pd.DataFrame(trader.log)
    entries = log[(log["kind"] == "ORDER") & log["detail"].str.contains(" BUY ")]
    if (entries["mode"] != Mode.DATA_READY.value).any():
        bad.append("DATA_READY가 아닌 상태에서 신규 진입")
    pos, peak = 0, 0
    for o in sorted(broker.orders.values(), key=lambda x: x.order.sent):
        for _, qty, _ in o.order.fills:
            pos += qty if o.order.side == "BUY" else -qty
            peak = max(peak, pos)
    if peak > max_contracts:
        bad.append(f"보유 수량 {peak} > 한도 {max_contracts}")
    return bad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--scenario", default="normal")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--log", action="store_true", help="상태 변화 로그 전체 출력")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option("display.width", 200)

    src = LiveDB(Path(args.db))
    scfg = StrategyConfig.from_yaml(args.config)
    bcfg = BacktestConfig.from_yaml(args.config)
    model = Forecaster.load(MODEL_PATH)
    cache: dict = {}

    def decide_fn(ts):
        if ts not in cache:
            out = run_once(src, model, scfg, ts)
            cache[ts] = out[0] if out else None
        return cache[ts]

    names = (["default", "normal", "no_limits", "data_stale", "latency", "disconnect", "lost_ack", "reject", "loss_limit",
              "kill"] if args.scenario == "all" else [args.scenario])
    lo, hi = src.span()
    rows = []
    for name in names:
        with tempfile.TemporaryDirectory() as tmp:
            kill_path = Path(tmp) / "KILL"
            q, risk_over, faults = scenario(name, src.day, src.quotes, kill_path)
            risk = RiskConfig.from_yaml(args.config)
            for k, v in risk_over.items():
                setattr(risk, k, v)
            broker = PaperBroker(bcfg.latency_sec)
            tcfg = TraderConfig(qty=bcfg.qty, horizon_min=bcfg.horizon_min, entry_timeout_sec=bcfg.entry_timeout_sec,
                                exit_timeout_sec=bcfg.exit_timeout_sec, entry_ticks=bcfg.entry_ticks,
                                fee_rate=bcfg.fee_rate)
            trader = Trader(risk, tcfg, broker, HEARTBEAT, src.expiry.date(), KillSwitch(kill_path))
            replay.run(q, decide_fn, trader, broker, lo, hi, faults)
        log = pd.DataFrame(trader.log)
        log["time"] = pd.to_datetime(log["time"]).dt.strftime("%H:%M:%S.%f").str[:12]
        print(f"\n######## {name}")
        show = log if args.log or len(names) == 1 else log[log["kind"] != "PHASE"]
        print(show.to_string(index=False, max_colwidth=70))
        tr = pd.DataFrame(trader.trades)
        if len(tr):
            print(tr[["signal_time", "cp", "strike", "qty", "entry_px", "exit_px", "n_exit_orders", "exit_reason",
                      "pnl_krw", "status"]].assign(signal_time=tr["signal_time"].dt.strftime("%H:%M"))
                  .round(2).to_string(index=False))
        bad = check_invariants(trader, broker, risk.max_contracts)
        rejects = log[log["detail"].str.startswith("RISK_REJECT", na=False)]["detail"]
        rows.append({"시나리오": name, "거래": len(tr), "손익(원)": round(tr["pnl_krw"].sum()) if len(tr) else 0,
                     "위험거절": len(rejects), "최종 상태": trader.mode.value,
                     "브로커 잔고": sum(broker.position.values()), "규칙 위반": "; ".join(bad) or "없음"})
    print("\n== 요약")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
