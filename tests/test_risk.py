from datetime import datetime, time, timedelta

import pandas as pd
import pytest

from optlab.execution import replay
from optlab.execution.paper import PaperBroker
from optlab.execution.trader import IllegalTransition, Mode, Phase, Trader, TraderConfig
from optlab.risk.health import HealthMonitor, KillSwitch
from optlab.risk.limits import Exposure, Intent, RiskConfig, check_entry
from optlab.strategy.signal import Signal

T0 = pd.Timestamp("2026-10-29 10:00:00")


# ── 한도 검사 ────────────────────────────────────────────
def intent(**kw):
    base = dict(now=datetime(2026, 10, 29, 10, 0), code="X", qty=1, limit=5.0, delta=0.5)
    return Intent(**{**base, **kw})


@pytest.mark.parametrize("kw, ex, reason", [
    ({}, Exposure(open_qty=1), "MAX_CONTRACTS"),
    ({}, Exposure(pending_qty=1), "MAX_CONTRACTS"),          # 미체결 주문도 한도에 포함
    ({"limit": 8.5}, Exposure(), "MAX_PREMIUM"),             # 8.5 × 250,000 = 212.5만 원
    ({"delta": None}, Exposure(), "NO_DELTA"),
    ({"now": datetime(2026, 10, 29, 15, 0)}, Exposure(), "LATE_ENTRY"),
    ({}, Exposure(day_pnl_krw=-1_000_000), "DAILY_LOSS"),
    ({}, Exposure(recent_orders=[datetime(2026, 10, 29, 9, 59, 30)] * 6), "ORDER_RATE"),
])
def test_check_entry_rejects(kw, ex, reason):
    assert reason in check_entry(intent(**kw), ex, RiskConfig())


def test_check_entry_approves_and_delta_limit():
    assert check_entry(intent(), Exposure(), RiskConfig()) == []
    cfg = RiskConfig(max_contracts=3, max_net_delta=1.0)
    assert "NET_DELTA" in check_entry(intent(qty=1, delta=0.6), Exposure(open_qty=1, net_delta=0.6), cfg)
    assert check_entry(intent(qty=1, delta=-0.5), Exposure(open_qty=1, net_delta=0.6), cfg) == []  # 풋은 델타를 줄인다


def test_expiry_day_switch():
    assert "EXPIRY_DAY" in check_entry(intent(expiry_day=True), Exposure(), RiskConfig(skip_expiry_day=True))
    assert check_entry(intent(expiry_day=True), Exposure(), RiskConfig(skip_expiry_day=False)) == []


# ── 시세 건강 상태 ───────────────────────────────────────
def test_health_heartbeat_latency_and_recovery():
    h = HealthMonitor(RiskConfig(), "F")
    t = datetime(2026, 10, 29, 10, 0)
    assert not h.check(t) and h.reason == "NO_DATA"
    for s in range(11):                                   # 10초 연속 정상이어야 풀린다
        h.on_message("F", t + timedelta(seconds=s), t + timedelta(seconds=s, milliseconds=500))
        ok = h.check(t + timedelta(seconds=s, milliseconds=500))
    assert ok
    assert not h.check(t + timedelta(seconds=16)) and h.reason == "HEARTBEAT_LOST"
    h.on_message("F", t + timedelta(seconds=13), t + timedelta(seconds=17))   # 4초 지연
    assert not h.check(t + timedelta(seconds=17)) and h.reason == "LATENCY"


def test_kill_switch_file(tmp_path):
    k = KillSwitch(tmp_path / "KILL")
    assert not k.poll()
    (tmp_path / "KILL").touch()
    assert k.poll()
    (tmp_path / "KILL").unlink()
    assert k.poll()                                       # 같은 날에는 풀리지 않는다


# ── 상태기계 + 가상 브로커 ───────────────────────────────
def market(until_min=6, ask=5.1, bid=5.0, size=5):
    """선물 F는 0.5초마다, 옵션 X는 2초마다 호가가 오는 장."""
    rows = []
    for i in range(int(until_min * 60 * 2)):
        t = T0 + pd.Timedelta(seconds=i * 0.5)
        rows.append({"code": "F", "received_time": t + pd.Timedelta(seconds=0.3), "exchange_time": t,
                     "bid1": 940.0, "ask1": 940.05, "bid_size1": 5, "ask_size1": 5, "depth": None})
        if i % 4 == 0:
            rows.append({"code": "X", "received_time": t + pd.Timedelta(seconds=0.3), "exchange_time": t,
                         "bid1": bid, "ask1": ask, "bid_size1": size, "ask_size1": size, "depth": None})
    return pd.DataFrame(rows)


def buy_x(when_min=(1,), price=5.1):
    times = {T0 + pd.Timedelta(minutes=m) for m in when_min}
    return lambda ts: (Signal(ts, "CALL_BUY", "T", code="X", cp="C", strike=940.0, entry=price, exp_pnl=0.1,
                              delta=0.5) if ts in times else None)


def make(risk=None, kill=None, **kw):
    b = PaperBroker(0.5)
    t = Trader(risk or RiskConfig(), TraderConfig(**{"horizon_min": 1, **kw}), b, "F", kill=kill)
    return t, b


def kinds(t):
    return [x["kind"] for x in t.log]


def test_normal_lifecycle_reaches_flat():
    t, b = make()
    replay.run(market(), buy_x(), t, b, T0, T0 + pd.Timedelta(minutes=4))
    phases = [x["phase"] for x in t.log if x["kind"] == "PHASE"]
    assert phases == ["SIGNAL_READY", "RISK_APPROVED", "ORDER_SUBMITTED", "POSITION_MONITORED", "EXIT_ORDER", "FLAT"]
    assert len(t.trades) == 1 and t.phase == Phase.FLAT and b.positions() == {}


def test_illegal_transition_raises():
    t, _ = make()
    with pytest.raises(IllegalTransition):
        t._go(Phase.EXIT_ORDER)


def test_same_signal_twice_is_blocked():
    # Ask가 올라 첫 주문이 체결되지 않음 → FLAT. 같은 신호가 다시 오면(재시작·중복 호출) 같은 client_id라 막는다
    t, b = make()
    replay.run(market(ask=5.3), buy_x(), t, b, T0, T0 + pd.Timedelta(minutes=4))
    sig = buy_x()(T0 + pd.Timedelta(minutes=1))
    t.on_signal(T0 + pd.Timedelta(minutes=1), sig)
    assert "DUPLICATE_BLOCKED" in kinds(t)
    assert len(b.orders) == 1


def test_lost_ack_is_resolved_without_resending():
    t, b = make()
    b.fail_next("lost_ack")
    replay.run(market(), buy_x(), t, b, T0, T0 + pd.Timedelta(minutes=4))
    assert "ACK_TIMEOUT" in kinds(t) and "FOUND" in kinds(t)
    assert len([o for o in b.orders.values() if o.order.tag == "ENTRY"]) == 1
    assert len(t.trades) == 1


def test_three_rejects_trigger_risk_stop():
    t, b = make()
    faults = [(T0 + pd.Timedelta(minutes=m, seconds=-1), "REJECT", lambda tr, br: br.fail_next("reject"))
              for m in (1, 2, 3)]
    replay.run(market(), buy_x((1, 2, 3, 4)), t, b, T0, T0 + pd.Timedelta(minutes=4), faults)
    assert t.mode == Mode.RISK_STOP and not t.trades and len(b.orders) == 0


def test_disconnect_missed_fill_is_reconciled():
    # 청산 매도 주문이 도착·체결되는 순간 연결이 끊겨 체결 통지를 잃는다 → 재연결 후 브로커 기록으로 맞춘다
    t, b = make()
    exit_sent = T0 + pd.Timedelta(minutes=2, seconds=0.8)     # 진입 체결(1분 0.8초) + 1분
    faults = [(exit_sent + pd.Timedelta(seconds=0.1), "DOWN", lambda tr, br: br.disconnect()),
              (exit_sent + pd.Timedelta(seconds=20), "UP", lambda tr, br: br.reconnect())]
    replay.run(market(), buy_x(), t, b, T0, T0 + pd.Timedelta(minutes=4), faults)
    assert {"BROKER_DOWN", "RECONCILE", "MISSED_FILLS"} <= set(kinds(t))
    assert len(t.trades) == 1 and t.trades[0]["status"] == "OK" and b.positions() == {}


def test_kill_switch_flattens_and_blocks(tmp_path):
    t, b = make(kill=KillSwitch(tmp_path / "KILL"))
    faults = [(T0 + pd.Timedelta(minutes=1, seconds=20), "KILL", lambda tr, br: (tmp_path / "KILL").touch())]
    replay.run(market(), buy_x((1, 3)), t, b, T0, T0 + pd.Timedelta(minutes=4), faults)
    assert t.mode == Mode.KILLED
    assert len(t.trades) == 1 and t.trades[0]["exit_reason"] == "FORCE_KILLED"
    assert len([o for o in b.orders.values() if o.order.tag == "ENTRY"]) == 1   # 3분 신호는 무시


def test_data_stale_blocks_entry():
    q = market()
    gap = q["received_time"].between(T0 + pd.Timedelta(seconds=50), T0 + pd.Timedelta(seconds=70))
    q = q[~((q["code"] == "F") & gap)]                    # 선물 20초 끊김 → 1분 신호 시점 DATA_STALE
    t, b = make()
    replay.run(q, buy_x(), t, b, T0, T0 + pd.Timedelta(minutes=4))
    assert "DECISION_SKIPPED" in kinds(t) and len(b.orders) == 0


def test_flatten_by_time():
    t, b = make(risk=RiskConfig(flatten_by=time(10, 1, 30)), horizon_min=3)
    replay.run(market(), buy_x(), t, b, T0, T0 + pd.Timedelta(minutes=4))
    assert t.trades[0]["exit_reason"] == "FLATTEN_BY"


def test_matches_chapter9_backtest_without_limits():
    # 위험 한도에 걸리지 않으면, 상태기계 + 가상 브로커의 거래는 9장 백테스트 엔진과 같아야 한다
    from optlab.backtest import engine
    q = market()
    q.loc[q["received_time"] > T0 + pd.Timedelta(minutes=2), ["bid1", "ask1"]] = [5.4, 5.5]
    t, b = make()
    replay.run(q, buy_x(), t, b, T0, T0 + pd.Timedelta(minutes=4))
    bt = engine.run(q, buy_x(), engine.BacktestConfig(horizon_min=1), T0, T0 + pd.Timedelta(minutes=4))
    assert len(t.trades) == len(bt.trades) == 1
    assert t.trades[0]["pnl_pt"] == pytest.approx(bt.trades.iloc[0]["pnl_pt"])
    assert t.trades[0]["entry_px"] == bt.trades.iloc[0]["entry_px"]
