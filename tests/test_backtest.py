import json

import pandas as pd
import pytest

from optlab.backtest import engine
from optlab.backtest.fills import Order, match
from optlab.strategy.signal import Signal

T0 = pd.Timestamp("2026-10-29 10:00:00")


def sec(s: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(seconds=s)


def quote(t, bid, ask, bs=1, as_=1, code="X", depth=None):
    return {"code": code, "received_time": t, "bid1": bid, "ask1": ask, "bid_size1": bs, "ask_size1": as_,
            "depth": json.dumps(depth) if depth else None}


def buy_at(ask, code="X"):
    return lambda ts: Signal(ts, "CALL_BUY", "TEST", code=code, cp="C", strike=940.0, entry=ask, exp_pnl=0.0)


def run(rows, decide, **kw):
    cfg = engine.BacktestConfig(horizon_min=1, fee_rate=0.0, **kw)
    q = pd.DataFrame(rows)
    return engine.run(q, decide, cfg, T0 + pd.Timedelta(minutes=1), T0 + pd.Timedelta(minutes=1))


def test_match_walks_levels_up_to_limit():
    q = pd.Series(quote(T0, 5.0, 5.1, depth=[[5.0], [5.1, 5.2, 5.4], [1], [1, 2, 9]]))
    o = Order("BUY", "X", 5, 5.2, T0, T0, T0)
    assert match(o, q, T0) == 3                  # 5.1 × 1 + 5.2 × 2, 5.4는 지정가 초과
    assert o.remaining == 2
    assert o.avg_price == pytest.approx((5.1 + 2 * 5.2) / 3)


def test_order_waits_for_latency_and_misses_when_ask_runs_away():
    rows = [quote(sec(50), 5.0, 5.1),
            quote(sec(60.2), 5.2, 5.3),          # 결정(60초) 직후, 주문 도착(60.5초) 전에 매도호가가 오름
            quote(sec(70), 5.2, 5.3), quote(sec(200), 5.2, 5.3)]
    res = run(rows, buy_at(5.1), latency_sec=0.5)
    assert res.orders.iloc[0]["filled"] == 0     # 5.1 매수 지정가는 체결되지 않음
    assert res.trades.empty


def test_partial_fill_then_exit_chases_bid():
    rows = [quote(sec(50), 5.0, 5.1, as_=1),
            quote(sec(61), 5.0, 5.1, as_=1),        # 두 번째 계약
            quote(sec(65), 5.0, 5.3, as_=5),        # 매도호가가 올라 나머지는 만료 → 2계약 보유, 청산 예정 121초
            quote(sec(100), 5.5, 5.6, bs=1),        # 청산 시각(121초)에 서 있던 매수호가 5.5, 1계약
            quote(sec(122), 5.4, 5.6, bs=1),        # 남은 1계약은 5.5에 안 팔림
            quote(sec(130), 5.4, 5.6, bs=3),        # 만료(126.5초) 뒤 5.4로 재호가해 체결
            quote(sec(300), 5.4, 5.6)]
    res = run(rows, buy_at(5.1), qty=3, latency_sec=0.5, entry_timeout_sec=5, exit_timeout_sec=5)
    o = res.orders
    assert list(o["tag"]) == ["ENTRY", "EXIT", "EXIT"]
    assert o.iloc[0]["filled"] == 2
    t = res.trades.iloc[0]
    assert t["qty"] == 2 and t["entry_px"] == pytest.approx(5.1)
    assert t["exit_px"] == pytest.approx((5.5 + 5.4) / 2)
    assert t["n_exit_orders"] == 2
    assert t["pnl_pt"] == pytest.approx((5.5 + 5.4) - 2 * 5.1)


def test_unsold_position_is_marked_at_last_bid():
    rows = [quote(sec(50), 5.0, 5.1), quote(sec(100), 4.9, 5.6, bs=0),   # 매수 잔량이 없어 팔 수 없음
            quote(sec(121), 4.8, 5.6, bs=0), quote(sec(127), 4.7, 5.6, bs=0)]
    res = run(rows, buy_at(5.1), latency_sec=0.5)
    t = res.trades.iloc[0]
    assert t["status"] == "DATA_END" and t["exit_px"] == pytest.approx(4.7)
