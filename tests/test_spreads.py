"""13장 변동성 관점·스프레드·다리 체결 단위 테스트."""
from datetime import datetime

import numpy as np
import pandas as pd

from optlab.backtest.engine import BacktestConfig
from optlab.backtest.legs import QuoteBook, vertical
from optlab.strategy.spreads import Vertical
from optlab.strategy.volview import HarForecaster, iv_for_window

CFG = BacktestConfig(latency_sec=0.5, entry_timeout_sec=5, exit_timeout_sec=5, horizon_min=1, fee_rate=0.0)


def test_vertical_max_loss_profit_breakeven():
    bull = Vertical("C", 930, 935, debit=2.0)
    assert (bull.max_loss, bull.max_profit, bull.breakeven) == (2.0, 3.0, 932.0)
    assert list(bull.payoff([920, 932, 940])) == [-2.0, 0.0, 3.0]
    bear = Vertical("P", 930, 925, debit=4.35)                 # 13-2절: 호가로 들어가면 최대이익 0.65
    assert round(bear.max_profit, 2) == 0.65 and bear.breakeven == 925.65
    assert Vertical("P", 930, 925, debit=10.3).max_profit < 0  # 폭보다 비싸게 사면 만기에 반드시 잃는다


def test_iv_for_window_is_unchanged_with_flat_profile_and_higher_at_open():
    flat = pd.Series(1.0, index=range(1, 391))
    ts, exp = datetime(2026, 10, 29, 10, 0), datetime(2026, 11, 2, 15, 30)
    assert abs(iv_for_window(0.5, ts, exp, flat) - 0.5) < 1e-12
    morning = flat.copy()
    morning.loc[61:75] = 3.0                                    # 10:01~10:15만 세 배
    assert iv_for_window(0.5, ts, exp, morning) > 0.5


def _quotes(rows):
    return pd.DataFrame(rows, columns=["code", "received_time", "bid1", "ask1", "bid_size1", "ask_size1"]).assign(
        received_time=lambda d: pd.to_datetime(d["received_time"], format="ISO8601"))


def test_second_leg_not_filled_is_unwound():
    q = _quotes([
        ("K1", "2026-10-29 10:00:00", 4.9, 5.0, 5, 5),
        ("K2", "2026-10-29 10:00:00", 2.0, 2.1, 5, 5),
        ("K2", "2026-10-29 10:00:00.6", 1.8, 1.9, 5, 5),          # 첫 다리가 체결되는 사이 매수호가가 내려감
        ("K1", "2026-10-29 10:00:30", 4.8, 4.9, 5, 5),
        ("K2", "2026-10-29 10:20:00", 1.8, 1.9, 5, 5),
    ])
    book = QuoteBook(q)
    t = pd.Timestamp("2026-10-29 10:00:00.1")
    r = vertical(book, "K1", "K2", t, CFG, "together")       # 결정 시각의 매수호가 2.0으로 냈지만 1.8로 내려가 미체결
    assert r["status"] == "UNWIND"
    assert r["pnl_pt"] < 0                                     # 산 다리를 매수호가에 되팔아 스프레드만큼 잃음
    r = vertical(book, "K1", "K2", t, CFG, "sequential")      # 첫 다리 체결 순간의 매수호가(1.8)로 내면 체결
    assert r["status"] == "OK"


def test_har_on_constant_variance_predicts_that_variance():
    rng = np.random.default_rng(0)
    D = 1e-4 * np.exp(rng.normal(0, 0.1, 120))
    day = pd.DataFrame({"intra": D / 2, "night": D / 2, "D": D}, index=pd.date_range("2026-01-01", periods=120).date)
    har = HarForecaster().fit(day)
    assert abs(har.next_day(day) / 1e-4 - 1) < 0.1
    assert abs(har.intra_share - 0.5) < 1e-9
