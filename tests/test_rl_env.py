"""14장 강화학습 환경 단위 테스트."""
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from optlab.rl.env import EnvConfig, OptionTradingEnv  # noqa: E402
from optlab.rl.market import OptionDay, next_weekly_expiry  # noqa: E402
from optlab.rl.policies import evaluate, round_trips  # noqa: E402
from optlab.strategy import clock  # noqa: E402


def _day(n=60, seed=0, flat=False):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-10-27 09:15", periods=n, freq="min")
    F = np.full(n, 930.0) if flat else 930 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    expiry = datetime(2026, 10, 29, 15, 30)
    T = np.array([clock.years(t.to_pydatetime(), expiry) for t in times])
    rows = pd.DataFrame({"time": times, "p_up": 0.5, "q50": 0.001, "q90": 0.002, "ret_5": 0.0, "ret_15": 0.0,
                         "ret_30": 0.0, "rv_15": 0.002, "ret_from_open": 0.0, "minutes_to_close": 100,
                         "is_mon": 0, "is_thu": 0})
    return OptionDay(date(2026, 10, 27), times.to_numpy(), F, T, 0.5, expiry, rows)


def _run(env, actions):
    env.reset(options={"day": 0})
    total, done, k = 0.0, False, 0
    while not done:
        _, r, done, _, _ = env.step(actions[k] if k < len(actions) else 0)
        total, k = total + r, k + 1
    return total


def test_next_weekly_expiry_skips_today():
    assert next_weekly_expiry(date(2026, 10, 29)) == date(2026, 11, 2)   # 목 → 다음 월
    assert next_weekly_expiry(date(2026, 10, 30)) == date(2026, 11, 2)   # 금 → 월
    assert next_weekly_expiry(date(2026, 11, 2)) == date(2026, 11, 5)    # 월 → 목


def test_reward_sum_equals_round_trip_cash():
    """보상의 합(패널티 0) = 체결가로 계산한 실현손익. 중간가 평가는 도중의 배분일 뿐이다."""
    env = OptionTradingEnv([_day()], EnvConfig())
    rng = np.random.default_rng(1)
    total = _run(env, list(rng.integers(0, 4, 60)))
    assert env.entries > 3
    assert abs(total - sum(x["pnl_pt"] for x in round_trips(env.fills, env.cfg.fee_rate))) < 1e-9
    assert not env.lots                                                  # 15:15 강제 청산


def test_immediate_round_trip_costs_spread_and_fees():
    d = _day(flat=True)
    env = OptionTradingEnv([d], EnvConfig(rel_spread=0.1, fee_rate=0.001))
    total = _run(env, [1, 3])
    bid, ask = d.quote(0, d.atm(0), True, 0.1)
    theta = d.mid(0, 930.0, True) - d.mid(1, 930.0, True)               # 1분 시간가치 감소
    bid1, _ = d.quote(1, d.atm(0), True, 0.1)
    assert abs(total - (bid1 - ask - 0.001 * (ask + bid1))) < 1e-9
    assert ask - bid > 0.09 * d.mid(0, 930.0, True) and theta > 0


def test_invalid_discrete_actions_are_noops_and_hold_is_zero():
    env = OptionTradingEnv([_day()], EnvConfig())
    _run(env, [1, 1, 2, 1])                                              # 보유 중 매수는 무시
    assert env.entries == 1 and env.turnover == 2
    assert _run(env, [3, 0, 3]) == 0.0 and env.turnover == 0            # 포지션 없을 때 청산도 무시


def test_continuous_target_and_sign_flip():
    env = OptionTradingEnv([_day()], EnvConfig(mode="continuous", max_qty=3))
    env.reset(options={"day": 0})
    for a, want in [(1.0, 3), (0.4, 1), (-0.5, -2), (0.0, 0)]:
        env.step(np.array([a]))
        assert env._qty() == want
    assert env.turnover == 3 + 2 + (1 + 2) + 2


def test_turnover_penalty_changes_reward_not_pnl():
    acts = [1, 0, 3, 2, 0, 3]
    a = OptionTradingEnv([_day()], EnvConfig())
    b = OptionTradingEnv([_day()], EnvConfig(turnover_penalty=0.5))
    ra, rb = _run(a, acts), _run(b, acts)
    assert abs(a.pnl - b.pnl) < 1e-12
    assert abs((ra - rb) - 0.5 * b.turnover) < 1e-9


def test_evaluate_reports_krw():
    env = OptionTradingEnv([_day(), _day(seed=2)], EnvConfig())
    per_day, trades = evaluate(env, lambda e: 1 if not e.lots else 0)
    assert len(per_day) == 2 and len(trades) == 2
    assert abs(per_day["pnl_krw"].sum() - trades["pnl_krw"].sum()) < 1e-6
