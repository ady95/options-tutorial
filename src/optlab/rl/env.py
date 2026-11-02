"""옵션 매매 강화학습 환경(Gymnasium).

에피소드 = 하루(09:15~15:15, 1분마다 결정). 15:15에 남은 포지션은 매수호가에 모두 판다(밤을 넘기지 않음).

행동
- discrete(PPO):   0 대기, 1 ATM 콜 매수, 2 ATM 풋 매수, 3 전부 청산. 포지션이 있을 때의 매수와
                   포지션이 없을 때의 청산은 아무것도 하지 않는다(대기와 같음).
- continuous(SAC): a ∈ [-1, 1] → 목표 포지션 round(a × max_qty)계약. +는 콜, -는 풋.
                   부호가 바뀌면 기존 포지션을 모두 팔고 반대쪽 ATM을 산다.

보상 = (다음 분 중간가 평가액 - 지금 평가액) - 이번 분에 낸 거래비용 - 회전율 패널티
- 거래비용: 매수는 매도호가, 매도는 매수호가에 체결(중간가와의 차이) + 수수료(거래대금 × fee_rate)
- 회전율 패널티: turnover_penalty × 사고판 계약 수. 비용과 달리 손익에는 들어가지 않는 '학습용 벌점'
한 에피소드의 보상 합 = 그날의 비용 차감 손익(포인트, 1계약 기준) - 패널티 합.

체결은 그 분의 호가로 바로, 수량 제한 없이 된다고 둔다. 09장의 지연·잔량·미체결은 넣지 않았다.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from optlab.rl.market import OptionDay
from optlab.strategy.clock import ANNUAL_UNITS

N_OBS = 16


@dataclass
class EnvConfig:
    mode: str = "discrete"          # discrete | continuous
    rel_spread: float = 0.07        # (매도호가-매수호가)/중간가. 실제 기록: 만기일 오전 7%, 월 위클리 12%
    fee_rate: float = 0.001         # 09장과 같은 가정치
    turnover_penalty: float = 0.0   # 계약당 벌점(포인트)
    max_qty: int = 3                # continuous의 최대 계약 수


class OptionTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, days: list[OptionDay], cfg: EnvConfig | None = None, seed: int | None = None):
        self.days, self.cfg = days, cfg or EnvConfig()
        self.observation_space = gym.spaces.Box(-5.0, 5.0, (N_OBS,), np.float32)
        if self.cfg.mode == "discrete":
            self.action_space = gym.spaces.Discrete(4)
        else:
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self._rng = np.random.default_rng(seed)
        self.d: OptionDay | None = None
        self._cache: dict = {}

    # ---------- 포지션 ----------
    def _value(self, i: int) -> float:
        """중간가 평가액(포인트)."""
        return sum(q * self.d.mid(i, k, cp == "C") for cp, k, q, _ in self.lots)

    def _qty(self) -> int:
        q = sum(x[2] for x in self.lots)
        return q if not self.lots or self.lots[0][0] == "C" else -q

    def _fill(self, i: int, cp: str, k: float, q: int) -> float:
        """q > 0 매수, q < 0 매도. 중간가 대비 비용(포인트)을 돌려준다."""
        mid = self.d.mid(i, k, cp == "C")
        bid, ask = self.d.quote(i, k, cp == "C", self.cfg.rel_spread)
        px = ask if q > 0 else bid
        self.fills.append((self.d.times[i], cp, k, q, px))
        self.turnover += abs(q)
        return abs(q) * (abs(px - mid) + self.cfg.fee_rate * px)

    def _trade_to(self, i: int, target: int) -> float:
        cur, cost = self._qty(), 0.0
        if target == cur:
            return 0.0
        if cur != 0:
            keep = abs(target) if np.sign(target) == np.sign(cur) else 0
            total = abs(cur)
            while total > keep:                                    # 마지막에 산 것부터 판다
                cp, k, q, entry = self.lots.pop()
                sell = min(q, total - keep)
                cost += self._fill(i, cp, k, -sell)
                if q > sell:
                    self.lots.append((cp, k, q - sell, entry))
                total -= sell
            cur = self._qty()
        if abs(target) > abs(cur):
            cp, k = ("C" if target > 0 else "P"), self.d.atm(i)
            if not self.lots:
                self.opened_at, self.entries = i, self.entries + 1
            add = abs(target) - abs(cur)
            cost += self._fill(i, cp, k, add)
            self.lots.append((cp, k, add, self.d.quote(i, k, cp == "C", self.cfg.rel_spread)[1]))
        return cost

    def _target(self, action) -> int:
        cur = self._qty()
        if self.cfg.mode == "discrete":
            a = int(action)
            if a in (1, 2) and cur == 0:
                return 1 if a == 1 else -1
            if a == 3:
                return 0
            return cur
        return int(np.round(float(np.asarray(action).ravel()[0]) * self.cfg.max_qty))

    # ---------- 관측 ----------
    def _market(self, d: OptionDay) -> np.ndarray:
        """시장 관측(포지션과 무관한 12개)을 하루치 미리 계산한다."""
        if id(d) not in self._cache:
            r = d.rows
            s1 = d.iv * np.sqrt(1 / ANNUAL_UNITS)                  # IV가 말하는 1분 표준편차
            s15 = s1 * np.sqrt(15)
            cols = [
                (r["p_up"] - 0.5) / 0.1,
                r["q50"] / (0.6745 * s15),     # 예측 |15분 수익률| 중앙값 / IV가 말하는 중앙값
                r["q90"] / (1.645 * s15),
                r["ret_5"] / (s1 * np.sqrt(5)),
                r["ret_15"] / s15,
                r["ret_30"] / (s1 * np.sqrt(30)),
                r["rv_15"] / s15,              # 최근 15분 실현 / IV
                r["ret_from_open"] / (s1 * np.sqrt(390)),
                r["minutes_to_close"] / 420,
                r["is_mon"], r["is_thu"],
                d.T * 252 / 4.0,               # 남은 거래일(변동성 시계)
            ]
            m = np.column_stack([np.asarray(c, dtype=float) for c in cols])
            self._cache[id(d)] = np.clip(np.nan_to_num(m), -5, 5).astype(np.float32)
        return self._cache[id(d)]

    def _obs(self) -> np.ndarray:
        i, q = self.i, self._qty()
        upnl = 0.0
        if self.lots:
            basis = sum(x[2] * x[3] for x in self.lots)
            upnl = (self._value(i) - basis) / basis
        pos = [q / self.cfg.max_qty if self.cfg.mode == "continuous" else float(np.sign(q)),
               np.clip(upnl, -5, 5), min((i - self.opened_at) / 60, 5.0) if self.lots else 0.0,
               min(self.d.iv / 0.5, 5.0)]
        return np.concatenate([self._market(self.d)[i], np.asarray(pos, np.float32)])

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        k = (options or {}).get("day")
        self.d = self.days[k if k is not None else int(self._rng.integers(len(self.days)))]
        self.i, self.lots, self.fills = 0, [], []
        self.turnover, self.entries, self.opened_at, self.pnl, self.cost = 0, 0, 0, 0.0, 0.0
        return self._obs(), {}

    def step(self, action):
        i, before = self.i, self.turnover
        cost = self._trade_to(i, self._target(action))
        v0 = self._value(i)                                        # 거래 직후 평가액
        self.i = i + 1
        last = self.i >= len(self.d) - 1
        v1 = self._value(self.i)
        if last:
            cost += self._trade_to(self.i, 0)                      # 15:15 강제 청산
        pnl = v1 - v0 - cost
        self.pnl += pnl
        self.cost += cost
        reward = pnl - self.cfg.turnover_penalty * (self.turnover - before)
        obs = np.zeros(N_OBS, np.float32) if last else self._obs()
        return obs, float(reward), last, False, {"pnl": pnl}
