"""시세 건강 상태 감시와 Kill Switch.

- 선물은 장중 거의 끊김 없이 메시지가 온다(2026-10-29 실측: 메시지 간격 최대 2.1초). 그래서 선물을
  심장 박동(heartbeat)으로 쓴다. 옵션은 몇 분씩 조용한 종목이 있어 연결 상태 판단에 쓸 수 없다(04-3절).
- 수신 지연은 '수신 시각 - 거래소 시각'. 거래소 시각이 초 단위라 0~1초 과대 추정된다(04-2절).
- 한 번 DATA_STALE이 되면 recover_sec 동안 연속으로 정상이어야 풀린다(경계에서 깜빡이지 않게).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from optlab.risk.limits import RiskConfig


class HealthMonitor:
    def __init__(self, cfg: RiskConfig, heartbeat_code: str):
        self.cfg = cfg
        self.code = heartbeat_code
        self.last_beat: datetime | None = None
        self.last_latency = 0.0
        self.stale = True                    # 첫 심장 박동 전에는 신뢰하지 않는다
        self.reason = "NO_DATA"
        self.ok_since: datetime | None = None

    def on_message(self, code: str, exchange_time: datetime | None, received_time: datetime) -> None:
        if exchange_time is not None:
            self.last_latency = (received_time - exchange_time).total_seconds()
        if code == self.code:
            self.last_beat = received_time

    def check(self, now: datetime) -> bool:
        """지금 시세를 믿을 수 있으면 True. 상태가 바뀌면 self.reason에 사유를 남긴다."""
        bad = None
        if self.last_beat is None:
            bad = "NO_DATA"
        elif (now - self.last_beat).total_seconds() > self.cfg.heartbeat_sec:
            bad = "HEARTBEAT_LOST"
        elif self.last_latency > self.cfg.max_quote_age_sec:
            bad = "LATENCY"
        if bad:
            self.stale, self.reason, self.ok_since = True, bad, None
        elif self.stale:
            self.ok_since = self.ok_since or now
            if now - self.ok_since >= timedelta(seconds=self.cfg.recover_sec):
                self.stale, self.reason = False, ""
        return not self.stale


class KillSwitch:
    """사람이 시스템을 멈추는 장치. 파일을 만들거나 engage()를 부르면 걸리고, 같은 날에는 풀리지 않는다."""

    def __init__(self, path: Path | None):
        self.path = path
        self.engaged = False
        self.reason = ""

    def engage(self, reason: str) -> None:
        self.engaged, self.reason = True, self.reason or reason

    def poll(self) -> bool:
        if not self.engaged and self.path is not None and self.path.exists():
            self.engage(f"FILE {self.path.name}")
        return self.engaged
