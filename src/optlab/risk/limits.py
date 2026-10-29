"""독립 위험관리 — 신규 진입 전 한도 검사와 손실 한도 감시.

시그널 엔진(8장)이나 강화학습 정책(14장)이 무엇을 결정하든, 주문은 이 검사를 통과해야만 나간다.
검사는 '거절 사유 목록'을 돌려준다. 사유가 하나라도 있으면 주문하지 않는다.
이 모듈은 예측 모델·시그널 코드를 import하지 않는다(독립성).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime, time
from pathlib import Path

import yaml

from optlab.market_data.spec import MULTIPLIER_KRW


@dataclass
class RiskConfig:
    max_contracts: int = 1                 # 보유 + 미체결 매수 수량 합계 상한
    max_premium_krw: int = 2_000_000       # 주문 1건의 프리미엄(지정가 × 수량 × 승수) 상한
    daily_loss_limit_krw: int = 1_000_000  # 당일 손실(실현 + 평가)이 이만큼이면 RISK_STOP
    max_net_delta: float = 1.0             # |Σ 델타 × 계약 수| 상한(선물 1계약 환산 아님, 옵션 델타 합)
    max_orders_per_min: int = 6            # 최근 60초 주문 수 상한(폭주 방지)
    max_rejects: int = 3                   # 연속 거부가 이만큼이면 RISK_STOP
    max_quote_age_sec: float = 3.0         # 수신 지연(수신 시각 - 거래소 시각) 상한 → DATA_STALE
    heartbeat_sec: float = 5.0             # 선물 메시지가 이만큼 끊기면 DATA_STALE
    recover_sec: float = 10.0              # DATA_STALE 해제: 이만큼 연속으로 정상이어야 함
    no_entry_after: time = time(15, 0)     # 이 시각 이후 신규 진입 금지
    flatten_by: time = time(15, 10)        # 이 시각에는 보유 포지션을 강제 청산
    skip_expiry_day: bool = True           # 만기일 신규 진입 금지(00-2절 안전 원칙)
    kill_file: str = "KILL"                # 이 파일이 생기면 Kill Switch

    @classmethod
    def from_yaml(cls, path: Path) -> "RiskConfig":
        raw = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("risk", {})
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in raw.items() if k in names}
        for k in ("no_entry_after", "flatten_by"):
            if k in kw:
                kw[k] = time.fromisoformat(str(kw[k]))
        return cls(**kw)


@dataclass
class Intent:
    """시그널을 주문으로 옮기기 전의 '하고 싶은 일'."""
    now: datetime
    code: str
    qty: int
    limit: float
    delta: float | None
    expiry_day: bool = False


@dataclass
class Exposure:
    """위험 판단에 필요한 현재 상태(포지션 관리자가 만든다)."""
    open_qty: int = 0              # 보유 수량
    pending_qty: int = 0           # 미체결 매수 수량
    net_delta: float = 0.0         # Σ 델타 × 수량
    day_pnl_krw: float = 0.0       # 실현 + 평가(매수호가 기준)
    recent_orders: list[datetime] = field(default_factory=list)


def check_entry(intent: Intent, ex: Exposure, cfg: RiskConfig) -> list[str]:
    """신규 진입 주문 검사. 빈 목록이면 승인."""
    reasons = []
    if intent.now.time() >= cfg.no_entry_after:
        reasons.append("LATE_ENTRY")
    if cfg.skip_expiry_day and intent.expiry_day:
        reasons.append("EXPIRY_DAY")
    if ex.open_qty + ex.pending_qty + intent.qty > cfg.max_contracts:
        reasons.append("MAX_CONTRACTS")
    if intent.limit * intent.qty * MULTIPLIER_KRW > cfg.max_premium_krw:
        reasons.append("MAX_PREMIUM")
    if intent.delta is None:
        reasons.append("NO_DELTA")      # 델타를 모르면 노출을 판단할 수 없다
    elif abs(ex.net_delta + intent.delta * intent.qty) > cfg.max_net_delta:
        reasons.append("NET_DELTA")
    if ex.day_pnl_krw <= -cfg.daily_loss_limit_krw:
        reasons.append("DAILY_LOSS")
    recent = [t for t in ex.recent_orders if (intent.now - t).total_seconds() < 60]
    if len(recent) >= cfg.max_orders_per_min:
        reasons.append("ORDER_RATE")
    return reasons


def loss_breached(ex: Exposure, cfg: RiskConfig) -> bool:
    return ex.day_pnl_krw <= -cfg.daily_loss_limit_krw
