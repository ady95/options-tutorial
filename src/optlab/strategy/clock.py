"""변동성 시계 — 옵션의 시간가치가 '소모되는 시간'을 장중 분 단위로 잰다.

달력 시간으로 잔존만기를 재면 밤과 주말도 장중과 같은 속도로 흐른다고 가정하게 된다.
실제로 지수는 장중에 움직이고, 밤사이에는 다음 날 시가 갭 한 번으로 움직인다.
그래서 잔존만기를 다음 두 가지의 합으로 잰다.
- 남은 장중 시간(분): 코스피200 지수 정규장 09:00~15:30
- 남은 밤 수 × OVERNIGHT_MINUTES: 밤사이 갭 하나의 분산을 '장중 몇 분어치'로 환산한 값

OVERNIGHT_MINUTES=400은 2025-12~2026-10 선물 1분봉에서 잰 값이다(전일 종가→시가 갭 분산 ÷ 장중 1분 분산).
공휴일은 고려하지 않는다.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

SESSION_START = time(9, 0)
SESSION_END = time(15, 30)
SESSION_MINUTES = 390
OVERNIGHT_MINUTES = 400.0
ANNUAL_UNITS = 252 * (SESSION_MINUTES + OVERNIGHT_MINUTES)   # 1년 = 252거래일 × (장중 + 밤 1회)


def variance_minutes(start: datetime, end: datetime, overnight: float = OVERNIGHT_MINUTES) -> float:
    """start~end 사이의 변동성 시계 길이(장중 분 환산)."""
    if end <= start:
        return 0.0
    session, nights = 0.0, 0
    d = start.date()
    while d <= end.date():
        if d.weekday() < 5:
            open_, close = datetime.combine(d, SESSION_START), datetime.combine(d, SESSION_END)
            s, e = max(start, open_), min(end, close)
            if e > s:
                session += (e - s).total_seconds() / 60
            if start < open_ <= end:   # 이 장이 열리기 전의 밤(주말 포함)이 구간 안에 있다
                nights += 1
        d += timedelta(days=1)
    return session + nights * overnight


def years(start: datetime, end: datetime, overnight: float = OVERNIGHT_MINUTES) -> float:
    """옵션 공식에 넣을 잔존만기(연). IV도 이 시계로 계산해야 재평가가 일관된다."""
    return variance_minutes(start, end, overnight) / ANNUAL_UNITS
