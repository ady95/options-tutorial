"""코스피200 선물 과거 1분봉 — 최근월 연속 시계열 만들기.

KIS 선물옵션 분봉조회(FHKIF03020200)는 한 번에 최대 102개 분봉을 최신→과거 순으로 돌려준다.
기준 일시를 가장 오래된 분봉 직전으로 옮겨 가며 반복 호출한다.

2026-10 실측:
- 만기가 지난 월물도 조회된다(A01609 → 만기일 2026-09-10까지).
- A01603보다 이전 월물(A01512 등)은 빈 응답 → 연속 시계열은 2025-12 중순부터 만들 수 있다.
- 최근월이 아닌 기간에는 거래가 드물어 분봉이 띄엄띄엄 있다 → 월물마다 최근월이었던 기간만 쓴다.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from optlab.broker.kis import KisClient, KisError

PATH = "/uapi/domestic-futureoption/v1/quotations/inquire-time-fuopchartprice"


def quarterly_expiry(year: int, month: int) -> date:
    """코스피200 선물 만기 = 3·6·9·12월 두 번째 목요일."""
    first = date(year, month, 1)
    return first + timedelta(days=(3 - first.weekday()) % 7 + 7)


def front_month_schedule(start: date, end: date) -> list[tuple[str, date, date]]:
    """(종목코드, 시작일, 종료일) 목록. 각 월물은 직전 월물 만기 다음 날부터 자기 만기일까지 최근월이다."""
    out = []
    y, m = start.year, ((start.month - 1) // 3 + 1) * 3
    prev_exp = None
    while True:
        exp = quarterly_expiry(y, m)
        if exp >= start:
            s = max(start, prev_exp + timedelta(days=1)) if prev_exp else start
            out.append((f"A01{y % 10}{m:02d}", s, min(exp, end)))   # A01612 = 2026년 12월물
            if exp >= end:
                break
        prev_exp = exp
        y, m = (y + 1, 3) if m == 12 else (y, m + 3)
    return out


def fetch_contract(kis: KisClient, code: str, start: date, end: date, progress=None) -> pd.DataFrame:
    """한 월물의 [start, end] 1분봉을 모두 받는다."""
    rows: list[dict] = []
    cursor = datetime.combine(end, datetime.min.time()).replace(hour=15, minute=45)
    seen = set()
    while True:
        try:
            out = kis.get(PATH, "FHKIF03020200", {
                "FID_COND_MRKT_DIV_CODE": "F", "FID_INPUT_ISCD": code, "FID_HOUR_CLS_CODE": "60",
                "FID_PW_DATA_INCU_YN": "Y", "FID_FAKE_TICK_INCU_YN": "N",
                "FID_INPUT_DATE_1": cursor.strftime("%Y%m%d"), "FID_INPUT_HOUR_1": cursor.strftime("%H%M%S")},
                output="output2")
        except KisError as e:
            if "비어 있습니다" in str(e):
                break
            raise
        batch = [r for r in out if r.get("stck_bsop_date")]
        new = [r for r in batch if (r["stck_bsop_date"], r["stck_cntg_hour"]) not in seen]
        if not new:
            break
        for r in new:
            seen.add((r["stck_bsop_date"], r["stck_cntg_hour"]))
        rows += new
        oldest = min(datetime.strptime(r["stck_bsop_date"] + r["stck_cntg_hour"], "%Y%m%d%H%M%S") for r in new)
        if progress:
            progress(code, oldest, len(rows))
        if oldest.date() < start:
            break
        cursor = oldest - timedelta(minutes=1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["stck_bsop_date"] + df["stck_cntg_hour"], format="%Y%m%d%H%M%S")
    df = df.rename(columns={"futs_oprc": "open", "futs_hgpr": "high", "futs_lwpr": "low",
                            "futs_prpr": "close", "cntg_vol": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float})
    df["contract"] = code
    d = df["time"].dt.date
    return df[(d >= start) & (d <= end)].sort_values("time").reset_index(drop=True)


def fetch_continuous(kis: KisClient, start: date, end: date, progress=None) -> pd.DataFrame:
    parts = [fetch_contract(kis, code, s, e, progress) for code, s, e in front_month_schedule(start, end)]
    return pd.concat([p for p in parts if len(p)]).reset_index(drop=True)
