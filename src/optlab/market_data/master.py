"""지수 선물옵션 종목 마스터 — 위클리옵션 종목코드와 옵션 체인.

한국투자증권이 배포하는 마스터 파일(cp949, '|' 구분)을 내려받아 파싱한다.
    구분|단축코드|표준코드|한글종목명|ATM구분|행사가|...|기초자산코드|기초자산명
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import requests

MASTER_URL = "https://new.real.download.dws.co.kr/common/master/fo_idx_code_mts.mst.zip"

# 구분 코드 → (상품, 콜/풋). 2026-10 마스터 파일 실측 기준.
KINDS = {
    "1": ("KOSPI200_FUT", None),
    "5": ("KOSPI200_MONTHLY", "C"), "6": ("KOSPI200_MONTHLY", "P"),
    "L": ("KOSPI200_WEEKLY_THU", "C"), "M": ("KOSPI200_WEEKLY_THU", "P"),
    "N": ("KOSPI200_WEEKLY_MON", "C"), "O": ("KOSPI200_WEEKLY_MON", "P"),
}


@dataclass(frozen=True)
class Instrument:
    code: str        # 단축코드 (API tr_key로 사용)
    product: str     # KINDS의 상품명
    cp: str | None   # 'C' | 'P' | None(선물)
    series: str      # 예: '2610W5', '202612'
    strike: float | None
    name: str


def download_master(dest: Path) -> Path:
    """마스터 zip을 내려받아 dest에 .mst로 저장한다."""
    r = requests.get(MASTER_URL, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(name))
    return dest


def parse_line(line: str) -> Instrument | None:
    f = line.rstrip("\r\n").split("|")
    if len(f) < 6 or f[0] not in KINDS:
        return None
    product, cp = KINDS[f[0]]
    name = f[3].strip()
    if cp is None:
        series = name.split()[-1]          # 'F 202612' → '202612'
        strike = None
    else:
        series = name.split()[-2]          # '위클리C 2610W5   965.0' → '2610W5'
        strike = float(f[5])
    return Instrument(code=f[1], product=product, cp=cp, series=series, strike=strike, name=name)


def load_master(path: Path) -> list[Instrument]:
    text = path.read_bytes().decode("cp949")
    return [i for i in map(parse_line, text.splitlines()) if i is not None]


def nearest_futures(instruments: list[Instrument]) -> Instrument:
    futs = [i for i in instruments if i.product == "KOSPI200_FUT"]
    return min(futs, key=lambda i: i.series)


def series_list(instruments: list[Instrument], product: str) -> list[str]:
    """상품의 만기 시리즈 목록. 'YYMMWn' 형식은 문자열 정렬이 곧 만기순이다."""
    return sorted({i.series for i in instruments if i.product == product})


def weekly_expiry(product: str, series: str) -> date:
    """위클리 시리즈 'YYMMWn'의 만기일 = 그 달 n번째 목요일(월요일 위클리는 월요일).

    2026-10 마스터로 확인: 목 2610W5 → 10/29, 목 2611W1 → 11/5, 월 2611W1 → 11/2.
    """
    yy, mm, n = int(series[:2]), int(series[2:4]), int(series[5])
    weekday = 0 if product == "KOSPI200_WEEKLY_MON" else 3
    first = date(2000 + yy, mm, 1)
    first_hit = first + timedelta(days=(weekday - first.weekday()) % 7)
    return first_hit + timedelta(weeks=n - 1)


def option_chain(instruments: list[Instrument], product: str, series: str,
                 atm: float, width: int = 4) -> list[tuple[float, Instrument, Instrument]]:
    """ATM ± width 행사가의 (행사가, 콜, 풋) 목록."""
    calls = {i.strike: i for i in instruments if i.product == product and i.series == series and i.cp == "C"}
    puts = {i.strike: i for i in instruments if i.product == product and i.series == series and i.cp == "P"}
    strikes = sorted(set(calls) & set(puts), key=lambda k: abs(k - atm))[: 2 * width + 1]
    return [(k, calls[k], puts[k]) for k in sorted(strikes)]
