"""13장 예제 공통: 실제 기록 세 개, 선물 분봉, 만기 결제가격."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "ch08"))

from market import LiveDB  # noqa: E402

DATA = ROOT / "data"
# 이름: (기록 파일, 설명). 04장 기록 두 개와 11장 Live 실행 기록
SESSIONS = {
    "thu_expiry_am": (DATA / "ticks_20261029_live.db", "목 위클리 만기일 오전(2610W5, 만기 당일)"),
    "mon_4d_pm": (DATA / "ticks_20261029_mon.db", "월 위클리 오후(2611W1, 만기 11/2)"),
    "thu_6d_fri": (ROOT / "runs" / "20261030-live" / "ticks.db", "목 위클리 금요일(2611W1, 만기 11/5, 거래 적음)"),
}


def sessions(names=None):
    for name, (db, desc) in SESSIONS.items():
        if (names and name not in names) or not db.exists():
            continue
        yield name, desc, LiveDB(db)


def all_bars() -> pd.DataFrame:
    parts = [pd.read_parquet(DATA / "futures_1min_continuous.parquet")]
    parts += [pd.read_parquet(p) for p in sorted(DATA.glob("futures_1min_2*.parquet"))]
    return pd.concat(parts).drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)


def settlement(day) -> float | None:
    """그날 코스피200 종가 = 그날이 최종거래일인 옵션의 결제가격. KIS 일봉으로 받아 data/ch13/settle.json에 둔다."""
    path = DATA / "ch13" / "settle.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = pd.Timestamp(day).strftime("%Y%m%d")
    if key not in cache:
        today, now = pd.Timestamp(day).normalize(), pd.Timestamp.now()
        if today > now.normalize() or (today == now.normalize() and now.hour < 16):
            return None                                     # 아직 종가가 없다(만기 전)
        from optlab.broker.kis import KisClient, KisConfig, load_dotenv_if_present
        load_dotenv_if_present(ROOT / ".env")
        rows = KisClient(KisConfig.from_env("real")).index_daily(key, key)
        if not rows or rows[0].get("stck_bsop_date") != key:
            return None
        cache[key] = float(rows[0]["bstp_nmix_prpr"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[key]
