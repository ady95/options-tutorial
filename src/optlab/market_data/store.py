"""실시간 이벤트 저장소 — SQLite.

거래소 시각과 수신 시각을 모두 저장한다. 시각은 ISO 문자열(마이크로초까지)로 둔다.
대용량·장기 보관이 필요하면 같은 스키마를 PostgreSQL/TimescaleDB로 옮긴다(04장 선택 경로).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
    code TEXT PRIMARY KEY, product TEXT, cp TEXT, series TEXT, strike REAL);
CREATE TABLE IF NOT EXISTS quotes (
    code TEXT, exchange_time TEXT, received_time TEXT,
    bid1 REAL, ask1 REAL, bid_size1 INTEGER, ask_size1 INTEGER, depth TEXT);
CREATE TABLE IF NOT EXISTS trades (
    code TEXT, exchange_time TEXT, received_time TEXT,
    price REAL, last_qty REAL, acml_vol REAL, bid1 REAL, ask1 REAL, iv REAL);
CREATE TABLE IF NOT EXISTS events (
    received_time TEXT, type TEXT, msg TEXT);
CREATE INDEX IF NOT EXISTS ix_quotes ON quotes(code, received_time);
CREATE INDEX IF NOT EXISTS ix_trades ON trades(code, received_time);
"""


def _ts(t) -> str | None:
    return t.isoformat(timespec="microseconds") if t else None


class EventStore:
    def __init__(self, path: Path, batch: int = 200):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.batch = batch
        self._q: list[tuple] = []
        self._t: list[tuple] = []

    def add_instrument(self, code: str, product: str, cp: str | None, series: str, strike: float | None) -> None:
        self.conn.execute("INSERT OR REPLACE INTO instruments VALUES (?,?,?,?,?)",
                          (code, product, cp, series, strike))
        self.conn.commit()

    def add(self, ev: dict) -> None:
        if ev["type"] == "quote":
            self._q.append((ev["code"], _ts(ev["exchange_time"]), _ts(ev["received_time"]),
                            ev["bids"][0], ev["asks"][0], ev["bid_sizes"][0], ev["ask_sizes"][0],
                            json.dumps([ev["bids"], ev["asks"], ev["bid_sizes"], ev["ask_sizes"]])))
        elif ev["type"] == "trade":
            self._t.append((ev["code"], _ts(ev["exchange_time"]), _ts(ev["received_time"]),
                            ev.get("price"), ev.get("last_qty"), ev.get("acml_vol"),
                            ev.get("bid1"), ev.get("ask1"), ev.get("iv")))
        else:  # error, disconnected 등 운영 이벤트도 남긴다
            self.conn.execute("INSERT INTO events VALUES (?,?,?)",
                              (_ts(ev["received_time"]), ev["type"], ev.get("msg")))
        if len(self._q) + len(self._t) >= self.batch:
            self.flush()

    def flush(self) -> None:
        if self._q:
            self.conn.executemany("INSERT INTO quotes VALUES (?,?,?,?,?,?,?,?)", self._q)
        if self._t:
            self.conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)", self._t)
        self.conn.commit()
        self._q, self._t = [], []

    def close(self) -> None:
        self.flush()
        self.conn.close()


def load(path: Path) -> dict[str, pd.DataFrame]:
    """저장된 테이블을 DataFrame으로 읽는다(시각 컬럼은 datetime으로 변환)."""
    conn = sqlite3.connect(path)
    out = {}
    for name in ("instruments", "quotes", "trades", "events"):
        df = pd.read_sql(f"SELECT * FROM {name}", conn)
        for col in ("exchange_time", "received_time"):
            if col in df:
                df[col] = pd.to_datetime(df[col])
        out[name] = df
    conn.close()
    return out
