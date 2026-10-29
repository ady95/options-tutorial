"""8장 예제 공용: 시점별 체인 호가·선도가격·분봉을 꺼내 주는 데이터 소스.

- LiveDB: 04장에서 기록한 실제 호가(SQLite) + 6장 선물 분봉(+ 기록한 날의 분봉)
- Synthetic: data_samples/synthetic 합성 데이터(1분 간격 체인)
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.broker.replay import synthetic_code  # noqa: E402
from optlab.market_data import store  # noqa: E402
from optlab.market_data.master import weekly_expiry  # noqa: E402
from optlab.strategy import clock  # noqa: E402
from optlab.strategy.signal import chain_at, expiry_datetime, implied_forward  # noqa: E402

BARS_PATH = ROOT / "data" / "futures_1min_continuous.parquet"


class LiveDB:
    def __init__(self, db: Path):
        d = store.load(db)
        self.quotes, self.inst = d["quotes"], d["instruments"]
        opts = self.inst.dropna(subset=["strike"])
        self.product, self.series = opts["product"].iloc[0], opts["series"].iloc[0]
        self.expiry = expiry_datetime(weekly_expiry(self.product, self.series))
        self.day = self.quotes["received_time"].iloc[0].date()
        day_bars = ROOT / "data" / f"futures_1min_{self.day:%Y%m%d}.parquet"
        if not day_bars.exists():
            raise SystemExit(f"{day_bars.name} 없음. 먼저 실행: python examples/ch06/download_futures.py "
                             f"--start {self.day} --end {self.day} --out data/{day_bars.name}")
        bars = pd.concat([pd.read_parquet(BARS_PATH), pd.read_parquet(day_bars)])
        self.bars = bars.drop_duplicates("time", keep="last").sort_values("time")
        self.label = f"실제 기록 {self.product} {self.series}"

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        t = self.quotes["received_time"]
        return t.min().ceil("min"), t.max().floor("min")

    def chain(self, ts: pd.Timestamp) -> pd.DataFrame:
        return chain_at(self.quotes, self.inst, ts)

    def bars_until(self, ts: pd.Timestamp) -> pd.DataFrame:
        # 분봉은 시작 시각으로 표시된다. ts 시점에 완성된 분봉은 ts-1분까지
        return self.bars[self.bars["time"] <= ts - timedelta(minutes=1)]


class Synthetic:
    def __init__(self, data_dir: Path = ROOT / "data_samples" / "synthetic"):
        self.fut = pd.read_csv(data_dir / "futures_1min.csv", parse_dates=["time"]).assign(contract="SYN")
        self.opt = pd.read_csv(data_dir / "option_quotes_1min.csv", parse_dates=["time", "expiry"])
        self.label = "합성 데이터"

    def span(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.fut["time"].min(), self.fut["time"].max()

    def expiry_at(self, ts: pd.Timestamp) -> datetime:
        return expiry_datetime(self.opt.loc[self.opt["time"] == ts, "expiry"].iloc[0].date())

    def chain(self, ts: pd.Timestamp) -> pd.DataFrame:
        c = self.opt[self.opt["time"] == ts]
        return pd.DataFrame({"code": [synthetic_code(cp, k) for cp, k in zip(c["cp"], c["strike"])],
                             "bid1": c["bid"].values, "ask1": c["ask"].values, "age": 0.0,
                             "cp": c["cp"].values, "strike": c["strike"].values})

    def bars_until(self, ts: pd.Timestamp) -> pd.DataFrame:
        # 합성 데이터의 ts 분봉 종가 = ts 시점 체인을 만든 가격
        return self.fut[self.fut["time"] <= ts]


def open_source(db: str | None):
    return LiveDB(Path(db)) if db else Synthetic()


def market_state(src, ts: pd.Timestamp, r: float = 0.03):
    """(체인, 선도가격 F, 만기 시각)."""
    expiry = src.expiry if isinstance(src, LiveDB) else src.expiry_at(ts)
    ch = src.chain(ts)
    F = implied_forward(ch, clock.years(ts.to_pydatetime(), expiry), r)
    return ch, F, expiry
