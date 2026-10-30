"""11장 Live 실행 기록 폴더(audit.jsonl, ticks.db)로 하루의 운영 상태를 점검한다."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from optlab.market_data import store


def read_audit(path: Path) -> pd.DataFrame:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    return df


def run_health(run_dir: Path, decide_delay_sec: float = 2.0, flip_min: int = 5) -> dict:
    """하루 실행의 안정성 지표.

    - starts: RUN_START 수. 1보다 크면 그날 다시 켠 적이 있다
    - stale_events: 시세 상태가 DATA_READY가 아닌 쪽으로 바뀐 횟수
    - recenters / quick_flips: 체인 이동 횟수, 그중 flip_min분 안에 직전 범위로 되돌아온 횟수
    - fut_msgs_per_min, opt_quotes_per_min: 장중 분당 수신 메시지(선물 체결, 옵션 호가)
    - lag_p50 / lag_p95: 수신 시각 - 거래소 시각(초). 거래소 시각은 초 단위라 평균 0.5초가 더해져 있다
    - late_minutes: 다음 분 첫 선물 체결이 결정 시각(decide_delay_sec)보다 늦게 도착한 분(직전 분봉 미완성)
    """
    run_dir = Path(run_dir)
    out = {}
    if run_dir.suffix == ".db":                    # 04장 기록 파일: 시세 지표만 계산한다
        ticks = run_dir
    else:
        ticks = run_dir / "ticks.db"
        if (run_dir / "audit.jsonl").exists():
            out.update(audit_health(read_audit(run_dir / "audit.jsonl"), flip_min))
    d = store.load(ticks)
    q, t, inst = d["quotes"], d["trades"], d["instruments"]
    fut = inst.loc[inst["strike"].isna(), "code"].iloc[0]
    f = t[t["code"] == fut]
    day = f["exchange_time"].dt.normalize().iloc[0]
    out["day"] = str(day.date())
    session = (f["exchange_time"] >= day + pd.Timedelta("9h")) & (f["exchange_time"] < day + pd.Timedelta("15h20min"))
    minutes = session.sum() and f.loc[session, "exchange_time"].dt.floor("min").nunique()
    oq = q[(q["code"] != fut) & (q["exchange_time"] >= day + pd.Timedelta("9h"))
           & (q["exchange_time"] < day + pd.Timedelta("15h20min"))]
    lag = (f["received_time"] - f["exchange_time"]).dt.total_seconds()
    first = f.groupby(f["exchange_time"].dt.floor("min"))["received_time"].min()
    late = (first - first.index).dt.total_seconds() > decide_delay_sec
    out.update(fut_msgs_per_min=float(session.sum() / max(minutes, 1)),
               opt_quotes_per_min=float(len(oq) / max(minutes, 1)),
               lag_p50=float(lag.median()), lag_p95=float(lag.quantile(0.95)),
               late_minutes=int(late[(first.index >= day + pd.Timedelta("9h"))
                                     & (first.index < day + pd.Timedelta("15h20min"))].sum()))
    return out


def audit_health(a: pd.DataFrame, flip_min: int = 5) -> dict:
    out = {"starts": int((a["kind"] == "RUN_START").sum()), "orders": int((a["kind"] == "ORDER").sum()),
           "trades": int((a["kind"] == "TRADE").sum()),
           "stale_events": int((a.loc[a["kind"] == "MODE", "mode"] != "DATA_READY").sum())}
    rs = a[a["kind"] == "RESUBSCRIBE"].drop_duplicates("time")
    flips = 0
    if len(rs):
        ranges = rs["detail"].str.split(",").str[0].str.split(" → ", expand=True)
        for i in range(1, len(rs)):
            back = ranges.iloc[i, 1] == ranges.iloc[i - 1, 0]
            fast = rs["time"].iloc[i] - rs["time"].iloc[i - 1] <= pd.Timedelta(minutes=flip_min)
            flips += int(back and fast)
    out.update(recenters=len(rs), quick_flips=flips)
    return out
