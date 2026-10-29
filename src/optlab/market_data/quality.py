"""시세 데이터 품질 검증.

store.load()로 읽은 DataFrame을 입력으로 받는다. 검사 항목:
1. 수신 지연        received_time - exchange_time
2. 호가 이상        한쪽 호가 없음, 매수 >= 매도(교차)
3. 넓은 스프레드    (매도 - 매수) / 중간가
4. 행사가 역전      같은 시점의 콜(풋) 호가가 행사가에 대해 단조롭지 않음
5. 오래된 체결가    마지막 체결가가 현재 호가 범위 밖에 있음
6. 무소식 구간      종목별 메시지 사이 최대 공백
"""
from __future__ import annotations

import pandas as pd


def latency(df: pd.DataFrame) -> pd.Series:
    """수신 지연(초). 거래소 시각이 초 단위로 잘려 오므로 0~1초 과대 추정된다."""
    return (df["received_time"] - df["exchange_time"]).dt.total_seconds()


def latency_summary(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in (("quote", quotes), ("trade", trades)):
        lag = latency(df.dropna(subset=["exchange_time"]))
        if len(lag):
            rows.append({"type": name, "n": len(lag), "p50": lag.quantile(.5), "p95": lag.quantile(.95),
                         "p99": lag.quantile(.99), "max": lag.max(), "over_3s": int((lag > 3).sum())})
    return pd.DataFrame(rows)


def quote_flags(quotes: pd.DataFrame, wide_ratio: float = 0.2) -> pd.DataFrame:
    """호가 한 건마다 이상 여부를 표시한다."""
    q = quotes.copy()
    q["one_sided"] = (q["bid1"] <= 0) | (q["ask1"] <= 0)
    q["crossed"] = ~q["one_sided"] & (q["bid1"] >= q["ask1"])
    mid = (q["bid1"] + q["ask1"]) / 2
    q["spread"] = (q["ask1"] - q["bid1"]).where(~q["one_sided"])
    q["spread_ratio"] = (q["spread"] / mid).where(~q["one_sided"])
    q["wide"] = q["spread_ratio"] > wide_ratio
    return q


def snapshots(quotes: pd.DataFrame, instruments: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    """일정 간격 시점마다 종목별 '그때까지의 최신 호가'를 모은 표 (as-of 스냅숏).

    반환: time, code, cp, strike, bid1, ask1, age(마지막 호가 이후 경과 초)
    """
    opts = instruments.dropna(subset=["strike"])
    q = quotes[quotes["code"].isin(opts["code"])].sort_values("received_time")
    if q.empty:
        return pd.DataFrame()
    # 끝점도 올림해야 마지막 메시지 이후의 상태가 스냅숏에 들어간다
    grid = pd.date_range(q["received_time"].min().ceil(freq), q["received_time"].max().ceil(freq), freq=freq)
    frames = []
    for code, g in q.groupby("code"):
        s = g.set_index("received_time")[["bid1", "ask1"]]
        s = s[~s.index.duplicated(keep="last")]
        last_seen = pd.Series(s.index, index=s.index).reindex(grid, method="ffill")
        snap = s.reindex(grid, method="ffill")
        snap["age"] = (grid - last_seen).dt.total_seconds().values
        snap["code"] = code
        frames.append(snap.rename_axis("time").reset_index())
    out = pd.concat(frames).dropna(subset=["bid1"])
    return out.merge(opts[["code", "cp", "strike"]], on="code")


def strike_inversions(snap: pd.DataFrame) -> pd.DataFrame:
    """행사가 역전을 찾는다.

    콜은 행사가가 높을수록, 풋은 낮을수록 싸야 한다. 이웃한 두 행사가 사이에서
    - soft: 비싸야 할 쪽의 매수호가 < 싸야 할 쪽의 매수호가   (호가가 얇아 생기는 왜곡)
    - hard: 비싸야 할 쪽의 매도호가 < 싸야 할 쪽의 매수호가   (그대로라면 무위험 차익 — 실제로는 체결 불가 의심)
    """
    rows = []
    valid = snap[(snap["bid1"] > 0) & (snap["ask1"] > 0)]
    for (t, cp), g in valid.groupby(["time", "cp"]):
        g = g.sort_values("strike", ascending=(cp == "C"))   # 비싼 쪽 → 싼 쪽 순서
        rich, cheap = g.iloc[:-1].reset_index(drop=True), g.iloc[1:].reset_index(drop=True)
        for a, b in zip(rich.itertuples(), cheap.itertuples()):
            if a.bid1 < b.bid1:
                rows.append({"time": t, "cp": cp, "rich_strike": a.strike, "cheap_strike": b.strike,
                             "rich_bid": a.bid1, "rich_ask": a.ask1, "cheap_bid": b.bid1, "cheap_ask": b.ask1,
                             "kind": "hard" if a.ask1 < b.bid1 else "soft"})
    return pd.DataFrame(rows)


def stale_trades(trades: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """스냅숏 시점의 마지막 체결가가 그 시점 호가 범위(매수~매도) 밖에 있는 경우."""
    if trades.empty or snap.empty:
        return pd.DataFrame()
    t = trades.sort_values("received_time")[["code", "received_time", "price"]]
    s = snap.sort_values("time")
    m = pd.merge_asof(s, t.rename(columns={"received_time": "trade_time"}),
                      left_on="time", right_on="trade_time", by="code", direction="backward")
    m = m.dropna(subset=["price"])
    m["trade_age"] = (m["time"] - m["trade_time"]).dt.total_seconds()
    out = m[(m["price"] < m["bid1"]) | (m["price"] > m["ask1"])]
    return out[["time", "code", "cp", "strike", "price", "bid1", "ask1", "trade_age"]]


def silence(quotes: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """종목별 메시지 사이 최대 공백(초)."""
    ev = pd.concat([quotes[["code", "received_time"]], trades[["code", "received_time"]]]).sort_values("received_time")
    gaps = ev.groupby("code")["received_time"].diff().dt.total_seconds()
    return (ev.assign(gap=gaps).groupby("code")
              .agg(messages=("gap", "size"), max_gap=("gap", "max"), median_gap=("gap", "median"))
              .sort_values("max_gap", ascending=False))
