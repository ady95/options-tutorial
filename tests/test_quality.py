from datetime import datetime, timedelta

import pandas as pd

from optlab.market_data import quality

T0 = datetime(2026, 10, 29, 10, 0, 0)


def _q(code, sec, bid, ask):
    t = T0 + timedelta(seconds=sec)
    return {"code": code, "exchange_time": t.replace(microsecond=0), "received_time": t + timedelta(seconds=0.8),
            "bid1": bid, "ask1": ask, "bid_size1": 1, "ask_size1": 1}


INST = pd.DataFrame([
    {"code": "C950", "product": "W", "cp": "C", "series": "S", "strike": 950.0},
    {"code": "C952", "product": "W", "cp": "C", "series": "S", "strike": 952.5},
    {"code": "C955", "product": "W", "cp": "C", "series": "S", "strike": 955.0},
])


def test_quote_flags():
    q = pd.DataFrame([_q("C950", 0, 5.0, 5.2), _q("C952", 0, 0.0, 4.0), _q("C955", 0, 3.5, 3.4)])
    f = quality.quote_flags(q, wide_ratio=0.2)
    assert list(f["one_sided"]) == [False, True, False]
    assert list(f["crossed"]) == [False, False, True]


def test_latency():
    q = pd.DataFrame([_q("C950", 0.3, 5.0, 5.2)])
    assert quality.latency(q).iloc[0] == 1.1  # 0.3초 잘림 + 0.8초 지연


def test_strike_inversions_soft_and_hard():
    q = pd.DataFrame([
        _q("C950", 0, 5.00, 5.20),
        _q("C952", 0, 5.10, 6.00),   # 952.5 매수 5.10 > 950 매수 5.00 → soft
        _q("C955", 0, 3.00, 3.20),
        _q("C955", 1, 6.30, 6.50),   # 1초 뒤 955 매수 6.30 > 952.5 매도 6.00 → hard
    ])
    snap = quality.snapshots(q, INST, "1s")
    inv = quality.strike_inversions(snap)
    kinds = set(zip(inv["rich_strike"], inv["cheap_strike"], inv["kind"]))
    assert (950.0, 952.5, "soft") in kinds
    assert (952.5, 955.0, "hard") in kinds


def test_stale_trades():
    q = pd.DataFrame([_q("C950", 0, 5.0, 5.2), _q("C950", 5, 4.0, 4.2)])
    t = pd.DataFrame([{"code": "C950", "received_time": T0 + timedelta(seconds=1), "price": 5.1}])
    snap = quality.snapshots(q, INST, "1s")
    st = quality.stale_trades(t, snap)
    assert len(st) >= 1 and (st["price"] == 5.1).all() and (st["bid1"] == 4.0).all()
