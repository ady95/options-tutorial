"""13-2 실습: 수직 스프레드의 손익 구조, 호가 비용, 두 다리 체결 위험.

    python examples/ch13/spread.py
    python examples/ch13/spread.py --session thu_expiry_am --at 10:30

매분 F에 가장 가까운 행사가 K1을 사고 K1 ± 5, ± 10을 판다(콜은 위, 풋은 아래).
1) 한 시점의 스프레드 표(중간가·호가 순지불액, 최대손실·최대이익·손익분기)
2) 15분 보유: 단일 옵션 vs 스프레드, 다리 내는 방식(sequential·together)별 체결 결과
3) 만기까지 보유(결제가격이 있는 기록만)
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from common import ROOT, sessions, settlement

from optlab.backtest.engine import BacktestConfig
from optlab.backtest.legs import QuoteBook, done_time, fill, vertical
from optlab.strategy import clock
from optlab.strategy.signal import implied_forward
from optlab.strategy.spreads import Vertical

CFG = BacktestConfig.from_yaml(ROOT / "configs" / "default.yaml")
WIDTHS = (5.0, 10.0)
MAX_AGE = pd.Timedelta(seconds=60)


def candidates(src, book, t):
    """t에 만들 수 있는 (종류, cp, K1, K2, 코드1, 코드2). K2=None이면 단일 옵션."""
    F = implied_forward(src.chain(t), clock.years(t.to_pydatetime(), src.expiry), 0.03)
    if F is None:
        return
    inst = src.inst.dropna(subset=["strike"])
    code = {(r.cp, r.strike): r.code for r in inst.itertuples()}
    for kind, cp, sgn in (("bull_call", "C", 1), ("bear_put", "P", -1)):
        k1 = min((k for c, k in code if c == cp), key=lambda k: abs(k - F))
        q1 = book.at(code[(cp, k1)], t)
        if q1 is None or min(q1.bid1, q1.ask1) <= 0 or t - q1.received_time > MAX_AGE:
            continue
        yield kind, cp, k1, None, code[(cp, k1)], None
        for w in WIDTHS:
            k2 = k1 + sgn * w
            q2 = book.at(code[(cp, k2)], t) if (cp, k2) in code else None
            if q2 is None or min(q2.bid1, q2.ask1) <= 0 or t - q2.received_time > MAX_AGE:
                continue
            yield kind, cp, k1, k2, code[(cp, k1)], code[(cp, k2)]


def at_expiry(book, c1, c2, cp, k1, k2, t, S) -> float | None:
    """t에 진입(sequential)해 만기 결제까지 들고 간 손익(원). 한쪽만 체결되면 None(15분 표에서 따로 본다)."""
    q1 = book.at(c1, t)
    o1 = fill(book, "BUY", c1, 1, q1.ask1, t, CFG.latency_sec, CFG.entry_timeout_sec)
    if o1.remaining:
        return None
    intrinsic = (lambda k: max(S - k, 0.0)) if cp == "C" else (lambda k: max(k - S, 0.0))
    if c2 is None:
        return (intrinsic(k1) - o1.avg_price - CFG.fee_rate * o1.avg_price) * CFG.multiplier
    q2 = book.at(c2, done_time(o1))
    o2 = fill(book, "SELL", c2, 1, q2.bid1, done_time(o1), CFG.latency_sec, CFG.entry_timeout_sec)
    if o2.remaining:
        return None
    pnl = (intrinsic(k1) - intrinsic(k2)) - (o1.avg_price - o2.avg_price) - CFG.fee_rate * (o1.avg_price + o2.avg_price)
    return pnl * CFG.multiplier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--session", nargs="*", help="기록 이름(기본: 전부)")
    p.add_argument("--at", default=None, help="1)의 시점 HH:MM(기본: 기록 가운데)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220)

    for name, desc, src in sessions(args.session):
        book = QuoteBook(src.quotes)
        lo, hi = src.span()
        print(f"===== {desc}")
        t_mid = pd.Timestamp(f"{lo.date()} {args.at}") if args.at else lo + (hi - lo) / 2
        t_mid = t_mid.floor("min")
        rows = []
        for kind, cp, k1, k2, c1, c2 in candidates(src, book, t_mid):
            if k2 is None:
                continue
            q1, q2 = book.at(c1, t_mid), book.at(c2, t_mid)
            for label, debit in (("중간가", (q1.bid1 + q1.ask1) / 2 - (q2.bid1 + q2.ask1) / 2), ("호가", q1.ask1 - q2.bid1)):
                v = Vertical(cp, k1, k2, debit)
                rows.append({"spread": f"{kind} {k1:g}/{k2:g}", "가격": label, "debit": round(debit, 2), "max_loss": round(v.max_loss, 2),
                             "max_profit": round(v.max_profit, 2), "breakeven": round(v.breakeven, 2)})
        print(f"1) {t_mid:%H:%M} 스프레드(포인트)")
        print(pd.DataFrame(rows).to_string(index=False) if rows else "  만들 수 있는 스프레드 없음")

        res = []
        S = settlement(src.expiry.date())
        for t in pd.date_range(lo + pd.Timedelta(minutes=1), hi - pd.Timedelta(minutes=CFG.horizon_min + 2), freq="1min"):
            for kind, cp, k1, k2, c1, c2 in candidates(src, book, t):
                w = 0.0 if k2 is None else abs(k2 - k1)
                for mode in (("single",) if k2 is None else ("sequential", "together")):
                    r = vertical(book, c1, c2, t, CFG, mode if k2 else "sequential")
                    q1, q2 = book.at(c1, t), book.at(c2, t) if c2 else None
                    res.append({"kind": kind, "w": w, "mode": mode, "t": t, "status": r["status"], "pnl": r.get("pnl_krw", np.nan),
                                "debit_mid": (q1.bid1 + q1.ask1) / 2 - ((q2.bid1 + q2.ask1) / 2 if q2 is not None else 0),
                                "debit_quote": q1.ask1 - (q2.bid1 if q2 is not None else 0),
                                "expiry_pnl": at_expiry(book, c1, c2, cp, k1, k2, t, S) if S and mode in ("single", "sequential") else np.nan})
        d = pd.DataFrame(res)
        g = d.groupby(["kind", "w", "mode"]).agg(
            n=("t", "size"), ok=("status", lambda s: int((s == "OK").sum())), unwind=("status", lambda s: int((s == "UNWIND").sum())),
            debit_mid=("debit_mid", "mean"), debit_quote=("debit_quote", "mean"), pnl_15m=("pnl", "mean"),
            win_15m=("pnl", lambda s: (s > 0).mean()))
        g["cost/width"] = np.where(g.index.get_level_values("w") > 0,
                                   (g["debit_quote"] - g["debit_mid"]) / g.index.get_level_values("w"), np.nan)
        print("2) 15분 보유(신호마다 따로, 원). w=0은 단일 옵션. cost/width = 호가로 더 낸 금액 ÷ 폭")
        print(g.round({"debit_mid": 2, "debit_quote": 2, "pnl_15m": 0, "win_15m": 2, "cost/width": 2}).to_string())
        if S:
            e = d[d["mode"].isin(["single", "sequential"])].dropna(subset=["expiry_pnl"])
            h = e.groupby(["kind", "w"]).agg(n=("t", "size"), mean=("expiry_pnl", "mean"), win=("expiry_pnl", lambda s: (s > 0).mean()),
                                             worst=("expiry_pnl", "min"), best=("expiry_pnl", "max"))
            print(f"3) 만기까지 보유(결제가 {S:.2f}, 원)")
            print(h.round({"mean": 0, "win": 2, "worst": 0, "best": 0}).to_string())
        print()


if __name__ == "__main__":
    main()
