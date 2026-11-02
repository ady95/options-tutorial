"""13-3 실습: A형(방향 예측 + 단일 옵션 매수)과 A+B형(방향 + 변동성 관점으로 단일·스프레드 선택) 비교.

    python examples/ch13/combined.py

두 가지 보유 기간으로 본다.
1) 15분(08~11장의 기본). A = 08장 시그널 엔진의 신호. A+B = 같은 신호에서
   07장 15분 예측 변동성 < 그 15분 구간의 IV(13-1절)이면 옵션이 비싸다고 보고 K1±5 스프레드로 바꾼다.
2) 만기까지(결제가격이 있는 기록만). 방향은 07장 15분 상승확률(0.5 이상이면 콜)로 매분 정한다.
   A = 그 방향의 ATM 단일 옵션. A+B = HAR 만기까지 예측 < ATM IV이면 K1±10 스프레드, 아니면 단일.
모든 체결은 호가 기반(09장 규칙, legs.py).
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from common import ROOT, all_bars, sessions, settlement

from optlab.analysis.performance import trade_stats
from optlab.backtest.engine import BacktestConfig
from optlab.backtest.legs import QuoteBook, vertical
from optlab.features.build import build_features
from optlab.models.forecast import Forecaster
from optlab.strategy import clock
from optlab.strategy.scenarios import annualized_vol, forecast_scenarios
from optlab.strategy.signal import StrategyConfig, decide, implied_forward
from optlab.strategy.spreads import skew_slope, smile
from optlab.strategy.volview import HarForecaster, daily_variance, intraday_profile, iv_for_window
from spread import at_expiry

CFG = BacktestConfig.from_yaml(ROOT / "configs" / "default.yaml")
SCFG = StrategyConfig.from_yaml(ROOT / "configs" / "default.yaml")
MODEL = Forecaster.load(ROOT / "models_store" / "forecaster.joblib")


def partner(src, cp, k1, w):
    """K1에서 바깥쪽(콜은 위, 풋은 아래)으로 w만큼 떨어진 행사가의 코드."""
    k2 = k1 + w if cp == "C" else k1 - w
    m = src.inst[(src.inst["cp"] == cp) & (src.inst["strike"] == k2)]
    return (k2, m["code"].iloc[0]) if len(m) else (None, None)


def fresh(book, codes, t, max_age=60):
    qs = [book.at(c, t) for c in codes]
    return all(q is not None and q.bid1 > 0 and q.ask1 > 0 and (t - q.received_time).total_seconds() <= max_age for q in qs)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)
    bars = all_bars()
    profile = intraday_profile(bars[bars["time"] < "2026-10-01"])
    day = daily_variance(bars)
    out15, outx = [], []

    for name, desc, src in sessions():
        book = QuoteBook(src.quotes)
        lo, hi = src.span()
        S = settlement(src.expiry.date())
        hist = day[day.index < lo.date()]
        har = HarForecaster().fit(hist)
        D = har.next_day(hist)
        for t in pd.date_range(lo + pd.Timedelta(minutes=1), hi - pd.Timedelta(minutes=CFG.horizon_min + 2), freq="1min"):
            ch = src.chain(t)
            T = clock.years(t.to_pydatetime(), src.expiry)
            F = implied_forward(ch, T, SCFG.risk_free)
            if F is None:
                continue
            b = src.bars_until(t)
            b = b[b["time"].dt.date.isin(sorted(b["time"].dt.date.unique())[-2:])]
            fc = next(f for f in MODEL.predict(build_features(b)) if f.horizon == SCFG.horizon_min)
            fit = skew_slope(smile(ch, F, T, SCFG.risk_free))
            if fit is None:
                continue
            atm_iv = fit[0]
            fvol = annualized_vol(*forecast_scenarios(fc, 100, SCFG.direction_weight), SCFG.horizon_min)
            rich_15m = fvol < iv_for_window(atm_iv, t.to_pydatetime(), src.expiry, profile)

            # 1) 15분: 08장 시그널 엔진
            sig = decide(ch, F, t.to_pydatetime(), src.expiry, fc, SCFG)
            if sig.action in ("CALL_BUY", "PUT_BUY") and fresh(book, [sig.code], t):
                a = vertical(book, sig.code, None, t, CFG)
                k2, c2 = partner(src, sig.cp, sig.strike, 5.0)
                use_spread = rich_15m and c2 is not None and fresh(book, [c2], t)
                ab = vertical(book, sig.code, c2, t, CFG) if use_spread else a
                out15.append({"session": name, "t": t, "A": a.get("pnl_krw", np.nan), "AB": ab.get("pnl_krw", np.nan),
                              "AB_kind": "spread" if use_spread else "single", "AB_status": ab["status"]})

            # 2) 만기까지
            if S is None:
                continue
            cp = "C" if fc.p_up >= 0.5 else "P"
            ks = sorted(src.inst.loc[src.inst["cp"] == cp, "strike"].dropna())
            k1 = min(ks, key=lambda k: abs(k - F))
            c1 = src.inst[(src.inst["cp"] == cp) & (src.inst["strike"] == k1)]["code"].iloc[0]
            if not fresh(book, [c1], t):
                continue
            rich_x = har.vol_to_expiry(D, t.to_pydatetime(), src.expiry, profile) < atm_iv
            k2, c2 = partner(src, cp, k1, 10.0)
            a = at_expiry(book, c1, None, cp, k1, None, t, S)
            use_spread = rich_x and c2 is not None and fresh(book, [c2], t)
            ab = at_expiry(book, c1, c2, cp, k1, k2, t, S) if use_spread else a
            outx.append({"session": name, "t": t, "dir": cp, "A": a, "AB": ab, "AB_kind": "spread" if use_spread else "single"})

    def table(d):
        rows = {}
        for (s, k), g in d.groupby(["session", "col"]):
            st = trade_stats(g["pnl"].dropna())
            rows[(s, k)] = {"trades": st["trades"], "avg": st.get("avg_krw"), "net": st.get("net_krw"),
                            "win": st.get("win_rate"), "pf": st.get("profit_factor")}
        return pd.DataFrame(rows).T

    for label, d in (("1) 15분 보유: 08장 신호 → A(단일) / A+B(비싸면 스프레드)", pd.DataFrame(out15)),
                     ("2) 만기까지 보유: 15분 상승확률 방향 → A(단일) / A+B(HAR < IV이면 스프레드)", pd.DataFrame(outx))):
        print(f"== {label}")
        if not len(d):
            print("  거래 없음\n")
            continue
        long = d.melt(id_vars=["session", "t", "AB_kind"], value_vars=["A", "AB"], var_name="col", value_name="pnl")
        t = table(long)
        for c in ("avg", "net"):
            t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}")
        t["win"] = t["win"].map(lambda v: "-" if pd.isna(v) else f"{v:.0%}")
        t["pf"] = t["pf"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
        print(t.to_string())
        print("  A+B가 스프레드를 고른 비율:", d.groupby("session")["AB_kind"].apply(lambda s: f"{(s == 'spread').mean():.0%}").to_dict())
        if "dir" in d:
            print("  방향(콜 비율):", d.groupby("session")["dir"].apply(lambda s: f"{(s == 'C').mean():.0%}").to_dict())
        print()


if __name__ == "__main__":
    main()
