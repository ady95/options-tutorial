"""8장 실습 1: 예측분포로 옵션 한 종목을 h분 뒤 시점에서 재평가하기.

    python examples/ch08/reprice.py --db data/ticks_20261029_live.db --time 10:10 --cp P --strike 940
    python examples/ch08/reprice.py --time 2026-01-08T10:30 --cp C          # 합성 데이터

1. 07장 예측기로 h분 뒤 |수익률| 분포(q50, q90)와 상승확률
2. 지수 시나리오별 h분 뒤 옵션 가치(IV 그대로)
3. 기대 가치 비교: 예측분포 / 시장 IV가 맞다고 할 때 / 지수가 그대로일 때
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from market import ROOT, LiveDB, market_state, open_source

from optlab.features.build import build_features
from optlab.models.forecast import Forecaster
from optlab.pricing import black76
from optlab.strategy import clock
from optlab.strategy.scenarios import annualized_vol, forecast_scenarios, implied_scenarios

MODEL_PATH = ROOT / "models_store" / "forecaster.joblib"
R = 0.03


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", help="04장 기록 파일. 없으면 합성 데이터")
    p.add_argument("--time", required=True, help="HH:MM(기록 파일) 또는 YYYY-MM-DDTHH:MM(합성)")
    p.add_argument("--cp", choices=["C", "P"], default="C")
    p.add_argument("--strike", type=float, help="생략하면 ATM")
    p.add_argument("--horizon", type=int, default=15, choices=[5, 15, 30])
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    src = open_source(args.db)
    ts = pd.Timestamp(f"{src.day} {args.time}") if isinstance(src, LiveDB) else pd.Timestamp(args.time)
    chain, F, expiry = market_state(src, ts, R)
    now = ts.to_pydatetime()
    h = args.horizon
    fc = next(f for f in Forecaster.load(MODEL_PATH).predict(build_features(src.bars_until(ts))) if f.horizon == h)

    strike = args.strike or float(chain.loc[(chain["strike"] - F).abs().idxmin(), "strike"])
    o = chain[(chain["cp"] == args.cp) & (chain["strike"] == strike)].iloc[0]
    mid = (o.bid1 + o.ask1) / 2
    is_call = args.cp == "C"
    T0 = clock.years(now, expiry)
    T1 = clock.years(now + timedelta(minutes=h), expiry)
    iv = black76.implied_vol(mid, F, strike, T0, R, is_call)

    print(f"[{src.label}] {ts:%Y-%m-%d %H:%M}  선도가격 F={F:.2f}, 만기 {expiry:%m-%d %H:%M}")
    print(f"잔존 변동성 시계 {clock.variance_minutes(now, expiry):.0f}분 -> {h}분 뒤 "
          f"{clock.variance_minutes(now + timedelta(minutes=h), expiry):.0f}분")
    print(f"\n== 1. 예측({h}분 뒤)  상승확률 {fc.p_up:.3f}, |수익률| 50% {fc.absret_q50:.2%}"
          f"({fc.absret_q50 * F:.2f}pt), 90% {fc.absret_q90:.2%}({fc.absret_q90 * F:.2f}pt)")

    print(f"\n== 2. {args.cp} {strike:g}  매수 {o.bid1:.2f} / 매도 {o.ask1:.2f} / 중간가 {mid:.2f}, IV {iv:.1%}")
    print(f"{'지수 변화':>10}{'F':>9}{'옵션 가치':>10}{'변화':>8}")
    for name, move in (("-q90", -fc.absret_q90), ("-q50", -fc.absret_q50), ("0", 0.0),
                       ("+q50", fc.absret_q50), ("+q90", fc.absret_q90)):
        F1 = F * np.exp(move)
        v = black76.price(F1, strike, T1, R, iv, is_call)
        print(f"{name:>10}{F1:>9.2f}{v:>10.2f}{v - mid:>+8.2f}")

    r_f, w_f = forecast_scenarios(fc, direction_weight=0.0)
    r_i, w_i = implied_scenarios(iv, h)
    val = lambda r, w: float(np.sum(w * black76.price_array(F * np.exp(r), strike, T1, R, np.full(len(r), iv), is_call)))
    print(f"\n== 3. {h}분 뒤 기대 가치(IV 그대로, 방향 중립)")
    print(f"{'가정':<22}{'연 변동성':>8}{'기대 가치':>10}{'중간가 대비':>10}")
    print(f"{'지수 그대로(시간만 흐름)':<22}{'-':>8}{black76.price(F, strike, T1, R, iv, is_call):>10.2f}"
          f"{black76.price(F, strike, T1, R, iv, is_call) - mid:>+10.2f}")
    print(f"{'시장 IV가 맞다면':<22}{iv:>8.1%}{val(r_i, w_i):>10.2f}{val(r_i, w_i) - mid:>+10.2f}")
    print(f"{'07장 예측분포':<22}{annualized_vol(r_f, w_f, h):>8.1%}{val(r_f, w_f):>10.2f}{val(r_f, w_f) - mid:>+10.2f}")


if __name__ == "__main__":
    main()
