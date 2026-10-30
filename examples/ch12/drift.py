"""12-3 실습: 재학습 정책 비교와 드리프트 감지.

    python examples/ch12/drift.py              # 정책 세 가지 학습에 몇 분 걸린다(결과는 data/ch12/에 저장)
    python examples/ch12/drift.py --refresh

정책: fixed(2026-03-31까지로 한 번만 학습), expanding(매달 그 이전 전체, 07-4절), rolling(매달 직전 3개월),
deployed(07장에서 저장한 예측기, 2026-08-31까지 학습). 모두 15분 예측.
"""
from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from common import load_dataset, predictions

from optlab.analysis.drift import PSI_FEATURES, VolCalibrator, forecast_metrics, pinball, psi, rolling_monitor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--alarm-cov90", type=float, default=0.85, help="20일 cov90이 이 값 아래면 경보")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)

    df = load_dataset()
    preds = {k: predictions(df, k, args.refresh) for k in ["fixed", "expanding", "rolling", "deployed"]}

    print("== 1. 정책별 월간 성적 (auc / cov90 / 중앙값 pinball×1e4)")
    rows = []
    for k, pr in preds.items():
        for m, g in pr.groupby(pd.to_datetime(pr["date"]).dt.to_period("M")):
            s = forecast_metrics(g)
            rows.append({"policy": k, "month": str(m), "v": f"{s['auc']:.3f} / {s['cov90']:.3f} / "
                                                            f"{pinball(g['y_abs'], g['q50'], 0.5) * 1e4:.2f}"})
    print(pd.DataFrame(rows).pivot(index="month", columns="policy", values="v").fillna("").to_string(), "\n")

    print("== 2. 특징 분포 변화(PSI). 기준: 그 달 이전 전체(9월부터는 배포 모델의 학습 기간)")
    ym = pd.to_datetime(df["date"]).dt.to_period("M")
    trained = ym <= pd.Period("2026-08", "M")
    table = {}
    for m in ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]:
        m = pd.Period(m, "M")
        ref = df[trained] if m > pd.Period("2026-08", "M") else df[ym < m]
        table[str(m)] = {f: psi(ref[f], df.loc[ym == m, f]) for f in PSI_FEATURES}
    t = pd.DataFrame(table).round(2)
    print(t.to_string())
    print("0.25 이상인 특징 수:", (t >= 0.25).sum().to_dict())
    print("15분 |수익률| 월평균(bp):", (df.groupby(ym)["y_absret_15"].mean() * 1e4).round(1).to_dict(), "\n")

    print(f"== 3. 20거래일 감시: cov90 < {args.alarm_cov90} 이거나 AUC < 0.5인 날")
    cal = VolCalibrator()
    for k in ["fixed", "expanding"]:
        for label, pr in [(k, preds[k]), (k + "+보정", cal.apply(preds[k]))]:
            mon = rolling_monitor(pr)
            alarm = mon[mon["cov90"] < args.alarm_cov90]
            span = f"{alarm['date'].min()} ~ {alarm['date'].max()}" if len(alarm) else "-"
            print(f"  {label:14s} 전체 cov90 {forecast_metrics(pr)['cov90']:.3f}, cov90 경보 {len(alarm):3d}/{len(mon)}일 ({span}), "
                  f"AUC<0.5 {int((mon['auc'] < 0.5).sum())}일, q90 폭 ×{pr['q90'].mean() / preds[k]['q90'].mean():.3f}")
    k = cal.factors(preds["expanding"])
    print("  expanding 보정 배율 k 월평균:", k.groupby(pd.to_datetime(k.index).to_period("M")).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
