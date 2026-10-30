"""12-2 실습: 국면별 예측 품질.

    python examples/ch12/regimes.py            # 처음 한 번은 월별 재학습에 1분 남짓
    python examples/ch12/regimes.py --refresh  # 분봉 파일을 갱신한 뒤 다시 학습

2026-04~10 워크포워드(매달 그 이전 전체로 재학습, 07-4절) 15분 예측을 국면별로 나눠 본다.
괄호 안은 거래일을 다시 뽑은 부트스트랩 95% 구간이다.
"""
from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from common import load_dataset, predictions

from optlab.analysis.drift import forecast_metrics
from optlab.analysis.regimes import by_day_bootstrap, label_regimes

AUC = lambda s: forecast_metrics(s)["auc"]      # noqa: E731
COV90 = lambda s: forecast_metrics(s)["cov90"]  # noqa: E731


def table(p: pd.DataFrame, col: str, boot: bool) -> pd.DataFrame:
    rows = []
    for k, g in p.groupby(col, observed=True):
        m = forecast_metrics(g)
        r = {col: k, "days": int(m["days"]), "auc": round(m["auc"], 3), "cov50": round(m["cov50"], 3),
             "cov90": round(m["cov90"], 3), "mean_abs_bp": round(m["mean_abs_bp"], 1)}
        if boot:
            r["auc_95"] = "[%.3f, %.3f]" % by_day_bootstrap(g, AUC)
            r["cov90_95"] = "[%.3f, %.3f]" % by_day_bootstrap(g, COV90)
        rows.append(r)
    return pd.DataFrame(rows).set_index(col)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-boot", action="store_true", help="부트스트랩 구간을 생략(빠름)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)

    pr = label_regimes(predictions(load_dataset(), "expanding", args.refresh))
    print(f"워크포워드 15분 예측 {len(pr):,}개 분봉, {pr['date'].nunique()}거래일 ({pr['date'].min()} ~ {pr['date'].max()})\n")
    print(table(pr.assign(all="전체"), "all", not args.no_boot).to_string(), "\n")
    for col in ["day_type", "vol_regime", "time_bucket"]:
        print(table(pr, col, not args.no_boot).to_string(), "\n")
    thu = pr[pr["day_type"] == "목(위클리 만기)"].assign(kind=lambda d: d["monthly_expiry"].map({True: "월물 만기(둘째 목)", False: "그 밖의 목"}))
    print(table(thu, "kind", not args.no_boot).to_string(), "\n")

    late = pr[pr["time_bucket"] == "14:45~15:14"]
    monthly = late.groupby(pd.to_datetime(late["date"]).dt.to_period("M")).apply(AUC).round(3)
    print("14:45~15:14 방향 AUC, 월별:", monthly.to_dict())
    for f in ["gap_open", "ret_from_open", "vwap_dev"]:
        print(f"  {f:14s} 실제 상승과 상관 {late[f].corr(late['y_up']):+.3f} / 예측 확률과 상관 {late[f].corr(late['p_up']):+.3f}")


if __name__ == "__main__":
    main()
