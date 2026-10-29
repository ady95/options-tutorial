"""7장 실습: 코스피200 선물 5·15·30분 방향·변동폭 예측 — 베이스라인과 LightGBM 비교.

    python examples/ch06/download_futures.py --end 2026-10-28    # 6장에서 1회
    python examples/ch07/train_eval.py

분할(거래일 단위, 시간순): 학습 ~2026-06-30 / 검증 2026-07-01~08-31 / 테스트 2026-09-01~
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.features.build import FEATURES, HORIZONS, add_labels, build_features, time_split  # noqa: E402
from optlab.models.evaluate import calibration_table, classification_report, regression_report  # noqa: E402
from optlab.models.forecast import interval_coverage  # noqa: E402

LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=Path, default=ROOT / "data" / "futures_1min_continuous.parquet")
    p.add_argument("--train-end", default="2026-06-30")
    p.add_argument("--valid-end", default="2026-08-31")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", message=".*eval_set.*")   # LightGBM 4.7의 인자 이름 변경 안내
    pd.set_option("display.width", 140)

    df = add_labels(build_features(pd.read_parquet(args.bars)))
    df = df.dropna(subset=["ret_30"])               # 장 초반 30분은 특징이 덜 채워져 제외
    tr, va, te = time_split(df, args.train_end, args.valid_end)
    print(f"학습 {tr['date'].nunique()}일 {len(tr):,}행 / 검증 {va['date'].nunique()}일 {len(va):,}행 / "
          f"테스트 {te['date'].nunique()}일 {len(te):,}행")

    rows, calib, imp = [], None, None
    for h in HORIZONS:
        y = f"y_up_{h}"
        a, b, c = tr.dropna(subset=[y]), va.dropna(subset=[y]), te.dropna(subset=[y])
        X = lambda d: d[FEATURES].fillna(0.0)    # noqa: E731
        res = {}
        base = a[y].mean()
        res["항상 학습기간 상승비율"] = np.full(len(c), base)
        res["모멘텀(ret_15 부호)"] = np.where(c["ret_15"] > 0, 0.52, np.where(c["ret_15"] < 0, 0.48, 0.5))
        logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        logit.fit(X(a), a[y])
        res["로지스틱 회귀"] = logit.predict_proba(X(c))[:, 1]
        clf = lgb.LGBMClassifier(random_state=args.seed, **LGB_PARAMS)
        clf.fit(X(a), a[y], eval_set=[(X(b), b[y])], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(100, verbose=False)])
        res["LightGBM"] = clf.predict_proba(X(c))[:, 1]
        for name, prob in res.items():
            rows.append({"h": h, "model": name, **classification_report(c[y].values, prob)})
        if h == 15:
            calib = calibration_table(c[y].values, res["LightGBM"])
            calib_lr = calibration_table(c[y].values, res["로지스틱 회귀"])
            imp = pd.Series(clf.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
            best_iter = clf.best_iteration_

    print("\n== 방향 예측 (테스트 기간)")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"\n== LightGBM 15분 확률 보정 (조기 종료 반복 {best_iter})")
    print(calib.round(4).to_string())
    print("\n== 로지스틱 회귀 15분 확률 보정")
    print(calib_lr.round(4).to_string())
    print("\n== 15분 특성 중요도(gain) 상위 8")
    print((imp / imp.sum()).head(8).round(3).to_string())

    print("\n== 변동폭 예측: |h분 수익률| 중앙값(mae·corr)과 90% 분위수 커버리지 — 테스트 기간")
    reg_rows = []
    for h in HORIZONS:
        y = f"y_absret_{h}"
        a, b, c = tr.dropna(subset=[y]), va.dropna(subset=[y]), te.dropna(subset=[y])
        X = lambda d: d[FEATURES].fillna(0.0)    # noqa: E731
        naive = c[f"rv_{h}"] * 0.6745     # 최근 h분 실현변동성 × 정규분포 |Z|의 중앙값
        reg = lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=args.seed, **LGB_PARAMS)
        reg.fit(X(a), a[y], eval_set=[(X(b), b[y])], callbacks=[lgb.early_stopping(100, verbose=False)])
        q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.9, random_state=args.seed, **LGB_PARAMS)
        q90.fit(X(a), a[y], eval_set=[(X(b), b[y])], callbacks=[lgb.early_stopping(100, verbose=False)])
        naive90 = c[f"rv_{h}"] * 1.645                 # 정규분포 |Z|의 90% 분위수
        reg_rows.append({"h": h, "model": "최근 RV 기반 정규근사", **regression_report(c[y].values, naive.values),
                         "cover90": interval_coverage(c[y].values, naive90.values)})
        reg_rows.append({"h": h, "model": "LightGBM 분위수", **regression_report(c[y].values, reg.predict(X(c))),
                         "cover90": interval_coverage(c[y].values, q90.predict(X(c)))})
    out = pd.DataFrame(reg_rows)
    out["mae"] = out["mae"] * 1e4    # bp 단위
    print(out.rename(columns={"mae": "mae(bp)"}).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
