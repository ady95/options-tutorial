"""10-1: 시그널 엔진의 매수 신호를 하나씩 위험 한도에 비춰 본다(포지션이 없다고 가정).

    python examples/ch10/risk_check.py --db data/ticks_20261029_live.db
    python examples/ch10/risk_check.py --db data/ticks_20261029_live.db --allow-expiry-day
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch08"))
from market import ROOT, LiveDB  # noqa: E402
from signal_engine import MODEL_PATH, run_once  # noqa: E402

from optlab.market_data.spec import MULTIPLIER_KRW  # noqa: E402
from optlab.models.forecast import Forecaster  # noqa: E402
from optlab.risk.limits import Exposure, Intent, RiskConfig, check_entry  # noqa: E402
from optlab.strategy.signal import StrategyConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--max-premium", type=int)
    p.add_argument("--allow-expiry-day", action="store_true", help="만기일 신규 진입 허용(기본은 금지)")
    args = p.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    src = LiveDB(Path(args.db))
    scfg = StrategyConfig.from_yaml(args.config)
    risk = RiskConfig.from_yaml(args.config)
    if args.allow_expiry_day:
        risk.skip_expiry_day = False
    if args.max_premium:
        risk.max_premium_krw = args.max_premium
    model = Forecaster.load(MODEL_PATH)
    lo, hi = src.span()
    rows, why = [], Counter()
    for ts in pd.date_range(lo, hi, freq="1min"):
        out = run_once(src, model, scfg, ts)
        if not out or out[0].code is None:
            continue
        sig = out[0]
        it = Intent(now=ts.to_pydatetime(), code=sig.code, qty=1, limit=sig.entry, delta=sig.delta,
                    expiry_day=ts.date() == src.expiry.date())
        r = check_entry(it, Exposure(), risk)
        why.update(r or ["APPROVED"])
        rows.append({"time": f"{ts:%H:%M}", "signal": f"{sig.cp}{sig.strike:g}", "ask": sig.entry,
                     "premium_krw": round(sig.entry * MULTIPLIER_KRW), "delta": round(sig.delta, 3),
                     "result": ",".join(r) or "APPROVED"})
    print(f"[{src.label}] 한도: 프리미엄 {risk.max_premium_krw:,}원, 순델타 {risk.max_net_delta}, "
          f"계약 {risk.max_contracts}, 만기일 진입 {'금지' if risk.skip_expiry_day else '허용'}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n매수 신호 {len(rows)}건: {dict(why)}")


if __name__ == "__main__":
    main()
