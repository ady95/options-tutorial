"""백테스트 결과 요약 — 비용 차감 손익, 체결률, 슬리피지, MDD."""
from __future__ import annotations

import pandas as pd

from optlab.backtest.engine import Result


def summarize(res: Result, multiplier: int) -> dict:
    tr, od = res.trades, res.orders
    entry = od[od["tag"] == "ENTRY"] if len(od) else od
    out = {
        "signals": int(res.decisions["action"].isin(["CALL_BUY", "PUT_BUY"]).sum()) if len(res.decisions) else 0,
        "entry_orders": len(entry),
        "filled_full": int((entry["filled"] == entry["qty"]).sum()) if len(entry) else 0,
        "filled_partial": int(((entry["filled"] > 0) & (entry["filled"] < entry["qty"])).sum()) if len(entry) else 0,
        "unfilled": int((entry["filled"] == 0).sum()) if len(entry) else 0,
        "trades": len(tr),
    }
    if not len(tr):
        return out
    qty = tr["qty"]
    cum = tr["pnl_krw"].cumsum()
    out.update({
        # 한 계약 기준 포인트. 스프레드 비용은 결정 시각 중간가 대비, 슬리피지는 결정 시각 호가 대비
        "spread_entry_pt": float((tr["sig_ask"] - tr["sig_mid"]).mean()),
        "spread_exit_pt": float((tr["exit_mid"] - tr["exit_bid"]).mean()),
        "slip_entry_pt": float((tr["entry_px"] - tr["sig_ask"]).mean()),
        "slip_exit_pt": float((tr["exit_bid"] - tr["exit_px"]).mean()),
        "gross_mid_pt": float(((tr["exit_mid"] - tr["sig_mid"]) * qty).sum()),   # 중간가에 사고팔았다면
        "fees_krw": float(tr["fees"].sum() * multiplier),
        "net_krw": float(tr["pnl_krw"].sum()),
        "avg_krw": float(tr["pnl_krw"].mean()),
        "win_rate": float((tr["pnl_krw"] > 0).mean()),
        "mdd_krw": float((cum - cum.cummax().clip(lower=0)).min()),
        "exp_pnl_pt": float(tr["exp_pnl"].mean()),
        "real_pnl_pt": float((tr["pnl_pt"] / qty).mean()),
        "data_end": int((tr["status"] == "DATA_END").sum()),
    })
    return out
