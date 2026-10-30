"""12-4 실습: 성과·안정성 대시보드(HTML 한 장).

    python examples/ch12/dashboard.py --runs runs/20261030-live data/ticks_20261029_live.db data/ticks_20261029_mon.db \
        --trades thu_signal=runs/ch12/thu_signal.csv fri_tightest=runs/ch12/fri_tightest.csv

결과: runs/ch12/dashboard.html. 외부 라이브러리 없이 표와 SVG 선그래프로 그린다.
매일 장이 끝난 뒤 한 번 만들어 보는 용도다. 경보 기준은 12-3절.
"""
from __future__ import annotations

import argparse
import html
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import ROOT, load_dataset, predictions
from performance import load as load_trades

from optlab.analysis.drift import PSI_FEATURES, VolCalibrator, psi, rolling_monitor
from optlab.analysis.health import run_health
from optlab.analysis.performance import trade_stats

CSS = """body{font-family:sans-serif;margin:24px;color:#222}table{border-collapse:collapse;margin:8px 0 20px}
td,th{border:1px solid #ccc;padding:3px 6px;text-align:right;font-size:13px;white-space:nowrap}
.wrap{overflow-x:auto}svg{display:block;margin:6px 0}th{background:#f3f3f3}td:first-child{text-align:left}
.bad{background:#fde2e2}.ok{background:#e3f4e3}h2{margin-top:28px}small{color:#666}"""


def svg_lines(series: dict[str, pd.Series], guides=(), width=720, height=220, lo=None, hi=None) -> str:
    """날짜 인덱스 시계열 몇 개를 선그래프로. guides는 가로 기준선 값."""
    xs = sorted(set().union(*[s.index for s in series.values()]))
    vals = pd.concat(series.values())
    lo = vals.min() if lo is None else lo
    hi = vals.max() if hi is None else hi
    pad = 30
    X = {x: pad + i * (width - 2 * pad) / max(len(xs) - 1, 1) for i, x in enumerate(xs)}
    Y = lambda v: height - pad - (v - lo) / (hi - lo) * (height - 2 * pad)  # noqa: E731
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    out = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    for g in guides:
        out.append(f'<line x1="{pad}" x2="{width - pad}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" stroke="#999" stroke-dasharray="4"/>'
                   f'<text x="2" y="{Y(g) + 4:.1f}" font-size="10">{g}</text>')
    for i, (name, s) in enumerate(series.items()):
        pts = " ".join(f"{X[x]:.1f},{Y(v):.1f}" for x, v in s.items())
        out.append(f'<polyline fill="none" stroke="{colors[i % 4]}" stroke-width="1.5" points="{pts}"/>'
                   f'<text x="{width - pad - 150}" y="{14 + 14 * i}" font-size="11" fill="{colors[i % 4]}">{html.escape(name)}</text>')
    out.append(f'<text x="{pad}" y="{height - 8}" font-size="10">{xs[0]}</text>'
               f'<text x="{width - pad - 60}" y="{height - 8}" font-size="10">{xs[-1]}</text></svg>')
    return "".join(out)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"
    return str(v)


def frame_html(df: pd.DataFrame, bad=None) -> str:
    """bad(행, 열) → True이면 그 칸을 붉게."""
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    body = []
    for idx, row in df.iterrows():
        cells = [f"<td>{html.escape(str(idx))}</td>"]
        for c, v in row.items():
            cls = ' class="bad"' if bad and bad(idx, c, v) else ""
            cells.append(f"<td{cls}>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='wrap'><table><tr>{head}</tr>{''.join(body)}</table></div>"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[], help="11장 실행 기록 폴더 또는 04장 기록 파일(.db)")
    p.add_argument("--trades", nargs="*", default=[], help="이름=경로(09장 --save CSV 또는 실행 기록 폴더)")
    p.add_argument("--out", default=str(ROOT / "runs" / "ch12" / "dashboard.html"))
    p.add_argument("--alarm-cov90", type=float, default=0.85)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    warnings.filterwarnings("ignore")
    parts = [f"<h1>optlab 성과·안정성 대시보드</h1><small>생성 {datetime.now():%Y-%m-%d %H:%M}</small>"]

    # 1. 운영 안정성
    if args.runs:
        h = pd.DataFrame([{"run": Path(r).name, **run_health(Path(r))} for r in args.runs]).set_index("run")
        h = h.T                          # 지표가 행, 실행이 열(실행이 늘어도 옆으로 넘치지 않게)
        h = h.apply(lambda col: col.map(_fmt))
        rules = {"starts": lambda v: v > 1, "stale_events": lambda v: v > 0,
                 "opt_quotes_per_min": lambda v: v < 20, "lag_p50": lambda v: v > 1.0, "late_minutes": lambda v: v > 10}
        parts.append("<h2>1. 운영 안정성</h2><small>붉은 칸: 다시 켬, 시세 끊김, 옵션 호가 분당 20건 미만(거래 안 되는 만기물), "
                     "수신 지연 중앙값 1초 초과(PC 시계 확인), 미완성 분봉 10분 초과</small>")
        parts.append(frame_html(h, lambda i, c, v: i in rules and v != "-" and rules[i](float(str(v).replace(",", "")))))

    # 2. 거래 성과
    if args.trades:
        rows = {}
        for item in args.trades:
            name, path = item.split("=", 1)
            t = load_trades(Path(path))
            s = trade_stats(t["pnl_krw"]) if len(t) else {"trades": 0}
            s["days"] = t["day"].nunique() if len(t) else 0
            rows[name] = s
        t = pd.DataFrame(rows).T
        for c in ["trades", "days", "max_losing_streak"]:
            if c in t:
                t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{int(v)}")
        for c in ["net_krw", "avg_krw", "avg_win_krw", "avg_loss_krw", "mdd_krw"]:
            if c in t:
                t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}")
        for c in ["payoff", "profit_factor", "win_rate"]:
            if c in t:
                t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
        t.index.name = "이름"
        parts.append("<h2>2. 거래 성과</h2><small>days가 1이면 신뢰구간을 낼 수 없다(12-1절). 거래일이 쌓일 때까지 결론을 미룬다.</small>")
        parts.append(frame_html(t.fillna("-"), lambda i, c, v: c == "profit_factor" and v != "-" and float(v) < 1))

    # 3. 모델 감시
    df = load_dataset()
    dep = predictions(df, "deployed")
    exp = predictions(df, "expanding")
    cal = VolCalibrator().apply(exp)
    m_raw, m_cal, m_dep = rolling_monitor(exp), rolling_monitor(cal), rolling_monitor(dep) if dep["date"].nunique() >= 20 else None
    s = {"cov90 매달 재학습": m_raw.set_index("date")["cov90"], "cov90 재학습+보정": m_cal.set_index("date")["cov90"]}
    if m_dep is not None:
        s["cov90 배포 모델"] = m_dep.set_index("date")["cov90"]
    parts.append(f"<h2>3. 모델 감시(15분 예측, 직전 20거래일)</h2><small>점선 0.9는 목표, {args.alarm_cov90}은 경보 기준</small>")
    parts.append(svg_lines(s, guides=(0.9, args.alarm_cov90), lo=0.75, hi=1.0))
    parts.append(svg_lines({"방향 AUC 매달 재학습": m_raw.set_index("date")["auc"]}, guides=(0.5,), lo=0.44, hi=0.58))
    last = pd.DataFrame({"매달 재학습": m_raw.iloc[-1], "재학습+보정": m_cal.iloc[-1]}
                        | ({"배포 모델": m_dep.iloc[-1]} if m_dep is not None else {})).T.drop(columns="date").astype(float).round(3)
    last.index.name = f"마지막 날 {m_raw['date'].iloc[-1]}"
    parts.append(frame_html(last, lambda i, c, v: (c == "cov90" and v < args.alarm_cov90) or (c == "auc" and v < 0.5)))

    ym = pd.to_datetime(df["date"]).dt.to_period("M")
    cur, ref = df[ym == ym.max()], df[ym < ym.max()]
    ps = pd.Series({f: psi(ref[f], cur[f]) for f in PSI_FEATURES}).round(2).sort_values(ascending=False).to_frame("PSI")
    ps.index.name = f"{ym.max()} vs 이전 전체"
    parts.append("<small>특징 분포 변화(PSI). 변동성 특징은 거의 매달 0.25를 넘으므로 경보로 쓰지 않고 참고로만 본다(12-3절).</small>")
    parts.append(frame_html(ps))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>optlab 대시보드</title>"
                   f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")
    print(f"대시보드: {out}")
    if args.runs:
        print(h.to_string())


if __name__ == "__main__":
    main()
