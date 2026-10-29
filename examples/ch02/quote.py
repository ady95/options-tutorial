"""2장 실습: 코스피200 지수·선물·위클리옵션 현재가 조회 (REST).

    python examples/ch02/quote.py                 # 실전 도메인 시세
    python examples/ch02/quote.py --mode paper    # 모의투자 도메인

필요: 프로젝트 루트 .env 의 KIS_REAL_APP_KEY / KIS_REAL_APP_SECRET
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from optlab.broker.kis import KisClient, KisConfig, load_dotenv_if_present  # noqa: E402
from optlab.market_data import master  # noqa: E402

MASTER_PATH = ROOT / ".cache" / "fo_idx_code_mts.mst"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["real", "paper"], default="real")
    args = p.parse_args()

    load_dotenv_if_present(ROOT / ".env")
    kis = KisClient(KisConfig.from_env(args.mode))

    idx = kis.index_price("2001")
    print(f"코스피200 지수   {float(idx['bstp_nmix_prpr']):>9.2f}  (전일 대비 {idx['bstp_nmix_prdy_vrss']})")

    if not MASTER_PATH.exists():
        master.download_master(MASTER_PATH)
    inst = master.load_master(MASTER_PATH)
    fut = master.nearest_futures(inst)
    f = kis.futopt_price(fut.code, market="F")
    print(f"선물 {fut.code} ({fut.series})  {float(f['futs_prpr']):>9.2f}")

    # 가장 가까운 목요일 위클리 시리즈의 ATM 콜·풋
    atm = float(idx["bstp_nmix_prpr"])
    series = master.series_list(inst, "KOSPI200_WEEKLY_THU")[0]
    k, call, put = min(master.option_chain(inst, "KOSPI200_WEEKLY_THU", series, atm, width=0),
                       key=lambda row: abs(row[0] - atm))
    print(f"\n위클리 {series} 행사가 {k:.1f}")
    print(f"{'':6}{'코드':>11}{'현재가':>9}{'매수1':>9}{'매도1':>9}{'IV':>8}{'거래량':>8}")
    for label, opt in (("콜", call), ("풋", put)):
        q = kis.futopt_price(opt.code)
        ob = kis.futopt_orderbook(opt.code)
        print(f"{label:6}{opt.code:>11}{float(q['futs_prpr']):>9.2f}{float(ob['futs_bidp1']):>9.2f}"
              f"{float(ob['futs_askp1']):>9.2f}{float(q['hts_ints_vltl']):>8.2f}{int(q['acml_vol']):>8,}")


if __name__ == "__main__":
    main()
