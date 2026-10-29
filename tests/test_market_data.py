from datetime import datetime

import pytest

from optlab.market_data import master, realtime

MASTER_LINES = [
    "1|A01612|KR4A016C0004|F 202612| |00000.00|1|2001|KOSPI200",
    "L|B09FJW962|KR4B09FJ9620|위클리C 2610W5   962.5|2|00962.50| |2001|KOSPI200",
    "L|B09FJW965|KR4B09FJ9653|위클리C 2610W5   965.0|1|00965.00| |2001|KOSPI200",
    "M|C09FJW962|KR4C09FJ9622|위클리P 2610W5   962.5|3|00962.50| |2001|KOSPI200",
    "M|C09FJW965|KR4C09FJ9655|위클리P 2610W5   965.0|1|00965.00| |2001|KOSPI200",
    "N|BAFC4W965|KR4BAFC49653|위클리M C 2611W1   965.0|1|00965.00| |2001|KOSPI200",
    "J|B06611001|KR4B066B0010|코스닥150C 202611 1,275|2|01275.00| |3003|KSQ150",
]


def test_parse_master_lines():
    inst = [i for i in map(master.parse_line, MASTER_LINES) if i]
    assert len(inst) == 6  # 코스닥150 제외
    fut = master.nearest_futures(inst)
    assert (fut.code, fut.series) == ("A01612", "202612")
    mon = next(i for i in inst if i.product == "KOSPI200_WEEKLY_MON")
    assert (mon.cp, mon.series, mon.strike) == ("C", "2611W1", 965.0)
    chain = master.option_chain(inst, "KOSPI200_WEEKLY_THU", "2610W5", atm=964.0, width=1)
    assert [(k, c.code, p.code) for k, c, p in chain] == [
        (962.5, "B09FJW962", "C09FJW962"), (965.0, "B09FJW965", "C09FJW965")]


def test_parse_quote_frame():
    asks = ["4.66", "4.67", "4.78", "4.79", "4.91"]
    bids = ["4.55", "4.54", "4.50", "4.48", "4.43"]
    fields = ["B09FJW965", "093734", *asks, *bids] + ["0"] * 10 + ["3", "1", "1", "2", "5", "7", "2", "1", "1", "1"] + ["0"] * 6
    assert len(fields) == 38
    ev = realtime.parse_frame("0|H0IOASP0|001|" + "^".join(fields), datetime(2026, 10, 29, 9, 37, 35))[0]
    assert ev["type"] == "quote" and ev["code"] == "B09FJW965"
    assert ev["asks"][0] == 4.66 and ev["bids"][0] == 4.55
    assert ev["ask_sizes"][0] == 3 and ev["bid_sizes"][0] == 7
    assert ev["exchange_time"] == datetime(2026, 10, 29, 9, 37, 34)


def test_parse_trade_frame_multi_records():
    rec = ["0"] * 58
    rec[0], rec[1], rec[2], rec[9], rec[10], rec[33], rec[41], rec[42] = \
        "B09FJW965", "093912", "3.82", "1", "669", "56.02", "3.95", "3.82"
    evs = realtime.parse_frame("0|H0IOCNT0|002|" + "^".join(rec + rec), datetime(2026, 10, 29, 9, 39, 13))
    assert len(evs) == 2
    assert evs[0]["price"] == pytest.approx(3.82) and evs[0]["iv"] == pytest.approx(56.02)
    assert evs[0]["ask1"] == pytest.approx(3.95) and evs[0]["acml_vol"] == 669


def test_subscription_limit():
    async def run():
        async for _ in realtime.stream("dummy", [("H0IOASP0", str(i)) for i in range(41)]):
            pass
    import asyncio
    with pytest.raises(ValueError):
        asyncio.run(run())


def test_weekly_expiry_dates():
    from datetime import date
    assert master.weekly_expiry("KOSPI200_WEEKLY_THU", "2610W5") == date(2026, 10, 29)
    assert master.weekly_expiry("KOSPI200_WEEKLY_THU", "2611W1") == date(2026, 11, 5)
    assert master.weekly_expiry("KOSPI200_WEEKLY_MON", "2611W1") == date(2026, 11, 2)


def test_front_month_schedule():
    from datetime import date
    from optlab.market_data.history import front_month_schedule, quarterly_expiry
    assert quarterly_expiry(2026, 9) == date(2026, 9, 10)
    sched = front_month_schedule(date(2026, 1, 5), date(2026, 10, 29))
    assert [c for c, _, _ in sched] == ["A01603", "A01606", "A01609", "A01612"]
    assert sched[2] == ("A01609", date(2026, 6, 12), date(2026, 9, 10))
    assert sched[3][1] == date(2026, 9, 11)
