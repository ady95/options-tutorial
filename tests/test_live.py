import asyncio

import pandas as pd

from optlab.execution import replay
from optlab.execution.paper import PaperBroker
from optlab.execution.recovery import catch_up, restore
from optlab.execution.scheduler import Scheduler
from optlab.execution.trader import Phase, Trader, TraderConfig
from optlab.live import audit, pipeline
from optlab.live.market import BarBuilder, MarketBook, Trade
from optlab.risk.limits import RiskConfig
from optlab.strategy.signal import Signal, chain_at

T0 = pd.Timestamp("2026-10-29 10:00:00")
INST = pd.DataFrame({"code": ["F", "X"], "cp": [None, "C"], "strike": [None, 940.0]})
NO_TRADES = pd.DataFrame(columns=["code", "exchange_time", "received_time", "price", "acml_vol"])


def market(until_min=8, rise_after_min=None):
    """선물 F는 0.5초마다, 옵션 X는 2초마다 호가. rise_after_min 뒤로 X가 0.5 오른다."""
    rows = []
    for i in range(int(until_min * 60 * 2)):
        t = T0 + pd.Timedelta(seconds=i * 0.5)
        up = 0.5 if rise_after_min is not None and t >= T0 + pd.Timedelta(minutes=rise_after_min) else 0.0
        rows.append({"code": "F", "received_time": t + pd.Timedelta(seconds=0.3), "exchange_time": t,
                     "bid1": 940.0, "ask1": 940.05, "bid_size1": 5, "ask_size1": 5, "depth": None})
        if i % 4 == 0:
            rows.append({"code": "X", "received_time": t + pd.Timedelta(seconds=0.3), "exchange_time": t,
                         "bid1": 5.0 + up, "ask1": 5.1 + up, "bid_size1": 5, "ask_size1": 5, "depth": None})
    return pd.DataFrame(rows)


def buy_x(when=(T0 + pd.Timedelta(minutes=1),)):
    return lambda ts: (Signal(ts, "CALL_BUY", "T", code="X", cp="C", strike=940.0, entry=5.1, exp_pnl=0.1,
                              delta=0.5) if ts in set(when) else None)


def make(state_path=None):
    b = PaperBroker(0.5, state_path)
    return Trader(RiskConfig(), TraderConfig(horizon_min=2), b, "F"), b


def run_pipeline(q, decide_fn, trader, broker, decisions, after=None, stop_at=None):
    book = MarketBook(INST, "F", lambda m: pd.DataFrame())
    sched = Scheduler(trader, broker, decide_fn, decisions, start=after)
    feed = pipeline.replay_feed(q, NO_TRADES, after=after)
    return asyncio.run(pipeline.run(feed, book, sched, stop_at=stop_at))


def pnl(trader):
    return [(t["signal_time"], round(t["pnl_krw"], 2)) for t in trader.trades]


# ── 분봉 만들기 ──────────────────────────────────────────
def test_bar_builder_volume_from_cumulative_and_skips_first_minute():
    bb = BarBuilder("F")
    ticks = [("10:00:30", 940.0, 100), ("10:00:59", 941.0, 103),          # 첫 분: 앞부분이 빠져 있어 버린다
             ("10:01:05", 941.5, 110), ("10:01:40", 939.0, 115), ("10:01:59", 940.5, 120),
             ("10:02:01", 940.0, 121)]
    for hhmmss, px, acml in ticks:
        t = pd.Timestamp(f"2026-10-29 {hhmmss}")
        bb.on_trade(Trade("F", t, t + pd.Timedelta(seconds=0.5), px, acml))
    assert [b["time"].strftime("%H:%M") for b in bb.done] == ["10:01"]
    b = bb.done[0]
    assert (b["open"], b["high"], b["low"], b["close"], b["volume"]) == (941.5, 941.5, 939.0, 940.5, 17)  # 120 - 103
    assert not bb.complete_until(pd.Timestamp("2026-10-29 10:02"))      # 다음 분 체결이 없으면 강제로 닫는다
    assert bb.done[-1]["volume"] == 1


def test_market_book_chain_matches_chain_at():
    q = market(until_min=1)
    book = MarketBook(INST, "F", lambda m: pd.DataFrame())
    for ev in q.itertuples(index=False):
        book.on_quote(ev)
    ts = T0 + pd.Timedelta(seconds=45)
    book2 = MarketBook(INST, "F", lambda m: pd.DataFrame())
    for ev in q[q["received_time"] <= ts].itertuples(index=False):
        book2.on_quote(ev)
    expect = chain_at(q, INST, ts).reset_index(drop=True)
    got = book2.chain(ts)[expect.columns]
    pd.testing.assert_frame_equal(got, expect, check_dtype=False)


# ── asyncio 파이프라인 = 10장 재생기 ─────────────────────
def test_pipeline_matches_replay_run():
    q = market(rise_after_min=2)
    t1, b1 = make()
    replay.run(q, buy_x(), t1, b1, T0, T0 + pd.Timedelta(minutes=5))
    t2, b2 = make()
    dec = [T0 + pd.Timedelta(minutes=m) for m in range(1, 6)]
    run_pipeline(q, buy_x(), t2, b2, dec)
    t2.finish(q["received_time"].max())
    assert pnl(t1) == pnl(t2) and len(t1.trades) == 1
    assert [x["kind"] for x in t1.log] == [x["kind"] for x in t2.log]


# ── 브로커 상태 저장과 재시작 복구 ───────────────────────
def test_broker_state_roundtrip(tmp_path):
    t, b = make(tmp_path / "broker.json")
    q = market(rise_after_min=2)
    run_pipeline(q, buy_x(), t, b, [T0 + pd.Timedelta(minutes=1)], stop_at=T0 + pd.Timedelta(seconds=90))
    b2 = PaperBroker.load(tmp_path / "broker.json")
    assert b2.positions() == b.positions() == {"X": 1}
    assert [bo.order.fills for bo in b2.orders.values()] == [bo.order.fills for bo in b.orders.values()]


def crash_and_resume(tmp_path, crash_after_sec, resume_after_sec):
    q = market(rise_after_min=2)
    dec = [T0 + pd.Timedelta(minutes=m) for m in range(1, 6)]
    full, fb = make()
    run_pipeline(q, buy_x(), full, fb, dec)
    full.finish(q["received_time"].max())

    log = audit.AuditLog(tmp_path / "audit.jsonl", "r1")
    t1, b1 = make(tmp_path / "broker.json")
    t1.sink = log.write
    crash = T0 + pd.Timedelta(seconds=crash_after_sec)
    run_pipeline(q, buy_x(), t1, b1, dec, stop_at=crash)               # 정리 없이 끝(강제 종료)
    log.close()

    resume = T0 + pd.Timedelta(seconds=resume_after_sec)
    b2 = PaperBroker.load(tmp_path / "broker.json")
    lost = catch_up(b2, q, resume)
    t2 = Trader(RiskConfig(), TraderConfig(horizon_min=2), b2, "F")
    notes = restore(t2, b2, audit.read(tmp_path / "audit.jsonl"), resume)
    run_pipeline(q, buy_x(), t2, b2, [d for d in dec if d > resume], after=resume)
    t2.finish(q["received_time"].max())
    return full, t2, lost, notes


def test_restore_while_holding_gives_same_trade(tmp_path):
    full, t2, lost, notes = crash_and_resume(tmp_path, 90, 120)
    assert pnl(full) == pnl(t2) and lost == 0
    assert any("보유 1계약" in n for n in notes)


def test_restore_entry_in_flight_adopts_fill(tmp_path):
    # 진입 주문을 보낸 직후(도착 전) 죽고, 꺼져 있던 사이 체결됐다 → 통지는 잃었지만 브로커 기록으로 되살린다
    full, t2, lost, notes = crash_and_resume(tmp_path, 60.6, 75)
    assert pnl(full) == pnl(t2) and lost == 2                         # FILL, DONE 통지
    assert t2.phase == Phase.FLAT and t2.broker.positions() == {}


def test_audit_read_skips_torn_last_line(tmp_path):
    p = tmp_path / "a.jsonl"
    log = audit.AuditLog(p, "r")
    log.write({"time": T0, "kind": "MODE"})
    log.close()
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"time": "2026-10-29T10:00:01", "ki')                  # 쓰다가 죽은 줄
    assert [r["kind"] for r in audit.read(p)] == ["MODE"]


def test_restore_after_exit_due_exits_now_and_clock_never_goes_back(tmp_path):
    # 청산 예정(1분 진입 + 2분) 뒤에 재시작 → 지난 예정 시각으로 돌아가지 않고 재시작 시각 이후에 판다
    full, t2, lost, notes = crash_and_resume(tmp_path, 90, 200)
    times = [x["time"] for x in t2.log]
    assert times == sorted(times)
    assert t2.trades[0]["status"] == "OK" and t2.broker.positions() == {}


# ── 구독 체인 이동 ───────────────────────────────────────
def test_chain_tracker_moves_after_two_strikes_and_keeps_held_code():
    ks = [930.0 + 2.5 * i for i in range(13)]                         # 930 ~ 960
    uni = pd.DataFrame({"code": [f"C{k:g}" for k in ks], "cp": "C", "strike": ks})
    tr = pipeline.ChainTracker(uni, width=2, center=945.0, recenter_steps=2)
    assert tr.check(943.0) is None                                     # 한 칸(942.5)은 기다린다
    subs = [("H0IOASP0", c) for c in tr.codes(tr.window(945.0))]       # 940 ~ 950
    q = asyncio.Queue()

    class FakeTrader:
        pos = {"code": "C950"}                                         # 950 콜 보유 중
        log = []

        def _log(self, kind, detail=""):
            self.log.append((kind, detail))

    t = FakeTrader()
    book = MarketBook(uni, "F", lambda m: pd.DataFrame())
    book.fut_quote = type("Q", (), {"bid1": 939.9, "ask1": 940.1})()   # ATM 940 → 두 칸 이동
    pipeline.resubscriber(tr, book, lambda: t, q, subs, limit=5)(T0)
    sent = [q.get_nowait() for _ in range(q.qsize())]
    assert [x for x in sent if not x[2]] == [("H0IOASP0", "C947.5", False)]   # 950은 보유 중이라 유지
    assert [x[1] for x in sent if x[2]] == ["C937.5"]                   # 한도 5건: ATM에 가까운 937.5만 추가
    assert len(subs) == 5 and ("H0IOASP0", "C950") in subs
    assert t.log[-1][0] == "RESUBSCRIBE"
