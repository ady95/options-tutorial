# 11장. 실시간 AI 옵션 자동 모의매매 시스템 완성하기

준비: 8~10장과 같습니다(7장 예측기, 기록 파일, 그날의 선물 분봉). 주문은 가상 브로커로만 처리합니다.

```bash
# 11-1 Replay: 기록 파일을 실시간처럼 흘려 전체 파이프라인 실행(체결로 분봉 생성, 매분 2초에 결정)
python examples/ch11/paper_trade.py --mode replay --db data/ticks_20261029_live.db --allow-expiry-day --run-dir runs/demo
# 10장 재생기와 같은 조건(완성된 분봉 파일, 지연 0초) → 10장 normal과 같은 결과
python examples/ch11/paper_trade.py --mode replay --db data/ticks_20261029_live.db --allow-expiry-day --bars rest --delay 0 --run-dir runs/same10
# 20배속: Live처럼 타이머가 도는 경로
python examples/ch11/paper_trade.py --mode replay --db data/ticks_20261029_live.db --allow-expiry-day --speed 20 --run-dir runs/speed20

# 11-3 강제 종료와 재시작 복구
python examples/ch11/paper_trade.py --mode replay --db data/ticks_20261029_live.db --allow-expiry-day --run-dir runs/crash --crash-at 10:21:59
python examples/ch11/paper_trade.py --mode replay --db data/ticks_20261029_live.db --allow-expiry-day --run-dir runs/crash --resume --resume-at 10:30:00

# 11-4 Live: 장중 KIS 실시간 시세(실전 앱키 필요, 시세 조회만). 주문은 가상 브로커
python examples/ch11/paper_trade.py --mode live --minutes 60

# 단위 테스트(분봉 생성, 체인, 파이프라인 = 10장 재생기, 브로커 상태 저장, 재시작 복구)
python -m pytest tests/test_live.py -v
```

| 옵션 | 내용 |
|---|---|
| `--mode replay\|live` | 기본값은 설정 파일의 `mode` |
| `--bars stream\|rest` | Replay 분봉 출처. stream은 선물 체결로 직접 생성(Live와 같음) |
| `--delay` | 결정 지연(초). 기본은 `run.decide_delay_sec`(2) |
| `--speed` | Replay 배속. 0이면 기다리지 않음 |
| `--crash-at`, `--resume`, `--resume-at` | 강제 종료 흉내, 이어 가기 |
| `--no-restore` | (비교용) 복구 없이 재시작 — 하면 안 되는 방법 |
| `--allow-expiry-day` | 만기일 신규 진입 허용(설정 기본값은 금지) |

실행 기록 폴더(`runs/...`)에 남는 것
- `audit.jsonl`: 감사 기록. 상태기계의 모든 기록 + 진입 의도(INTENT)·확정 거래(TRADE). 한 줄 JSON, 덧붙여 쓰기만 함
- `broker.json`: 가상 브로커 상태(주문·체결·잔고). 재시작 대조의 기준
- `ticks.db`: Live에서 받은 시세 그대로(04장 형식). 12장 분석과 Replay 재현에 쓴다

본체
- `src/optlab/live/market.py`: 체인·선물 분봉(MarketBook, BarBuilder)
- `src/optlab/live/pipeline.py`: asyncio 파이프라인(수신·시계 작업), 결정 함수, Replay·Live 수신기, 구독 체인 이동(ChainTracker, resubscriber)
- `src/optlab/live/audit.py`: 감사 기록
- `src/optlab/execution/scheduler.py`: 시각 순서 스케줄러(10장 재생기와 공용)
- `src/optlab/execution/recovery.py`: 재시작 복구(restore), 꺼져 있던 동안의 거래소 흉내(catch_up, Replay 전용)

구독 체인: Live는 선물 2건 + ATM ± 9 행사가 옵션 호가 38건 = 40건을 구독하고, 선물 가격이 가운데 행사가에서 2칸(`run.recenter_steps`) 이상 멀어지면 해제 후 추가로 체인을 옮깁니다. 보유 중인 종목은 해제하지 않습니다. Replay는 옮겨야 할 시점만 기록합니다(기록 파일에 새 행사가의 호가가 없음).

Live 전에: `data/futures_1min_continuous.parquet`가 전 거래일까지 있어야 합니다(06장 `download_futures.py`).
기본 설정은 만기일 신규 진입을 막으므로, 목요일 위클리를 목요일에 돌리면 신호가 모두 EXPIRY_DAY로 거절됩니다.
