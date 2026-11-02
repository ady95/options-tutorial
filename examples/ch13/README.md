# 13장. AI 변동성 예측과 옵션 스프레드 전략 확장하기 (선택)

준비: 7장 예측기, 6장 선물 분봉과 날짜별 분봉(`data/futures_1min_YYYYMMDD.parquet`), 실제 기록 세 개
(`data/ticks_20261029_live.db`, `data/ticks_20261029_mon.db`, 11장 Live `runs/20261030-live/ticks.db`).
만기 결제가격(그날 코스피200 종가)은 KIS 지수 일봉으로 한 번 받아 `data/ch13/settle.json`에 둡니다(실전 앱키, 시세 조회만).

```bash
# 13-1 장중 변동성 패턴, IV를 15분 구간으로 맞추기, HAR 실현분산 예측, 만기까지 IV·예측·실제, 스큐
python examples/ch13/iv_rv.py

# 13-2 수직 스프레드: 손익 구조, 호가 비용, 다리 내는 방식(sequential·together), 15분·만기 보유
python examples/ch13/spread.py
python examples/ch13/spread.py --session thu_expiry_am --at 10:25

# 13-3 A형(방향 + 단일 옵션) vs A+B형(방향 + 변동성 관점으로 단일·스프레드 선택)
python examples/ch13/combined.py

python -m pytest tests/test_spreads.py -v
```

| 파일 | 내용 |
|---|---|
| `iv_rv.py` | 13-1 |
| `spread.py` | 13-2. 스프레드 표, 15분 보유 체결 결과, 만기 보유 손익 |
| `combined.py` | 13-3 |
| `common.py` | 기록 목록, 분봉, 결제가격 |

모듈: `src/optlab/strategy/volview.py`(장중 패턴, IV 구간 변환, HAR), `src/optlab/strategy/spreads.py`(스마일, 스큐, 수직 스프레드),
`src/optlab/backtest/legs.py`(여러 다리 가상체결, 09장과 같은 규칙)
