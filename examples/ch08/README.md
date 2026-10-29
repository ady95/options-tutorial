# 8장. AI 예측으로 옵션 기대손익과 매매신호 계산하기

준비: 7장 예측기(`models_store/forecaster.joblib`), 기록 파일을 쓸 때는 그날의 선물 분봉

```bash
python examples/ch06/download_futures.py --start 2026-10-29 --end 2026-10-29 --out data/futures_1min_20261029.parquet
```

```bash
# 08-1 예측분포로 옵션 한 종목 재평가
python examples/ch08/reprice.py --db data/ticks_20261029_live.db --time 10:10 --cp P --strike 940
python examples/ch08/reprice.py --time 2026-01-08T10:30 --cp C                 # 합성 데이터

# 08-2~08-4 시그널 엔진
python examples/ch08/signal_engine.py --db data/ticks_20261029_live.db --time 10:10   # 한 시점 자세히
python examples/ch08/signal_engine.py --db data/ticks_20261029_live.db --check        # 기록 구간 전체 + 사후 확인
python examples/ch08/signal_engine.py --start 2026-01-08T09:30 --end 2026-01-08T11:30 --every 10   # 합성 데이터
```

| 파일 | 내용 |
|---|---|
| `market.py` | 기록 파일·합성 데이터에서 시점별 체인 호가, 선도가격, 분봉을 꺼내는 공용 코드 |
| `reprice.py` | 지수 시나리오별 옵션 가치, 기대 가치 비교(지수 그대로 / 시장 IV / 예측분포) |
| `signal_engine.py` | 후보 필터, 기대순손익 분해, CALL BUY·PUT BUY·HOLD 신호 |

엔진 본체: `src/optlab/strategy/` (clock.py 변동성 시계, scenarios.py 예측→시나리오, signal.py 기대순손익과 신호)
주문은 내지 않습니다.
