# 12장. 거래 성과 분석과 시스템 안정성 검증하기

준비: 7장 예측기(`models_store/forecaster.joblib`), 6장 선물 분봉(`data/futures_1min_continuous.parquet`).
그 뒤에 받은 날짜별 분봉(`data/futures_1min_YYYYMMDD.parquet`)이 있으면 함께 씁니다.
워크포워드 예측은 처음 한 번 학습한 뒤 `data/ch12/`에 저장해 다시 씁니다.

```bash
# 12-1 성과 지표: 09장 백테스트 거래 내역(이름=경로) 또는 11장 실행 기록 폴더
python examples/ch09/backtest.py --db data/ticks_20261029_live.db --each --save runs/ch12/thu_signal.csv
python examples/ch09/backtest.py --db data/ticks_20261029_live.db --each --strategy tightest --save runs/ch12/thu_tightest.csv
python examples/ch12/performance.py thu_signal=runs/ch12/thu_signal.csv thu_tightest=runs/ch12/thu_tightest.csv

# 12-2 국면별 예측 품질(요일·전날 변동성·시간대, 거래일 부트스트랩 구간). 4분 남짓
python examples/ch12/regimes.py

# 12-3 재학습 정책 비교, PSI, 20거래일 감시와 변동폭 보정
python examples/ch12/drift.py

# 12-4 대시보드(HTML 한 장): runs/ch12/dashboard.html
python examples/ch12/dashboard.py --runs runs/20261030-live data/ticks_20261029_live.db --trades thu_signal=runs/ch12/thu_signal.csv

python -m pytest tests/test_analysis.py -v
```

| 파일 | 내용 |
|---|---|
| `performance.py` | 승률·손익비·손익 배수·MDD·최대 연속 손실, 거래/거래일 부트스트랩 구간, 필요한 거래 수 |
| `regimes.py` | 2026-04~10 워크포워드 15분 예측을 국면별로 평가 |
| `drift.py` | fixed·expanding·rolling·deployed 정책 비교, 특징 PSI, cov90·AUC 20일 감시, `VolCalibrator` |
| `dashboard.py` | 운영 안정성(실행 기록·시세), 거래 성과, 모델 감시를 HTML로 |
| `common.py` | 분봉 → 특징·레이블, 워크포워드 예측 캐시 |

모듈 본체: `src/optlab/analysis/` (performance.py, regimes.py, drift.py, health.py)
