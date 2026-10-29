# 10장. 위험관리 엔진과 자동주문 상태관리 구현하기

준비: 8·9장과 같습니다(7장 예측기, 기록 파일, 그날의 선물 분봉). 주문은 내지 않습니다(가상 브로커).

```bash
# 10-1 매수 신호를 위험 한도에 비춰 보기
python examples/ch10/risk_check.py --db data/ticks_20261029_live.db                      # 기본: 만기일 진입 금지
python examples/ch10/risk_check.py --db data/ticks_20261029_live.db --allow-expiry-day   # 나머지 한도 보기

# 10-2~10-4 시그널 → 위험관리 → 상태기계 → 가상 브로커, 장애 주입
python examples/ch10/fault_replay.py --db data/ticks_20261029_live.db --scenario normal --log
python examples/ch10/fault_replay.py --db data/ticks_20261029_live.db --scenario all

# 단위 테스트(한도, 시세 건강 상태, 상태기계, 중복주문, 장애)
python -m pytest tests/test_risk.py -v
```

| 파일 | 내용 |
|---|---|
| `risk_check.py` | 매수 신호마다 한도 검사 결과(프리미엄·델타·사유) |
| `fault_replay.py` | 시나리오: default, normal, no_limits, data_stale, latency, disconnect, lost_ack, reject, loss_limit, kill, all. default 외에는 만기일 진입을 허용(기록 파일이 만기일) |

엔진 본체
- `src/optlab/risk/limits.py`: 한도 설정(RiskConfig)과 신규 진입 검사(check_entry)
- `src/optlab/risk/health.py`: 시세 건강 상태(HealthMonitor), Kill Switch
- `src/optlab/execution/trader.py`: 운영 상태(Mode)·포지션 단계(Phase) 상태기계, 멱등 주문, 대조(reconcile)
- `src/optlab/execution/paper.py`: 가상 브로커(장애 주입 가능)
- `src/optlab/execution/replay.py`·`scheduler.py`: 기록 재생과 시각 순서 스케줄러(11장 실시간 파이프라인과 공용)

설정: `configs/default.yaml`의 `risk` 절. Kill Switch는 실행 폴더에 `KILL` 파일을 만들면 걸립니다.
