# 9장. 호가 기반 이벤트 백테스트 구현하기

준비: 8장과 같습니다(7장 예측기, 기록 파일을 쓸 때는 그날의 선물 분봉). 기록 파일이 없으면 합성 데이터로 실행됩니다.

```bash
# 09-1 같은 매매를 마지막 체결가·중간가·호가로 계산하면
python examples/ch09/price_bias.py --db data/ticks_20261029_live.db
python examples/ch09/price_bias.py --db data/ticks_20261029_mon.db

# 09-2 주문 지연·수량·지정가에 따른 체결
cd examples/ch09
python fill_sweep.py --db ../../data/ticks_20261029_live.db
python fill_sweep.py --db ../../data/ticks_20261029_mon.db --strategy tightest
cd ../..

# 09-3·09-4 백테스트 리포트
python examples/ch09/backtest.py --db data/ticks_20261029_live.db                 # 포지션 하나로 운용
python examples/ch09/backtest.py --db data/ticks_20261029_live.db --each          # 신호마다 따로
python examples/ch09/backtest.py --db data/ticks_20261029_live.db --each --strategy tightest   # 비교 기준
python examples/ch09/backtest.py --db data/ticks_20261029_mon.db --each --strategy tightest
python examples/ch09/backtest.py --strategy tightest --start 2026-01-05T09:00 --end 2026-01-09T15:30   # 합성 데이터
```

| 파일 | 내용 |
|---|---|
| `price_bias.py` | 모든 옵션·시점의 왕복을 마지막 체결가·중간가·호가로 계산해 비교 |
| `backtest.py` | 이벤트 백테스트 실행과 리포트. `--strategy signal`(8장 엔진) / `tightest`(예측 없이 스프레드가 좁은 ATM 매수) |
| `fill_sweep.py` | 같은 신호로 지연·수량·진입 틱만 바꿔 체결률·슬리피지·손익 비교 |

엔진 본체: `src/optlab/backtest/` (fills.py 가상체결, engine.py 이벤트 루프, report.py 요약)
설정: `configs/default.yaml`의 `backtest` 절. 주문은 내지 않습니다(가상체결).
