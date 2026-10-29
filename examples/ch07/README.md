# 7장. LightGBM으로 코스피200 단기 방향과 변동폭 예측하기

```bash
python examples/ch07/train_eval.py          # 베이스라인·LightGBM 비교, 보정, 변동폭
python examples/ch07/walk_forward.py        # 월별 워크포워드
python examples/ch07/predict_service.py --train --train-end 2026-08-31
python examples/ch07/predict_service.py --mode replay --date 2026-10-28 --at 10:30
python examples/ch07/predict_service.py --train && python examples/ch07/predict_service.py --mode live --minutes 3
```
