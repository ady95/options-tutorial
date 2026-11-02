# 14장. PPO·SAC 강화학습으로 매매 전략 최적화하기

실제 선물 1분봉(2026-04~10) 위에 Black-76으로 만든 **합성 옵션 시장**에서 PPO(이산 행동)와 SAC(연속 비중)를
학습하고, 규칙·LightGBM(08장 시그널 엔진)과 같은 비용·같은 체결 가정으로 비교합니다.
합성 옵션 시장은 실제 옵션 호가가 아닙니다. 결과를 수익성 근거로 쓰지 마세요.

## 준비

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[rl]"
```

12장 워크포워드 예측(`data/ch12/wf_expanding.parquet`)이 필요합니다. 없으면 처음 실행할 때 만듭니다(몇 분).
합성 시장은 `data/ch14/days.pkl`에 저장해 두고 다시 씁니다.

## 실행 순서

| 절 | 명령 | 내용 |
|---|---|---|
| 14-1 | `python examples/ch14/env_check.py` | 관측·행동, 비용과 15분 움직임, 생각 없는 정책 |
| 14-2 | `python examples/ch14/train.py --algo ppo --seed 0` | PPO 30만 걸음(CPU 1코어 약 10분) |
| 14-3 | `python examples/ch14/train.py --algo sac --seed 0` | SAC 6만 걸음(CPU 1코어 약 20분) |
| 14-4 | `bash examples/ch14/run_all.sh` 후 `python examples/ch14/compare.py` | 시드 3개 × 비용 2가지 전체 비교 |

`train.py` 옵션: `--cost real|wide|none`(스프레드 7%·12%·1틱), `--penalty 0.5`(회전율 패널티, 계약당 포인트),
`--steps N`.

결과: `runs/ch14/models/`(검증 기간 손익이 가장 좋았던 모델과 마지막 모델), `runs/ch14/results/*.json`(학습 곡선과
학습·검증·테스트 성과), `runs/ch14/baselines.json`(기준 정책).

## 기간

| 용도 | 기간 | 날 수 |
|---|---|---|
| 학습 | 2026-04~07 | 83 |
| 검증(모델 고르기) | 2026-08 | 20 |
| 테스트(한 번만) | 2026-09~10 | 40 |

## 메모리

학습 프로세스 하나가 메모리를 약 1GB 씁니다. `run_all.sh`는 3개씩 돌립니다. 메모리가 부족하면 스왑 때문에
수십 배 느려집니다(집필 중 12개를 한꺼번에 돌렸다가 겪었습니다).
