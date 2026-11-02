#!/usr/bin/env bash
# 14-4절: 실험 12개(비용 2가지 × PPO·SAC × 시드 3개)를 3개씩 돌린다. 프로세스 하나에 메모리 약 1GB
#   bash examples/ch14/run_all.sh                       # 파이썬이 python이 아니면 PYTHON=경로 로 지정
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHON="${PYTHON:-python}"
mkdir -p runs/ch14/logs
for cost in real none; do for algo in sac ppo; do for seed in 0 1 2; do echo "$algo $seed $cost"; done; done; done |
  xargs -P 3 -L 1 bash -c '"$PYTHON" examples/ch14/train.py --algo $0 --seed $1 --cost $2 > runs/ch14/logs/$0_$2_s$1.log 2>&1'
echo ALL_DONE
