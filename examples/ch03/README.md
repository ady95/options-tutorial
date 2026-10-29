# 3장. 코스피200 위클리옵션 실시간 데이터 수집하기

```bash
# Replay Mode (합성 데이터, API 키 불필요)
python data_samples/generate_synthetic.py
python examples/ch03/chain_monitor.py --mode replay

# Live Data Mode (장중, KIS 실전 키 필요, 주문 없음)
python examples/ch03/chain_monitor.py --mode live --width 4 --seconds 60
```

구독 수 = (2*width+1)*2 + 1 이며 40건 이하여야 합니다.
