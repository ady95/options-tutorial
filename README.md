# AI로 코스피200 위클리옵션 분석·자동매매 따라하기 — 예제 코드

위키독스 도서 「[AI로 코스피200 위클리옵션 분석·자동매매 따라하기](https://wikidocs.net/book/21896)」의 실습 예제 저장소입니다.

> 한국투자증권 Open API와 LightGBM으로 구현하는 옵션 가격평가·기대손익 예측·실시간 모의매매, PPO·SAC 최적화

## ⚠️ 먼저 읽어 주세요

- 이 저장소의 코드는 **학습용**이며 투자 수익이나 자동주문의 안전성을 보장하지 않습니다.
- 기본 실습은 **실계좌로 주문하지 않습니다.** 실시간 시세는 수집에만 쓰고, 주문은 가상체결 엔진이 처리합니다.
- `data_samples/`의 데이터는 **합성 데이터(실제 시장 데이터 아님)**입니다. 이 데이터로 얻은 성과를 실전 수익성의 근거로 쓰지 마세요.
- 거래승수·호가단위·거래시간 등 상품명세는 바뀔 수 있습니다. [한국거래소](https://www.krx.co.kr/) 최신 공고로 확인하세요.

## 빠른 시작

```bash
git clone https://github.com/ady95/options-tutorial.git
cd options-tutorial
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python data_samples/generate_synthetic.py          # Replay용 합성 데이터 생성
python examples/ch01/payoff.py                      # 1장: 만기 손익
python examples/ch05/greeks_chain.py                # 5장: IV·Greeks 분석기
pytest
```

Live Data Mode(한국투자증권 실시간 시세)를 쓰려면 `.env.example`을 `.env`로 복사해 앱키를 넣으세요. `.env`는 절대 커밋하지 마세요.

## 실행 모드

| 모드 | 데이터 | 주문 | 용도 |
|---|---|---|---|
| Replay Mode | `data_samples/` 샘플 재생 | 가상체결 | 휴장일·API 미연결 상태에서도 모든 장 실습 |
| Live Data Mode | 한국투자증권 Open API 실시간 시세 | 가상체결 | 실시간 모의매매 |

## 장별 예제

| 장 | 주제 | 예제 | 상태 |
|---|---|---|---|
| 01 | 위클리옵션과 AI 자동매매 이해 | `examples/ch01` | ✅ |
| 02 | 개발 환경과 한국투자증권 Open API | `examples/ch02` | ✅ |
| 03 | 실시간 데이터 수집 | `examples/ch03` | ✅ |
| 04 | 데이터 저장과 품질 검증 | `examples/ch04` | ✅ |
| 05 | 이론가격·IV·Greeks | `examples/ch05` | ✅ |
| 06 | 학습 데이터와 특징 | `examples/ch06` | ✅ |
| 07 | LightGBM 단기 방향·변동폭 예측 | `examples/ch07` | ✅ |
| 08 | 기대손익과 매매신호 | `examples/ch08` | ✅ |
| 09 | 호가 기반 이벤트 백테스트 | `examples/ch09` | 집필 중 |
| 10 | 위험관리 엔진과 주문 상태관리 | `examples/ch10` | 집필 중 |
| 11 | 실시간 자동 모의매매 시스템 | `examples/ch11` | 집필 중 |
| 12 | 성과 분석과 안정성 검증 | `examples/ch12` | 집필 중 |
| 13 | 변동성 예측과 스프레드 전략 (선택) | `examples/ch13` | 집필 중 |
| 14 | PPO·SAC 강화학습 (선택) | `examples/ch14` | 집필 중 |

## 저장소 구조

```text
options-tutorial/
├── examples/chNN/     # 장별 실행 예제 (책의 장 번호와 1:1)
├── data_samples/      # Replay용 합성 샘플과 생성 스크립트
├── src/optlab/
│   ├── broker/        # 증권사 API 추상화 (KIS, Replay)
│   ├── market_data/   # 상품명세, 시세 수집
│   ├── features/      # 특징 생성
│   ├── models/        # LightGBM, PPO·SAC
│   ├── pricing/       # Black-76 가격평가·IV·Greeks
│   ├── strategy/      # 기대순손익·CALL/PUT/HOLD 신호
│   ├── backtest/      # 호가 기반 이벤트 백테스트
│   ├── execution/     # 가상 주문·체결
│   └── risk/          # 독립 위험관리·Kill Switch
├── tests/
└── configs/
```

## 참고 자료

- 한국거래소: <https://www.krx.co.kr/>
- 한국투자증권 KIS Developers: <https://apiportal.koreainvestment.com/>
- 한국투자증권 공식 Open API 예제: <https://github.com/koreainvestment/open-trading-api>

## 라이선스

MIT License
