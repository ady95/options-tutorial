"""한국투자증권(KIS) Open API REST 어댑터 — 시세 조회 전용.

이 책의 실습은 주문 API를 호출하지 않는다. 주문은 execution 패키지의 가상체결 엔진이 처리한다.

환경 변수(.env):
    KIS_REAL_APP_KEY, KIS_REAL_APP_SECRET     실전 도메인(실시간 시세 수집용)
    KIS_PAPER_APP_KEY, KIS_PAPER_APP_SECRET   모의투자 도메인(선택)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

DOMAINS = {
    "real": "https://openapi.koreainvestment.com:9443",
    "paper": "https://openapivts.koreainvestment.com:29443",
}
# 모의투자 도메인은 초당 호출 한도가 낮다(2026-10 실측: 연속 호출 시 "초당 거래건수를 초과").
MIN_INTERVAL_SEC = {"real": 0.06, "paper": 0.55}
TOKEN_CACHE = Path(".cache") / "kis_token_{mode}.json"


class KisError(RuntimeError):
    pass


@dataclass(frozen=True)
class KisConfig:
    mode: str
    app_key: str
    app_secret: str

    @classmethod
    def from_env(cls, mode: str = "real") -> "KisConfig":
        prefix = f"KIS_{mode.upper()}_"
        key, secret = os.getenv(prefix + "APP_KEY"), os.getenv(prefix + "APP_SECRET")
        if not key or not secret:
            raise KisError(f"{prefix}APP_KEY / {prefix}APP_SECRET 환경 변수가 없습니다 (.env 확인)")
        return cls(mode=mode, app_key=key, app_secret=secret)


class KisClient:
    def __init__(self, config: KisConfig, session: requests.Session | None = None):
        self.cfg = config
        self.base = DOMAINS[config.mode]
        self.session = session or requests.Session()
        self._last_call = 0.0
        self._token: str | None = None

    # ── 인증 ─────────────────────────────────────────────
    def access_token(self) -> str:
        """접근토큰(24시간 유효). 재발급 제한이 있으므로 파일에 캐시해 재사용한다."""
        if self._token:
            return self._token
        cache = Path(str(TOKEN_CACHE).format(mode=self.cfg.mode))
        if cache.exists():
            c = json.loads(cache.read_text(encoding="utf-8"))
            if c.get("app_key_tail") == self.cfg.app_key[-4:] and c["expires_at"] > time.time() + 600:
                self._token = c["token"]
                return self._token
        r = self.session.post(f"{self.base}/oauth2/tokenP", json={
            "grant_type": "client_credentials",
            "appkey": self.cfg.app_key, "appsecret": self.cfg.app_secret}, timeout=10)
        j = r.json()
        if "access_token" not in j:
            raise KisError(f"토큰 발급 실패: {j.get('error_code')} {j.get('error_description')}")
        self._token = j["access_token"]
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps({
            "token": self._token, "app_key_tail": self.cfg.app_key[-4:],
            "expires_at": time.time() + int(j.get("expires_in", 86400))}), encoding="utf-8")
        return self._token

    def approval_key(self) -> str:
        """WebSocket 접속키."""
        r = self.session.post(f"{self.base}/oauth2/Approval", json={
            "grant_type": "client_credentials",
            "appkey": self.cfg.app_key, "secretkey": self.cfg.app_secret}, timeout=10)
        key = r.json().get("approval_key")
        if not key:
            raise KisError(f"WebSocket 접속키 발급 실패: {r.text[:120]}")
        return key

    # ── 공통 GET ────────────────────────────────────────
    def get(self, path: str, tr_id: str, params: dict, output: str = "output") -> dict:
        wait = MIN_INTERVAL_SEC[self.cfg.mode] - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        headers = {
            "authorization": f"Bearer {self.access_token()}",
            "appkey": self.cfg.app_key, "appsecret": self.cfg.app_secret,
            "tr_id": tr_id, "custtype": "P",
        }
        r = self.session.get(self.base + path, headers=headers, params=params, timeout=10)
        self._last_call = time.time()
        j = r.json()
        if j.get("rt_cd") != "0":
            raise KisError(f"{tr_id} 실패: {j.get('msg_cd')} {j.get('msg1', '').strip()}")
        data = j.get(output)
        # 잘못된 종목코드도 rt_cd=0으로 응답하고 output만 비어 온다(2026-10 실측) → 반드시 내용 검사
        if not data or (isinstance(data, dict) and not any(data.values())):
            raise KisError(f"{tr_id} 응답이 비어 있습니다. 종목코드를 확인하세요: {params}")
        return data

    # ── 시세 조회 ───────────────────────────────────────
    def index_price(self, index_code: str = "2001") -> dict:
        """업종 지수 현재가. 2001 = 코스피200."""
        return self.get("/uapi/domestic-stock/v1/quotations/inquire-index-price", "FHPUP02100000",
                        {"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": index_code})

    def futopt_price(self, code: str, market: str = "O") -> dict:
        """선물(market='F', 코드 예 A01612)·옵션(market='O', 위클리 포함 9자리 코드) 현재가."""
        return self.get("/uapi/domestic-futureoption/v1/quotations/inquire-price", "FHMIF10000000",
                        {"FID_COND_MRKT_DIV_CODE": market, "FID_INPUT_ISCD": code}, output="output1")

    def futopt_orderbook(self, code: str, market: str = "O") -> dict:
        """5단계 호가."""
        return self.get("/uapi/domestic-futureoption/v1/quotations/inquire-asking-price", "FHMIF10010000",
                        {"FID_COND_MRKT_DIV_CODE": market, "FID_INPUT_ISCD": code}, output="output2")


def load_dotenv_if_present(path: str | Path | None = None) -> None:
    """python-dotenv가 있으면 .env를 읽는다(이미 설정된 환경 변수는 덮어쓰지 않음).

    path가 없거나 존재하지 않으면 현재 폴더에서 상위 폴더로 올라가며 .env를 찾는다.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    if path is None or not Path(path).exists():
        path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)
