"""감사 기록 — 상태기계의 모든 기록을 한 줄 JSON으로 덧붙여 쓴다(append-only).

- 한 줄씩 쓰고 바로 flush·fsync한다. 프로그램이 죽어도 마지막 줄까지 남는다.
- 고치거나 지우지 않는다. 재시작 복구는 이 파일을 처음부터 읽어 필요한 것만 꺼낸다(optlab.execution.recovery).
- 시각은 ISO 문자열, 숫자는 그대로 둔다. pandas 없이도 읽을 수 있다.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def _default(o):
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if hasattr(o, "item"):                    # numpy 숫자
        return o.item()
    return str(o)


class AuditLog:
    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.f = open(self.path, "a", encoding="utf-8")

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps({"run": self.run_id, **rec}, ensure_ascii=False, default=_default) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()


def read(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:          # 쓰다가 죽은 마지막 줄
            break
    return out
