from __future__ import annotations

"""按 upstream 分文件加载账号池。"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("rogator")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ROOT_CSV = PROJECT_ROOT / "accounts.csv"

_UPSTREAM_CSV: Dict[str, Path] = {
    "qwen": PROJECT_ROOT / "persist" / "qwen" / "accounts.csv",
    "deepseek": PROJECT_ROOT / "persist" / "deepseek" / "accounts.csv",
}


@dataclass
class Account:
    username: str
    password: str
    token: str = ""
    user_id: str = ""
    password_hash: str = ""
    token_expires: float = 0.0
    memory_disabled: bool = False
    context_length: Optional[int] = None
    is_login: bool = False
    last_login: float = 0.0
    area_code: str = ""


def accounts_csv_path(upstream: str) -> Path:
    key = upstream.strip().lower()
    if key not in _UPSTREAM_CSV:
        raise KeyError(f"unknown upstream for accounts: {upstream}")
    return _UPSTREAM_CSV[key]


def _read_csv(path: Path) -> List[Account]:
    if not path.exists():
        return []
    out: List[Account] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            email = (row.get("email") or "").strip()
            phone = (row.get("phone") or row.get("mobile") or "").strip()
            username = email or phone
            password = (row.get("password") or "").strip()
            area_code = (row.get("area_code") or "").strip()
            if username and password:
                out.append(Account(username=username, password=password, area_code=area_code))
    return out


def accounts_for_upstream(upstream: str) -> List[Account]:
    key = upstream.strip().lower()
    path = _UPSTREAM_CSV.get(key)
    if path is not None and path.exists():
        return _read_csv(path)
    if key == "qwen" and _ROOT_CSV.exists():
        return _read_csv(_ROOT_CSV)
    return []


ACCOUNTS: List[Account] = accounts_for_upstream("qwen")
ACCOUNTS_CSV_PATH: Path = accounts_csv_path("qwen")
