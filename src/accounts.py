from __future__ import annotations

"""账户池：从 accounts.csv 加载账户信息。"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ACCOUNTS_CSV_PATH: Path = Path(__file__).resolve().parent.parent / "accounts.csv"


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


def _load_accounts() -> List[Account]:
    """从 accounts.csv 读取账户 (列: email,password,name)。"""
    if not ACCOUNTS_CSV_PATH.exists():
        return []
    accounts: List[Account] = []
    with ACCOUNTS_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            email = (row.get("email") or "").strip()
            password = (row.get("password") or "").strip()
            if email and password:
                accounts.append(Account(username=email, password=password))
    return accounts


ACCOUNTS: List[Account] = _load_accounts()
