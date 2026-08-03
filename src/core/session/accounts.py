from __future__ import annotations

"""按 upstream 分文件加载账号池。"""

import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from core.persist.paths import persist_root

logger = logging.getLogger("rogator")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ROOT_CSV = PROJECT_ROOT / "accounts.csv"

KNOWN_ACCOUNT_UPSTREAMS: tuple[str, ...] = ("qwen", "deepseek")


def accounts_csv_path_for_root(upstream: str, root: Path) -> Path:
    key = upstream.strip().lower()
    if key not in KNOWN_ACCOUNT_UPSTREAMS:
        raise KeyError(f"unknown upstream for accounts: {upstream}")
    return root / "config" / "upstream" / key / "accounts.csv"


def accounts_csv_path(upstream: str) -> Path:
    return accounts_csv_path_for_root(upstream, PROJECT_ROOT)


def _legacy_persist_accounts_csv(upstream: str, root: Path) -> Path:
    return persist_root(root) / upstream.strip().lower() / "accounts.csv"


def _move_if_missing(src: Path, dest: Path, *, label: str) -> None:
    if not src.is_file() or dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    logger.info("已迁移账号 CSV [%s] → %s", label, dest)


def migrate_accounts_csv_layout(root: Path | None = None) -> None:
    """一次性迁移：persist/<upstream>/accounts.csv、根 accounts.csv → config/upstream/<upstream>/。"""
    base = root or PROJECT_ROOT
    for upstream in KNOWN_ACCOUNT_UPSTREAMS:
        dest = accounts_csv_path_for_root(upstream, base)
        _move_if_missing(
            _legacy_persist_accounts_csv(upstream, base),
            dest,
            label=upstream,
        )
    _move_if_missing(
        base / "accounts.csv",
        accounts_csv_path_for_root("qwen", base),
        label="qwen (root)",
    )


_UPSTREAM_CSV: Dict[str, Path] = {
    upstream: accounts_csv_path_for_root(upstream, PROJECT_ROOT)
    for upstream in KNOWN_ACCOUNT_UPSTREAMS
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


migrate_accounts_csv_layout()

ACCOUNTS: List[Account] = accounts_for_upstream("qwen")
ACCOUNTS_CSV_PATH: Path = accounts_csv_path("qwen")
