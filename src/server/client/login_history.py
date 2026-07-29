from __future__ import annotations

"""账号登录历史与选号：持久化 UTC+8 时间戳，按「未登录优先 / 最久未登录池」随机。"""

import json
import logging
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from accounts import Account
from server.client.session_store import _atomic_write_text

logger = logging.getLogger("rogator")

LOGIN_HISTORY_FILE: str = "persist/login_history.json"
STALE_PICK_POOL: int = 20
_UTC8 = timezone(timedelta(hours=8))
_save_lock = threading.Lock()


def format_utc8(ts: float) -> str:
    """Unix 秒 → ``YYYY-MM-DD HH:MM:SS``（UTC+8）。"""
    return datetime.fromtimestamp(ts, tz=_UTC8).strftime("%Y-%m-%d %H:%M:%S")


def _stale_rank(ts: Optional[float]) -> float:
    """未登录视为最久未登录（排序最前）。"""
    return ts if ts is not None else float("-inf")


class LoginHistoryStore:
    """username → {at_unix, at_utc8}；选号策略与历史同源。"""

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self.load()

    def load(self) -> None:
        path = Path(LOGIN_HISTORY_FILE)
        if not path.is_file():
            self._records = {}
            self._dirty = False
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("logins") or {}
            self._records = (
                {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}
                if isinstance(raw, dict) else {}
            )
        except Exception as exc:
            logger.warning("Failed to load login history: %s", exc)
            self._records = {}
        self._dirty = False

    def last_login_unix(self, username: str) -> Optional[float]:
        entry = self._records.get(username)
        if not entry:
            return None
        val = entry.get("at_unix")
        return float(val) if val is not None else None

    def record(self, username: str, *, at: Optional[float] = None) -> None:
        ts = time.time() if at is None else at
        self._records[username] = {"at_unix": ts, "at_utc8": format_utc8(ts)}
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        payload = json.dumps(
            {"updated_at": format_utc8(time.time()), "logins": self._records},
            ensure_ascii=False,
            indent=2,
        )
        with _save_lock:
            try:
                _atomic_write_text(Path(LOGIN_HISTORY_FILE), payload)
                self._dirty = False
            except OSError as exc:
                logger.debug("Failed to save login history: %s", exc)

    def pick_account(
        self,
        accounts: List[Account],
        *,
        eligible: Callable[[Account], bool],
    ) -> Optional[Account]:
        """未登录过优先随机；否则在最久未登录池（≥20 账号时为 20）中随机。"""
        pool = [a for a in accounts if eligible(a)]
        if not pool:
            return None

        fresh = [a for a in pool if self.last_login_unix(a.username) is None]
        if fresh:
            return random.choice(fresh)

        if len(accounts) >= STALE_PICK_POOL:
            stale_names = {
                a.username
                for a in sorted(
                    accounts,
                    key=lambda a: _stale_rank(self.last_login_unix(a.username)),
                )[:STALE_PICK_POOL]
            }
            pool = [a for a in pool if a.username in stale_names] or pool

        return random.choice(pool)
