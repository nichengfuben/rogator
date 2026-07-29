from __future__ import annotations

"""按 upstream 分桶的登录历史与选号策略。"""

import json
import logging
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.persist.migrate import migrate_login_history_upstream
from core.persist.paths import PROJECT_ROOT, login_history_path as upstream_login_history_path
from core.session.accounts import Account
from core.session.io import atomic_write_text

logger = logging.getLogger("rogator")

STALE_PICK_POOL: int = 20
_UTC8 = timezone(timedelta(hours=8))
_save_lock = threading.Lock()
_migrated_upstreams: set[str] = set()


def login_history_path(upstream: str) -> Path:
    return upstream_login_history_path(upstream, PROJECT_ROOT)


def format_utc8(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=_UTC8).strftime("%Y-%m-%d %H:%M:%S")


def _stale_rank(ts: Optional[float]) -> float:
    return ts if ts is not None else float("-inf")


def _maybe_migrate_upstream_login_history(upstream: str) -> None:
    key = upstream.strip().lower()
    if key in _migrated_upstreams:
        return
    _migrated_upstreams.add(key)
    migrate_login_history_upstream(key, PROJECT_ROOT, archive_unified=False)


class LoginHistoryStore:
    """upstream 内 username → {at_unix, at_utc8}。"""

    def __init__(self, upstream: str) -> None:
        self._upstream = upstream.strip().lower()
        self._path = login_history_path(self._upstream)
        self._records: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self.load()

    def load(self) -> None:
        _maybe_migrate_upstream_login_history(self._upstream)
        path = self._path
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
            logger.warning("Failed to load login history [%s]: %s", self._upstream, exc)
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
                self._path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(self._path, payload)
                self._dirty = False
            except OSError as exc:
                logger.debug("Failed to save login history [%s]: %s", self._upstream, exc)

    def pick_account(
        self,
        accounts: List[Account],
        *,
        eligible: Callable[[Account], bool],
    ) -> Optional[Account]:
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
