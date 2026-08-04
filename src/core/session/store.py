from __future__ import annotations

"""平台共享 session 持久化与分桶读写。"""

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.persist.migrate import migrate_sessions_upstream
from core.persist.paths import PROJECT_ROOT, sessions_path as upstream_sessions_path
from core.session.accounts import Account
from core.session.io import atomic_write_text

logger = logging.getLogger("rogator")

CLEANUP_INTERVAL: float = 60.0
MUTE_LOGIN_BLOCK_SECONDS: float = 86400.0
_save_lock = threading.Lock()
_migrated_upstreams: set[str] = set()


@dataclass
class SessionStoreMeta:
    current_index: int = 0
    blocked_accounts: Dict[str, float] = field(default_factory=dict)
    muted_accounts: Dict[str, float] = field(default_factory=dict)


@dataclass
class PlatformSession:
    account: Account
    token: str
    user_id: str
    upstream: str = "qwen"
    login_time: float = field(default_factory=time.time)
    is_valid: bool = True

    @property
    def username(self) -> str:
        return self.account.username

    def is_expired(self) -> bool:
        if not self.token or not self.is_valid:
            return True
        exp = _jwt_exp(self.token)
        if exp is not None:
            return time.time() >= exp - 30
        if self.upstream == "deepseek":
            ttl = _deepseek_session_ttl()
            return time.time() >= self.login_time + ttl - 30
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.account.username,
            "password": self.account.password,
            "token": self.token,
            "user_id": self.user_id,
            "upstream": self.upstream,
            "login_time": self.login_time,
            "is_valid": self.is_valid,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], *, upstream: str) -> "PlatformSession":
        account = Account(username=data["username"], password=data["password"])
        return PlatformSession(
            account=account,
            token=data["token"],
            user_id=data.get("user_id", ""),
            upstream=str(data.get("upstream") or upstream),
            login_time=data.get("login_time", time.time()),
            is_valid=data.get("is_valid", True),
        )


def sessions_file(upstream: str) -> Path:
    return upstream_sessions_path(upstream, PROJECT_ROOT)


def _deepseek_session_ttl() -> float:
    try:
        from server.config.app_config import _load_upstream_toml

        raw = _load_upstream_toml("deepseek")
        session = raw.get("session") if isinstance(raw, dict) else None
        if isinstance(session, dict) and session.get("token_ttl_seconds") is not None:
            return float(session["token_ttl_seconds"])
    except Exception:
        pass
    return 3600.0


def _jwt_exp(token: str) -> Optional[float]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return float(exp) if exp is not None else None
    except Exception:
        return None


def _empty_upstream_store() -> Dict[str, Any]:
    return {
        "sessions": [],
        "current_index": 0,
        "blocked_accounts": {},
        "muted_accounts": {},
        "updated_at": 0,
    }


def _maybe_migrate_upstream_sessions(upstream: str) -> None:
    key = upstream.strip().lower()
    if key in _migrated_upstreams:
        return
    _migrated_upstreams.add(key)
    migrate_sessions_upstream(key, PROJECT_ROOT, archive_unified=False)


def _read_upstream_store(upstream: str) -> Dict[str, Any]:
    _maybe_migrate_upstream_sessions(upstream)
    path = sessions_file(upstream)
    if not path.exists():
        return _empty_upstream_store()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "sessions" in raw:
            return raw
    except Exception as exc:
        logger.warning("Failed to load sessions store [%s]: %s", upstream, exc)
    return _empty_upstream_store()


def load_upstream_sessions(upstream: str) -> Tuple[List[PlatformSession], SessionStoreMeta]:
    meta = SessionStoreMeta()
    bucket = _read_upstream_store(upstream)
    meta.current_index = int(bucket.get("current_index") or 0)
    raw_blocked = bucket.get("blocked_accounts") or {}
    if isinstance(raw_blocked, dict):
        meta.blocked_accounts = {str(k): float(v) for k, v in raw_blocked.items()}
    raw_muted = bucket.get("muted_accounts") or {}
    if isinstance(raw_muted, dict):
        meta.muted_accounts = {str(k): float(v) for k, v in raw_muted.items()}
    sessions = [
        PlatformSession.from_dict(item, upstream=upstream)
        for item in bucket.get("sessions") or []
        if isinstance(item, dict)
    ]
    restored = [s for s in sessions if not s.is_expired() and s.is_valid]
    if restored:
        logger.info("Restored %d %s session(s) from disk", len(restored), upstream)
    return restored, meta


def save_upstream_sessions(
    upstream: str,
    sessions: List[PlatformSession],
    *,
    current_index: int = 0,
    blocked_accounts: Optional[Dict[str, float]] = None,
    muted_accounts: Optional[Dict[str, float]] = None,
) -> List[str]:
    return _save_upstream_sessions_impl(
        upstream,
        sessions,
        current_index=current_index,
        blocked_accounts=blocked_accounts,
        muted_accounts=muted_accounts,
    )


def _save_upstream_sessions_impl(
    upstream: str,
    sessions: List[PlatformSession],
    *,
    current_index: int = 0,
    blocked_accounts: Optional[Dict[str, float]] = None,
    muted_accounts: Optional[Dict[str, float]] = None,
) -> List[str]:
    cleaned, removed = clean_expired(sessions)
    sessions[:] = cleaned
    if removed:
        logger.info("Cleanup [%s]: removed %d expired/invalid session(s)", upstream, len(removed))
    now = time.time()
    blocked = {k: v for k, v in (blocked_accounts or {}).items() if v > now}
    muted = {
        k: float(v)
        for k, v in (muted_accounts or {}).items()
        if now - float(v) < MUTE_LOGIN_BLOCK_SECONDS
    }
    if current_index >= len(sessions):
        current_index = 0
    payload_dict = {
        "sessions": [s.to_dict() for s in sessions],
        "current_index": current_index,
        "blocked_accounts": blocked,
        "muted_accounts": muted,
        "updated_at": int(now),
    }
    payload = json.dumps(payload_dict, ensure_ascii=False)
    path = sessions_file(upstream)
    with _save_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, payload)
        except OSError as exc:
            logger.debug("Failed to save sessions [%s]: %s", upstream, exc)
    return removed


async def save_upstream_sessions_async(
    upstream: str,
    sessions: List[PlatformSession],
    *,
    current_index: int = 0,
    blocked_accounts: Optional[Dict[str, float]] = None,
    muted_accounts: Optional[Dict[str, float]] = None,
) -> List[str]:
    from core.transport.blocking import run_blocking

    return await run_blocking(
        _save_upstream_sessions_impl,
        upstream,
        sessions,
        current_index=current_index,
        blocked_accounts=blocked_accounts,
        muted_accounts=muted_accounts,
    )


def clean_expired(sessions: List[PlatformSession]) -> Tuple[List[PlatformSession], List[str]]:
    removed: List[str] = []
    valid: List[PlatformSession] = []
    for s in sessions:
        if s.is_expired():
            removed.append(s.username)
        elif not s.is_valid:
            removed.append(s.username)
        else:
            valid.append(s)
    return valid, removed


def mark_invalid(sessions: List[PlatformSession], username: str) -> bool:
    found = False
    for s in sessions:
        if s.username == username:
            s.is_valid = False
            found = True
    return found


def replace_or_append(sessions: List[PlatformSession], new_session: PlatformSession) -> List[PlatformSession]:
    for i, s in enumerate(sessions):
        if s.username == new_session.username:
            sessions[i] = new_session
            return sessions
    sessions.append(new_session)
    return sessions


def mask_username(username: str) -> str:
    return username[:6] if username else ""


def valid_session_count(sessions: List[PlatformSession]) -> int:
    return sum(1 for s in sessions if s.is_valid and not s.is_expired())


def find_session_index(sessions: List[PlatformSession], username: str) -> Optional[int]:
    for i, s in enumerate(sessions):
        if s.username == username:
            return i
    return None


def oldest_session_username(sessions: List[PlatformSession]) -> Optional[str]:
    if not sessions:
        return None
    oldest = min(sessions, key=lambda s: s.login_time)
    return mask_username(oldest.username)


def usernames_in_use(sessions: List[PlatformSession]) -> List[str]:
    return [s.username for s in sessions]


def remove_by_username(sessions: List[PlatformSession], username: str) -> List[PlatformSession]:
    return [s for s in sessions if s.username != username]


def describe_sessions(sessions: List[PlatformSession]) -> Dict[str, Any]:
    now = time.time()
    valid = [s for s in sessions if s.is_valid and not s.is_expired()]
    return {
        "total": len(sessions),
        "valid": len(valid),
        "invalid": sum(1 for s in sessions if not s.is_valid),
        "expired": sum(1 for s in sessions if s.is_valid and s.is_expired()),
        "usernames": [s.username[:6] for s in sessions],
        "oldest_login_age_seconds": (
            now - min(s.login_time for s in sessions) if sessions else 0.0
        ),
    }


def is_account_mute_blocked(
    muted_accounts: Dict[str, float],
    username: str,
    *,
    now: Optional[float] = None,
) -> bool:
    """``muted_accounts`` 中记录在 24h 内则不可登录/选为 session。"""
    muted_at = muted_accounts.get(username)
    if muted_at is None:
        return False
    ts = now if now is not None else time.time()
    if ts - float(muted_at) >= MUTE_LOGIN_BLOCK_SECONDS:
        return False
    return True


def prune_expired_muted_accounts(
    muted_accounts: Dict[str, float],
    *,
    now: Optional[float] = None,
) -> Dict[str, float]:
    ts = now if now is not None else time.time()
    return {
        k: float(v)
        for k, v in muted_accounts.items()
        if ts - float(v) < MUTE_LOGIN_BLOCK_SECONDS
    }


def is_session_fatal_error(text: str) -> bool:
    lower = text.lower()
    if "ratelimited" in lower or "daily usage" in lower:
        return True
    if "unauthorized" in lower:
        return True
    if "expired" in lower and "token" in lower:
        return True
    if "log in" in lower:
        return True
    return False
