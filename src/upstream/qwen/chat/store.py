from __future__ import annotations

"""Qwen 上游 session 类型与持久化接口。"""

from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from core.session.store import (
    CLEANUP_INTERVAL,
    PlatformSession as QwenSession,
    SessionStoreMeta,
    clean_expired,
    describe_sessions,
    is_session_fatal_error,
    load_upstream_sessions,
    mark_invalid,
    mask_username,
    replace_or_append,
    save_upstream_sessions,
    valid_session_count,
    find_session_index,
    oldest_session_username,
    usernames_in_use,
    remove_by_username,
)
from upstream.qwen.chat.routes import USER_AGENT

_UPSTREAM = "qwen"
SESSIONS_FILE = "persist/qwen/sessions.json"


def load_session_store() -> Tuple[List[QwenSession], SessionStoreMeta]:
    return load_upstream_sessions(_UPSTREAM)


def load_sessions() -> List[QwenSession]:
    sessions, _ = load_session_store()
    return sessions


def save_sessions(
    sessions: List[QwenSession],
    *,
    current_index: int = 0,
    blocked_accounts: Optional[Dict[str, float]] = None,
) -> List[str]:
    return save_upstream_sessions(
        _UPSTREAM,
        sessions,
        current_index=current_index,
        blocked_accounts=blocked_accounts,
    )


async def fetch_user_id(session: aiohttp.ClientSession, token: str, auth_base_url: str, proxy: Optional[str] = None) -> str:
    try:
        async with session.get(
            f"{auth_base_url}/api/v2/user",
            headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT},
            ssl=False,
            proxy=proxy,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as ur:
            if ur.status == 200:
                return str((await ur.json()).get("data", {}).get("id", ""))
    except Exception:
        pass
    return ""


__all__ = [
    "CLEANUP_INTERVAL",
    "QwenSession",
    "SessionStoreMeta",
    "SESSIONS_FILE",
    "clean_expired",
    "describe_sessions",
    "fetch_user_id",
    "find_session_index",
    "is_session_fatal_error",
    "load_session_store",
    "load_sessions",
    "mark_invalid",
    "mask_username",
    "oldest_session_username",
    "remove_by_username",
    "replace_or_append",
    "save_sessions",
    "usernames_in_use",
    "valid_session_count",
]
