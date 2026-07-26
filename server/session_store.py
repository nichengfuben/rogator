from __future__ import annotations

"""Qwen session 数据模型与磁盘持久化。

包含 session 的 dataclass 定义、原子化磁盘读写（临时文件 + os.replace 防止写入过程中
进程被杀导致文件损坏）、过期/失效清理，以及登录时用来拉取用户信息的辅助函数。
从 server/qwen_client.py 中拆出，避免该文件承载过多不相关职责。
"""

import json
import logging
import os
import time
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from accounts import Account
from server.formats import DEFAULT_USER_AGENT

logger = logging.getLogger("rogator")

SESSIONS_FILE: str = "persist/sessions.json"
LEGACY_SESSIONS_FILE: str = "persist/qwen/sessions.json"
CLEANUP_INTERVAL: float = 60.0


def _migrate_legacy_sessions_file() -> None:
    """启动时将旧路径 persist/qwen/sessions.json 迁移到 persist/sessions.json。"""
    new_path = Path(SESSIONS_FILE)
    old_path = Path(LEGACY_SESSIONS_FILE)
    if new_path.exists() or not old_path.exists():
        return
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(old_path), str(new_path))
        logger.info("Migrated sessions file: %s -> %s", old_path, new_path)
    except Exception as e:
        logger.warning("Failed to migrate sessions file from %s: %s", old_path, e)


def _jwt_exp(token: str) -> Optional[float]:
    """解析 JWT payload 中的 exp 字段，返回过期时间戳（Unix 秒）。"""
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


@dataclass
class QwenSession:
    account: Account
    token: str
    user_id: str
    login_time: float = field(default_factory=time.time)
    is_valid: bool = True

    @property
    def username(self) -> str:
        return self.account.username

    def is_expired(self) -> bool:
        """通过 JWT exp 字段判断 token 是否过期（提前 30 秒清理）。"""
        if not self.token:
            return True
        exp = _jwt_exp(self.token)
        if exp is not None:
            return time.time() >= exp - 30
        # JWT 解析失败则视为已过期，强制重新登录
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.account.username,
            "password": self.account.password,
            "token": self.token,
            "user_id": self.user_id,
            "login_time": self.login_time,
            "is_valid": self.is_valid,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QwenSession":
        account = Account(username=data["username"], password=data["password"])
        return QwenSession(
            account=account,
            token=data["token"],
            user_id=data.get("user_id", ""),
            login_time=data.get("login_time", time.time()),
            is_valid=data.get("is_valid", True),
        )


@dataclass
class SessionStoreMeta:
    """sessions.json 元数据（索引、封禁账号）。"""

    current_index: int = 0
    account_index: int = 0
    blocked_accounts: Dict[str, float] = field(default_factory=dict)


def load_session_store() -> tuple[List[QwenSession], SessionStoreMeta]:
    """启动时从磁盘恢复 session 池与元数据。"""
    _migrate_legacy_sessions_file()
    meta = SessionStoreMeta()
    try:
        p = Path(SESSIONS_FILE)
        if not p.exists():
            return [], meta
        data = json.loads(p.read_text(encoding="utf-8"))
        meta.current_index = int(data.get("current_index", 0))
        meta.account_index = int(data.get("account_index", 0))
        raw_blocked = data.get("blocked_accounts") or {}
        if isinstance(raw_blocked, dict):
            meta.blocked_accounts = {
                str(k): float(v) for k, v in raw_blocked.items()
            }
        sessions = [QwenSession.from_dict(item) for item in data.get("sessions", [])]
        restored = [s for s in sessions if not s.is_expired() and s.is_valid]
        if restored:
            logger.info("Restored %d session(s) from disk", len(restored))
        return restored, meta
    except Exception as e:
        logger.warning("Failed to load sessions: %s", e)
        return [], meta


def load_sessions() -> List[QwenSession]:
    """兼容旧接口：仅返回 session 列表。"""
    sessions, _ = load_session_store()
    return sessions


def is_session_fatal_error(text: str) -> bool:
    """判断 API 错误是否表示当前 session 不可再用（过期、鉴权失败、限流）。"""
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


def save_sessions(
    sessions: List[QwenSession],
    *,
    current_index: int = 0,
    account_index: int = 0,
    blocked_accounts: Optional[Dict[str, float]] = None,
) -> List[str]:
    """清理过期/失效 session 后原子写入磁盘，原地更新列表并返回被移除的 username。"""
    cleaned, removed = clean_expired(sessions)
    sessions[:] = cleaned
    if removed:
        logger.info("Cleanup: removed %d expired/invalid session(s)", len(removed))
    now = time.time()
    blocked = {
        k: v for k, v in (blocked_accounts or {}).items()
        if v > now
    }
    if current_index >= len(sessions):
        current_index = 0
    try:
        Path(SESSIONS_FILE).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sessions": [s.to_dict() for s in sessions],
            "current_index": current_index,
            "account_index": account_index,
            "blocked_accounts": blocked,
            "updated_at": int(now),
            "count": len(sessions),
        }
        tmp_path = f"{SESSIONS_FILE}.tmp"
        Path(tmp_path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, SESSIONS_FILE)
    except Exception as e:
        logger.warning("Failed to save sessions: %s", e)
    return removed


async def fetch_user_id(session: aiohttp.ClientSession, token: str, auth_base_url: str) -> str:
    """登录成功后拉取 user_id，用于文件上传等接口的 user_id 字段。"""
    try:
        async with session.get(
            f"{auth_base_url}/api/v2/user",
            headers={"Authorization": f"Bearer {token}", "User-Agent": DEFAULT_USER_AGENT},
            ssl=False,
        ) as ur:
            if ur.status == 200:
                return str((await ur.json()).get("data", {}).get("id", ""))
    except Exception:
        pass
    return ""


def clean_expired(sessions: List[QwenSession]) -> tuple[List[QwenSession], List[str]]:
    """清理过期和无效的 session，返回 (剩余 session 列表, 被移除的 username 列表)"""
    removed = []
    valid_sessions = []
    for s in sessions:
        if s.is_expired():
            logger.debug("Session %s expired (jwt exp-30s), removing", s.username[:6])
            removed.append(s.username)
        elif not s.is_valid:
            logger.debug("Session %s invalid, removing", s.username[:6])
            removed.append(s.username)
        else:
            valid_sessions.append(s)
    return valid_sessions, removed


def mark_invalid(sessions: List[QwenSession], username: str) -> bool:
    """按 username 精确标记单个 session 失效（不落盘，由调用方决定何时 save_sessions）。"""
    found = False
    for s in sessions:
        if s.username == username:
            s.is_valid = False
            found = True
    return found


def describe_sessions(sessions: List[QwenSession]) -> Dict[str, Any]:
    """汇总当前 session 池状态，供管理端点 / 日志排障使用。"""
    now = time.time()
    valid = [s for s in sessions if s.is_valid and not s.is_expired()]
    invalid = [s for s in sessions if not s.is_valid]
    expired = [s for s in sessions if s.is_valid and s.is_expired()]
    return {
        "total": len(sessions),
        "valid": len(valid),
        "invalid": len(invalid),
        "expired": len(expired),
        "usernames": [s.username[:6] for s in sessions],
        "oldest_login_age_seconds": (
            now - min(s.login_time for s in sessions) if sessions else 0.0
        ),
    }


def find_session_index(sessions: List[QwenSession], username: str) -> Optional[int]:
    """按 username 查找 session 在列表中的索引，找不到返回 None。"""
    for i, s in enumerate(sessions):
        if s.username == username:
            return i
    return None


def replace_or_append(
    sessions: List[QwenSession], new_session: QwenSession,
) -> List[QwenSession]:
    """用新登录的 session 替换同用户名的旧 session（若存在），否则追加。"""
    idx = find_session_index(sessions, new_session.username)
    if idx is not None:
        sessions[idx] = new_session
    else:
        sessions.append(new_session)
    return sessions


def mask_username(username: str) -> str:
    """日志中统一使用的用户名掩码格式，避免完整邮箱出现在日志里。"""
    return username[:6] if username else ""


def oldest_session_username(sessions: List[QwenSession]) -> Optional[str]:
    """返回登录时间最早的 session 对应的用户名（掩码后），无 session 时返回 None。"""
    if not sessions:
        return None
    oldest = min(sessions, key=lambda s: s.login_time)
    return mask_username(oldest.username)


def valid_session_count(sessions: List[QwenSession]) -> int:
    """统计当前有效（未过期且未失效）的 session 数量。"""
    return sum(1 for s in sessions if s.is_valid and not s.is_expired())


def usernames_in_use(sessions: List[QwenSession]) -> List[str]:
    """返回所有已登录账号的 username 列表，用于 prelogin 阶段去重判断。"""
    return [s.username for s in sessions]


def remove_by_username(sessions: List[QwenSession], username: str) -> List[QwenSession]:
    """按 username 从 session 列表中彻底移除（区别于 mark_invalid 仅置无效标记）。"""
    return [s for s in sessions if s.username != username]
