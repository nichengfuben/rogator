from __future__ import annotations

"""平台共享 session 池：预登、最少在途选号、换号、封禁与落盘。"""

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

if TYPE_CHECKING:
    from state_sched import RequestScheduler

from core.session.accounts import Account, accounts_for_upstream
from core.session.pool_switch import SessionReplenishMixin, SessionSwitchMixin
from core.session.store import (
    CLEANUP_INTERVAL,
    MUTE_LOGIN_BLOCK_SECONDS,
    PlatformSession,
    clean_expired,
    is_account_mute_blocked,
    load_upstream_sessions,
    mark_invalid,
    mask_username,
    prune_expired_muted_accounts,
    remove_by_username,
    replace_or_append,
    save_upstream_sessions,
    valid_session_count,
)
from server.formats import UpstreamUnavailableError
from server.records.login_history import LoginHistoryStore

logger = logging.getLogger("rogator")

# 单账号并发软顶；全池饱和时仍允许选用（退化为最少在途）
MAX_INFLIGHT_PER_ACCOUNT: int = 2


class SessionLoginMixin(SessionReplenishMixin, SessionSwitchMixin):
    """按 upstream 分桶的 session 池；子类实现 ``_perform_login``。"""

    UPSTREAM_NAME: str = ""

    _sessions: List[PlatformSession]
    _current_index: int
    _blocked_accounts: Dict[str, float]
    _muted_accounts: Dict[str, float]
    _lock: asyncio.Lock
    _prelogin_target: int
    _login_interval: float
    _login_history: LoginHistoryStore
    _last_cleanup: float
    _inflight: Dict[str, int]
    _replenish_event: asyncio.Event

    def _init_session_pool(self) -> None:
        sessions, meta = load_upstream_sessions(self.UPSTREAM_NAME)
        self._sessions = sessions
        self._current_index = meta.current_index
        self._blocked_accounts = dict(meta.blocked_accounts)
        self._muted_accounts = prune_expired_muted_accounts(dict(meta.muted_accounts))
        self._login_history = LoginHistoryStore(self.UPSTREAM_NAME)
        self._last_cleanup = 0.0
        self._inflight = {}
        self._replenish_event = asyncio.Event()

    def _inflight_count(self, username: str) -> int:
        return max(0, self._inflight.get(username, 0))

    def _bump_inflight(self, username: str, delta: int) -> None:
        if not username:
            return
        count = self._inflight_count(username) + delta
        if count <= 0:
            self._inflight.pop(username, None)
        else:
            self._inflight[username] = count

    @asynccontextmanager
    async def lease_valid_session(
        self,
        *,
        exclude_username: Optional[str] = None,
    ) -> AsyncIterator[Optional[PlatformSession]]:
        """租用 session：在途 +1，退出时 −1（含 cancel / 流式 aclose）。"""
        session = await self.get_valid_session(exclude_username=exclude_username)
        if session is None:
            yield None
            return
        self._bump_inflight(session.username, 1)
        try:
            yield session
        finally:
            self._bump_inflight(session.username, -1)

    def _save_meta(self) -> List[str]:
        return save_upstream_sessions(
            self.UPSTREAM_NAME,
            self._sessions,
            current_index=self._current_index,
            blocked_accounts=self._blocked_accounts,
            muted_accounts=self._muted_accounts,
        )

    def block_account(self, username: str, block_seconds: float) -> None:
        until = time.time() + max(block_seconds, 60.0)
        self._blocked_accounts[username] = until
        self._save_meta()
        logger.info("Blocked account %s for %.0fs", mask_username(username), block_seconds)

    def _is_account_blocked(self, username: str) -> bool:
        until = self._blocked_accounts.get(username)
        if until is None:
            return False
        if time.time() >= until:
            self._blocked_accounts.pop(username, None)
            return False
        return True

    def _is_account_muted(self, username: str) -> bool:
        return is_account_mute_blocked(self._muted_accounts, username)

    def handle_account_muted(
        self,
        username: str,
        *,
        mute_at: Optional[float] = None,
    ) -> None:
        """上游 mute：清理 session、记录 mute 时间戳，24h 内不再登录。"""
        ts = float(mute_at if mute_at is not None else time.time())
        self._muted_accounts[username] = ts
        self._muted_accounts = prune_expired_muted_accounts(self._muted_accounts)
        mark_invalid(self._sessions, username)
        self._sessions[:] = remove_by_username(self._sessions, username)
        self._fix_current_index(username)
        self._save_meta()
        logger.warning(
            "Account muted [%s]: %s, login blocked %.0fh",
            self.UPSTREAM_NAME,
            mask_username(username),
            MUTE_LOGIN_BLOCK_SECONDS / 3600,
        )

    def _session_username_at_current(self) -> Optional[str]:
        if self._sessions and self._current_index < len(self._sessions):
            return self._sessions[self._current_index].username
        return None

    @property
    def current_session_username(self) -> Optional[str]:
        return self._session_username_at_current()

    def _fix_current_index(self, previous_username: Optional[str] = None) -> None:
        username = previous_username
        if username is None and self._sessions and self._current_index < len(self._sessions):
            username = self._sessions[self._current_index].username
        if username and not any(s.username == username for s in self._sessions):
            valid_indices = [
                i for i, s in enumerate(self._sessions)
                if s.is_valid and not s.is_expired()
            ]
            self._current_index = random.choice(valid_indices) if valid_indices else 0
        elif self._current_index >= len(self._sessions):
            self._current_index = 0

    def prune_expired_sessions(self) -> List[str]:
        previous_username = self._session_username_at_current()
        self._sessions, removed = clean_expired(self._sessions)
        if removed:
            self._fix_current_index(previous_username)
        return removed

    def cleanup_expired_sessions(self) -> List[str]:
        previous_username = self._session_username_at_current()
        removed = self._save_meta()
        if removed:
            self._fix_current_index(previous_username)
            logger.info(
                "Session cleanup [%s]: removed %d expired/invalid session(s)",
                self.UPSTREAM_NAME,
                len(removed),
            )
        return removed

    def _persist_sessions(self) -> List[str]:
        previous_username = self._session_username_at_current()
        self._login_history.flush()
        removed = self._save_meta()
        self._fix_current_index(previous_username)
        return removed

    async def _ensure_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        self.cleanup_expired_sessions()

    @property
    def current_session(self) -> Optional[PlatformSession]:
        if not self._sessions or self._current_index >= len(self._sessions):
            return None
        session = self._sessions[self._current_index]
        if not session.is_valid or session.is_expired():
            return None
        return session

    def _index_of_username(self, username: str) -> Optional[int]:
        for i, s in enumerate(self._sessions):
            if s.username == username:
                return i
        return None

    async def _perform_login(self, account: Account) -> Optional[PlatformSession]:
        raise NotImplementedError

    async def login_account(self, account: Account) -> Optional[PlatformSession]:
        await self._ensure_cleanup()
        try:
            ps = await self._perform_login(account)
            if ps is None:
                return None
            async with self._lock:
                replace_or_append(self._sessions, ps)
            self._login_history.record(account.username)
            self._persist_sessions()
            logger.info(
                "Logged in [%s]: %s (total: %d)",
                self.UPSTREAM_NAME,
                account.username[:6],
                len(self._sessions),
            )
            return ps
        except asyncio.TimeoutError:
            logger.warning("Login [%s] %s timed out", self.UPSTREAM_NAME, account.username[:6])
            return None
        except UpstreamUnavailableError:
            return None
        except Exception as exc:
            logger.warning(
                "Login exception [%s] for %s: %s",
                self.UPSTREAM_NAME,
                account.username[:6],
                exc,
            )
            return None

    def _pool_accounts(self) -> List[Account]:
        return accounts_for_upstream(self.UPSTREAM_NAME)

    def _login_pool_available(self) -> bool:
        """号池为空时不尝试登录、不刷 replenish 日志。"""
        return bool(self._pool_accounts())

    def _active_usernames(self) -> set[str]:
        return {s.username for s in self._sessions if s.is_valid and not s.is_expired()}


    async def _on_session_selected(self, session: PlatformSession) -> None:
        """子类可在选号后同步 vendor 内部状态。"""
