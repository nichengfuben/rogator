from __future__ import annotations

"""平台共享 session 池：预登、随机选号、换号、封禁与落盘。"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Optional

from core.session.accounts import Account, accounts_for_upstream
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
from server.records.login_history import LoginHistoryStore

logger = logging.getLogger("rogator")


class SessionLoginMixin:
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

    def _init_session_pool(self) -> None:
        sessions, meta = load_upstream_sessions(self.UPSTREAM_NAME)
        self._sessions = sessions
        self._current_index = meta.current_index
        self._blocked_accounts = dict(meta.blocked_accounts)
        self._muted_accounts = prune_expired_muted_accounts(dict(meta.muted_accounts))
        self._login_history = LoginHistoryStore(self.UPSTREAM_NAME)
        self._last_cleanup = 0.0

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

    async def replenish_sessions(self, count: Optional[int] = None) -> None:
        """有效 session 不足 pool target 时补登（LinUCB 门控，可 skip）。"""
        target = self._prelogin_target if count is None else count
        await self._ensure_cleanup()
        if not self._pool_accounts():
            logger.warning("No accounts available for %s", self.UPSTREAM_NAME)
            return
        await self._gated_replenish(target)

    async def _gated_replenish(self, target: int) -> None:
        from server.schedule.hooks import (
            build_login_features,
            login_gate,
            login_hard,
            reward_login,
            schedule_enabled,
        )
        from server.schedule.loops import gated_tick

        valid = valid_session_count(self._sessions)
        need = max(0, target - valid)
        gate = login_gate(self.UPSTREAM_NAME)
        force_act, force_skip = login_hard(valid, need)
        fill = valid / max(1, target)
        x = build_login_features(self, gate, need, target)

        async def _act() -> float:
            logged = await self._login_until(need)
            gate.note_outcome(logged > 0)
            return reward_login(True, logged, need, fill)

        await gated_tick(
            gate,
            x,
            enabled=schedule_enabled(),
            force_act=force_act,
            force_skip=force_skip,
            act=_act,
            skip_reward=lambda: reward_login(False, 0, need, fill),
        )

    async def _login_until(self, need: int) -> int:
        if need <= 0:
            return 0
        logged = 0
        interval = max(0.0, self._login_interval)
        for attempt in range(need):
            account = self._pick_account_for_login()
            if account is None:
                if valid_session_count(self._sessions) == 0:
                    logger.debug(
                        "No login-eligible accounts for %s (all blocked/muted/in-use)",
                        self.UPSTREAM_NAME,
                    )
                break
            if await self.login_account(account):
                logged += 1
            if interval > 0 and attempt < need - 1:
                await asyncio.sleep(interval)
        if logged:
            logger.info(
                "Session pool [%s]: +%d new, %d ready",
                self.UPSTREAM_NAME,
                logged,
                valid_session_count(self._sessions),
            )
        return logged

    async def ensure_prelogin(self) -> None:
        """兼容旧调用；等同 ``replenish_sessions``。"""
        await self.replenish_sessions(self._prelogin_target)

    async def prelogin_accounts(self, count: Optional[int] = None) -> None:
        """兼容旧调用；等同 ``replenish_sessions``。"""
        await self.replenish_sessions(count)

    def _active_usernames(self) -> set[str]:
        return {s.username for s in self._sessions if s.is_valid and not s.is_expired()}

    def _pick_account_for_login(self, *, skip: Optional[set[str]] = None) -> Optional[Account]:
        pool = self._pool_accounts()
        if not pool:
            return None
        skip = skip or set()

        def eligible(account: Account) -> bool:
            return (
                account.username not in skip
                and account.username not in self._active_usernames()
                and not self._is_account_blocked(account.username)
                and not self._is_account_muted(account.username)
            )

        return self._login_history.pick_account(pool, eligible=eligible)

    def _valid_sessions(
        self,
        *,
        exclude_username: Optional[str] = None,
    ) -> List[PlatformSession]:
        return [
            s for s in self._sessions
            if s.is_valid and not s.is_expired()
            and not self._is_account_blocked(s.username)
            and not self._is_account_muted(s.username)
            and (exclude_username is None or s.username != exclude_username)
        ]

    def _select_valid_session(
        self,
        *,
        exclude_username: Optional[str] = None,
    ) -> Optional[PlatformSession]:
        valid = self._valid_sessions(exclude_username=exclude_username)
        if not valid:
            return None
        selected = random.choice(valid)
        idx = self._index_of_username(selected.username)
        if idx is not None:
            self._current_index = idx
        return selected

    async def _commit_session(self, session: PlatformSession) -> PlatformSession:
        self._save_meta()
        await self._on_session_selected(session)
        return session

    async def switch_to_next(
        self,
        exclude_username: Optional[str] = None,
    ) -> Optional[PlatformSession]:
        skip: set[str] = {exclude_username} if exclude_username else set()

        async with self._lock:
            self.prune_expired_sessions()

        await self._ensure_cleanup()

        async with self._lock:
            session = self._select_valid_session(exclude_username=exclude_username)
            if session is not None:
                return await self._commit_session(session)
            account = self._pick_account_for_login(skip=skip)

        if account is None:
            async with self._lock:
                session = self._select_valid_session(exclude_username=exclude_username)
                if session is not None:
                    return await self._commit_session(session)
            return None

        skip.add(account.username)
        pool = self._pool_accounts()
        for _ in range(len(pool)):
            ps = await self.login_account(account)
            if ps and ps.username != exclude_username:
                async with self._lock:
                    idx = self._index_of_username(ps.username)
                    self._current_index = idx if idx is not None else 0
                await self._on_session_selected(ps)
                return ps
            async with self._lock:
                account = self._pick_account_for_login(skip=skip)
            if account is None:
                break
            skip.add(account.username)

        async with self._lock:
            session = self._select_valid_session(exclude_username=exclude_username)
            if session is not None:
                return await self._commit_session(session)
        return None

    async def get_valid_session(
        self,
        *,
        exclude_username: Optional[str] = None,
    ) -> Optional[PlatformSession]:
        self.prune_expired_sessions()
        await self._ensure_cleanup()
        async with self._lock:
            session = self._select_valid_session(exclude_username=exclude_username)
            if session is not None:
                await self._on_session_selected(session)
                return session
        return await self.switch_to_next(exclude_username=exclude_username)

    async def _on_session_selected(self, session: PlatformSession) -> None:
        """子类可在选号后同步 vendor 内部状态。"""
