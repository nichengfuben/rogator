from __future__ import annotations

"""Session 池换号与选号逻辑（从 pool.py 拆出以满足 achecker）。"""

import random
from typing import List, Optional

from core.session.accounts import Account
from core.session.store import PlatformSession

MAX_INFLIGHT_PER_ACCOUNT: int = 2


class SessionSwitchMixin:
    """选号、换号与登录重试。"""

    _sessions: List[PlatformSession]
    _current_index: int
    _blocked_accounts: dict
    _muted_accounts: dict
    _lock: object
    _inflight: dict
    _login_history: object

    def _pick_account_for_login(self, *, skip: Optional[set] = None) -> Optional[Account]:
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
        under_cap = [
            s for s in valid
            if self._inflight_count(s.username) < MAX_INFLIGHT_PER_ACCOUNT
        ]
        pool = under_cap or valid
        selected = min(
            pool,
            key=lambda s: (self._inflight_count(s.username), random.random()),
        )
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
        skip: set = {exclude_username} if exclude_username else set()

        async with self._lock:
            self.prune_expired_sessions()

        await self._ensure_cleanup()

        async with self._lock:
            session = self._select_valid_session(exclude_username=exclude_username)
            if session is not None:
                return await self._commit_session(session)
            account = self._pick_account_for_login(skip=skip)

        if account is None:
            return await self._fallback_valid_session(exclude_username)

        return await self._login_until_valid(account, skip, exclude_username)

    async def _fallback_valid_session(
        self, exclude_username: Optional[str],
    ) -> Optional[PlatformSession]:
        async with self._lock:
            session = self._select_valid_session(exclude_username=exclude_username)
            if session is not None:
                return await self._commit_session(session)
        return None

    async def _login_until_valid(
        self,
        account: Account,
        skip: set,
        exclude_username: Optional[str],
    ) -> Optional[PlatformSession]:
        skip.add(account.username)
        pool = self._pool_accounts()
        for _ in range(len(pool)):
            ps = await self.login_account(account)
            if ps and ps.username != exclude_username:
                async with self._lock:
                    idx = self._index_of_username(ps.username)
                    self._current_index = idx if idx is not None else 0
                await self._on_session_selected(ps)
                self.signal_replenish()
                return ps
            async with self._lock:
                account = self._pick_account_for_login(skip=skip)
            if account is None:
                break
            skip.add(account.username)
        return await self._fallback_valid_session(exclude_username)

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
