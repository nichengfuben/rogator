from __future__ import annotations

"""Session 池换号、预登与选号逻辑（从 pool.py 拆出以满足 achecker）。"""

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from state_sched import RequestScheduler

from core.session.accounts import Account
from core.session.store import PlatformSession, valid_session_count

import random

logger = logging.getLogger("rogator")

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


MIN_EFFECTIVE_PRELOGIN: int = 4
MIN_BOOTSTRAP_SESSIONS: int = 1


class SessionReplenishMixin:
    """预登目标计算与补登循环。"""

    UPSTREAM_NAME: str
    _sessions: List[PlatformSession]
    _prelogin_target: int
    _login_interval: float
    _replenish_event: asyncio.Event
    _inflight: dict[str, int]

    def _prelogin_cap(self) -> int:
        base = max(0, int(self._prelogin_target))
        try:
            from server.config import CONFIG

            mc = int(CONFIG.max_concurrent)
        except Exception:
            return base
        if mc <= 0:
            return base
        return min(base, max(MIN_EFFECTIVE_PRELOGIN, mc // 2))

    def _prelogin_headroom(self) -> int:
        try:
            from server.config import CONFIG

            mc = int(CONFIG.max_concurrent)
        except Exception:
            return 4
        if mc <= 0:
            return 8
        return max(2, mc // 4)

    def _pool_demand(self, scheduler: Optional["RequestScheduler"] = None) -> int:
        inflight = sum(self._inflight.values())
        if scheduler is None:
            return inflight
        active = getattr(scheduler, "active", 0)
        pending = getattr(scheduler, "pending", 0)
        return max(inflight, int(active) + int(pending))

    def _replenish_is_urgent(self, *, valid: int, need: int, demand: int) -> bool:
        if need <= 0:
            return False
        if valid == 0:
            return demand > 0 or self._replenish_event.is_set()
        if demand <= 0 and not self._replenish_event.is_set():
            return False
        if self._replenish_event.is_set():
            return True
        return demand >= valid * MAX_INFLIGHT_PER_ACCOUNT

    def _effective_prelogin_target(
        self,
        count: Optional[int] = None,
        *,
        scheduler: Optional["RequestScheduler"] = None,
    ) -> int:
        if count is not None:
            return max(0, int(count))
        cap = self._prelogin_cap()
        demand = self._pool_demand(scheduler)
        if demand <= 0 and not self._replenish_event.is_set():
            return min(cap, MIN_BOOTSTRAP_SESSIONS)
        headroom = self._prelogin_headroom()
        return min(cap, max(MIN_BOOTSTRAP_SESSIONS, demand + headroom))

    def signal_replenish(self) -> None:
        self._replenish_event.set()

    async def wait_for_replenish_or_timeout(self, timeout: float) -> None:
        if self._replenish_event.is_set():
            self._replenish_event.clear()
            return
        try:
            await asyncio.wait_for(self._replenish_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            self._replenish_event.clear()

    async def _replenish_login_batch(self, need: int, interval: float) -> int:
        if not self._login_pool_available():
            return 0
        logged = 0
        for attempt in range(need):
            account = self._pick_account_for_login()
            if account is None:
                break
            ps = await self.login_account(account)
            if ps:
                logged += 1
            if interval > 0 and attempt < need - 1:
                await asyncio.sleep(interval)
        return logged

    def _log_replenish_result(self, logged: int, target: int) -> None:
        if not logged:
            return
        logger.info(
            "Session pool [%s]: +%d new, %d ready (target=%d)",
            self.UPSTREAM_NAME,
            logged,
            valid_session_count(self._sessions),
            target,
        )

    async def replenish_sessions(
        self,
        count: Optional[int] = None,
        *,
        scheduler: Optional["RequestScheduler"] = None,
    ) -> None:
        """有效 session 不足目标时补登；空闲仅保 bootstrap，有负载再扩池。"""
        if not self._login_pool_available():
            return
        target = self._effective_prelogin_target(count, scheduler=scheduler)
        await self._ensure_cleanup()
        valid = valid_session_count(self._sessions)
        need = max(0, target - valid)
        if need <= 0:
            return
        demand = self._pool_demand(scheduler)
        urgent = self._replenish_is_urgent(valid=valid, need=need, demand=demand)
        interval = 0.0 if urgent else max(0.0, self._login_interval)
        if urgent and need > 1:
            logger.debug(
                "Session pool [%s] urgent replenish: valid=%d need=%d demand=%d",
                self.UPSTREAM_NAME, valid, need, demand,
            )
        logged = await self._replenish_login_batch(need, interval)
        self._log_replenish_result(logged, target)

    async def ensure_prelogin(self) -> None:
        await self.replenish_sessions()

    async def prelogin_accounts(self, count: Optional[int] = None) -> None:
        await self.replenish_sessions(count)
