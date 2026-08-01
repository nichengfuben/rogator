from __future__ import annotations

"""Rogator DeepSeek 上游客户端。"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

from core.transport.conn_retry import run_with_connection_retry
from core.transport.owned import HttpTransportMixin
from core.session.accounts import Account, accounts_for_upstream
from core.session.models_cache import ModelsCacheMixin
from core.session.pool import SessionLoginMixin
from core.session.store import PlatformSession
from upstream.deepseek.lib.adapter.client import DeepseekClient
from upstream.deepseek.lib.adapter.helpers.pmtutil import Account as DsAccount
from upstream.deepseek.lib.guard.hif import fetch_hif_tokens
from upstream.deepseek.lib.protocol.consts import MODELS
from upstream.deepseek.lib.adapter.helpers.biz_error import DeepSeekUserMutedError, DeepSeekWafChallengeError
from upstream.deepseek.lib.user.userapi import login
from server.formats import UpstreamUnavailableError

logger = logging.getLogger("rogator")


class DeepSeekClient(HttpTransportMixin, ModelsCacheMixin, SessionLoginMixin):
    UPSTREAM_NAME = "deepseek"

    def __init__(self, splitter: Any = None) -> None:
        self._splitter = splitter
        self._init_session_pool()
        self._init_http_transport()
        self._init_models_cache(list(MODELS))
        self._lock = asyncio.Lock()
        self._inner: Optional[DeepseekClient] = None
        self._startup_done: bool = False
        self._startup_lock = asyncio.Lock()
        from server.config import CONFIG

        self._prelogin_target: int = CONFIG.prelogin
        self._login_interval: float = CONFIG.login_interval

    def load_models_cache(self) -> List[str]:
        return list(self._models)

    async def fetch_models(self, *, use_cache: bool = True) -> List[str]:
        self._models_fetch_time = time.time()
        return list(self._models)

    def _sync_inner_account(self, session: PlatformSession) -> None:
        if self._inner is None:
            return
        for account in self._inner._accounts:  # noqa: SLF001
            if account.username == session.username:
                account.token = session.token
                account.user_id = session.user_id
                break
        self._inner._rebuild_candidates()  # noqa: SLF001

    def _apply_restored_sessions(self) -> None:
        if self._inner is None:
            return
        for ps in self._sessions:
            if not ps.is_valid or ps.is_expired():
                continue
            self._sync_inner_account(ps)

    async def _on_session_selected(self, session: PlatformSession) -> None:
        self._sync_inner_account(session)

    def handle_account_muted(
        self,
        username: str,
        *,
        mute_at: Optional[float] = None,
    ) -> None:
        SessionLoginMixin.handle_account_muted(self, username, mute_at=mute_at)
        if self._inner is not None:
            for account in self._inner._accounts:  # noqa: SLF001
                if account.username == username:
                    account.token = ""
            self._inner._rebuild_candidates()  # noqa: SLF001

    def _ds_accounts(self) -> List[DsAccount]:
        pool = accounts_for_upstream(self.UPSTREAM_NAME)
        return [DsAccount(username=a.username, password=a.password) for a in pool]

    def _on_http_session_created(self, session: aiohttp.ClientSession) -> None:
        if self._inner is not None:
            self._inner.rebind_http_session(session)

    def _should_recreate_http_on_reset(self) -> bool:
        return self._inner is not None

    async def _init_minimal_unlocked(self) -> None:
        """仅初始化 HTTP + inner，不跑 background_setup（供预登）。"""
        ds_accounts = self._ds_accounts()
        if not ds_accounts:
            return
        self._ensure_http_unlocked()
        if self._inner is None:
            self._inner = DeepseekClient()
            await self._inner.init_immediate(self._http, accounts=ds_accounts)
            self._apply_restored_sessions()
        else:
            self._inner.rebind_http_session(self._http)

    async def _startup_unlocked(self) -> None:
        if self._startup_done:
            return
        ds_accounts = self._ds_accounts()
        if not ds_accounts:
            logger.warning("DeepSeek: 无可用账号")
            self._startup_done = True
            return
        async with self._transport_lock:
            await self._init_minimal_unlocked()
        if self._inner is None:
            self._startup_done = True
            return
        await self._inner.background_setup(login_accounts=False)
        self._startup_done = True
        logger.info(
            "DeepSeek startup: %d account(s), %d candidate(s), %d session(s)",
            len(ds_accounts),
            len(await self._inner.candidates()),
            len(self._sessions),
        )

    async def _login_once(self, account: Account) -> Optional[PlatformSession]:
        inner = await self._ensure_ready(skip_background=True)
        http = await self._ensure_http_session()
        try:
            token, user_id = await login(http, account.username, account.password)
        except (DeepSeekUserMutedError, DeepSeekWafChallengeError):
            self.handle_account_muted(account.username, mute_at=time.time())
            return None
        mgr = inner._hif_managers.get(account.username)  # noqa: SLF001
        if mgr is not None:
            try:
                leim, dliq, expire = await fetch_hif_tokens(http)
                mgr._leim = leim
                mgr._dliq = dliq
                mgr._expire_at = expire
            except Exception as exc:
                logger.warning("首次 HIF 令牌获取失败 %s: %s", account.username, exc)
        ps = PlatformSession(
            account=account,
            token=token,
            user_id=user_id,
            upstream="deepseek",
        )
        self._sync_inner_account(ps)
        return ps

    async def _perform_login(self, account: Account) -> Optional[PlatformSession]:
        async def _run() -> Optional[PlatformSession]:
            return await self._login_once(account)

        return await run_with_connection_retry(
            "login", _run, upstream="deepseek", transport_owner=self,
        )

    async def startup(self) -> None:
        async with self._startup_lock:
            await self._startup_unlocked()

    async def _ensure_ready(self, *, skip_background: bool = False) -> DeepseekClient:
        if skip_background:
            async with self._startup_lock:
                if not self._startup_done:
                    async with self._transport_lock:
                        await self._init_minimal_unlocked()
                else:
                    async with self._transport_lock:
                        self._ensure_http_unlocked()
            if self._inner is None:
                raise UpstreamUnavailableError(
                    "DeepSeek 未初始化或无可用账号",
                    upstream="deepseek",
                )
            return self._inner
        await self.startup()
        if self._inner is None:
            raise UpstreamUnavailableError(
                "DeepSeek 未初始化或无可用账号",
                upstream="deepseek",
            )
        async with self._transport_lock:
            self._ensure_http_unlocked()
        return self._inner

    async def pick_candidate(self) -> Any:
        session = await self.get_valid_session()
        if session is None:
            raise UpstreamUnavailableError(
                "DeepSeek 无可用会话，请检查账号配置与登录状态",
                upstream="deepseek",
            )
        inner = await self._ensure_ready()
        for cand in await inner.candidates():
            if cand.meta.get("identifier") == session.username:
                return cand
        self._sync_inner_account(session)
        for cand in await inner.candidates():
            if cand.meta.get("identifier") == session.username:
                return cand
        raise UpstreamUnavailableError(
            f"DeepSeek 账号 {session.username[:6]} 无可用候选",
            upstream="deepseek",
        )

    async def switch_to_next(self, exclude_username: Optional[str] = None) -> Optional[PlatformSession]:
        return await SessionLoginMixin.switch_to_next(self, exclude_username=exclude_username)

    async def shutdown(self) -> None:
        if self._inner is not None:
            await self._inner.close()
            self._inner = None
        await self.close_http_transport()
        self._startup_done = False
        logger.debug("DeepSeek client shut down")
