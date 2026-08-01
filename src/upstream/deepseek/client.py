from __future__ import annotations

"""Rogator DeepSeek 上游客户端。"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

from server.retry.http_client import client_session

from core.transport.http import reset_upstream_transport
from core.session.accounts import Account, accounts_for_upstream
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


class DeepSeekClient(SessionLoginMixin):
    UPSTREAM_NAME = "deepseek"

    def __init__(self, splitter: Any = None) -> None:
        self._splitter = splitter
        self._init_session_pool()
        self._lock = asyncio.Lock()
        self._http: Optional[aiohttp.ClientSession] = None
        self._inner: Optional[DeepseekClient] = None
        self._models: List[str] = list(MODELS)
        self._model_meta: Dict[str, Any] = {}
        self._models_fetch_time: float = 0.0
        self._startup_done: bool = False
        self._startup_lock = asyncio.Lock()
        from server.config import CONFIG

        self._prelogin_target: int = CONFIG.prelogin
        self._login_interval: float = CONFIG.login_interval

    def load_models_cache(self) -> List[str]:
        return list(self._models)

    def models_refresh_due(self, interval: float) -> bool:
        if interval <= 0:
            return True
        if self._models_fetch_time <= 0:
            return True
        return (time.time() - self._models_fetch_time) >= interval

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

    async def _perform_login(self, account: Account) -> Optional[PlatformSession]:
        inner = await self._ensure_ready(skip_background=True)
        try:
            token, user_id = await login(  # noqa: SLF001
                inner._session, account.username, account.password
            )
        except (DeepSeekUserMutedError, DeepSeekWafChallengeError):
            self.handle_account_muted(account.username, mute_at=time.time())
            return None
        mgr = inner._hif_managers.get(account.username)  # noqa: SLF001
        if mgr is not None:
            try:
                leim, dliq, expire = await fetch_hif_tokens(inner._session)  # noqa: SLF001
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

    async def startup(self) -> None:
        async with self._startup_lock:
            if self._startup_done:
                return
            pool = accounts_for_upstream(self.UPSTREAM_NAME)
            if not pool:
                logger.warning("DeepSeek: 无可用账号")
                self._startup_done = True
                return
            self._http = client_session()
            ds_accounts = [
                DsAccount(username=a.username, password=a.password)
                for a in pool
            ]
            self._inner = DeepseekClient()
            await self._inner.init_immediate(self._http, accounts=ds_accounts)
            self._apply_restored_sessions()
            await self._inner.background_setup(login_accounts=False)
            self._startup_done = True
            logger.info(
                "DeepSeek startup: %d account(s), %d candidate(s), %d session(s)",
                len(pool),
                len(await self._inner.candidates()),
                len(self._sessions),
            )

    async def _ensure_ready(self, *, skip_background: bool = False) -> DeepseekClient:
        if not self._startup_done:
            if skip_background:
                if self._http is None:
                    self._http = client_session()
                if self._inner is None:
                    pool = accounts_for_upstream(self.UPSTREAM_NAME)
                    ds_accounts = [
                        DsAccount(username=a.username, password=a.password)
                        for a in pool
                    ]
                    self._inner = DeepseekClient()
                    await self._inner.init_immediate(self._http, accounts=ds_accounts)
                    self._apply_restored_sessions()
            else:
                await self.startup()
        if self._inner is None:
            raise UpstreamUnavailableError(
                "DeepSeek 未初始化或无可用账号",
                upstream="deepseek",
            )
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

    async def reset_http_transport(self) -> None:
        await reset_upstream_transport(self._http)
        self._http = None
        if self._inner is not None and self._startup_done:
            self._http = client_session()
            self._inner._session = self._http  # noqa: SLF001

    async def shutdown(self) -> None:
        if self._inner is not None:
            await self._inner.close()
            self._inner = None
        await self.reset_http_transport()
        self._startup_done = False
        logger.debug("DeepSeek client shut down")
