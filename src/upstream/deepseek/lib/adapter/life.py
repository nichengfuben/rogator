from __future__ import annotations

"""DeepSeek 客户端后台生命周期：登录、WASM 定期更新、HIF 令牌刷新。"""

import asyncio
import logging
from typing import List

from upstream.deepseek.lib.adapter.helpers.pmtutil import Account
from upstream.deepseek.lib.guard.hif import fetch_hif_tokens
from upstream.deepseek.lib.guard.pow import WasmPow, download_wasm
from upstream.deepseek.lib.protocol.consts import HIF_REFRESH_INTERVAL
from upstream.deepseek.lib.runtime.user.userapi import login

logger = logging.getLogger(__name__)


class _ClientLifecycleMixin:
    _session: object
    _pow: WasmPow
    _accounts: List[Account]
    _hif_managers: dict
    _closing: bool
    _bg_tasks: List[asyncio.Task]

    def _spawn_bg(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.append(task)
        task.add_done_callback(self._discard_bg_task)

    def _discard_bg_task(self, task: asyncio.Task) -> None:
        try:
            self._bg_tasks.remove(task)
        except ValueError:
            pass

    async def _cancel_bg_tasks(self) -> None:
        pending = [t for t in self._bg_tasks if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._bg_tasks.clear()

    async def background_setup(self, *, login_accounts: bool = True) -> None:
        """后台完善：下载 WASM；可选并发登录所有尚无 token 的账号。"""
        if self._closing:
            return
        self._spawn_bg(download_wasm(self._session))
        self._spawn_bg(self._bg_wasm_check())
        self._spawn_bg(self._bg_hif_refresh())

        if not login_accounts:
            return

        pending = [account for account in self._accounts if not account.token]
        if not pending:
            logger.info("deepseek 跳过登录：全部账号已有 token")
            return

        tasks = [
            asyncio.ensure_future(self._login_account(account))
            for account in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for account, result in zip(pending, results):
            if isinstance(result, Exception):
                logger.error("deepseek 登录失败 %s: %s", account.username, result)

    async def _bg_wasm_check(self) -> None:
        while not self._closing:
            try:
                await asyncio.sleep(86400)
            except asyncio.CancelledError:
                break
            if self._closing:
                break
            try:
                await download_wasm(self._session)
                self._pow = WasmPow()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("WASM 更新失败: %s", exc)

    async def _bg_hif_refresh(self) -> None:
        while not self._closing:
            try:
                await asyncio.sleep(HIF_REFRESH_INTERVAL)
            except asyncio.CancelledError:
                break
            if self._closing:
                break
            for account in self._accounts:
                if self._closing:
                    break
                if not account.token:
                    continue
                mgr = self._hif_managers.get(account.username)
                if mgr is None:
                    continue
                try:
                    leim, dliq, expire = await fetch_hif_tokens(self._session)
                    mgr._leim = leim
                    mgr._dliq = dliq
                    mgr._expire_at = expire
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("HIF 刷新失败 %s: %s", account.username, exc)

    async def _login_account(self, account: Account) -> None:
        token, user_id = await login(
            self._session, account.username, account.password
        )
        account.token = token
        account.user_id = user_id
        logger.info("deepseek 登录成功: %s (id=%s)", account.username, user_id)

        mgr = self._hif_managers.get(account.username)
        if mgr is not None:
            try:
                leim, dliq, expire = await fetch_hif_tokens(self._session)
                mgr._leim = leim
                mgr._dliq = dliq
                mgr._expire_at = expire
            except Exception as exc:
                logger.warning("首次 HIF 令牌获取失败 %s: %s", account.username, exc)

        self._rebuild_candidates()
