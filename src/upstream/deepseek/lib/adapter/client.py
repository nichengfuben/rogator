from __future__ import annotations

# src/platforms/deepseek/core/client.py
"""DeepSeek HTTP 客户端——管理账号登录、PoW、HIF、流式补全"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp

from upstream.deepseek.lib.adapter.helpers.client_helpers import (
    prepare_full_request,
    stream_initial_response,
)
from upstream.deepseek.lib.adapter.helpers.pmtutil import Account
from upstream.deepseek.lib.adapter.life import _ClientLifecycleMixin
from upstream.deepseek.lib.adapter.strmrun import _StreamRunMixin
from upstream.deepseek.lib.adapter.util import Candidate
from upstream.deepseek.lib.adapter.util import make_candidate_id as make_id
from upstream.deepseek.lib.guard.hif import HifTokenManager
from upstream.deepseek.lib.guard.pow import WasmPow
from upstream.deepseek.lib.protocol.consts import (
    CAPS,
    DEFAULT_HOST,
    MAX_RETRIES,
    MODELS,
)
from upstream.deepseek.lib.stream.strmpars import StreamParser

logger = logging.getLogger(__name__)

__all__ = ["Account", "Candidate", "DeepseekClient", "make_id"]


# ── 客户端主类 ─────────────────────────────────────────────────────────────────


class DeepseekClient(_ClientLifecycleMixin, _StreamRunMixin):
    """DeepSeek HTTP 客户端，管理账号登录、PoW、HIF 与流式补全。"""

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._pow: WasmPow = WasmPow()
        self._models: List[str] = list(MODELS)
        self._candidates: List[Any] = []
        # 每个账号对应一个 HIF 令牌管理器
        self._hif_managers: Dict[str, HifTokenManager] = {}
        self._proxy_override: Optional[bool] = None
        self._closing: bool = False
        self._bg_tasks: List[Any] = []

    async def init_immediate(
        self,
        session: aiohttp.ClientSession,
        accounts: Optional[List[Account]] = None,
    ) -> None:
        self._session = session
        if accounts is not None:
            self._accounts = list(accounts)
        else:
            try:
                from upstream.deepseek.accounts import accounts_for_upstream

                self._accounts = [
                    Account(username=a.username, password=a.password)
                    for a in accounts_for_upstream("deepseek")
                    if a.username and a.password
                ]
            except (ImportError, ModuleNotFoundError):
                self._accounts = []
        # 为每个账号预建 HIF 管理器
        for account in self._accounts:
            mgr = HifTokenManager()
            mgr.bind_session(session)
            self._hif_managers[account.username] = mgr
        self._rebuild_candidates()
        logger.info("deepseek 客户端已初始化（等待后台登录）")

    def rebind_http_session(self, session: aiohttp.ClientSession) -> None:
        """transport 重建后同步 ClientSession 与全部 HIF 管理器。"""
        self._session = session
        for mgr in self._hif_managers.values():
            mgr.bind_session(session)

    def set_proxy_enabled(self, enabled: bool) -> None:
        self._proxy_override = bool(enabled)

    def is_proxy_enabled(self) -> bool:
        return bool(self._proxy_override)

    def _get_proxy_kwarg(self) -> Optional[str]:
        if self._proxy_override is True:
            from upstream.deepseek.lib.adapter.util import load_use_proxy

            if not load_use_proxy():
                return None
            from .runtime import get_proxy_server

            return get_proxy_server() or None
        return None

    def update_models(self, models: List[str]) -> None:
        merged = list(
            dict.fromkeys(list(models) + [m for m in MODELS if m not in models])
        )
        self._models = merged
        for cand in self._candidates:
            cand.models = list(self._models)

    def _rebuild_candidates(self) -> None:
        self._candidates = [
            Candidate(
                id=make_id("deepseek", account.username[:20]),
                platform="deepseek",
                resource_id=account.username[:20],
                models=list(self._models),
                context_length=account.context_length,
                meta={
                    "identifier": account.username,
                    "token": account.token,
                    "user_id": account.user_id,
                },
                **CAPS,
            )
            for account in self._accounts
            if account.token
        ]

    async def candidates(self) -> List[Any]:
        return list(self._candidates)

    async def ensure_candidates(self, count: int) -> int:
        return len(self._candidates)

    async def complete(
        self,
        candidate: Any,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        *,
        thinking: bool = False,
        search: bool = False,
        **kw: Any,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        async for chunk in self._complete_with_retry(
            candidate,
            messages,
            model,
            stream,
            thinking,
            search,
            ref_file_ids=kw.get("ref_file_ids"),
        ):
            yield chunk

    async def _complete_with_retry(
        self,
        candidate: Any,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        effective_thinking: bool,
        effective_search: bool,
        *,
        ref_file_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """按 MAX_RETRIES 重试执行 ``_do_complete``。"""
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                await asyncio.sleep(1.0 * (2 ** (attempt - 1)))
            try:
                async for chunk in self._do_complete(
                    candidate,
                    messages,
                    model,
                    stream,
                    thinking=effective_thinking,
                    search=effective_search,
                    ref_file_ids=ref_file_ids,
                ):
                    yield chunk
                return
            except Exception as exc:
                last_exc = exc
                logger.warning("deepseek 重试 %d/%d: %s", attempt + 1, MAX_RETRIES, exc)
        if last_exc:
            raise last_exc

    async def _do_complete(
        self,
        candidate: Any,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        *,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        (
            ctx,
            session_id,
            hif_leim,
            hif_dliq,
            post_kw,
            parser,
        ) = await prepare_full_request(
            self._session,
            self._hif_managers,
            self._pow,
            candidate,
            messages,
            model,
            self._proxy_override,
            self._get_proxy_kwarg,
            StreamParser,
            ref_file_ids=ref_file_ids,
            thinking_enabled=thinking,
            search_enabled=search,
            include_thinking=thinking,
        )
        try:
            async for chunk in self._stream_and_continue(
                ctx, session_id, hif_leim, hif_dliq, post_kw, parser
            ):
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            await self._abort_upstream_on_cancel(ctx["token"], session_id, parser)
            raise

    async def stop_upstream_generation(
        self,
        token: str,
        chat_session_id: str,
        message_id: Any,
    ) -> bool:
        """POST /api/v0/chat/stop_stream，终止上游仍在进行的生成。"""
        if not token or not chat_session_id or message_id is None:
            return False
        if self._session is None:
            return False
        from upstream.deepseek.lib.session.sessapi import stop_stream

        try:
            return await stop_stream(
                self._session,
                token,
                str(chat_session_id),
                str(message_id),
            )
        except Exception as exc:
            logger.debug("stop_upstream_generation failed: %s", exc)
            return False

    async def _abort_upstream_on_cancel(
        self,
        token: str,
        chat_session_id: Any,
        parser: Any,
    ) -> None:
        message_id = getattr(parser, "message_id", None)
        if message_id is None:
            return
        stopped = await self.stop_upstream_generation(
            token,
            str(chat_session_id),
            message_id,
        )
        if stopped:
            logger.info(
                "Stopped DeepSeek upstream generation session=%s msg=%s",
                str(chat_session_id)[:8],
                message_id,
            )

    async def _stream_and_continue(
        self,
        ctx: Dict[str, Any],
        session_id: Any,
        hif_leim: str,
        hif_dliq: str,
        post_kw: Dict[str, Any],
        parser: Any,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """发起初次流式请求，按需触发续写，最后附加 usage。"""
        token = ctx["token"]
        prompt = ctx["prompt"]
        url = "https://{}/api/v0/chat/completion".format(DEFAULT_HOST)

        parser.begin_stream(is_continuation=False)
        state: Dict[str, bool] = {"needs_continue": False}
        async for chunk in stream_initial_response(
            self._session, url, post_kw, parser, self._parse_sse_stream, state
        ):
            yield chunk

        needs_continue = state["needs_continue"] or parser.should_continue

        async for chunk in self._run_continue_loop(
            parser, session_id, token, hif_leim, hif_dliq, needs_continue
        ):
            yield chunk

        for chunk in self._compute_usage(parser, prompt):
            yield chunk

    async def close(self) -> None:
        """清理资源并取消后台协程。"""
        self._closing = True
        cancel = getattr(self, "_cancel_bg_tasks", None)
        if callable(cancel):
            await cancel()
