from __future__ import annotations

# src/platforms/deepseek/core/client.py
"""DeepSeek HTTP 客户端——管理账号登录、PoW、HIF、流式补全"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any as _Any

import aiohttp

from upstream.deepseek.lib.protocol.consts import (
    CAPS,
    DEFAULT_HOST,
    MAX_RETRIES,
    MODELS,
)
from upstream.deepseek.lib.adapter.life import _ClientLifecycleMixin
from upstream.deepseek.lib.adapter.strmrun import _StreamRunMixin
from upstream.deepseek.lib.adapter.helpers.pmtutil import Account
from upstream.deepseek.lib.adapter.helpers.client_helpers import (
    prepare_full_request,
    stream_initial_response,
)
from upstream.deepseek.lib.guard.hif import HifTokenManager
from upstream.deepseek.lib.guard.pow import WasmPow
from upstream.deepseek.lib.runtime.stream.strmpars import StreamParser

logger = logging.getLogger(__name__)


# ── 本地替代：Candidate / make_id（原 src.core.dispatch.cand）──────────────────

@dataclass
class Candidate:
    """轻量候选项，替代 src.core.dispatch.cand.Candidate。"""
    id: str
    platform: str
    resource_id: str
    models: list
    context_length: int | None = None
    meta: dict = field(default_factory=dict)
    chat: bool = False
    completions: bool = False
    responses: bool = False
    thinking: bool = False
    search: bool = False
    tools: bool = False
    continuation: bool = False
    vision: bool = False

    def __init__(self, *, id: str, platform: str, resource_id: str, models: list,
                 context_length: int | None = None, meta: dict | None = None,
                 **caps: bool) -> None:
        self.id = id
        self.platform = platform
        self.resource_id = resource_id
        self.models = list(models)
        self.context_length = context_length
        self.meta = dict(meta) if meta else {}
        for k, v in caps.items():
            setattr(self, k, v)


def make_id(platform: str, suffix: str) -> str:
    """生成候选项 ID（替代 src.core.dispatch.cand.make_id）。"""
    raw = "{}:{}".format(platform, suffix)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

# 重新导出 Account，保持 `from .client import Account` 等旧引用路径可用
__all__ = ["Account", "DeepseekClient"]


# ── 客户端主类 ─────────────────────────────────────────────────────────────────

class DeepseekClient(_ClientLifecycleMixin, _StreamRunMixin):
    """DeepSeek HTTP 客户端（管理账号登录、PoW、HIF、流式补全）。

    单次请求生命周期中的上下文提取/HIF与PoW获取/会话与请求头准备/
    payload与post参数构造/初次响应流式解析等纯函数拆分至
    ``client_helpers.py``。
    """

    def __init__(self) -> None:
        """初始化客户端。"""
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
        """立即初始化（不阻塞）。

        Args:
            session: 共享的 aiohttp ClientSession。
            accounts: 账号列表（可选）。未提供时尝试从 accounts 模块加载。
        """
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

    def set_proxy_enabled(self, enabled: bool) -> None:
        """设置此平台的代理覆盖开关。

        Args:
            enabled: True 强制使用代理，False 强制不使用。
        """
        self._proxy_override = bool(enabled)

    def is_proxy_enabled(self) -> bool:
        """返回此平台当前是否启用代理覆盖。

        Returns:
            是否启用代理。
        """
        return bool(self._proxy_override)

    def _get_proxy_kwarg(self) -> Optional[str]:
        """获取应传递给 session.request 的 proxy 值。"""
        if self._proxy_override is True:
            from upstream.deepseek.lib.adapter.util import load_use_proxy

            if not load_use_proxy():
                return None
            from .runtime import get_proxy_server
            return get_proxy_server() or None
        return None

    def update_models(self, models: List[str]) -> None:
        """更新模型列表，同步刷新所有候选项的 models 字段。

        Args:
            models: 新的模型列表。
        """
        merged = list(dict.fromkeys(list(models) + [m for m in MODELS if m not in models]))
        self._models = merged
        for cand in self._candidates:
            cand.models = list(self._models)

    def _rebuild_candidates(self) -> None:
        """根据当前账号状态重建候选项列表。"""
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
        """返回当前候选项列表。

        Returns:
            候选项列表。
        """
        return list(self._candidates)

    async def ensure_candidates(self, count: int) -> int:
        """返回可用候选项数量。

        Args:
            count: 期望数量（此处仅返回当前实际数量）。

        Returns:
            当前可用候选项数量。
        """
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
        """执行聊天补全（含重试）。

        Args:
            candidate: 候选项。
            messages: 消息列表。
            model: 模型名（deepseek-v4-pro / deepseek-v4-flash / deepseek-v4-vision）。
            stream: 是否流式。
            thinking: 是否启用思考模式（两个模型均支持）。
            search: 是否启用联网搜索（两个模型均支持）。
            **kw: 额外参数透传。

        Yields:
            str（文本增量）或 dict（thinking/usage）。
        """
        async for chunk in self._complete_with_retry(
            candidate, messages, model, stream, False, False,
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
                ):
                    yield chunk
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "deepseek 重试 %d/%d: %s", attempt + 1, MAX_RETRIES, exc
                )
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
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """执行单次完整会话请求。

        Args:
            candidate: 候选项（含 token）。
            messages: 消息列表。
            model: 模型名。
            stream: 是否流式。
            thinking: 是否启用思考模式。
            search: 是否启用联网搜索。

        Yields:
            str（文本增量）或 dict（thinking/usage）。
        """
        ctx, session_id, hif_leim, hif_dliq, post_kw, parser = await prepare_full_request(
            self._session,
            self._hif_managers,
            self._pow,
            candidate,
            messages,
            model,
            self._proxy_override,
            self._get_proxy_kwarg,
            StreamParser,
        )
        async for chunk in self._stream_and_continue(
            ctx, session_id, hif_leim, hif_dliq, post_kw, parser
        ):
            yield chunk

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
