from __future__ import annotations

"""DeepSeek 上游 OpenAI 聊天流（原生 complete API）。"""

import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from echotools.logger import get_logger

from handlers import extract_system_for_inject
from handlers.fncall_inject import inject_fncall_for_request
from handlers.openai.protocol import _inject_protocol_options
from handlers.openai.thinking import protocol_thinking_level
from handlers.openai.tools import convert_tools_to_openai
from server.formats import (
    UpstreamTimeoutError,
    as_upstream_connection_error,
)
from server.model.model_thinking import resolve_qwen_thinking
from upstream.deepseek.lib.adapter.helpers.biz_error import (
    DeepSeekAccountsExhaustedError,
    DeepSeekUserMutedError,
)

logger = get_logger("rogator")

_MAX_MUTE_SWITCH = 8


def _normalize_chunk(chunk: Any) -> Optional[Dict[str, Any]]:
    if isinstance(chunk, str):
        return {"type": "answer", "content": chunk} if chunk else None
    if not isinstance(chunk, dict):
        return None
    if "usage" in chunk:
        return {"type": "usage", "data": chunk["usage"]}
    thinking = chunk.get("thinking")
    if thinking:
        return {"type": "thinking", "content": thinking}
    return None


def _prepare_messages(
    state: Any,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    model: str,
    protocol_options: Optional[Dict[str, Any]],
    prompt_api: str,
) -> Tuple[List[Dict[str, Any]], str]:
    _, _, use_entml = resolve_qwen_thinking(model, protocol_thinking_level(protocol_options))
    inject_options = _inject_protocol_options(protocol_options, use_entml)
    user_system_prompt, messages = extract_system_for_inject(messages)
    injected = inject_fncall_for_request(
        messages,
        convert_tools_to_openai(tools),
        state.protocol,
        req_id=req_id,
        api=prompt_api,
        model=model,
        lang="zh",
        user_system_prompt=user_system_prompt,
        protocol_options=inject_options,
    )
    send_text = injected[0].get("content") or ""
    if state.splitter.send_full_prompt or len(send_text) <= state.splitter.max_chars:
        return injected, send_text
    send_text = send_text[-state.splitter.max_chars:]
    return [{**injected[0], "content": send_text}], send_text


async def _on_user_muted(
    client: Any, username: str, exc: DeepSeekUserMutedError, attempt: int
) -> None:
    if username and hasattr(client, "handle_account_muted"):
        client.handle_account_muted(username, mute_at=time.time())
        logger.warning(
            "DeepSeek muted %s: %s (attempt %d/%d)",
            username[:6],
            exc.biz_msg,
            attempt + 1,
            _MAX_MUTE_SWITCH,
        )
    switched = await client.switch_to_next(exclude_username=username or None)
    if switched is None:
        raise DeepSeekAccountsExhaustedError(
            "All DeepSeek accounts are muted or unavailable"
        ) from exc


def _reraise_transport_error(exc: BaseException) -> None:
    """将超时/连接类异常映射为 session_retry 可识别类型。"""
    if isinstance(exc, asyncio.TimeoutError):
        raise UpstreamTimeoutError("DeepSeek upstream timeout") from exc
    conn_err = as_upstream_connection_error(exc, upstream="deepseek")
    if conn_err is not None:
        raise conn_err from exc
    raise exc


async def _iter_complete_events(
    inner: Any,
    candidate: Any,
    messages: List[Dict[str, Any]],
    model: str,
    *,
    thinking: bool,
) -> AsyncGenerator[Dict[str, Any], None]:
    async for chunk in inner.complete(
        candidate, messages, model, stream=True, thinking=thinking, search=False,
    ):
        event = _normalize_chunk(chunk)
        if event is not None:
            yield event


async def stream_openai_chat(
    state: Any,
    client: Any,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    *,
    protocol_options: Optional[Dict[str, Any]] = None,
    prompt_api: str = "openai",
    files: Optional[List[Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    qwen_enabled, _, _ = resolve_qwen_thinking(
        model, protocol_thinking_level(protocol_options)
    )
    final_messages, send_text = _prepare_messages(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )
    if files:
        logger.debug(
            "DeepSeek: ignoring %d uploaded file(s) for req %s", len(files), req_id
        )
    yield {"type": "prompt_meta", "prompt_chars": len(send_text)}
    inner = await client._ensure_ready()  # noqa: SLF001
    for attempt in range(_MAX_MUTE_SWITCH):
        candidate = await client.pick_candidate()
        username = str(candidate.meta.get("identifier") or "")
        try:
            async for event in _iter_complete_events(
                inner, candidate, final_messages, model, thinking=qwen_enabled,
            ):
                yield event
            return
        except DeepSeekUserMutedError as exc:
            await _on_user_muted(client, username, exc, attempt)
        except Exception as exc:
            _reraise_transport_error(exc)
    raise DeepSeekAccountsExhaustedError("DeepSeek mute switch limit exceeded")
