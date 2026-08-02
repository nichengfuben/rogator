from __future__ import annotations

"""DeepSeek 上游 OpenAI 聊天流（原生 complete API）。"""

import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from echotools.base.logger import get_logger

from core.transport.conn_retry import reraise_transport_error
from handlers.chat_request import apply_prompt_budget, prepare_injected_messages
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
) -> Tuple[List[Dict[str, Any]], str, bool]:
    injected, full_content, qwen_enabled, _mode, _entml = prepare_injected_messages(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )
    final_messages, send_text, _filename, _file_bytes = apply_prompt_budget(
        state, injected, full_content,
    )
    return final_messages, send_text, qwen_enabled


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
    final_messages, send_text, qwen_enabled = _prepare_messages(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )
    if files:
        logger.debug(
            "DeepSeek: ignoring %d uploaded file(s) for req %s", len(files), req_id
        )
    yield {"type": "prompt_meta", "prompt_chars": len(send_text)}
    inner = await client._ensure_ready()  # noqa: SLF001
    for attempt in range(_MAX_MUTE_SWITCH):
        muted_exc: Optional[DeepSeekUserMutedError] = None
        muted_user = ""
        async with client.lease_valid_session() as session:
            if session is None:
                raise DeepSeekAccountsExhaustedError(
                    "DeepSeek 无可用会话，请检查账号配置与登录状态"
                )
            candidate = await client.pick_candidate(session)
            muted_user = str(candidate.meta.get("identifier") or "")
            try:
                async for event in _iter_complete_events(
                    inner, candidate, final_messages, model, thinking=qwen_enabled,
                ):
                    yield event
                return
            except DeepSeekUserMutedError as exc:
                muted_exc = exc
            except Exception as exc:
                reraise_transport_error(
                    exc,
                    upstream="deepseek",
                    timeout_message="DeepSeek upstream timeout",
                )
        if muted_exc is not None:
            await _on_user_muted(client, muted_user, muted_exc, attempt)
    raise DeepSeekAccountsExhaustedError("DeepSeek mute switch limit exceeded")
