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
from upstream.deepseek.lib.adapter.helpers.file_collect import collect_message_attachments_async
from upstream.deepseek.lib.adapter.helpers.file_upload import (
    resolve_model_type,
    upload_attachments,
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
    # 思考与工具走 entml；忽略 DS 原生 THINK 增量
    if chunk.get("thinking"):
        return None
    return None


def _prepare_messages(
    state: Any,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    model: str,
    protocol_options: Optional[Dict[str, Any]],
    prompt_api: str,
) -> Tuple[List[Dict[str, Any]], str, Optional[str], Optional[bytes]]:
    injected, full_content, _route = prepare_injected_messages(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )
    final_messages, send_text, filename, file_bytes = apply_prompt_budget(
        state, injected, full_content, use_file_split=True,
    )
    return final_messages, send_text, filename, file_bytes


def _extract_preuploaded_ids(files: Optional[List[Any]]) -> List[str]:
    if not files:
        return []
    ids: List[str] = []
    for item in files:
        if isinstance(item, str) and item:
            ids.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("id", "file_id", "fileId"):
            raw = item.get(key)
            if raw:
                ids.append(str(raw))
                break
    return ids


async def _upload_message_files(
    inner: Any,
    candidate: Any,
    messages: List[Dict[str, Any]],
    model: str,
    *,
    filename: Optional[str],
    file_bytes: Optional[bytes],
    preuploaded: Optional[List[Any]],
) -> List[str]:
    ref_ids = _extract_preuploaded_ids(preuploaded)
    attachments = await collect_message_attachments_async(
        messages, filename=filename, file_bytes=file_bytes,
    )
    if not attachments:
        return ref_ids

    token = str(candidate.meta.get("token") or "")
    username = str(candidate.meta.get("identifier") or "")
    if not token:
        logger.warning("DeepSeek: 无 token，跳过 %d 个附件上传", len(attachments))
        return ref_ids

    model_type = resolve_model_type(model)
    uploaded = await upload_attachments(
        inner._session,  # noqa: SLF001
        token,
        username,
        attachments,
        hif_managers=inner._hif_managers,  # noqa: SLF001
        pow_solver=inner._pow,  # noqa: SLF001
        model_type=model_type,
        thinking_enabled=False,
    )
    ref_ids.extend(uploaded)
    logger.info(
        "DeepSeek: uploaded %d file(s), ref_file_ids=%d",
        len(uploaded),
        len(ref_ids),
    )
    return ref_ids


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
    ref_file_ids: Optional[List[str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    async for chunk in inner.complete(
        candidate,
        messages,
        model,
        stream=True,
        thinking=False,
        search=False,
        ref_file_ids=ref_file_ids,
    ):
        event = _normalize_chunk(chunk)
        if event is not None:
            yield event


async def _stream_with_mute_retry(
    client: Any,
    inner: Any,
    final_messages: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    model: str,
    *,
    filename: Optional[str],
    file_bytes: Optional[bytes],
    preuploaded: Optional[List[Any]],
) -> AsyncGenerator[Dict[str, Any], None]:
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
                ref_file_ids = await _upload_message_files(
                    inner, candidate, messages, model,
                    filename=filename, file_bytes=file_bytes,
                    preuploaded=preuploaded,
                )
                async for event in _iter_complete_events(
                    inner, candidate, final_messages, model,
                    ref_file_ids=ref_file_ids or None,
                ):
                    yield event
                return
            except DeepSeekUserMutedError as exc:
                muted_exc = exc
            except Exception as exc:
                reraise_transport_error(
                    exc, upstream="deepseek",
                    timeout_message="DeepSeek upstream timeout",
                )
        if muted_exc is not None:
            await _on_user_muted(client, muted_user, muted_exc, attempt)
    raise DeepSeekAccountsExhaustedError("DeepSeek mute switch limit exceeded")


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
    final_messages, send_text, filename, file_bytes = _prepare_messages(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )
    if final_messages:
        final_messages[0]["content"] = send_text
    yield {"type": "prompt_meta", "prompt_chars": len(send_text)}
    inner = await client._ensure_ready()  # noqa: SLF001
    async for event in _stream_with_mute_retry(
        client, inner, final_messages, messages, model,
        filename=filename, file_bytes=file_bytes,
        preuploaded=files,
    ):
        yield event
