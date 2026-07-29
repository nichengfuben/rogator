from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from echotools.exec.fncall.protocols.entml_think.parse import split_entml_thinking
from echotools.logger import get_logger

from handlers import EmptyResponseError, extract_system_for_inject
from handlers.fncall_inject import inject_fncall_for_request
from handlers.openai.protocol import _inject_protocol_options
from handlers.openai.thinking import protocol_thinking_level
from handlers.openai.tools import _parse_tool_calls, convert_tools_to_openai
from server.formats import (
    TokenExpiredError,
    UpstreamUsageTracker,
    build_openai_response,
    log_qwen_upstream_usage,
)
from server.model.model_thinking import resolve_qwen_thinking
from server.records.response_record import record_raw_response
from server.retry import run_with_session_retry

logger = get_logger("rogator")


async def _upload_base64_images(state, session, image_uris: List[str]) -> List[Any]:
    files: List[Any] = []
    for uri in image_uris:
        try:
            _, image_obj = await state.client.upload_file_from_base64(session, uri)
            files.append(image_obj)
        except Exception as e:
            logger.warning("Image upload failed: %s", e)
    return files


async def _upload_remote_images(state, session, image_urls: List[str]) -> List[Any]:
    files: List[Any] = []
    for image_url in image_urls:
        try:
            _, image_obj = await state.client.upload_file_from_url(session, image_url)
            files.append(image_obj)
        except Exception as e:
            logger.debug("Remote image upload failed: %s", e)
    return files


async def _upload_text_attachment(
    state, session, filename: Optional[str], file_bytes: Optional[bytes],
) -> List[Any]:
    if not filename or not file_bytes:
        return []
    try:
        _, file_obj = await state.client.upload_file(session, file_bytes, filename)
        return [file_obj]
    except Exception as e:
        logger.warning("Upload failed: %s, sending truncated text without attachment", e)
        return []


async def _collect_uploaded_files(
    state, session, messages, image_uris, image_urls, filename, file_bytes,
) -> List[Any]:
    files: List[Any] = []
    files.extend(await _upload_base64_images(state, session, image_uris))
    files.extend(await _upload_remote_images(state, session, image_urls))
    files.extend(await _upload_text_attachment(state, session, filename, file_bytes))
    return files


async def _prepare_stream(
    state,
    messages,
    model,
    tools,
    req_id,
    protocol_options=None,
    *,
    prompt_api: str = "openai",
) -> Tuple[Any, List, List, str, bool, str]:
    """准备工作（获取session、构建消息、上传文件、创建chat），单次尝试"""
    qwen_enabled, qwen_mode, use_entml = resolve_qwen_thinking(
        model, protocol_thinking_level(protocol_options),
    )
    inject_options = _inject_protocol_options(protocol_options, use_entml)

    session = await state.client.get_valid_session()
    if not session:
        raise TokenExpiredError("No valid session available")

    image_uris = state.client.extract_base64_images(messages)
    image_urls = state.client.extract_image_urls(messages)
    user_system_prompt, messages = extract_system_for_inject(messages)
    openai_tools = convert_tools_to_openai(tools)
    injected = inject_fncall_for_request(
        messages,
        openai_tools,
        state.protocol,
        req_id=req_id,
        api=prompt_api,
        model=model,
        lang="zh",
        user_system_prompt=user_system_prompt,
        protocol_options=inject_options,
    )
    full_content = injected[0]["content"]
    final_messages = injected
    send_text, filename, file_bytes = state.splitter.split(full_content)
    files = await _collect_uploaded_files(
        state, session, messages, image_uris, image_urls, filename, file_bytes,
    )
    final_messages[0]["content"] = send_text
    chat_id = await state.client.create_chat(session, model)
    return session, final_messages, files, chat_id, qwen_enabled, qwen_mode


async def _run_prepare_once(
    state, messages, model, tools, req_id, protocol_options=None, *, prompt_api: str = "openai",
):
    """单次准备，失败向上抛出 TokenExpiredError。"""
    return await _prepare_stream(
        state, messages, model, tools, req_id, protocol_options, prompt_api=prompt_api,
    )


async def _chat_once(
    state,
    messages,
    model,
    tools,
    req_id,
    files=None,
    protocol_options=None,
    *,
    prompt_api: str = "openai",
) -> AsyncGenerator[Dict[str, Any], None]:
    """单次聊天（不含换号重试，由 session_retry 包装）。"""
    session, final_messages, uploaded_files, chat_id, qwen_enabled, qwen_mode = await _run_prepare_once(
        state, messages, model, tools, req_id, protocol_options, prompt_api=prompt_api,
    )
    if files is not None:
        uploaded_files = files
    send_text = final_messages[0].get("content") or ""
    yield {"type": "prompt_meta", "prompt_chars": len(send_text)}
    async for event in state.client.chat_completion(
        session, chat_id, final_messages, model, uploaded_files,
        qwen_thinking_enabled=qwen_enabled,
        qwen_thinking_mode=qwen_mode,
    ):
        yield event


async def _collect_non_stream_response(
    state, messages, model, tools, req_id, protocol_options,
) -> Dict[str, Any]:
    response_parts: List[str] = []
    think_parts: List[str] = []
    event_count = 0
    usage_tracker = UpstreamUsageTracker()
    with record_raw_response(req_id) as raw_recorder:
        async for event in _chat_once(
            state, messages, model, tools, req_id, protocol_options=protocol_options,
            prompt_api="openai",
        ):
            event_count += 1
            usage_tracker.ingest_event(event)
            raw_recorder.ingest_event(event)
            if event.get("type") in ("response_created", "usage", "prompt_meta"):
                continue
            if event.get("type") == "answer":
                response_parts.append(event.get("content", ""))
            elif event.get("type") == "thinking":
                think_parts.append(event.get("content", ""))
    if event_count == 0:
        logger.warning("No events received from qwen for req %s", req_id)
        raise EmptyResponseError(f"No events received from qwen for {req_id}")
    full_text = "".join(response_parts)
    reasoning = "".join(think_parts)
    display_text, tool_calls = _parse_tool_calls(state, full_text, tools)
    display_text, entml_thinking = split_entml_thinking(display_text)
    if entml_thinking:
        reasoning = f"{reasoning}\n{entml_thinking}".strip() if reasoning else entml_thinking
    log_qwen_upstream_usage(req_id, usage_tracker)
    return build_openai_response(
        model, display_text, reasoning=reasoning, tool_calls=tool_calls,
        usage=usage_tracker.openai_stream_usage(),
    )


async def _process_openai_non_stream(state, messages, model, req_id, tools, protocol_options=None):
    """非流式处理 - 含换号重试"""
    async def _run():
        return await _collect_non_stream_response(
            state, messages, model, tools, req_id, protocol_options,
        )

    return await run_with_session_retry(req_id, state, _run)
