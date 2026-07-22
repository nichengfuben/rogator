from __future__ import annotations

"""OpenAI-compatible chat completion handlers."""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from aiohttp import web

from echotools.fncall import FncallStreamParser
from echotools.logger import get_logger

from handlers import (
    TOOL_INSTRUCTION,
    EmptyResponseError,
    _patched_inject_fncall as inject_fncall,
    fold_system_into_user,
    get_state,
)
from server.formats import (
    MAX_CHARS,
    MAX_QUEUE_SIZE,
    TokenExpiredError,
    _error_response,
    _fix_tool_call_id,
    _gen_chatcmpl_id,
    _gen_request_id,
    _json_response,
    build_openai_chunk,
    build_openai_response,
)
from state import AppState, QueueFullError

logger = get_logger("rogator")


def convert_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not tools:
        return tools
    converted = []
    for tool in tools:
        if "type" in tool and tool.get("type") == "function":
            converted.append(tool)
            continue
        params = tool.get("input_schema", {})
        if not params:
            params = {"type": "object", "properties": {}}
        elif "type" not in params:
            params["type"] = "object"
        converted.append({"type": "function", "function": {
            "name": tool.get("name", ""), "description": tool.get("description", ""), "parameters": params}})
    return converted


def _parse_tool_calls(state: AppState, full_answer: str, tools: List[Dict]) -> Tuple[str, List[Dict[str, Any]]]:
    if not tools or not full_answer:
        return full_answer, []
    try:
        # echotools>=2.3.x 的 protocol.parse 已内置 normalize_tool_calls
        clean_text, parsed_calls = state.protocol.parse(full_answer, tools)
        tool_calls = [_fix_tool_call_id(tc) for tc in parsed_calls]
        if tool_calls:
            logger.info("protocol.parse: parsed %d tool calls", len(tool_calls))
        return clean_text, tool_calls
    except Exception as e:
        logger.warning("protocol.parse failed: %s", e)
        return full_answer, []


async def _prepare_stream(state, messages, model, tools, req_id):
    """准备工作（获取session、构建消息、上传文件、创建chat），单次尝试"""
    session = await state.client.get_valid_session()
    if not session:
        raise TokenExpiredError("No valid session available")
    image_uris = state.client.extract_base64_images(messages)
    image_urls = state.client.extract_image_urls(messages)
    messages = fold_system_into_user(messages)
    openai_tools = convert_tools_to_openai(tools)
    injected = inject_fncall(messages, openai_tools, state.protocol, lang="zh")
    full_content = injected[0]["content"]
    final_messages = injected
    send_text, filename, file_bytes = state.splitter.split(full_content)
    files = []
    for uri in image_uris:
        try:
            _, image_obj = await state.client.upload_file_from_base64(session, uri)
            files.append(image_obj)
        except Exception as e:
            logger.warning("Image upload failed: %s", e)
    for image_url in image_urls:
        try:
            _, image_obj = await state.client.upload_file_from_url(session, image_url)
            files.append(image_obj)
        except Exception as e:
            logger.debug("Remote image upload failed: %s", e)
    if filename and file_bytes:
        try:
            _, file_obj = await state.client.upload_file(session, file_bytes, filename)
            files.append(file_obj)
            if tools:
                send_text = TOOL_INSTRUCTION + send_text
        except Exception as e:
            logger.warning("Upload failed: %s, truncating to max chars", e)
            send_text = full_content[:MAX_CHARS]
            if tools:
                send_text = TOOL_INSTRUCTION + send_text
            files = []
    final_messages[0]["content"] = send_text
    chat_id = await state.client.create_chat(session, model)
    return session, final_messages, files, chat_id


async def _run_prepare_once(state, messages, model, tools, req_id):
    """单次准备，失败后切号再向上抛出。"""
    try:
        return await _prepare_stream(state, messages, model, tools, req_id)
    except (TokenExpiredError, EmptyResponseError, ConnectionError) as e:
        logger.warning("Prepare failed with %s: %s, switching session", type(e).__name__, e)
        await state.client.switch_to_next()
        raise


async def _chat_once(state, messages, model, tools, req_id, files=None) -> AsyncGenerator[Dict[str, Any], None]:
    """单次聊天，TokenExpiredError 切号后原样抛给上层。"""
    session, final_messages, uploaded_files, chat_id = await _run_prepare_once(
        state, messages, model, tools, req_id
    )
    if files is not None:
        uploaded_files = files
    try:
        async for event in state.client.chat_completion(
            session, chat_id, final_messages, model, uploaded_files
        ):
            yield event
    except TokenExpiredError as e:
        logger.warning("Chat failed with TokenExpiredError: %s, switching session", e)
        await state.client.switch_to_next()
        raise


async def _process_openai_non_stream(state, messages, model, req_id, tools):
    """非流式处理 - 单次尝试"""
    response_parts = []
    think_parts = []
    event_count = 0
    async for event in _chat_once(state, messages, model, tools, req_id):
        event_count += 1
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
    return build_openai_response(model, display_text, reasoning=reasoning, tool_calls=tool_calls)


# ============================================================
# 流式辅助函数
# ============================================================

async def _write_openai_stream_error(
    resp, message: str, disconnected: list, *, error_type: str = "server_error", code: int = 500,
) -> None:
    payload = {"error": {"message": message, "type": error_type, "code": code}}
    await _safe_write(resp, f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"), disconnected)


async def _safe_write(resp, data: bytes, disconnected: list) -> bool:
    if disconnected[0]:
        return False
    try:
        await resp.write(data)
        return True
    except (ConnectionError, OSError, asyncio.CancelledError):
        disconnected[0] = True
        return False


async def _process_stream_event(resp, event, parser, model, chunk_id, last_safe_len, disconnected):
    """处理单个流式事件，返回 (answer_chunk, thinking_chunk, last_safe_len, ok)。"""
    etype = event.get("type")
    content = event.get("content", "")
    if etype == "thinking":
        if not content:
            return "", content, last_safe_len, True
        chunk = build_openai_chunk(model, chunk_id=chunk_id, reasoning=content)
        ok = await _safe_write(resp, f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"), disconnected)
        return "", content, last_safe_len, ok
    if etype == "answer":
        parser.feed(content)
        safe_text = parser.partial_text
        if len(safe_text) > last_safe_len:
            new_text = safe_text[last_safe_len:]
            last_safe_len = len(safe_text)
            if new_text:
                chunk = build_openai_chunk(model, chunk_id=chunk_id, content=new_text)
                ok = await _safe_write(resp, f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"), disconnected)
                return content, "", last_safe_len, ok
        return content, "", last_safe_len, True
    return "", "", last_safe_len, True


# ============================================================
# 主处理器
# ============================================================

async def openai_chat_handler(request: web.Request) -> web.StreamResponse:
    state = get_state()
    if state.is_shutting_down:
        return web.Response(status=503, text="Shutting down")
    if state.scheduler.pending >= MAX_QUEUE_SIZE:
        return web.Response(status=503, text="Busy")
    body = await request.json() if request.can_read_body else {}
    messages = body.get("messages", [])
    model = body.get("model", state.model)
    stream = body.get("stream", False)
    tools = body.get("tools", [])
    if not messages:
        return _error_response(400, "messages is required")
    logger.info("OpenAI: %d messages, model=%s, stream=%s, tools=%d",
                len(messages), model, stream, len(tools))
    req_id = _gen_request_id()
    if not stream:
        return await _handle_non_stream(state, messages, model, req_id, tools)
    return await _handle_stream(request, state, messages, model, req_id, tools)


async def _handle_non_stream(state, messages, model, req_id, tools):
    try:
        result = await state.scheduler.submit(
            lambda: _process_openai_non_stream(state, messages, model, req_id, tools))
        return _json_response(result)
    except QueueFullError as e:
        return web.Response(status=503, text=str(e))
    except asyncio.CancelledError:
        return web.Response(status=503, text="Shutting down")
    except TokenExpiredError as e:
        logger.warning("OpenAI non-stream token expired: %s", e)
        return _error_response(429, str(e), "rate_limited")
    except Exception as e:
        logger.error("OpenAI non-stream error: %s", e, exc_info=True)
        return _error_response(500, str(e))


async def _send_stream_chunks(resp, state, full_answer, tools, model, chunk_id, disconnected):
    """发送工具调用块和结束块"""
    _, tool_calls = _parse_tool_calls(state, full_answer, tools)
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            openai_tc = [{
                "index": i, "id": tc.get("id"), "type": "function",
                "function": {"name": tc.get("function", {}).get("name"),
                             "arguments": tc.get("function", {}).get("arguments", "")},
            }]
            chunk = build_openai_chunk(model, chunk_id=chunk_id, tool_calls=openai_tc)
            if not await _safe_write(resp, f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"), disconnected):
                break
    finish_reason = "tool_calls" if tool_calls else "stop"
    chunk = build_openai_chunk(model, chunk_id=chunk_id, finish_reason=finish_reason)
    await _safe_write(resp, f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"), disconnected)
    await _safe_write(resp, b"data: [DONE]\n\n", disconnected)


async def _handle_stream(request, state, messages, model, req_id, tools):
    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                 "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
    await resp.prepare(request)
    chunk_id = _gen_chatcmpl_id()
    disconnected = [False]
    full_answer = ""
    full_thinking = ""
    parser = FncallStreamParser(protocol=state.protocol, tools=tools)
    last_safe_len = 0
    try:
        async for event in _chat_once(state, messages, model, tools, req_id):
            if disconnected[0]:
                break
            answer_chunk, think_chunk, last_safe_len, ok = await _process_stream_event(
                resp, event, parser, model, chunk_id, last_safe_len, disconnected
            )
            full_answer += answer_chunk
            full_thinking += think_chunk
            if not ok:
                break
    except asyncio.CancelledError:
        await _safe_write(resp, b"data: [DONE]\n\n", disconnected)
        return resp
    except TokenExpiredError as e:
        logger.warning("OpenAI stream token expired: %s", e)
        await _write_openai_stream_error(resp, str(e), disconnected, error_type="rate_limited", code=429)
        return resp
    except Exception as e:
        logger.error("OpenAI stream error: %s", e, exc_info=True)
        await _write_openai_stream_error(resp, str(e), disconnected)
        return resp
    if disconnected[0]:
        logger.info("Client disconnected during stream %s", req_id)
        return resp
    await _send_stream_chunks(resp, state, full_answer, tools, model, chunk_id, disconnected)
    return resp
