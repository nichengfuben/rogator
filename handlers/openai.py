from __future__ import annotations

"""OpenAI-compatible chat completion handlers."""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from aiohttp import web

from echotools.fncall import FncallStreamParser, inject_fncall
from echotools.logger import get_logger
from echotools.exec.fncall.protocols.entml_thinking import normalize_thinking_mode

from handlers import EmptyResponseError, fold_system_into_user, get_state
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

_ENTML_USER_MARKER = "<current_user_message>"


def _build_protocol_options(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从请求体提取 thinking 设置并构建 protocol_options。"""
    raw = body.get("thinking")
    if raw is None:
        return None
    if isinstance(raw, bool):
        mode = "on" if raw else "off"
    elif isinstance(raw, dict):
        t = raw.get("type", "")
        mode = "on" if t == "enabled" else "off"
    else:
        mode = normalize_thinking_mode(raw) or "off"
    if mode == "off":
        return None
    return {"thinking_mode": mode}


def _entml_prompt_header(prompt: str) -> str:
    """提取 inject_fncall 产出的 entml 工具说明头（echotools 2.3+ render_prompt）。"""
    idx = prompt.find(_ENTML_USER_MARKER)
    return prompt[:idx] if idx > 0 else ""


def _ensure_entml_prompt_header(prompt: str, send_text: str) -> str:
    """长文本尾部截断后补回工具说明头，避免重复拼接过时指令。"""
    header = _entml_prompt_header(prompt)
    if header and not send_text.startswith(header):
        return header + send_text
    return send_text


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


async def _prepare_stream(state, messages, model, tools, req_id, protocol_options=None):
    """准备工作（获取session、构建消息、上传文件、创建chat），单次尝试"""
    session = await state.client.get_valid_session()
    if not session:
        raise TokenExpiredError("No valid session available")
    image_uris = state.client.extract_base64_images(messages)
    image_urls = state.client.extract_image_urls(messages)
    messages = fold_system_into_user(messages)
    openai_tools = convert_tools_to_openai(tools)
    injected = inject_fncall(messages, openai_tools, state.protocol, lang="zh",
                             protocol_options=protocol_options)
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
                send_text = _ensure_entml_prompt_header(full_content, send_text)
        except Exception as e:
            logger.warning("Upload failed: %s, truncating to max chars", e)
            send_text = full_content[:MAX_CHARS]
            files = []
    final_messages[0]["content"] = send_text
    chat_id = await state.client.create_chat(session, model)
    return session, final_messages, files, chat_id


async def _run_prepare_once(state, messages, model, tools, req_id, protocol_options=None):
    """单次准备，失败后切号再向上抛出。"""
    try:
        return await _prepare_stream(state, messages, model, tools, req_id, protocol_options)
    except (TokenExpiredError, EmptyResponseError, ConnectionError) as e:
        logger.warning("Prepare failed with %s: %s, switching session", type(e).__name__, e)
        await state.client.switch_to_next()
        raise


async def _chat_once(state, messages, model, tools, req_id, files=None, protocol_options=None) -> AsyncGenerator[Dict[str, Any], None]:
    """单次聊天，TokenExpiredError 切号后原样抛给上层。"""
    session, final_messages, uploaded_files, chat_id = await _run_prepare_once(
        state, messages, model, tools, req_id, protocol_options
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


async def _process_openai_non_stream(state, messages, model, req_id, tools, protocol_options=None):
    """非流式处理 - 单次尝试"""
    response_parts = []
    think_parts = []
    event_count = 0
    async for event in _chat_once(state, messages, model, tools, req_id, protocol_options=protocol_options):
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


async def _emit_chunk(resp, chunk: Dict[str, Any], disconnected: list) -> bool:
    return await _safe_write(
        resp,
        f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"),
        disconnected,
    )


async def _send_stream_chunks(
    resp, state, full_answer, tools, model, chunk_id, disconnected,
    already_sent_tc_count: int = 0,
):
    """发送尚未流式发送的 tool_calls 及结束块。"""
    _, all_tool_calls = _parse_tool_calls(state, full_answer, tools)
    remaining = all_tool_calls[already_sent_tc_count:]
    for i, tc in enumerate(remaining):
        openai_tc = [{
            "index": already_sent_tc_count + i,
            "id": tc.get("id"),
            "type": "function",
            "function": {
                "name": tc.get("function", {}).get("name"),
                "arguments": tc.get("function", {}).get("arguments", ""),
            },
        }]
        chunk = build_openai_chunk(model, chunk_id=chunk_id, tool_calls=openai_tc)
        if not await _emit_chunk(resp, chunk, disconnected):
            break
    finish_reason = "tool_calls" if all_tool_calls else "stop"
    chunk = build_openai_chunk(model, chunk_id=chunk_id, finish_reason=finish_reason)
    await _emit_chunk(resp, chunk, disconnected)
    await _safe_write(resp, b"data: [DONE]\n\n", disconnected)


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
    protocol_options = _build_protocol_options(body)
    logger.info("OpenAI: %d messages, model=%s, stream=%s, tools=%d, thinking_mode=%s",
                len(messages), model, stream, len(tools),
                (protocol_options or {}).get("thinking_mode", "off"))
    req_id = _gen_request_id()
    if not stream:
        return await _handle_non_stream(state, messages, model, req_id, tools, protocol_options)
    return await _handle_stream(request, state, messages, model, req_id, tools, protocol_options)


async def _handle_non_stream(state, messages, model, req_id, tools, protocol_options=None):
    try:
        result = await state.scheduler.submit(
            lambda: _process_openai_non_stream(state, messages, model, req_id, tools, protocol_options))
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


async def _handle_stream(request, state, messages, model, req_id, tools, protocol_options=None):
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
    last_thinking_len = 0
    pending_tc_index = 0  # 已流式发送的 tool call 数量
    try:
        async for event in _chat_once(state, messages, model, tools, req_id,
                                      protocol_options=protocol_options):
            if disconnected[0]:
                break
            etype = event.get("type")
            content = event.get("content", "")

            if etype == "thinking":
                # 上游原生 thinking 事件
                full_thinking += content
                if content:
                    chunk = build_openai_chunk(model, chunk_id=chunk_id, reasoning=content)
                    if not await _emit_chunk(resp, chunk, disconnected):
                        break
                continue

            if etype != "answer":
                continue

            full_answer += content
            parser.feed(content)

            # 发送 <entml:thinking> 块中新产生的思考内容
            pt = parser.partial_thinking
            if len(pt) > last_thinking_len:
                new_thinking = pt[last_thinking_len:]
                last_thinking_len = len(pt)
                full_thinking += new_thinking
                chunk = build_openai_chunk(model, chunk_id=chunk_id, reasoning=new_thinking)
                if not await _emit_chunk(resp, chunk, disconnected):
                    break

            # 发送新产生的可见文本
            safe_text = parser.partial_text
            if len(safe_text) > last_safe_len:
                new_text = safe_text[last_safe_len:]
                last_safe_len = len(safe_text)
                chunk = build_openai_chunk(model, chunk_id=chunk_id, content=new_text)
                if not await _emit_chunk(resp, chunk, disconnected):
                    break

            # 增量发送已完整解析的 tool calls
            for tc in parser.get_ready_tool_calls():
                tc = _fix_tool_call_id(tc)
                openai_tc = [{
                    "index": pending_tc_index,
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }]
                chunk = build_openai_chunk(model, chunk_id=chunk_id, tool_calls=openai_tc)
                if not await _emit_chunk(resp, chunk, disconnected):
                    break
                pending_tc_index += 1

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
    await _send_stream_chunks(resp, state, full_answer, tools, model, chunk_id, disconnected,
                              already_sent_tc_count=pending_tc_index)
    return resp
