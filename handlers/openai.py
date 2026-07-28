from __future__ import annotations

"""OpenAI-compatible chat completion handlers."""

import asyncio
import json
from contextlib import aclosing
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from aiohttp import web

from echotools.fncall import FncallStreamParser
from echotools.logger import get_logger
from echotools.exec.fncall.protocols.entml_think.core import (
    normalize_thinking_level,
    normalize_thinking_mode,
    parse_max_thinking_length,
    resolve_thinking_injection,
)
from echotools.exec.fncall.protocols.entml_think.parse import split_entml_thinking

from handlers import EmptyResponseError, extract_system_for_inject, get_state
from handlers.fncall_inject import inject_fncall_for_request
from server.response_record import record_raw_response
from server.model_thinking import always_qwen_thinking, resolve_qwen_thinking
from server.session_retry import run_with_session_retry, stream_with_session_retry
from server.formats import (
    MAX_QUEUE_SIZE,
    TokenExpiredError,
    UpstreamUsageTracker,
    ClientDisconnectedError,
    _error_response,
    _fix_tool_call_id,
    _gen_chatcmpl_id,
    _gen_request_id,
    _gen_tool_id,
    _json_response,
    build_openai_chunk,
    build_openai_response,
    build_openai_stream_usage_chunk,
    client_disconnected_response,
    openai_stream_include_usage,
    read_request_json,
)
from state import AppState, QueueFullError

logger = get_logger("rogator")

_STREAM_CHUNK_SIZE = 20


# effort 别名 → echotools thinking_level
_EFFORT_LEVEL_ALIASES = {
    "minimal": "low",
    "default": "medium",
}


def _map_to_thinking_level(raw: Any) -> Optional[str]:
    """映射请求侧 thinking / effort 值为 echotools thinking_level。"""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "medium" if raw else "none"
    if isinstance(raw, (int, float)):
        return "none" if raw == 0 else "medium"
    key = str(raw).strip().lower()
    if not key:
        return None
    level = normalize_thinking_level(key)
    if level is not None:
        return level
    if key in _EFFORT_LEVEL_ALIASES:
        return _EFFORT_LEVEL_ALIASES[key]
    mode = normalize_thinking_mode(key)
    if mode == "off":
        return "none"
    if mode == "on":
        return "medium"
    if mode == "auto":
        return "auto"
    return None


def protocol_thinking_level(protocol_options: Optional[Dict[str, Any]]) -> str:
    """从 protocol_options 读取 thinking_level；兼容旧 thinking_mode。"""
    opts = protocol_options or {}
    if opts.get("thinking_level") is not None:
        normalized = normalize_thinking_level(opts.get("thinking_level"))
        if normalized is not None:
            return normalized
    legacy = normalize_thinking_mode(opts.get("thinking_mode"))
    if legacy == "off":
        return "none"
    if legacy == "on":
        return "medium"
    if legacy == "auto":
        return "auto"
    return "none"


def thinking_level_is_active(level: str) -> bool:
    return level not in ("none", "")


def _build_protocol_options(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从请求体提取 thinking 设置并构建 protocol_options。

    兼容：
    - thinking_level: none|low|medium|high|xhigh|max|auto
    - thinking: true/false/"on"/"auto"/"off"/effort 档位
    - thinking: {type, effort, mode, budget_tokens, max_thinking_length, ...}
    - 顶层 max_thinking_length / thinking_mode
    - reasoning_effort / reasoning（OpenAI / kimi-code）
    """
    raw = body.get("thinking")
    level: Optional[str] = normalize_thinking_level(body.get("thinking_level"))
    max_len: Optional[int] = None

    if isinstance(raw, bool):
        level = "medium" if raw else "none"
    elif isinstance(raw, dict):
        if level is None and "level" in raw:
            level = normalize_thinking_level(raw.get("level"))
        if level is None and "mode" in raw:
            level = _map_to_thinking_level(raw.get("mode"))
        if level is None and "type" in raw:
            level = _map_to_thinking_level(raw.get("type"))
        if level is None and "enabled" in raw:
            level = "medium" if raw.get("enabled") else "none"
        if level is None and "effort" in raw:
            level = _map_to_thinking_level(raw.get("effort"))
        for key in ("budget_tokens", "max_tokens", "max_thinking_length"):
            if key in raw:
                max_len = parse_max_thinking_length(raw.get(key))
                if max_len is not None:
                    break
    elif raw is not None and level is None:
        level = _map_to_thinking_level(raw)

    if level is None:
        level = _map_to_thinking_level(body.get("thinking_mode"))

    if level is None and "reasoning_effort" in body:
        level = _map_to_thinking_level(body.get("reasoning_effort"))

    if level is None:
        reasoning = body.get("reasoning")
        if isinstance(reasoning, dict):
            if "level" in reasoning:
                level = normalize_thinking_level(reasoning.get("level"))
            if level is None and "effort" in reasoning:
                level = _map_to_thinking_level(reasoning.get("effort"))
            if level is None and "mode" in reasoning:
                level = _map_to_thinking_level(reasoning.get("mode"))
            if level is None and "enabled" in reasoning:
                level = "medium" if reasoning.get("enabled") else "none"
            if level is None and "type" in reasoning:
                level = _map_to_thinking_level(reasoning.get("type"))
            if max_len is None:
                for key in ("budget_tokens", "max_tokens", "max_thinking_length"):
                    if key in reasoning:
                        max_len = parse_max_thinking_length(reasoning.get(key))
                        if max_len is not None:
                            break
        elif reasoning is not None:
            level = _map_to_thinking_level(reasoning)

    if max_len is None:
        max_len = parse_max_thinking_length(body.get("max_thinking_length"))

    opts: Dict[str, Any] = {"include_thinking_in_history": True}

    if level is None and max_len is None:
        return opts

    if level == "none" or (level is None and max_len is not None):
        if level == "none":
            opts["thinking_level"] = "none"
        elif max_len is not None:
            opts["thinking_level"] = "medium"
            opts["max_thinking_length"] = max_len
        return opts

    if level is not None:
        opts["thinking_level"] = level
    if max_len is not None:
        opts["max_thinking_length"] = max_len
    return opts


def _inject_protocol_options(protocol_options: Optional[Dict[str, Any]], use_entml: bool) -> Dict[str, Any]:
    """合并 inject_fncall 选项：历史 thinking 始终交给 echotools 渲染。"""
    opts = dict(protocol_options or {})
    opts.setdefault("include_thinking_in_history", True)
    if not use_entml:
        opts.pop("thinking_level", None)
        opts.pop("thinking_mode", None)
        opts.pop("max_thinking_length", None)
    return opts


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


async def _prepare_stream(
    state,
    messages,
    model,
    tools,
    req_id,
    protocol_options=None,
    *,
    prompt_api: str = "openai",
):
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
    # inject 后超限：尾部 max_chars → send_text，剩余前缀 → 附件（不再回拼 header）
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
        except Exception as e:
            logger.warning("Upload failed: %s, sending truncated text without attachment", e)
            # 上传失败：仍用尾部 send_text，仅丢弃附件
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
    async for event in state.client.chat_completion(
        session, chat_id, final_messages, model, uploaded_files,
        qwen_thinking_enabled=qwen_enabled,
        qwen_thinking_mode=qwen_mode,
    ):
        yield event


async def _process_openai_non_stream(state, messages, model, req_id, tools, protocol_options=None):
    """非流式处理 - 含换号重试"""
    async def _run():
        response_parts = []
        think_parts = []
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
                if event.get("type") in ("response_created", "usage"):
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
        return build_openai_response(
            model, display_text, reasoning=reasoning, tool_calls=tool_calls,
            usage=usage_tracker.openai_stream_usage(),
        )

    return await run_with_session_retry(req_id, state, _run)


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


def _openai_tool_call_entry(index: int, tc: Dict[str, Any]) -> Dict[str, Any]:
    """单个 tool_calls delta 条目（对齐 mock.py OpenAIBuilder._build_tool_calls）。"""
    fixed = _fix_tool_call_id(tc)
    func = fixed.get("function", {})
    args = func.get("arguments", "{}")
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {
        "index": index,
        "id": fixed["id"],
        "type": "function",
        "function": {
            "name": func.get("name", ""),
            "arguments": args,
        },
    }


async def _emit_tool_call_chunks(
    resp,
    model: str,
    chunk_id: str,
    tool_calls: List[Dict[str, Any]],
    start_index: int,
    disconnected: list,
    *,
    include_usage: bool = False,
) -> int:
    """按 mock 规范：每个 tool call 单独一个 chunk，delta.content=null。"""
    index = start_index
    for tc in tool_calls:
        chunk = build_openai_chunk(
            model,
            chunk_id=chunk_id,
            tool_calls=[_openai_tool_call_entry(index, tc)],
            usage_null=include_usage,
        )
        if not await _emit_chunk(resp, chunk, disconnected):
            break
        index += 1
    return index


async def _emit_openai_streaming_tool_delta(
    resp,
    parser,
    model: str,
    chunk_id: str,
    stream_tool: Optional[Dict[str, Any]],
    tool_index: int,
    disconnected: list,
    *,
    include_usage: bool = False,
) -> Tuple[Optional[Dict[str, Any]], int, bool]:
    """invoke 开标签就绪后，增量发送 tool_calls.function.arguments（无需等 </entml:invoke>）。"""
    while True:
        delta_info = parser.consume_stream_delta()
        if not delta_info:
            break
        name, partial_json = delta_info
        if not partial_json:
            continue
        if stream_tool is not None and stream_tool.get("name") != name:
            stream_tool = None
            tool_index += 1
        if stream_tool is None:
            stream_tool = {
                "index": tool_index,
                "id": _gen_tool_id(),
                "name": name,
                "header_sent": False,
            }
        for i in range(0, len(partial_json), _STREAM_CHUNK_SIZE):
            piece = partial_json[i : i + _STREAM_CHUNK_SIZE]
            if not stream_tool["header_sent"]:
                entry: Dict[str, Any] = {
                    "index": tool_index,
                    "id": stream_tool["id"],
                    "type": "function",
                    "function": {"name": name, "arguments": piece},
                }
                stream_tool["header_sent"] = True
            else:
                entry = {
                    "index": stream_tool["index"],
                    "function": {"arguments": piece},
                }
            chunk = build_openai_chunk(
                model,
                chunk_id=chunk_id,
                tool_calls=[entry],
                usage_null=include_usage,
            )
            if not await _emit_chunk(resp, chunk, disconnected):
                return stream_tool, tool_index, False
    return stream_tool, tool_index, True


async def _emit_openai_streaming_tool_argument_pieces(
    resp,
    model: str,
    chunk_id: str,
    stream_tool: Dict[str, Any],
    partial_json: str,
    disconnected: list,
    *,
    include_usage: bool = False,
) -> bool:
    """向已打开的流式 tool call 追加 arguments 片段。"""
    for i in range(0, len(partial_json), _STREAM_CHUNK_SIZE):
        piece = partial_json[i : i + _STREAM_CHUNK_SIZE]
        entry = {
            "index": stream_tool["index"],
            "function": {"arguments": piece},
        }
        chunk = build_openai_chunk(
            model,
            chunk_id=chunk_id,
            tool_calls=[entry],
            usage_null=include_usage,
        )
        if not await _emit_chunk(resp, chunk, disconnected):
            return False
    return True


async def _send_stream_finish(
    resp,
    model: str,
    chunk_id: str,
    all_tool_calls: List[Dict[str, Any]],
    disconnected: list,
    already_sent_tc_count: int = 0,
    usage: Optional[Dict[str, Any]] = None,
    *,
    include_usage: bool = False,
) -> None:
    """补发未流式送出的 tool_calls，然后 finish + [DONE]（对齐 OpenAI 官方流式收尾）。"""
    remaining = all_tool_calls[already_sent_tc_count:]
    if remaining:
        await _emit_tool_call_chunks(
            resp, model, chunk_id, remaining, already_sent_tc_count, disconnected,
            include_usage=include_usage,
        )
    finish_reason = (
        "tool_calls" if (all_tool_calls or already_sent_tc_count > 0) else "stop"
    )
    if include_usage:
        chunk = build_openai_chunk(
            model, chunk_id=chunk_id, finish_reason=finish_reason, usage_null=True,
        )
        await _emit_chunk(resp, chunk, disconnected)
        if usage is not None:
            usage_chunk = build_openai_stream_usage_chunk(model, chunk_id, usage)
            await _emit_chunk(resp, usage_chunk, disconnected)
    else:
        chunk = build_openai_chunk(
            model, chunk_id=chunk_id, finish_reason=finish_reason, usage=usage,
        )
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
    try:
        body = await read_request_json(request)
    except ClientDisconnectedError:
        logger.info("OpenAI client disconnected while reading body from %s", request.remote)
        return client_disconnected_response()
    except json.JSONDecodeError:
        return _error_response(400, "Invalid JSON body")
    messages = body.get("messages", [])
    model = body.get("model", state.model)
    stream = body.get("stream", False)
    tools = body.get("tools", [])
    if not messages:
        return _error_response(400, "messages is required")
    protocol_options = _build_protocol_options(body)
    req_level = protocol_thinking_level(protocol_options)
    _, _, use_entml = resolve_qwen_thinking(model, req_level)
    qwen_thinking = not use_entml and (always_qwen_thinking(model) or thinking_level_is_active(req_level))
    logger.info(
        "OpenAI: %d messages, model=%s, stream=%s, tools=%d, thinking_level=%s, qwen_thinking=%s",
        len(messages), model, stream, len(tools), req_level, qwen_thinking,
    )
    req_id = _gen_request_id()
    if not stream:
        return await _handle_non_stream(state, messages, model, req_id, tools, protocol_options)
    return await _handle_stream(
        request, state, messages, model, req_id, tools, protocol_options,
        include_usage=openai_stream_include_usage(body),
    )


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


async def _handle_stream(
    request, state, messages, model, req_id, tools, protocol_options=None, *,
    include_usage: bool = False,
):
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
    parser = FncallStreamParser(protocol=state.protocol, tools=tools, protocol_options=protocol_options)
    last_safe_len = 0
    last_thinking_len = 0
    pending_tc_index = 0  # 已流式发送的 tool call 数量
    streamed_tool_calls: List[Dict[str, Any]] = []
    stream_tool: Optional[Dict[str, Any]] = None
    stream_tool_blocks_sent = 0
    usage_tracker = UpstreamUsageTracker()

    def _stream_chunk(**kwargs: Any) -> Dict[str, Any]:
        if include_usage:
            kwargs["usage_null"] = True
        return build_openai_chunk(model, chunk_id=chunk_id, **kwargs)

    with record_raw_response(req_id) as raw_recorder:
        try:
            async def _make_stream():
                async for event in _chat_once(
                    state, messages, model, tools, req_id, protocol_options=protocol_options,
                    prompt_api="openai",
                ):
                    yield event

            async with aclosing(
                stream_with_session_retry(req_id, state, _make_stream),
            ) as event_stream:
                async for event in event_stream:
                    if disconnected[0]:
                        break
                    usage_tracker.ingest_event(event)
                    raw_recorder.ingest_event(event)
                    etype = event.get("type")
                    if etype in ("response_created", "usage"):
                        continue
                    content = event.get("content", "")

                    if etype == "thinking":
                        full_thinking += content
                        if content:
                            chunk = _stream_chunk(reasoning=content)
                            if not await _emit_chunk(resp, chunk, disconnected):
                                break
                        continue

                    if etype != "answer":
                        continue

                    full_answer += content
                    ready_calls = parser.feed(content)

                    had_stream_tool = stream_tool is not None
                    stream_tool, pending_tc_index, ok = await _emit_openai_streaming_tool_delta(
                        resp, parser, model, chunk_id, stream_tool, pending_tc_index, disconnected,
                        include_usage=include_usage,
                    )
                    if not had_stream_tool and stream_tool is not None:
                        stream_tool_blocks_sent += 1
                    if not ok:
                        break

                    pt = parser.partial_thinking
                    if len(pt) > last_thinking_len:
                        new_thinking = pt[last_thinking_len:]
                        last_thinking_len = len(pt)
                        full_thinking += new_thinking
                        chunk = _stream_chunk(reasoning=new_thinking)
                        if not await _emit_chunk(resp, chunk, disconnected):
                            break

                    safe_text = parser.partial_text
                    if len(safe_text) > last_safe_len:
                        new_text = safe_text[last_safe_len:]
                        last_safe_len = len(safe_text)
                        chunk = _stream_chunk(content=new_text)
                        if not await _emit_chunk(resp, chunk, disconnected):
                            break

                    if ready_calls:
                        fixed = [_fix_tool_call_id(tc) for tc in ready_calls]
                        streamed_tool_calls.extend(fixed)
                        if stream_tool is not None:
                            stream_tool = None
                            pending_tc_index += len(fixed)
                        else:
                            pending_tc_index = await _emit_tool_call_chunks(
                                resp, model, chunk_id, fixed, pending_tc_index, disconnected,
                                include_usage=include_usage,
                            )

        except asyncio.CancelledError:
            logger.info("OpenAI stream cancelled %s", req_id)
            await _safe_write(resp, b"data: [DONE]\n\n", disconnected)
            raise
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

        all_tool_calls: List[Dict[str, Any]] = []
        final_text = parser.partial_text
        try:
            final_text, parsed_calls = parser.finalize()
            all_tool_calls = [_fix_tool_call_id(tc) for tc in parsed_calls]
        except Exception as e:
            logger.warning("stream parser.finalize failed: %s", e)

        if not disconnected[0]:
            pt = parser.partial_thinking
            if len(pt) > last_thinking_len:
                new_thinking = pt[last_thinking_len:]
                last_thinking_len = len(pt)
                full_thinking += new_thinking
                chunk = _stream_chunk(reasoning=new_thinking)
                await _emit_chunk(resp, chunk, disconnected)

        if not disconnected[0]:
            safe_text = parser.partial_text if parser.has_calls else (final_text or parser.partial_text)
            if len(safe_text) > last_safe_len:
                new_text = safe_text[last_safe_len:]
                last_safe_len = len(safe_text)
                if new_text:
                    chunk = _stream_chunk(content=new_text)
                    await _emit_chunk(resp, chunk, disconnected)

        if not disconnected[0]:
            late_ready = parser.get_ready_tool_calls()
            if late_ready:
                fixed = [_fix_tool_call_id(tc) for tc in late_ready]
                streamed_tool_calls.extend(fixed)
                if stream_tool is not None:
                    stream_tool = None
                    pending_tc_index += len(fixed)
                else:
                    pending_tc_index = await _emit_tool_call_chunks(
                        resp, model, chunk_id, fixed, pending_tc_index, disconnected,
                        include_usage=include_usage,
                    )
            elif stream_tool is not None:
                if not parser.streaming_invoke_closed:
                    final_delta = parser.complete_stream_delta_if_needed()
                    if final_delta:
                        _, piece = final_delta
                        if piece:
                            await _emit_openai_streaming_tool_argument_pieces(
                                resp, model, chunk_id, stream_tool, piece, disconnected,
                                include_usage=include_usage,
                            )
                stream_tool = None
                if parser.streaming_invoke_closed or all_tool_calls:
                    pending_tc_index += 1
                else:
                    logger.warning(
                        "OpenAI stream ended with incomplete invoke %s", req_id,
                    )

        if not all_tool_calls:
            all_tool_calls = streamed_tool_calls

        if all_tool_calls and stream_tool_blocks_sent:
            pending_tc_index = max(
                pending_tc_index,
                min(len(all_tool_calls), stream_tool_blocks_sent),
            )

        await _send_stream_finish(
            resp, model, chunk_id, all_tool_calls, disconnected,
            already_sent_tc_count=pending_tc_index,
            usage=usage_tracker.openai_stream_usage(),
            include_usage=include_usage,
        )
        return resp
