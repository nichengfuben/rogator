from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from handlers.api_errors import safe_write as _safe_write
from handlers.fncall_inject import STREAM_CHUNK_SIZE, emit_parser_stream_deltas, iter_text_chunks
from server.formats import build_openai_chunk, build_openai_stream_usage_chunk, _fix_tool_call_id, _gen_tool_id


async def _emit_chunk(resp, chunk: Dict[str, Any], disconnected: list) -> bool:
    return await _safe_write(
        resp,
        f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"),
        disconnected,
    )


async def _write_openai_stream_error(
    resp, message: str, disconnected: list, *, error_type: str = "server_error", code: int = 500,
) -> None:
    payload = {"error": {"message": message, "type": error_type, "code": code}}
    await _safe_write(resp, f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"), disconnected)


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


def _build_stream_tool_header_entry(tool_index: int, stream_tool: Dict[str, Any], name: str, piece: str) -> Dict[str, Any]:
    return {
        "index": tool_index,
        "id": stream_tool["id"],
        "type": "function",
        "function": {"name": name, "arguments": piece},
    }


def _build_stream_tool_args_entry(stream_tool: Dict[str, Any], piece: str) -> Dict[str, Any]:
    return {
        "index": stream_tool["index"],
        "function": {"arguments": piece},
    }


async def _emit_stream_tool_pieces(
    resp,
    model: str,
    chunk_id: str,
    stream_tool: Dict[str, Any],
    name: str,
    partial_json: str,
    disconnected: list,
    *,
    include_usage: bool = False,
) -> bool:
    for piece in iter_text_chunks(partial_json, STREAM_CHUNK_SIZE):
        if not stream_tool["header_sent"]:
            entry = _build_stream_tool_header_entry(stream_tool["index"], stream_tool, name, piece)
            stream_tool["header_sent"] = True
        else:
            entry = _build_stream_tool_args_entry(stream_tool, piece)
        chunk = build_openai_chunk(
            model,
            chunk_id=chunk_id,
            tool_calls=[entry],
            usage_null=include_usage,
        )
        if not await _emit_chunk(resp, chunk, disconnected):
            return False
    return True


def _init_stream_tool(tool_index: int, name: str) -> Dict[str, Any]:
    return {
        "index": tool_index,
        "id": _gen_tool_id(),
        "name": name,
        "header_sent": False,
    }


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
) -> tuple[Optional[Dict[str, Any]], int, bool]:
    """invoke 开标签就绪后，增量发送 tool_calls.function.arguments（无需等 </entml:invoke>）。"""

    async def _on_delta(name: str, partial_json: str) -> bool:
        nonlocal stream_tool, tool_index
        if stream_tool is not None and stream_tool.get("name") != name:
            stream_tool = None
            tool_index += 1
        if stream_tool is None:
            stream_tool = _init_stream_tool(tool_index, name)
        return await _emit_stream_tool_pieces(
            resp, model, chunk_id, stream_tool, name, partial_json, disconnected,
            include_usage=include_usage,
        )

    ok = await emit_parser_stream_deltas(parser, _on_delta)
    return stream_tool, tool_index, ok


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
    return await _emit_stream_tool_pieces(
        resp, model, chunk_id, stream_tool, stream_tool.get("name", ""), partial_json, disconnected,
        include_usage=include_usage,
    )


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
