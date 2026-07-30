from __future__ import annotations

import json
import queue
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from upstream.cursor.stream.exec import execute_tool, extract_tool_result_text
from upstream.cursor.stream.proto import StreamEvent
from upstream.cursor.stream.proto import (
    build_interaction_reply,
    parse_error,
    text_delta,
)


@dataclass
class AgentRunContext:
    q: queue.Queue
    conv_id: str
    start: float
    send_frame: Callable[[Dict[str, Any]], None]
    finish: Callable[..., None]
    touch: Callable[[], None]
    text_received: bool = False
    tool_completed_time: Optional[float] = None
    heartbeat_count: int = 0
    last_activity: float = field(default_factory=time.time)
    blob_store: Dict[str, str] = field(default_factory=dict)
    tool_handlers: Dict[str, Callable[..., Any]] = field(default_factory=dict)


def _emit_tool_call(ctx: AgentRunContext, tc: Dict[str, Any], elapsed: float) -> None:
    ctx.q.put(StreamEvent(type="tool_call", tool_call=tc, elapsed=elapsed))
    ctx.touch()
    ctx.heartbeat_count = 0


def _handle_heartbeat(ctx: AgentRunContext, elapsed: float) -> bool:
    ctx.heartbeat_count += 1
    idle = time.time() - ctx.last_activity
    if ctx.text_received and ctx.heartbeat_count >= 15 and idle > 120:
        ctx.finish(elapsed)
        return True
    if (
        ctx.tool_completed_time is not None
        and not ctx.text_received
        and (time.time() - ctx.tool_completed_time) > 15
    ):
        ctx.finish(elapsed)
        return True
    if not ctx.text_received and ctx.heartbeat_count >= 5 and idle > 30:
        ctx.finish(elapsed)
        return True
    return False


def _emit_text_delta(ctx: AgentRunContext, iu: Dict[str, Any], elapsed: float, *, nested: bool = False) -> bool:
    source = iu if not nested else (iu.get("message") or {})
    if nested and "textDelta" not in source:
        return False
    t = text_delta(source.get("textDelta") if nested else iu["textDelta"])
    if t:
        ctx.text_received = True
        ctx.q.put(StreamEvent(type="text", text=t, elapsed=elapsed))
        ctx.touch()
    return False


def _finish_turn(ctx: AgentRunContext, te: Dict[str, Any], elapsed: float) -> bool:
    ctx.finish(elapsed, usage={
        "prompt_tokens": int(te.get("inputTokens") or 0),
        "completion_tokens": int(te.get("outputTokens") or 0),
    })
    return True


def _handle_iu_tool_events(iu: Dict[str, Any], ctx: AgentRunContext, elapsed: float) -> bool:
    if "partialToolCall" in iu:
        ptc = iu.get("partialToolCall") or {}
        tc = ptc.get("toolCall") or {}
        ctx.q.put(StreamEvent(type="tool_partial", tool_name=tc.get("toolName", ""), elapsed=elapsed))
        ctx.touch()
        return True
    if "toolCallStarted" in iu:
        tcs = iu.get("toolCallStarted") or {}
        tc = tcs.get("toolCall") or {}
        tn = tc.get("toolName") or ""
        tc_id = tc.get("toolCallId") or str(uuid.uuid4())
        ctx.q.put(StreamEvent(type="tool_started", tool_name=tn, tool_call_id=tc_id, elapsed=elapsed))
        _emit_tool_call(ctx, {
            "id": tc_id,
            "type": "function",
            "function": {"name": tn, "arguments": tc.get("argsJson") or "{}"},
        }, elapsed)
        return True
    if "toolCallCompleted" in iu:
        ctx.q.put(StreamEvent(type="tool_completed", elapsed=elapsed))
        ctx.tool_completed_time = time.time()
        ctx.touch()
        return True
    return False


def _handle_iu_meta(iu: Dict[str, Any], ctx: AgentRunContext, elapsed: float) -> bool:
    if "summary" in iu:
        su = iu["summary"]
        s = su.get("summary", "") if isinstance(su, dict) else str(su)
        if s:
            ctx.q.put(StreamEvent(type="summary", text=s, elapsed=elapsed))
            ctx.touch()
        return True
    if "summaryStarted" in iu:
        ctx.q.put(StreamEvent(type="summary_started", elapsed=elapsed))
        ctx.touch()
        return True
    if "userMessageAppended" in iu:
        ctx.touch()
        return True
    if "toolCallDelta" in iu:
        tcd = iu["toolCallDelta"]
        ctx.q.put(StreamEvent(
            type="tool_delta",
            tool_call_id=tcd.get("callId", ""),
            data=tcd.get("toolCallDelta", {}),
            elapsed=elapsed,
        ))
        ctx.touch()
        return True
    if "tokenDelta" in iu:
        return True
    return False


def _handle_interaction_update(iu: Dict[str, Any], ctx: AgentRunContext, elapsed: float) -> bool:
    if "heartbeat" in iu:
        return _handle_heartbeat(ctx, elapsed)
    if "textDelta" in iu:
        return _emit_text_delta(ctx, iu, elapsed)
    if "thinkingDelta" in iu:
        t = text_delta(iu.get("thinkingDelta"))
        if t:
            ctx.q.put(StreamEvent(type="thinking", text=t, elapsed=elapsed))
            ctx.touch()
        return False
    if "thinkingCompleted" in iu:
        ctx.q.put(StreamEvent(type="thinking_done", elapsed=elapsed))
        ctx.touch()
        return False
    if _handle_iu_tool_events(iu, ctx, elapsed):
        return False
    if "turnEnded" in iu:
        return _finish_turn(ctx, iu.get("turnEnded") or {}, elapsed)
    if _handle_iu_meta(iu, ctx, elapsed):
        return False
    nested = iu.get("message") or {}
    if not nested:
        return False
    _emit_text_delta(ctx, iu, elapsed, nested=True)
    if "turnEnded" in nested:
        return _finish_turn(ctx, nested.get("turnEnded") or {}, elapsed)
    return False


def _handle_kv_server_message(msg: Dict[str, Any], ctx: AgentRunContext) -> None:
    kvm = msg.get("kvServerMessage") or {}
    kv_id = kvm.get("id", 0)
    kv_resp: Dict[str, Any] = {"kvClientMessage": {"id": kv_id}}
    if "getBlobArgs" in kvm:
        blob_key = (kvm.get("getBlobArgs") or {}).get("blobId", "")
        blob_data = ctx.blob_store.get(blob_key)
        kv_resp["kvClientMessage"]["getBlobResult"] = (
            {"blobData": blob_data} if blob_data else {}
        )
    elif "setBlobArgs" in kvm:
        sb = kvm.get("setBlobArgs") or {}
        ctx.blob_store[sb.get("blobId", "")] = sb.get("blobData", "")
        kv_resp["kvClientMessage"]["setBlobResult"] = {}
    ctx.send_frame(kv_resp)
    ctx.touch()


def _handle_exec_server_message(msg: Dict[str, Any], ctx: AgentRunContext, elapsed: float) -> None:
    exec_msg = msg["execServerMessage"]
    mcp = exec_msg.get("mcpArgs") or {}
    if mcp:
        name = mcp.get("toolName") or mcp.get("name") or ""
        tc_id = str(exec_msg.get("execId") or exec_msg.get("id") or uuid.uuid4())
        _emit_tool_call(ctx, {
            "id": tc_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(mcp.get("args") or {}, ensure_ascii=False),
            },
        }, elapsed)
    exec_id = exec_msg.get("id", 0)
    try:
        results = execute_tool(exec_msg, ctx.tool_handlers)
        for result in results:
            ctx.send_frame({"execClientMessage": result})
        if exec_id:
            ctx.send_frame({"execClientControlMessage": {"streamClose": {"id": exec_id}}})
        if results:
            result_text = extract_tool_result_text(results[0])
            if result_text:
                ctx.q.put(StreamEvent(type="tool_result", tool_result=result_text, elapsed=elapsed))
    except Exception as exc:
        if exec_id:
            ctx.send_frame({"execClientControlMessage": {"throw": {"id": exec_id, "error": str(exc)}}})
    ctx.touch()
    ctx.heartbeat_count = 0


def process_agent_message(msg: Dict[str, Any], ctx: AgentRunContext) -> bool:
    """处理单帧 Agent JSON；返回 True 表示应结束流。"""
    elapsed = time.time() - ctx.start

    if "error" in msg:
        ctx.q.put(StreamEvent(type="error", error=parse_error(msg), elapsed=elapsed))
        return True

    escm = msg.get("execServerControlMessage")
    if escm is not None:
        if "abort" in escm:
            ctx.q.put(StreamEvent(type="error", error="Server abort", elapsed=elapsed))
            return True
        ctx.touch()
        return False

    if "interactionQuery" in msg:
        ctx.send_frame(build_interaction_reply(msg["interactionQuery"]))
        ctx.touch()
        return False

    if "execServerMessage" in msg:
        _handle_exec_server_message(msg, ctx, elapsed)
        return False

    if "kvServerMessage" in msg:
        _handle_kv_server_message(msg, ctx)
        return False

    if "conversationCheckpointUpdate" in msg:
        ctx.touch()
        return False

    iu = msg.get("interactionUpdate") or {}
    if iu:
        return _handle_interaction_update(iu, ctx, elapsed)
    return False
