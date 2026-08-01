from __future__ import annotations

import queue
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

from upstream.cursor.stream.exec import execute_tool, extract_tool_result_text
from upstream.cursor.stream.exec.builtin_tools import (
    mcp_args_to_json,
    openai_tool_from_agent_tool_call,
)
from upstream.cursor.stream.proto import StreamEvent
from upstream.cursor.stream.proto import (
    build_interaction_reply,
    parse_error,
    text_delta,
)
from upstream.cursor.chat.tool_ids import normalize_tool_call_id

# 测试兼容别名
_openai_tool_from_agent_tool_call = openai_tool_from_agent_tool_call
_mcp_args_to_json = mcp_args_to_json


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
    # OpenAI 代理：MCP 由客户端执行，同流挂起等真实 mcpResult
    defer_mcp: bool = False
    emitted_tool_call_ids: Set[str] = field(default_factory=set)
    deferred_mcp_count: int = 0
    last_deferred_mcp_at: float = 0.0
    session_id: str = ""
    pending_mcp: Dict[str, Any] = field(default_factory=dict)
    should_park: bool = False
    last_checkpoint: Dict[str, Any] = field(default_factory=dict)
    workspace: str = ""


def _emit_tool_call(ctx: AgentRunContext, tc: Dict[str, Any], elapsed: float) -> None:
    tc_id = str((tc or {}).get("id") or "").strip()
    if tc_id:
        if tc_id in ctx.emitted_tool_call_ids:
            return
        ctx.emitted_tool_call_ids.add(tc_id)
    ctx.q.put(StreamEvent(type="tool_call", tool_call=tc, elapsed=elapsed))
    ctx.touch()
    ctx.heartbeat_count = 0


def _maybe_request_park(ctx: AgentRunContext) -> bool:
    """本批 MCP 已委托客户端：请求 loop 挂起，不关流、不发空 mcpResult。"""
    if not (ctx.defer_mcp and ctx.pending_mcp):
        return False
    since_defer = time.time() - (ctx.last_deferred_mcp_at or ctx.last_activity)
    idle = time.time() - ctx.last_activity
    if since_defer < 1.2 or idle < 0.8:
        return False
    ctx.should_park = True
    return True


def _handle_heartbeat(ctx: AgentRunContext, elapsed: float) -> bool:
    ctx.heartbeat_count += 1
    idle = time.time() - ctx.last_activity
    if _maybe_request_park(ctx):
        return False
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
        if ctx.defer_mcp and ctx.pending_mcp:
            return False
        ctx.finish(elapsed)
        return True
    return False


def _emit_text_delta(ctx: AgentRunContext, iu: Dict[str, Any], elapsed: float, *, nested: bool = False) -> bool:
    if ctx.defer_mcp and ctx.pending_mcp:
        _maybe_request_park(ctx)
        return False
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
        extracted = openai_tool_from_agent_tool_call(tc, "")
        name = (extracted or {}).get("function", {}).get("name") or tc.get("toolName") or ""
        ctx.q.put(StreamEvent(type="tool_partial", tool_name=name, elapsed=elapsed))
        ctx.touch()
        return True
    if "toolCallStarted" in iu:
        tcs = iu.get("toolCallStarted") or {}
        tc = tcs.get("toolCall") or {}
        tc_id = str(tcs.get("callId") or tc.get("toolCallId") or uuid.uuid4())
        extracted = openai_tool_from_agent_tool_call(tc, tc_id)
        if extracted:
            ctx.q.put(StreamEvent(
                type="tool_started",
                tool_name=extracted["function"]["name"],
                tool_call_id=extracted["id"],
                elapsed=elapsed,
            ))
            if not ctx.defer_mcp:
                _emit_tool_call(ctx, extracted, elapsed)
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
        if ctx.defer_mcp and ctx.pending_mcp:
            _maybe_request_park(ctx)
            return False
        t = text_delta(iu.get("thinkingDelta"))
        if t:
            ctx.q.put(StreamEvent(type="thinking", text=t, elapsed=elapsed))
            ctx.touch()
        return False
    if "thinkingCompleted" in iu:
        if ctx.defer_mcp and ctx.pending_mcp:
            _maybe_request_park(ctx)
            return False
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
    if _emit_text_delta(ctx, iu, elapsed, nested=True):
        return True
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
        if blob_data is None and getattr(ctx, "session_id", ""):
            from upstream.cursor.chat import session_db
            blob_data = session_db.get_blob(ctx.session_id, blob_key, session_db.cache_gen()) or None
            if blob_data:
                ctx.blob_store[blob_key] = blob_data
        kv_resp["kvClientMessage"]["getBlobResult"] = (
            {"blobData": blob_data} if blob_data else {}
        )
    elif "setBlobArgs" in kvm:
        sb = kvm.get("setBlobArgs") or {}
        blob_id = sb.get("blobId", "")
        blob_data = sb.get("blobData", "")
        ctx.blob_store[blob_id] = blob_data
        try:
            from upstream.cursor.chat.agent_session import note_blob
            note_blob(ctx, blob_id, blob_data)
        except Exception:
            pass
        kv_resp["kvClientMessage"]["setBlobResult"] = {}
    ctx.send_frame(kv_resp)
    ctx.touch()


def _emit_mcp_tool_call(ctx: AgentRunContext, mcp: Dict[str, Any], exec_msg: Dict[str, Any], elapsed: float) -> str:
    from upstream.cursor.chat.convert import strip_mcp_prefix

    provider = (mcp.get("providerIdentifier") or "").strip()
    short = (mcp.get("toolName") or "").strip()
    qualified = (mcp.get("name") or "").strip()
    if not qualified and provider and short:
        qualified = f"mcp__{provider}__{short}"
    cursor_name = qualified or short or provider
    openai_name = strip_mcp_prefix(cursor_name) if cursor_name.startswith("mcp__") else cursor_name
    tc_id = normalize_tool_call_id(
        mcp.get("toolCallId") or exec_msg.get("execId") or exec_msg.get("id") or uuid.uuid4()
    ) or str(uuid.uuid4())
    _emit_tool_call(ctx, {
        "id": tc_id,
        "type": "function",
        "function": {"name": openai_name, "arguments": mcp_args_to_json(mcp.get("args") or {})},
    }, elapsed)
    return tc_id


def _defer_mcp_pending(ctx: AgentRunContext, exec_msg: Dict[str, Any], tc_id: str) -> None:
    from upstream.cursor.stream.exec.common import base_msg as exec_base_msg
    from upstream.cursor.chat.agent_session import note_pending_mcp

    note_pending_mcp(
        ctx,
        tool_call_id=tc_id,
        base_msg=exec_base_msg(exec_msg),
        exec_id=int(exec_msg.get("id") or 0),
        result_field="mcpResult",
    )
    ctx.deferred_mcp_count += 1
    ctx.last_deferred_mcp_at = time.time()
    ctx.touch()
    ctx.heartbeat_count = 0


def _exec_local_tools(ctx: AgentRunContext, exec_msg: Dict[str, Any], elapsed: float) -> None:
    exec_id = exec_msg.get("id", 0)
    try:
        results = execute_tool(exec_msg, ctx.tool_handlers, defer_mcp=False)
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


def _handle_exec_server_message(msg: Dict[str, Any], ctx: AgentRunContext, elapsed: float) -> bool:
    """处理 execServerMessage。返回 True 表示应结束 Agent 流。"""
    exec_msg = msg["execServerMessage"]
    mcp = exec_msg.get("mcpArgs") or {}
    tc_id = ""
    if mcp:
        tc_id = _emit_mcp_tool_call(ctx, mcp, exec_msg, elapsed)
    if mcp and ctx.defer_mcp:
        _defer_mcp_pending(ctx, exec_msg, tc_id)
        return False
    _exec_local_tools(ctx, exec_msg, elapsed)
    return False


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
        return _handle_exec_server_message(msg, ctx, elapsed)
    if "kvServerMessage" in msg:
        _handle_kv_server_message(msg, ctx)
        return False
    if "conversationCheckpointUpdate" in msg:
        state = msg.get("conversationCheckpointUpdate")
        if isinstance(state, dict):
            try:
                from upstream.cursor.chat.agent_session import note_checkpoint
                note_checkpoint(ctx, state)
            except Exception:
                ctx.last_checkpoint = state
            ctx.q.put(StreamEvent(
                type="checkpoint", data=state,
                conversation_id=ctx.conv_id, elapsed=elapsed,
            ))
        ctx.touch()
        return False
    iu = msg.get("interactionUpdate") or {}
    if iu:
        return _handle_interaction_update(iu, ctx, elapsed)
    return False
