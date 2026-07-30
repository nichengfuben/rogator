from __future__ import annotations

import json
import queue
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

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
    # OpenAI 代理：MCP 由客户端执行，本地勿回 Unknown tool
    defer_mcp: bool = False
    emitted_tool_call_ids: Set[str] = field(default_factory=set)
    deferred_mcp_count: int = 0
    last_deferred_mcp_at: float = 0.0


def _emit_tool_call(ctx: AgentRunContext, tc: Dict[str, Any], elapsed: float) -> None:
    tc_id = str((tc or {}).get("id") or "").strip()
    if tc_id:
        if tc_id in ctx.emitted_tool_call_ids:
            return
        ctx.emitted_tool_call_ids.add(tc_id)
    ctx.q.put(StreamEvent(type="tool_call", tool_call=tc, elapsed=elapsed))
    ctx.touch()
    ctx.heartbeat_count = 0


def _finish_after_deferred_mcp(ctx: AgentRunContext, elapsed: float) -> bool:
    """已把 MCP 交给 OpenAI 客户端后结束流，避免空回执上继续生成。"""
    if not (ctx.defer_mcp and ctx.deferred_mcp_count > 0):
        return False
    ctx.finish(elapsed)
    return True


def _handle_heartbeat(ctx: AgentRunContext, elapsed: float) -> bool:
    ctx.heartbeat_count += 1
    idle = time.time() - ctx.last_activity
    # 并行工具：等一小段空闲，确认本批 mcpArgs 到齐后再结束
    if ctx.defer_mcp and ctx.deferred_mcp_count > 0:
        since_defer = time.time() - (ctx.last_deferred_mcp_at or ctx.last_activity)
        if since_defer >= 1.2 and idle >= 0.8:
            return _finish_after_deferred_mcp(ctx, elapsed)
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
    if _finish_after_deferred_mcp(ctx, elapsed):
        return True
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


def _openai_tool_from_agent_tool_call(tc: Dict[str, Any], fallback_id: str) -> Optional[Dict[str, Any]]:
    """从 interactionUpdate.toolCall（ToolCall oneof）提取 OpenAI function tool_call。"""
    if not tc:
        return None
    # ConversationHistory 风格（少见，但兼容）
    if tc.get("toolName") or tc.get("argsJson"):
        name = str(tc.get("toolName") or "").strip()
        if not name:
            return None
        return {
            "id": str(tc.get("toolCallId") or fallback_id),
            "type": "function",
            "function": {"name": name, "arguments": tc.get("argsJson") or "{}"},
        }
    mcp = tc.get("mcpToolCall") or {}
    args = mcp.get("args") or mcp
    if isinstance(args, dict) and (
        args.get("name") or args.get("toolName") or args.get("providerIdentifier")
    ):
        provider = str(args.get("providerIdentifier") or "").strip()
        short = str(args.get("toolName") or "").strip()
        qualified = str(args.get("name") or "").strip()
        if not qualified and provider and short:
            qualified = f"mcp__{provider}__{short}"
        name = qualified or short or provider
        if not name:
            return None
        raw_args = args.get("args") or {}
        if isinstance(raw_args, dict):
            args_json = _mcp_args_to_json(raw_args)
        else:
            args_json = "{}"
        return {
            "id": str(args.get("toolCallId") or fallback_id),
            "type": "function",
            "function": {"name": name, "arguments": args_json},
        }
    return None


def _handle_iu_tool_events(iu: Dict[str, Any], ctx: AgentRunContext, elapsed: float) -> bool:
    if "partialToolCall" in iu:
        ptc = iu.get("partialToolCall") or {}
        tc = ptc.get("toolCall") or {}
        extracted = _openai_tool_from_agent_tool_call(tc, "")
        name = (extracted or {}).get("function", {}).get("name") or tc.get("toolName") or ""
        ctx.q.put(StreamEvent(type="tool_partial", tool_name=name, elapsed=elapsed))
        ctx.touch()
        return True
    if "toolCallStarted" in iu:
        tcs = iu.get("toolCallStarted") or {}
        tc = tcs.get("toolCall") or {}
        tc_id = str(tcs.get("callId") or tc.get("toolCallId") or uuid.uuid4())
        extracted = _openai_tool_from_agent_tool_call(tc, tc_id)
        if extracted:
            ctx.q.put(StreamEvent(
                type="tool_started",
                tool_name=extracted["function"]["name"],
                tool_call_id=extracted["id"],
                elapsed=elapsed,
            ))
            # defer_mcp：tool_call 改由 mcpArgs/exec 发出，避免与 exec 重复且便于并行攒批
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
        if _finish_after_deferred_mcp(ctx, elapsed):
            return True
        t = text_delta(iu.get("thinkingDelta"))
        if t:
            ctx.q.put(StreamEvent(type="thinking", text=t, elapsed=elapsed))
            ctx.touch()
        return False
    if "thinkingCompleted" in iu:
        if _finish_after_deferred_mcp(ctx, elapsed):
            return True
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
        kv_resp["kvClientMessage"]["getBlobResult"] = (
            {"blobData": blob_data} if blob_data else {}
        )
    elif "setBlobArgs" in kvm:
        sb = kvm.get("setBlobArgs") or {}
        ctx.blob_store[sb.get("blobId", "")] = sb.get("blobData", "")
        kv_resp["kvClientMessage"]["setBlobResult"] = {}
    ctx.send_frame(kv_resp)
    ctx.touch()


def _unwrap_proto_value(value: Any) -> Any:
    """把 google.protobuf.Value / Struct 的 JSON 形态还原成普通 Python 值。"""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value and len(value) == 1:
        return value["stringValue"]
    if "numberValue" in value and len(value) == 1:
        return value["numberValue"]
    if "boolValue" in value and len(value) == 1:
        return value["boolValue"]
    if "nullValue" in value and len(value) == 1:
        return None
    if "structValue" in value:
        return _unwrap_proto_struct(value["structValue"])
    if "listValue" in value:
        items = (value["listValue"] or {}).get("values") or []
        return [_unwrap_proto_value(v) for v in items]
    if "fields" in value and all(isinstance(k, str) for k in value.keys()):
        # 可能是裸 Struct
        if set(value.keys()) <= {"fields"} or (
            "fields" in value and not any(k.endswith("Value") for k in value)
        ):
            return _unwrap_proto_struct(value)
    # 已是普通 JSON object
    return {k: _unwrap_proto_value(v) for k, v in value.items()}


def _unwrap_proto_struct(struct: Any) -> Any:
    if not isinstance(struct, dict):
        return struct
    fields = struct.get("fields")
    if isinstance(fields, dict):
        return {k: _unwrap_proto_value(v) for k, v in fields.items()}
    return {k: _unwrap_proto_value(v) for k, v in struct.items()}


def _mcp_args_to_json(args_obj: Any) -> str:
    if not isinstance(args_obj, dict):
        return "{}"
    plain = {k: _unwrap_proto_value(v) for k, v in args_obj.items()}
    return json.dumps(plain, ensure_ascii=False)


def _handle_exec_server_message(msg: Dict[str, Any], ctx: AgentRunContext, elapsed: float) -> bool:
    """处理 execServerMessage。返回 True 表示应结束 Agent 流。"""
    exec_msg = msg["execServerMessage"]
    mcp = exec_msg.get("mcpArgs") or {}
    if mcp:
        # McpArgs.name 多为合格名 mcp__provider__tool；toolName 仅为短名
        provider = (mcp.get("providerIdentifier") or "").strip()
        short = (mcp.get("toolName") or "").strip()
        qualified = (mcp.get("name") or "").strip()
        if not qualified and provider and short:
            qualified = f"mcp__{provider}__{short}"
        name = qualified or short or provider
        tc_id = (
            str(mcp.get("toolCallId") or "").strip()
            or str(exec_msg.get("execId") or exec_msg.get("id") or uuid.uuid4())
        )
        _emit_tool_call(ctx, {
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": _mcp_args_to_json(mcp.get("args") or {})},
        }, elapsed)
    exec_id = exec_msg.get("id", 0)
    try:
        results = execute_tool(exec_msg, ctx.tool_handlers, defer_mcp=ctx.defer_mcp)
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
    # OpenAI 代理：记录已委托的 MCP，等本批并行工具到齐（心跳空闲 / 后续正文）再结束
    if mcp and ctx.defer_mcp:
        ctx.deferred_mcp_count += 1
        ctx.last_deferred_mcp_at = time.time()
        return False
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
        ctx.touch()
        return False

    iu = msg.get("interactionUpdate") or {}
    if iu:
        return _handle_interaction_update(iu, ctx, elapsed)
    return False
