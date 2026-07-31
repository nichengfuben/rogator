from __future__ import annotations

"""ParkedRun 工具结果回灌（仅 mcpResult）。"""

import time
from typing import Any, Dict, List, Optional, Tuple

from echotools.logger import get_logger

from upstream.cursor.chat import session_db
from upstream.cursor.chat.tool_ids import normalize_tool_call_id, tool_call_id_aliases
from upstream.cursor.stream.exec.common import finish

logger = get_logger("rogator")

# 由 agent_session 注入，避免循环导入细节散落
PARK_TTL_SEC = 900.0


def tool_msg_text(msg: Dict[str, Any]) -> str:
    body = msg.get("content")
    if isinstance(body, list):
        parts = []
        for p in body:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(body or "")


def mcp_result_payload(text: str, *, is_error: bool = False) -> Dict[str, Any]:
    return {
        "success": {
            "content": [{"text": {"text": text or ""}}],
            "isError": bool(is_error),
        },
    }


def clear_pending_aliases(run, raw_tid: str, match_key: str, *, lock, unindex) -> None:
    for alias in tool_call_id_aliases(raw_tid) + tool_call_id_aliases(match_key):
        run.pending.pop(alias, None)
        if getattr(run.ctx, "pending_mcp", None) is not None:
            run.ctx.pending_mcp.pop(alias, None)
        session_db.delete_pending_tool(alias)
        with lock:
            unindex(alias)


def lookup_pending(run, raw_tid: str, tid: str) -> Tuple[Any, Dict[str, Any], int, str, str]:
    """返回 (pending, base, exec_id, result_field, match_key)。"""
    pending = None
    match_key = None
    for alias in tool_call_id_aliases(raw_tid):
        pending = run.pending.get(alias) or getattr(run.ctx, "pending_mcp", {}).get(alias)
        if pending is not None:
            match_key = alias
            break
    result_field = "mcpResult"
    from upstream.cursor.chat.agent_session import PendingMcp

    if isinstance(pending, PendingMcp):
        return pending, dict(pending.base_msg), pending.exec_id, pending.result_field or "mcpResult", match_key or pending.tool_call_id or tid
    if isinstance(pending, dict):
        return (
            pending,
            dict(pending.get("base_msg") or {"id": 0}),
            int(pending.get("exec_id") or 0),
            str(pending.get("result_field") or "mcpResult"),
            match_key or tid,
        )
    base = {"id": 0}
    exec_id = 0
    match_key = tid
    alias_set = set(tool_call_id_aliases(raw_tid))
    for item in session_db.pending_tools_dicts(run.session_id):
        item_tid = str(item.get("tool_call_id") or "")
        if item_tid in alias_set or normalize_tool_call_id(item_tid) in alias_set:
            base = dict(item.get("base_msg") or {"id": 0})
            exec_id = int(item.get("exec_id") or 0)
            result_field = str(item.get("result_field") or "mcpResult")
            match_key = item_tid
            break
    return None, base, exec_id, result_field, match_key


def inject_one_mcp(run, msg: Dict[str, Any], *, lock, unindex) -> Optional[bool]:
    """注入单条 tool 结果。返回 True/False 表示是否注入；None 表示跳过。"""
    raw_tid = str(msg.get("tool_call_id") or "").strip()
    if not raw_tid:
        return None
    tid = normalize_tool_call_id(raw_tid) or raw_tid
    text = tool_msg_text(msg)
    _pending, base, exec_id, result_field, match_key = lookup_pending(run, raw_tid, tid)
    is_error = bool(msg.get("is_error") or msg.get("isError"))
    if result_field != "mcpResult":
        clear_pending_aliases(run, raw_tid, match_key or tid, lock=lock, unindex=unindex)
        return None
    frame = finish(base, time.time(), "mcpResult", mcp_result_payload(text, is_error=is_error))
    try:
        run.ctx.send_frame({"execClientMessage": frame})
        if exec_id:
            run.ctx.send_frame({"execClientControlMessage": {"streamClose": {"id": exec_id}}})
    except Exception as exc:
        logger.error("cursor resume_exec send failed: %s", exc)
        return False
    clear_pending_aliases(run, raw_tid, match_key or tid, lock=lock, unindex=unindex)
    return True


def resume_with_tool_results(run, tool_messages: List[Dict[str, Any]], *, req_id: str = "") -> bool:
    """把 OpenAI tool 消息回灌为 mcpResult，并唤醒 loop（仅 MCP）。"""
    if not tool_messages:
        return False
    from upstream.cursor.chat import agent_session as ag

    injected = 0
    for msg in tool_messages:
        result = inject_one_mcp(run, msg, lock=ag._lock, unindex=ag._unindex_tool)
        if result is False:
            return False
        if result is True:
            injected += 1
    run.ctx.deferred_mcp_count = 0
    run.ctx.last_deferred_mcp_at = 0.0
    run.ctx.should_park = False
    run.ctx.touch()
    session_db.set_session_status(run.session_id, "running", ttl_sec=ag.PARK_TTL_SEC)
    session_db.log_request(
        kind="resume_exec",
        req_id=req_id,
        session_id=run.session_id,
        payload={"injected": injected, "tool_ids": [str(m.get("tool_call_id") or "") for m in tool_messages]},
    )
    logger.info(
        "cursor resume_exec session=%s injected=%d req=%s",
        run.session_id[:8], injected, req_id,
    )
    run.resume_event.set()
    return injected > 0
