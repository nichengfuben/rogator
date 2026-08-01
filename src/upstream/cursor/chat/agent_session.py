from __future__ import annotations

"""内存 ParkedRun + session_db 协作（活 H2 仅内存，元数据实时落库）。"""

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from echotools.logger import get_logger

from upstream.cursor.chat import session_db
from upstream.cursor.chat.tool_ids import (
    expand_tool_call_ids,
    normalize_tool_call_id,
    tool_call_id_aliases,
)
from upstream.cursor.stream.handlers import AgentRunContext
from upstream.cursor.stream.proto import StreamEvent

logger = get_logger("rogator")

PARK_TTL_SEC = 900.0


def messages_have_prior_turns(messages: List[Dict[str, Any]]) -> bool:
    return any((m.get("role") or "") in ("assistant", "tool") for m in messages or [])


def _index_tool(tid: str, session_id: str) -> None:
    for alias in tool_call_id_aliases(tid):
        _by_tool[alias] = session_id


def _unindex_tool(tid: str) -> None:
    for alias in tool_call_id_aliases(tid):
        _by_tool.pop(alias, None)


@dataclass
class PendingMcp:
    tool_call_id: str
    base_msg: Dict[str, Any]
    exec_id: int = 0
    result_field: str = "mcpResult"


@dataclass
class ParkedRun:
    session_id: str
    workspace: str
    ctx: AgentRunContext
    sock: Any
    conn: Any
    stream_id: int
    sock_lock: threading.Lock
    event_q: queue.Queue
    resume_event: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)
    pending: Dict[str, PendingMcp] = field(default_factory=dict)
    parked_at: float = field(default_factory=time.time)
    last_checkpoint: Dict[str, Any] = field(default_factory=dict)
    hb_stop: Optional[threading.Event] = None


_parked: Dict[str, ParkedRun] = {}
_by_tool: Dict[str, str] = {}
_lock = threading.RLock()


def register_running(
    *,
    session_id: str,
    conversation_id: str,
    workspace: str,
    req_id: str = "",
    prompt_head: str = "",
    history_len: int = 0,
) -> None:
    session_db.upsert_session(
        session_id=session_id,
        conversation_id=conversation_id,
        workspace=workspace or "",
        status="running",
        ttl_sec=PARK_TTL_SEC,
    )
    session_db.log_request(
        kind="start",
        req_id=req_id,
        session_id=session_id,
        prompt_head=prompt_head,
        history_len=history_len,
    )


def note_pending_mcp(
    ctx: AgentRunContext,
    *,
    tool_call_id: str,
    base_msg: Dict[str, Any],
    exec_id: int = 0,
    result_field: str = "mcpResult",
) -> None:
    sid = getattr(ctx, "session_id", "") or ""
    tid = normalize_tool_call_id(tool_call_id)
    if not sid or not tid:
        return
    pending = getattr(ctx, "pending_mcp", None)
    if pending is None:
        ctx.pending_mcp = {}
        pending = ctx.pending_mcp
    pending[tid] = PendingMcp(
        tool_call_id=tid,
        base_msg=dict(base_msg or {}),
        exec_id=int(exec_id or 0),
        result_field=str(result_field or "mcpResult"),
    )
    session_db.upsert_pending_tool(
        tool_call_id=tid,
        session_id=sid,
        base_msg=dict(base_msg or {}),
        exec_id=int(exec_id or 0),
    )
    with _lock:
        _index_tool(tid, sid)
        # 原文若含换行，仍登记别名以便客户端回传脏 id
        if str(tool_call_id or "").strip() != tid:
            _index_tool(str(tool_call_id), sid)


def note_checkpoint(ctx: AgentRunContext, state: Dict[str, Any]) -> None:
    sid = getattr(ctx, "session_id", "") or ""
    if not sid:
        return
    ctx.last_checkpoint = dict(state or {})
    session_db.update_checkpoint(sid, dict(state or {}), ttl_sec=PARK_TTL_SEC)


def note_blob(ctx: AgentRunContext, blob_id: str, blob_data: str) -> None:
    sid = getattr(ctx, "session_id", "") or ""
    if not sid or not blob_id:
        return
    ctx.blob_store[blob_id] = blob_data or ""
    session_db.upsert_blob(sid, blob_id, blob_data or "")


def park_run(run: ParkedRun) -> None:
    with _lock:
        _parked[run.session_id] = run
        for tid in list(run.pending.keys()):
            _index_tool(tid, run.session_id)
        # also index from ctx.pending_mcp
        for tid in getattr(run.ctx, "pending_mcp", {}) or {}:
            canon = normalize_tool_call_id(tid) or tid
            _index_tool(tid, run.session_id)
            _index_tool(canon, run.session_id)
            if canon not in run.pending:
                pm = run.ctx.pending_mcp[tid]
                if isinstance(pm, PendingMcp):
                    run.pending[canon] = PendingMcp(
                        tool_call_id=canon,
                        base_msg=pm.base_msg,
                        exec_id=pm.exec_id,
                        result_field=pm.result_field,
                    )
                elif isinstance(pm, dict):
                    run.pending[canon] = PendingMcp(
                        tool_call_id=canon,
                        base_msg=dict(pm.get("base_msg") or pm),
                        exec_id=int(pm.get("exec_id") or 0),
                        result_field=str(pm.get("result_field") or "mcpResult"),
                    )
    session_db.set_session_status(run.session_id, "parked", ttl_sec=PARK_TTL_SEC)
    session_db.log_request(
        kind="park",
        session_id=run.session_id,
        payload={"tool_call_ids": list(run.pending.keys())},
    )
    logger.info(
        "cursor park session=%s tools=%d conv=%s",
        run.session_id[:8],
        len(run.pending),
        (run.ctx.conv_id or "")[:8],
    )


def find_parked_by_tool_ids(tool_call_ids: List[str]) -> Optional[ParkedRun]:
    session_db.purge_expired()
    aliases = expand_tool_call_ids(tool_call_ids)
    with _lock:
        for tid in aliases:
            sid = _by_tool.get(tid) or session_db.get_session_id_by_tool_call(
                tid, session_db.cache_gen()
            )
            if not sid:
                # DB 可能只存了规范化 id
                sid = session_db.get_session_id_by_tool_call(
                    normalize_tool_call_id(tid), session_db.cache_gen()
                )
            if sid and sid in _parked:
                run = _parked[sid]
                if time.time() - run.parked_at > PARK_TTL_SEC:
                    _drop_locked(sid, reason="ttl")
                    continue
                return run
    return None


def find_parked_for_workspace(workspace: str) -> Optional[ParkedRun]:
    """精确 id 未命中时：同 workspace 最近一条仍有 pending 的 ParkedRun。"""
    session_db.purge_expired()
    ws = workspace or ""
    with _lock:
        candidates: List[ParkedRun] = []
        for run in _parked.values():
            if (run.workspace or "") != ws:
                continue
            if not run.pending:
                continue
            if time.time() - run.parked_at > PARK_TTL_SEC:
                continue
            candidates.append(run)
        if not candidates:
            return None
        candidates.sort(key=lambda r: r.parked_at, reverse=True)
        return candidates[0]


def find_completed_for_workspace(workspace: str) -> Optional[Dict[str, Any]]:
    session_db.purge_expired()
    sid = session_db.get_latest_session_for_workspace(workspace or "", session_db.cache_gen())
    if not sid:
        return None
    meta = session_db.session_dict(sid)
    if not meta or meta.get("status") not in ("done", "parked", "running"):
        return None
    meta["blobs"] = session_db.blobs_dict(sid)
    return meta


def _drop_locked(session_id: str, *, reason: str) -> None:
    run = _parked.pop(session_id, None)
    if run:
        for tid in list(run.pending.keys()):
            _unindex_tool(tid)
        for tid, mapped in list(_by_tool.items()):
            if mapped == session_id:
                _by_tool.pop(tid, None)
        run.stop_event.set()
        run.resume_event.set()
        try:
            if run.hb_stop:
                run.hb_stop.set()
            run.sock.close()
        except Exception:
            pass
    session_db.set_session_status(session_id, "expired", ttl_sec=1)
    logger.info("cursor drop park session=%s reason=%s", session_id[:8], reason)


def drop_parked(session_id: str, *, reason: str = "manual") -> None:
    with _lock:
        _drop_locked(session_id, reason=reason)


def mcp_result_payload(text: str, *, is_error: bool = False) -> Dict[str, Any]:
    from upstream.cursor.chat.resume import mcp_result_payload as _payload
    return _payload(text, is_error=is_error)


def resume_with_tool_results(
    run: ParkedRun,
    tool_messages: List[Dict[str, Any]],
    *,
    req_id: str = "",
) -> bool:
    from upstream.cursor.chat.resume import resume_with_tool_results as _resume
    return _resume(run, tool_messages, req_id=req_id)


def complete_run(run: ParkedRun, *, req_id: str = "") -> None:
    sid = run.session_id
    checkpoint = getattr(run.ctx, "last_checkpoint", None) or run.last_checkpoint or {}
    # persist blobs
    for bid, data in (run.ctx.blob_store or {}).items():
        session_db.upsert_blob(sid, bid, data or "")
    session_db.upsert_session(
        session_id=sid,
        conversation_id=run.ctx.conv_id or "",
        workspace=run.workspace or "",
        status="done",
        checkpoint=checkpoint if isinstance(checkpoint, dict) else {},
        ttl_sec=PARK_TTL_SEC,
    )
    session_db.clear_pending_tools(sid)
    session_db.log_request(kind="done", req_id=req_id, session_id=sid)
    with _lock:
        _parked.pop(sid, None)
        for tid, mapped in list(_by_tool.items()):
            if mapped == sid:
                _by_tool.pop(tid, None)


def trailing_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in reversed(messages or []):
        if (msg.get("role") or "") != "tool":
            break
        out.append(msg)
    out.reverse()
    return out
