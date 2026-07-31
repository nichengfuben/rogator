from __future__ import annotations

"""conversationHistory / Prior conversation / 本轮拆分。"""

from typing import Any, Dict, List, Optional, Set, Tuple

from upstream.cursor.chat.tool_ids import normalize_tool_call_id, tool_call_id_aliases
from upstream.cursor.chat.convert.text import (
    _is_meta_user_text,
    _last_real_user_text,
    _message_text,
    _user_text,
    messages_to_prompt,
)
from upstream.cursor.chat.convert.tools import restore_mcp_prefix_for_cursor


def _tool_name_index(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """tool_call_id → toolName（OpenAI tool 消息常缺 name）。"""
    out: Dict[str, str] = {}
    for msg in messages or []:
        if (msg.get("role") or "") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            tid = str(tc.get("id") or "").strip()
            if not tid:
                continue
            fn = tc.get("function") or {}
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            for alias in tool_call_id_aliases(tid):
                out[alias] = name
            canon = normalize_tool_call_id(tid)
            if canon:
                out[canon] = name
    return out


def _assistant_history_blocks(
    msg: Dict[str, Any],
    text: str,
    tool_originals: Optional[Set[str]],
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if text:
        blocks.append({"text": {"text": text}})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_name = str(fn.get("name") or "")
        blocks.append({
            "toolCall": {
                "toolCallId": normalize_tool_call_id(tc.get("id") or "") or (tc.get("id") or ""),
                "toolName": restore_mcp_prefix_for_cursor(raw_name, tool_originals),
                "argsJson": fn.get("arguments") or "{}",
            },
        })
    return blocks


def messages_to_cursor_history(
    messages: List[Dict[str, Any]],
    *,
    tool_originals: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """对齐 agent.v1.ConversationHistory（user / assistant / tool）。"""
    history: List[Dict[str, Any]] = []
    name_by_id = _tool_name_index(messages)
    for msg in messages:
        role = msg.get("role") or "user"
        text = _message_text(msg)
        if role == "system":
            continue
        if role == "user":
            if text.strip():
                history.append({"user": {"content": [{"text": {"text": text}}]}})
        elif role == "assistant":
            blocks = _assistant_history_blocks(msg, text, tool_originals)
            if blocks:
                history.append({"assistant": {"content": blocks}})
        elif role == "tool":
            tid = str(msg.get("tool_call_id") or "")
            tname = str(msg.get("name") or "").strip() or name_by_id.get(tid, "")
            history.append({
                "tool": {
                    "toolCallId": normalize_tool_call_id(tid) or tid,
                    "toolName": restore_mcp_prefix_for_cursor(tname, tool_originals),
                    "content": [{"text": {"text": text}}],
                    "isError": bool(msg.get("is_error") or msg.get("isError")),
                },
            })
    return history


def _trailing_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """末尾连续 tool 消息（OpenAI 工具回灌尾包）。"""
    out: List[Dict[str, Any]] = []
    for msg in reversed(messages or []):
        if (msg.get("role") or "") != "tool":
            break
        out.append(msg)
    out.reverse()
    return out


def format_tool_results_user_text(
    messages: List[Dict[str, Any]],
    *,
    tool_originals: Optional[Set[str]] = None,
) -> str:
    """跨请求工具续轮：把 tool 回执写进 UserMessage.text。"""
    trailing = _trailing_tool_messages(messages)
    if not trailing:
        return ""
    name_by_id = _tool_name_index(messages)
    blocks: List[str] = []
    for msg in trailing:
        tid = str(msg.get("tool_call_id") or "")
        raw = str(msg.get("name") or "").strip() or name_by_id.get(tid, "tool")
        name = restore_mcp_prefix_for_cursor(raw, tool_originals)
        body = _message_text(msg)
        blocks.append(f"Tool result for {name}:\n{body}")
    return "\n\n".join(blocks)


def _prior_block_for_msg(
    msg: Dict[str, Any],
    name_by_id: Dict[str, str],
    max_chars: int,
    tool_originals: Optional[Set[str]],
) -> List[str]:
    role = msg.get("role") or ""
    text = _message_text(msg).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    blocks: List[str] = []
    if role == "user":
        if text and not _is_meta_user_text(text):
            blocks.append(f"User: {text}")
    elif role == "assistant":
        if text:
            blocks.append(f"Assistant: {text}")
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = str(fn.get("arguments") or "{}")
            if len(args) > max_chars:
                args = args[:max_chars] + "…"
            raw_name = str(fn.get("name") or "tool")
            name = restore_mcp_prefix_for_cursor(raw_name, tool_originals)
            blocks.append(f"Assistant called {name}({args})")
    elif role == "tool":
        tid = str(msg.get("tool_call_id") or "")
        raw = str(msg.get("name") or "").strip() or name_by_id.get(tid, "tool")
        name = restore_mcp_prefix_for_cursor(raw, tool_originals)
        blocks.append(f"Tool result for {name}:\n{text}")
    return blocks


def format_prior_context_user_text(
    messages: List[Dict[str, Any]],
    *,
    limit: int = 10,
    max_chars: int = 400,
    tool_originals: Optional[Set[str]] = None,
) -> str:
    """跨请求多轮：把此前 user/assistant/tool 压进 UserMessage.text。"""
    if not messages:
        return ""
    name_by_id = _tool_name_index(messages)
    selected = [m for m in messages if (m.get("role") or "") in ("user", "assistant", "tool")]
    selected = selected[-limit:]
    if not any((m.get("role") or "") in ("assistant", "tool") for m in selected):
        return ""
    blocks: List[str] = []
    for msg in selected:
        blocks.extend(_prior_block_for_msg(msg, name_by_id, max_chars, tool_originals))
    if not blocks:
        return ""
    return "Prior conversation:\n" + "\n".join(blocks)


def _split_user_prompt(
    messages: List[Dict[str, Any]],
    last_text: str,
    tool_originals: Optional[Set[str]],
) -> Tuple[str, List[Dict[str, Any]]]:
    history = messages_to_cursor_history(messages[:-1], tool_originals=tool_originals)
    body = last_text
    if _is_meta_user_text(last_text):
        prior_goal = _last_real_user_text(messages[:-1])
        if prior_goal:
            body = f"{prior_goal}\n\n{last_text}"
    prior = format_prior_context_user_text(messages[:-1], tool_originals=tool_originals)
    if prior:
        return f"{prior}\n\nCurrent request:\n{body}", history
    return body, history


def _split_blank_user(
    messages: List[Dict[str, Any]],
    tool_originals: Optional[Set[str]],
) -> Tuple[str, List[Dict[str, Any]]]:
    for msg in reversed(messages[:-1]):
        if (msg.get("role") or "") == "user" and _user_text(msg):
            idx = messages.index(msg)
            return _user_text(msg), messages_to_cursor_history(
                messages[:idx] + messages[idx + 1 : -1],
                tool_originals=tool_originals,
            )
    return "", messages_to_cursor_history(messages[:-1], tool_originals=tool_originals)


def _split_tool_round(
    messages: List[Dict[str, Any]],
    tool_originals: Optional[Set[str]],
) -> Tuple[str, List[Dict[str, Any]]]:
    text = format_tool_results_user_text(messages, tool_originals=tool_originals)
    trailing = _trailing_tool_messages(messages)
    prior_only = format_prior_context_user_text(
        [m for m in messages if m not in trailing],
        tool_originals=tool_originals,
    )
    if prior_only and text and not text.startswith("Prior conversation:"):
        text = f"{prior_only}\n\n{text}"
    return text, messages_to_cursor_history(messages, tool_originals=tool_originals)


def split_prompt_and_history(
    messages: List[Dict[str, Any]],
    *,
    tool_originals: Optional[Set[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """拆出本轮 UserMessage.text + Cursor conversationHistory。"""
    if not messages:
        return "", []
    last = messages[-1]
    last_role = last.get("role") or ""
    last_text = _user_text(last) if last_role == "user" else ""
    if last_role == "user" and last_text:
        return _split_user_prompt(messages, last_text, tool_originals)
    if last_role == "user" and not last_text:
        return _split_blank_user(messages, tool_originals)
    if last_role == "tool" or (
        last_role != "user"
        and any((m.get("role") or "") == "tool" for m in messages)
    ):
        return _split_tool_round(messages, tool_originals)
    return messages_to_prompt(messages), []
