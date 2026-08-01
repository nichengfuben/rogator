from __future__ import annotations

"""消息文本 / system / 本轮 UserMessage 构造。"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from upstream.cursor.setup.config import load_cursor_upstream_config

IMPORTANT_MCP_TOOLS_ONLY = """IMPORTANT:
You may use ONLY the tools provided in this request's tool list, calling them by their exact names.
Treat every other tool as unavailable — including any Cursor built-in shell, file, terminal, browser, editor, search, or tools not in that list.
When you need to act, call a tool from the provided list. Never invent a tool name."""

IMPORTANT_NO_TOOLS = """IMPORTANT:
No tools are available in this turn.
You must not call any tool — including built-in shell, file, terminal, browser, editor, search, or MCP tools.
Do not attempt tool use. Respond with plain assistant text only."""


def _message_text(msg: Dict[str, Any]) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _message_text_parts(content)
    if isinstance(content, (dict, bool, int, float)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _message_text_parts(parts: List[Any]) -> str:
    out: List[str] = []
    for p in parts:
        if not isinstance(p, dict):
            out.append(str(p))
            continue
        text = p.get("text")
        if text is not None and str(text).strip():
            out.append(str(text))
            continue
        if p.get("type") in ("text", "input_text") and p.get("text") is not None:
            out.append(str(p.get("text") or ""))
            continue
        if isinstance(p.get("content"), str) and str(p.get("content") or "").strip():
            out.append(str(p.get("content") or ""))
            continue
        if p.get("type") in ("image_url", "image", "input_image", "file", "input_file"):
            continue
        if "text" in p:
            out.append(str(p.get("text") or ""))
        elif "content" in p and isinstance(p.get("content"), str):
            out.append(str(p.get("content") or ""))
    return "\n".join(out)


def _user_text(msg: Dict[str, Any]) -> str:
    """用户可见文本（去首尾空白）；空白视为空。"""
    return _message_text(msg).strip()


def _is_meta_user_text(text: str) -> bool:
    """客户端注入的 meta 轮（如 Plan mode system-reminder），本身不含用户任务。"""
    t = (text or "").lstrip()
    return t.startswith("<system-reminder>")


def _last_real_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if (msg.get("role") or "") != "user":
            continue
        text = _user_text(msg)
        if text and not _is_meta_user_text(text):
            return text
    return ""


def extract_system_texts(messages: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for msg in messages or []:
        if (msg.get("role") or "") != "system":
            continue
        text = _message_text(msg)
        if text:
            out.append(text)
    return out


def build_custom_system_prompt(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> str:
    """IMPORTANT 块置顶，其后拼接全部 system 消息。始终返回非空字符串。"""
    has_tools = bool(tools)
    preamble = IMPORTANT_MCP_TOOLS_ONLY if has_tools else IMPORTANT_NO_TOOLS
    systems = extract_system_texts(messages)
    if not systems:
        return preamble
    return preamble + "\n\n" + "\n\n".join(systems)


def prepend_system_to_prompt(system_text: str, prompt: str) -> str:
    """兼容旧路径：system 前缀 + body。"""
    block = (system_text or "").strip()
    if not block:
        return prompt or ""
    wrapped = f"<system>\n{block}\n</system>"
    body = (prompt or "").strip()
    if not body:
        return wrapped
    return f"{wrapped}\n\n{body}"


def messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return ""
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role") or "user"
        text = _message_text(msg)
        if not text and role != "assistant":
            continue
        if role == "system":
            continue
        if role == "assistant":
            if text:
                parts.append(f"<assistant>\n{text}\n</assistant>")
        elif role == "tool":
            continue
        else:
            parts.append(text)
    return "\n\n".join(parts)


def build_prepend_user_messages(system_text: str) -> List[Dict[str, Any]]:
    """对齐 agent.v1.UserMessageAction.prepend_user_messages。"""
    block = (system_text or "").strip()
    if not block:
        return []
    return [{
        "text": block,
        "messageId": str(uuid.uuid4()),
        "mode": 1,
    }]


def build_cursor_turn(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """构造 Cursor 本轮 UserMessage.text + conversationHistory + prependUserMessages。"""
    from upstream.cursor.chat.convert.history import split_prompt_and_history
    from upstream.cursor.chat.convert.tools import original_tool_names

    originals = original_tool_names(tools)
    prompt, history = split_prompt_and_history(messages, tool_originals=originals or None)
    system = build_custom_system_prompt(messages, tools).strip()
    prepend = build_prepend_user_messages(system)
    return (prompt or "").strip(), list(history or []), prepend


def map_model(model: Optional[str]) -> str:
    """将 Rogator 内键映射为 Cursor Agent ``modelId``（保留 effort 后缀）。"""
    cfg = load_cursor_upstream_config()
    cursor_cfg = cfg.get("cursor") or {}
    models_cfg = cfg.get("models") or {}
    default = str(models_cfg.get("default") or cursor_cfg.get("default_model") or "composer-2.5-fast")
    if not model:
        return default
    from upstream.cursor.models.identity import is_valid_model_id
    if is_valid_model_id(model):
        return model
    mapping = models_cfg.get("mapping") or {}
    if isinstance(mapping, dict) and model in mapping:
        mapped = str(mapping[model])
        return mapped if is_valid_model_id(mapped) else model
    return model
