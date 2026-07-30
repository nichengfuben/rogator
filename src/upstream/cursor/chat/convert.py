from __future__ import annotations

"""OpenAI ↔ Cursor 消息/模型转换。"""

import json
from typing import Any, Dict, List, Optional, Tuple

from upstream.cursor.setup.config import load_cursor_upstream_config


def _message_text(msg: Dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


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
            parts.append(f"<system>\n{text}\n</system>")
        elif role == "assistant":
            if text:
                parts.append(f"<assistant>\n{text}\n</assistant>")
        elif role == "tool":
            parts.append(f"<tool_result name=\"{msg.get('name', '')}\">\n{text}\n</tool_result>")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def messages_to_cursor_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role") or "user"
        text = _message_text(msg)
        if role == "user":
            if not text:
                continue
            history.append({"user": {"content": [{"text": {"text": text}}]}})
        elif role == "assistant":
            blocks: List[Dict[str, Any]] = []
            if text:
                blocks.append({"text": {"text": text}})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append({
                    "toolCall": {
                        "toolCallId": tc.get("id") or "",
                        "toolName": fn.get("name") or "",
                        "argsJson": fn.get("arguments") or "{}",
                    },
                })
            if blocks:
                history.append({"assistant": {"content": blocks}})
        elif role == "tool":
            history.append({
                "tool": {
                    "toolCallId": msg.get("tool_call_id") or "",
                    "toolName": msg.get("name") or "",
                    "content": [{"text": {"text": text}}],
                    "isError": False,
                },
            })
    return history


def split_prompt_and_history(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if not messages:
        return "", []
    last = messages[-1]
    if last.get("role") == "user" and _message_text(last):
        return _message_text(last), messages_to_cursor_history(messages[:-1])
    return messages_to_prompt(messages), []


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


def openai_tools_to_mcp(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "toolName": name,
            "description": fn.get("description") or "",
            "inputSchemaJson": json.dumps(
                fn.get("parameters") or {"type": "object", "properties": {}},
                ensure_ascii=False,
            ),
        })
    return out
