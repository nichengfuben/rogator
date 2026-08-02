from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from handlers.openai import convert_tools_to_openai


def _append_text_block(block: Dict[str, Any], text_parts: List[str]) -> None:
    btype = block.get("type")
    if btype == "text" or "text" in block and btype is None:
        text_parts.append(str(block.get("text") or ""))


def _append_thinking_block(block: Dict[str, Any], thinking_parts: List[str]) -> None:
    btype = block.get("type")
    if btype in ("thinking", "redacted_thinking"):
        t = str(block.get("thinking") or block.get("data") or "")
        if t:
            thinking_parts.append(t)
    elif btype == "reasoning":
        t = str(block.get("text") or block.get("reasoning") or "")
        if t:
            thinking_parts.append(t)


def _append_tool_use_block(block: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> None:
    tool_calls.append({
        "id": block.get("id") or "",
        "type": "function",
        "function": {
            "name": block.get("name") or "",
            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
        },
    })


def _tool_result_message(block: Dict[str, Any]) -> Dict[str, Any]:
    content = block.get("content")
    if isinstance(content, str):
        content_str = content
    elif content is not None:
        content_str = json.dumps(content, ensure_ascii=False)
    else:
        content_str = ""
    msg: Dict[str, Any] = {
        "role": "tool",
        "tool_call_id": block.get("tool_use_id") or block.get("tool_call_id") or "",
        "content": content_str,
    }
    if block.get("is_error"):
        msg["is_error"] = True
    return msg


def _process_content_block(
    block: Any,
    text_parts: List[str],
    thinking_parts: List[str],
    tool_calls: List[Dict[str, Any]],
    out: List[Dict[str, Any]],
) -> None:
    if not isinstance(block, dict):
        text_parts.append(str(block))
        return
    btype = block.get("type")
    if btype == "tool_result":
        out.append(_tool_result_message(block))
        return
    _append_text_block(block, text_parts)
    _append_thinking_block(block, thinking_parts)
    if btype == "tool_use":
        _append_tool_use_block(block, tool_calls)
    elif btype not in ("text", "thinking", "redacted_thinking", "reasoning") and "text" in block:
        text_parts.append(str(block.get("text") or ""))


def _build_assistant_message(
    text_parts: List[str],
    thinking_parts: List[str],
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    joined = "\n".join(p for p in text_parts if p) or None
    msg: Dict[str, Any] = {"role": "assistant", "content": joined}
    if thinking_parts:
        msg["reasoning"] = "\n".join(thinking_parts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _only_tool_results(content: List[Any]) -> bool:
    return bool(content) and all(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _normalize_message_blocks(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    role = msg.get("role") or "user"
    content = msg.get("content")
    if not isinstance(content, list):
        return [dict(msg)]

    text_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []
    for block in content:
        _process_content_block(block, text_parts, thinking_parts, tool_calls, out)

    if role == "assistant":
        out.append(_build_assistant_message(text_parts, thinking_parts, tool_calls))
    elif role == "tool":
        pass
    else:
        if _only_tool_results(content):
            return out
        joined = "\n".join(p for p in text_parts if p)
        out.append({"role": role, "content": joined})
    return out


def _normalize_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """鎶� Anthropic messages锛堝惈 content 鏁扮粍 / tool_use / tool_result锛夎浆鎴� OpenAI 椋庢牸銆�"""
    out: List[Dict[str, Any]] = []
    for msg in messages or []:
        out.extend(_normalize_message_blocks(msg))
    return out


def _normalize_anthropic_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Anthropic tools锛坣ame/input_schema锛夆啋 OpenAI function tools銆�"""
    return convert_tools_to_openai(tools or [])


from echotools.exec.fncall.protocols.entml_think.core import (
    normalize_thinking_level,
    parse_max_thinking_length,
)

_ANTHROPIC_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _parse_anthropic_effort(body: Dict[str, Any]) -> str:
    """璇诲彇 output_config.effort锛涚渷鐣ユ椂瀹樻柟榛樿�や负 high銆�"""
    output_config = body.get("output_config")
    if not isinstance(output_config, dict) or "effort" not in output_config:
        return "high"
    level = normalize_thinking_level(output_config["effort"])
    if level not in _ANTHROPIC_EFFORT_LEVELS:
        raise ValueError(f"invalid output_config.effort: {output_config['effort']!r}")
    return level


def _build_anthropic_protocol_options(body: Dict[str, Any]) -> Dict[str, Any]:
    """鎸� Anthropic Messages API 瑙ｆ瀽 thinking 涓� output_config.effort銆�"""
    effort = _parse_anthropic_effort(body)
    opts: Dict[str, Any] = {"include_thinking_in_history": True}

    thinking = body.get("thinking")
    if thinking is None:
        opts["thinking_level"] = effort
        return opts

    if not isinstance(thinking, dict):
        raise ValueError("thinking must be an object")

    thinking_type = thinking.get("type")
    if thinking_type is None:
        raise ValueError("thinking.type is required when thinking is set")

    mode = str(thinking_type).strip().lower()
    if mode == "disabled":
        opts["thinking_level"] = "none"
        return opts

    if mode == "adaptive":
        opts["thinking_level"] = effort
        return opts

    if mode == "enabled":
        opts["thinking_level"] = effort
        if "budget_tokens" in thinking:
            max_len = parse_max_thinking_length(thinking["budget_tokens"])
            if max_len is None:
                raise ValueError("thinking.budget_tokens must be a positive integer")
            opts["max_thinking_length"] = max_len
        return opts

    raise ValueError(f"unsupported thinking.type: {thinking_type!r}")
