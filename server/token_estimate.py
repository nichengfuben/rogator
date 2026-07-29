from __future__ import annotations

"""Token 估算：输入 len//3；输出 token 由上游 usage 提供。"""

import json
from typing import Any, Dict, List

__all__ = [
    "estimate_tokens_from_char_count",
    "estimate_anthropic_request_input_tokens",
    "estimate_openai_request_input_tokens",
]


def estimate_tokens_from_char_count(total_chars: int) -> int:
    """按字符数 // 3 估算 input token（与 count_tokens 一致）。"""
    return max(0, total_chars // 3)


def _serialize_for_estimate(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _anthropic_system_char_count(system: Any) -> int:
    if system is None or system == "":
        return 0
    if isinstance(system, str):
        return len(system)
    if isinstance(system, list):
        total = 0
        for block in system:
            if isinstance(block, dict):
                total += len(str(block.get("text") or block.get("thinking") or ""))
            else:
                total += len(str(block))
        return total
    return len(_serialize_for_estimate(system))


def _messages_char_count(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        total += len(_serialize_for_estimate(msg.get("content", "")))
    return total


def _tools_char_count(tools: Any) -> int:
    if not tools:
        return 0
    return len(_serialize_for_estimate(tools))


def estimate_anthropic_request_input_tokens(body: Dict[str, Any]) -> int:
    """Anthropic Messages / count_tokens 请求体 input 估算。"""
    total = _anthropic_system_char_count(body.get("system"))
    total += _messages_char_count(body.get("messages") or [])
    total += _tools_char_count(body.get("tools"))
    return estimate_tokens_from_char_count(total)


def estimate_openai_request_input_tokens(body: Dict[str, Any]) -> int:
    """OpenAI Chat Completions 请求体 input 估算。"""
    total = _messages_char_count(body.get("messages") or [])
    total += _tools_char_count(body.get("tools"))
    return estimate_tokens_from_char_count(total)
