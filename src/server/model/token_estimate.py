from __future__ import annotations

"""Token 估算：count_tokens 预检与 inject 后 prompt 字符估算。"""

import json
from typing import Any, Dict, List, Optional

__all__ = [
    "estimate_tokens_from_char_count",
    "estimate_stream_tokens_from_char_count",
    "estimate_anthropic_request_input_tokens",
    "estimate_openai_request_input_tokens",
    "estimate_anthropic_injected_input_tokens",
    "estimate_openai_injected_input_tokens",
]


def estimate_tokens_from_char_count(total_chars: int) -> int:
    """按字符数向上取整 // 3 估算 input token。"""
    if total_chars <= 0:
        return 0
    return (total_chars + 2) // 3


def estimate_stream_tokens_from_char_count(total_chars: int) -> int:
    """流式实时估算：已生成字符向上取整 // 4（单字符标点亦计 1 token）。"""
    if total_chars <= 0:
        return 0
    return (total_chars + 3) // 4


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
    """Anthropic Messages / count_tokens 请求体 input 估算（未 inject）。"""
    total = _anthropic_system_char_count(body.get("system"))
    total += _messages_char_count(body.get("messages") or [])
    total += _tools_char_count(body.get("tools"))
    return estimate_tokens_from_char_count(total)


def estimate_openai_request_input_tokens(body: Dict[str, Any]) -> int:
    """OpenAI Chat Completions 请求体 input 估算（未 inject）。"""
    total = _messages_char_count(body.get("messages") or [])
    total += _tools_char_count(body.get("tools"))
    return estimate_tokens_from_char_count(total)


def _inject_prompt_char_count(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    *,
    protocol: Any,
    model: str,
    api: str,
    user_system_prompt: str = "",
    protocol_options: Optional[Dict[str, Any]] = None,
) -> int:
    from handlers.shared.fncall_inject import inject_fncall_for_request

    injected = inject_fncall_for_request(
        messages,
        tools,
        protocol,
        req_id="count-tokens",
        api=api,
        model=model,
        lang="zh",
        user_system_prompt=user_system_prompt,
        protocol_options=protocol_options,
    )
    return len(injected[0].get("content") or "")


def estimate_anthropic_injected_input_tokens(
    body: Dict[str, Any],
    *,
    protocol: Any,
    model: str,
    protocol_options: Optional[Dict[str, Any]] = None,
) -> int:
    """Anthropic count_tokens：按 inject 后实际发给 Qwen 的 prompt 估算。"""
    from handlers import extract_system_for_inject, prepend_anthropic_system
    from handlers.anthropic.normalize import _normalize_anthropic_messages, _normalize_anthropic_tools
    from handlers.openai.tools import convert_tools_to_openai

    raw_messages = body.get("messages") or []
    system = body.get("system")
    messages = _normalize_anthropic_messages(raw_messages)
    messages = prepend_anthropic_system(messages, system)
    tools = _normalize_anthropic_tools(body.get("tools") or [])
    user_system, messages = extract_system_for_inject(messages)
    openai_tools = convert_tools_to_openai(tools)
    chars = _inject_prompt_char_count(
        messages,
        openai_tools,
        protocol=protocol,
        model=model,
        api="anthropic",
        user_system_prompt=user_system,
        protocol_options=protocol_options,
    )
    return estimate_tokens_from_char_count(chars)


def estimate_openai_injected_input_tokens(
    body: Dict[str, Any],
    *,
    protocol: Any,
    model: str,
    protocol_options: Optional[Dict[str, Any]] = None,
) -> int:
    """OpenAI count_tokens 等价估算：inject 后 prompt 字符 // 3。"""
    from handlers import extract_system_for_inject
    from handlers.openai.tools import convert_tools_to_openai

    messages = list(body.get("messages") or [])
    tools = convert_tools_to_openai(body.get("tools") or [])
    user_system, messages = extract_system_for_inject(messages)
    chars = _inject_prompt_char_count(
        messages,
        tools,
        protocol=protocol,
        model=model,
        api="openai",
        user_system_prompt=user_system,
        protocol_options=protocol_options,
    )
    return estimate_tokens_from_char_count(chars)
