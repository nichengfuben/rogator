from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from echotools.exec.fncall.protocols.entml_think.core import (
    normalize_thinking_level,
    parse_max_thinking_length,
)

from handlers.shared.fncall_inject import inject_fncall_for_request
from handlers.openai.thinking import _map_to_thinking_level


def _parse_thinking_dict(
    raw: dict, level: Optional[str],
) -> Tuple[Optional[str], Optional[int]]:
    max_len: Optional[int] = None
    if level is None and "level" in raw:
        level = normalize_thinking_level(raw.get("level"))
    if level is None and "mode" in raw:
        level = _map_to_thinking_level(raw.get("mode"))
    if level is None and "type" in raw:
        level = _map_to_thinking_level(raw.get("type"))
    if level is None and "enabled" in raw:
        level = "medium" if raw.get("enabled") else "none"
    if level is None and "effort" in raw:
        level = _map_to_thinking_level(raw.get("effort"))
    for key in ("budget_tokens", "max_tokens", "max_thinking_length"):
        if key in raw:
            max_len = parse_max_thinking_length(raw.get(key))
            if max_len is not None:
                break
    return level, max_len


def _parse_thinking_field(
    raw: Any, level: Optional[str],
) -> Tuple[Optional[str], Optional[int]]:
    if isinstance(raw, bool):
        return ("medium" if raw else "none"), None
    if isinstance(raw, dict):
        return _parse_thinking_dict(raw, level)
    if raw is not None and level is None:
        return _map_to_thinking_level(raw), None
    return level, None


def _parse_reasoning_dict(
    reasoning: dict, level: Optional[str], max_len: Optional[int],
) -> Tuple[Optional[str], Optional[int]]:
    if level is None and "level" in reasoning:
        level = normalize_thinking_level(reasoning.get("level"))
    if level is None and "effort" in reasoning:
        level = _map_to_thinking_level(reasoning.get("effort"))
    if level is None and "mode" in reasoning:
        level = _map_to_thinking_level(reasoning.get("mode"))
    if level is None and "enabled" in reasoning:
        level = "medium" if reasoning.get("enabled") else "none"
    if level is None and "type" in reasoning:
        level = _map_to_thinking_level(reasoning.get("type"))
    if max_len is None:
        for key in ("budget_tokens", "max_tokens", "max_thinking_length"):
            if key in reasoning:
                max_len = parse_max_thinking_length(reasoning.get(key))
                if max_len is not None:
                    break
    return level, max_len


def _parse_reasoning_field(
    body: Dict[str, Any], level: Optional[str], max_len: Optional[int],
) -> Tuple[Optional[str], Optional[int]]:
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        return _parse_reasoning_dict(reasoning, level, max_len)
    if reasoning is not None and level is None:
        return _map_to_thinking_level(reasoning), max_len
    return level, max_len


def _assemble_protocol_options(
    level: Optional[str], max_len: Optional[int],
) -> Dict[str, Any]:
    opts: Dict[str, Any] = {"include_thinking_in_history": True}
    if level is None and max_len is None:
        return opts
    if level == "none" or (level is None and max_len is not None):
        if level == "none":
            opts["thinking_level"] = "none"
        elif max_len is not None:
            opts["thinking_level"] = "medium"
            opts["max_thinking_length"] = max_len
        return opts
    if level is not None:
        opts["thinking_level"] = level
    if max_len is not None:
        opts["max_thinking_length"] = max_len
    return opts


def _build_protocol_options(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从请求体提取 thinking 设置并构建 protocol_options。"""
    level: Optional[str] = normalize_thinking_level(body.get("thinking_level"))
    max_len: Optional[int] = None

    level, parsed_len = _parse_thinking_field(body.get("thinking"), level)
    if parsed_len is not None:
        max_len = parsed_len

    if level is None:
        level = _map_to_thinking_level(body.get("thinking_mode"))
    if level is None and "reasoning_effort" in body:
        level = _map_to_thinking_level(body.get("reasoning_effort"))

    level, parsed_len = _parse_reasoning_field(body, level, max_len)
    if parsed_len is not None:
        max_len = parsed_len

    if max_len is None:
        max_len = parse_max_thinking_length(body.get("max_thinking_length"))

    return _assemble_protocol_options(level, max_len)


def _inject_protocol_options(protocol_options: Optional[Dict[str, Any]], use_entml: bool) -> Dict[str, Any]:
    """合并 inject_fncall 选项：历史 thinking 始终交给 echotools 渲染。"""
    opts = dict(protocol_options or {})
    opts.setdefault("include_thinking_in_history", True)
    if not use_entml:
        opts.pop("thinking_level", None)
        opts.pop("thinking_mode", None)
        opts.pop("max_thinking_length", None)
    return opts
