from __future__ import annotations

from typing import Any, Dict, Optional

from echotools.exec.fncall.protocols.entml_think.core import (
    normalize_thinking_level,
    normalize_thinking_mode,
)

# effort 别名 → echotools thinking_level
_EFFORT_LEVEL_ALIASES = {
    "minimal": "low",
    "default": "medium",
}


def _map_to_thinking_level(raw: Any) -> Optional[str]:
    """映射请求侧 thinking / effort 值为 echotools thinking_level。"""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "medium" if raw else "none"
    if isinstance(raw, (int, float)):
        return "none" if raw == 0 else "medium"
    key = str(raw).strip().lower()
    if not key:
        return None
    level = normalize_thinking_level(key)
    if level is not None:
        return level
    if key in _EFFORT_LEVEL_ALIASES:
        return _EFFORT_LEVEL_ALIASES[key]
    mode = normalize_thinking_mode(key)
    if mode == "off":
        return "none"
    if mode == "on":
        return "medium"
    if mode == "auto":
        return "auto"
    return None


def protocol_thinking_level(protocol_options: Optional[Dict[str, Any]]) -> str:
    """从 protocol_options 读取 thinking_level；兼容旧 thinking_mode。"""
    opts = protocol_options or {}
    if opts.get("thinking_level") is not None:
        normalized = normalize_thinking_level(opts.get("thinking_level"))
        if normalized is not None:
            return normalized
    legacy = normalize_thinking_mode(opts.get("thinking_mode"))
    if legacy == "off":
        return "none"
    if legacy == "on":
        return "medium"
    if legacy == "auto":
        return "auto"
    return "none"


def thinking_level_is_active(level: str) -> bool:
    return level not in ("none", "")
