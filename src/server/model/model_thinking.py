from __future__ import annotations

"""Qwen 上游思考模式解析（基于 model_registry 内键 + entml 开关）。"""

from typing import Optional, Tuple

from echotools.exec.fncall.protocols.entml_think.core import (
    normalize_thinking_level,
    normalize_thinking_mode,
    resolve_thinking_injection,
)

from server.model.model_registry import (
    ModelNotConfiguredError,
    uses_entml_thinking,
    uses_entml_tools,
)

# 上游原生思考且无法关闭（内键）
_ALWAYS_QWEN_THINKING_MODELS = frozenset({"qwen3.8-max-preview"})


def always_qwen_thinking(internal_model: str) -> bool:
    """True=上游永远 Thinking，忽略请求侧 off/none。"""
    return internal_model in _ALWAYS_QWEN_THINKING_MODELS


def uses_entml_protocol(internal_model: str) -> bool:
    """思考或工具走 entml inject 时为 True（含 DeepSeek 仅 entml 工具注册表）。"""
    return uses_entml_thinking(internal_model) or uses_entml_tools(internal_model)


def resolve_qwen_thinking(
    internal_model: str,
    request_thinking_level: Optional[str],
) -> Tuple[bool, str, bool]:
    """返回 (qwen_thinking_enabled, qwen_thinking_mode, use_entml_protocol)。

    *internal_model* 必须为注册表解析后的上游 ID。
    """
    level = normalize_thinking_level(request_thinking_level)
    if level is None and request_thinking_level is not None:
        legacy = normalize_thinking_mode(request_thinking_level)
        if legacy == "off":
            level = "none"
        elif legacy == "on":
            level = "medium"
        elif legacy == "auto":
            level = "auto"
    level = level or "none"

    if always_qwen_thinking(internal_model):
        return True, "Thinking", False

    if uses_entml_protocol(internal_model):
        return False, "Fast", True

    if level == "none" or resolve_thinking_injection({"thinking_level": level}) is None:
        return False, "Fast", False
    return True, "Thinking", False


def model_supports_thinking(internal_model: str) -> bool:
    if internal_model in _ALWAYS_QWEN_THINKING_MODELS:
        return True
    try:
        return uses_entml_protocol(internal_model)
    except ModelNotConfiguredError:
        return False
