from __future__ import annotations

"""思考路由：entml inject vs Qwen 上游原生思考（DeepSeek 仅 entml）。"""

from dataclasses import dataclass
from typing import Optional

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

# Qwen 上游原生思考且无法关闭（内键）
_ALWAYS_QWEN_NATIVE_THINKING = frozenset({"qwen3.8-max-preview"})


@dataclass(frozen=True)
class ThinkingRoute:
    """按注册表与请求挡位解析出的思考/工具路径。

    * ``use_entml`` — inject entml 思考与工具（Qwen / DeepSeek 网关主路径）。
    * ``qwen_native_*`` — 仅 Qwen 上游 ``thinking_enabled`` / ``thinking_mode``；
      DeepSeek 与其它上游忽略。
    """

    use_entml: bool
    qwen_native_enabled: bool
    qwen_native_mode: str


def always_qwen_native_thinking(internal_model: str) -> bool:
    """True=Qwen 上游永远 Thinking，忽略请求侧 off/none。"""
    return internal_model in _ALWAYS_QWEN_NATIVE_THINKING


def uses_entml_protocol(internal_model: str) -> bool:
    """思考或工具走 entml inject 时为 True。"""
    return uses_entml_thinking(internal_model) or uses_entml_tools(internal_model)


def resolve_thinking_route(
    internal_model: str,
    request_thinking_level: Optional[str],
) -> ThinkingRoute:
    """解析模型思考路由（*internal_model* 为注册表内键）。"""
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

    if always_qwen_native_thinking(internal_model):
        return ThinkingRoute(
            use_entml=False,
            qwen_native_enabled=True,
            qwen_native_mode="Thinking",
        )

    if uses_entml_protocol(internal_model):
        return ThinkingRoute(
            use_entml=True,
            qwen_native_enabled=False,
            qwen_native_mode="Fast",
        )

    if level == "none" or resolve_thinking_injection({"thinking_level": level}) is None:
        return ThinkingRoute(
            use_entml=False,
            qwen_native_enabled=False,
            qwen_native_mode="Fast",
        )

    return ThinkingRoute(
        use_entml=False,
        qwen_native_enabled=True,
        qwen_native_mode="Thinking",
    )


def model_supports_thinking(internal_model: str) -> bool:
    if internal_model in _ALWAYS_QWEN_NATIVE_THINKING:
        return True
    try:
        return uses_entml_protocol(internal_model)
    except ModelNotConfiguredError:
        return False
