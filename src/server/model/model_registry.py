from __future__ import annotations

"""模型注册表：外键 → 内键 → entml 思考 / entml 工具调用开关。"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from server.config.files import PROJECT_ROOT

logger = logging.getLogger("rogator")

MODEL_REGISTRY_FILE = PROJECT_ROOT / "persist" / "model_registry.jsonl"


@dataclass(frozen=True)
class ModelRegistryEntry:
    external_id: str
    internal_id: str
    uses_entml: bool
    uses_entml_tools: bool


class ModelResolveError(Exception):
    """模型 ID 解析失败（handler 映射为 HTTP 4xx）。"""

    status: int = 400
    error_type: str = "invalid_request_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotFoundError(ModelResolveError):
    status = 404


class ModelInternalIdError(ModelResolveError):
    """调用方使用了内键（上游 ID）而非外键。"""


class ModelNotConfiguredError(ModelResolveError):
    """上游模型已在列表中，但注册表缺少外键:内键:思考:工具 配置。"""


@dataclass
class ModelRegistry:
    entries_in_order: List[ModelRegistryEntry]
    by_external: Dict[str, ModelRegistryEntry]
    by_internal: Dict[str, ModelRegistryEntry]


def _parse_bool_flag(raw: str) -> Optional[bool]:
    lowered = raw.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    return None


def _parse_line(line: str) -> Optional[ModelRegistryEntry]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split(":")]
    if len(parts) not in (3, 4):
        return None
    external_id, internal_id = parts[0], parts[1]
    if not external_id or not internal_id:
        return None
    thinking_flag = _parse_bool_flag(parts[2])
    if thinking_flag is None:
        return None
    if len(parts) == 4:
        tools_flag = _parse_bool_flag(parts[3])
        if tools_flag is None:
            return None
    else:
        tools_flag = thinking_flag
    return ModelRegistryEntry(external_id, internal_id, thinking_flag, tools_flag)


def load_model_registry(path: Path | None = None) -> ModelRegistry:
    p = path or MODEL_REGISTRY_FILE
    entries: List[ModelRegistryEntry] = []
    by_external: Dict[str, ModelRegistryEntry] = {}
    by_internal: Dict[str, ModelRegistryEntry] = {}
    if not p.exists():
        logger.warning("Model registry not found: %s", p)
        return ModelRegistry([], {}, {})
    for line in p.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        entries.append(parsed)
        by_external[parsed.external_id] = parsed
        by_internal[parsed.internal_id] = parsed
    return ModelRegistry(entries, by_external, by_internal)


_MODEL_REGISTRY: ModelRegistry = load_model_registry()


def get_model_registry() -> ModelRegistry:
    return _MODEL_REGISTRY


def reload_model_registry(path: Path | None = None) -> ModelRegistry:
    global _MODEL_REGISTRY
    _MODEL_REGISTRY = load_model_registry(path)
    return _MODEL_REGISTRY


def _entry_for_internal(internal_model: str) -> ModelRegistryEntry:
    entry = _MODEL_REGISTRY.by_internal.get(internal_model)
    if entry is None:
        raise ModelNotConfiguredError(
            f"模型 ID {internal_model} 存在于上游列表但未配置，请检查 persist/model_registry.jsonl"
        )
    return entry


def list_external_models(available_internal: Sequence[str]) -> List[str]:
    """注册表顺序列出、且内键仍在上游模型列表中的外键。"""
    available = set(available_internal)
    return [
        entry.external_id
        for entry in _MODEL_REGISTRY.entries_in_order
        if entry.internal_id in available
    ]


def resolve_request_model(requested: str, available_internal: Iterable[str]) -> ModelRegistryEntry:
    """将 API 外键解析为注册表项；内键不可用，未配置的上游模型报错。"""
    model = requested.strip()
    if not model:
        raise ModelNotFoundError("model is required")

    available = set(available_internal)
    registry = get_model_registry()

    if model in registry.by_external:
        entry = registry.by_external[model]
        if entry.internal_id not in available:
            raise ModelNotConfiguredError(
                f"模型 {entry.external_id} 已配置但上游不可用，请检查 model_registry.jsonl 与模型列表"
            )
        return entry

    if model in registry.by_internal:
        entry = registry.by_internal[model]
        raise ModelInternalIdError(
            f"不能使用内键 {model} 调用，请使用外键 {entry.external_id}"
        )

    if model in available:
        raise ModelNotConfiguredError(
            f"模型 ID {model} 存在于上游列表但未配置，请检查 persist/model_registry.jsonl"
        )

    raise ModelNotFoundError(f"模型 {model} 不存在")


def uses_entml_thinking(internal_model: str) -> bool:
    return _entry_for_internal(internal_model).uses_entml


def uses_entml_tools(internal_model: str) -> bool:
    return _entry_for_internal(internal_model).uses_entml_tools


def uses_native_upstream_response(entry: ModelRegistryEntry) -> bool:
    """思考与工具调用均不走 entml 响应解析。"""
    return not entry.uses_entml and not entry.uses_entml_tools


def is_native_upstream_event(
    entry: Optional[ModelRegistryEntry],
    event: Dict[str, object],
) -> bool:
    """上游事件是否应绕过 entml 解析（按注册表或 upstream native 标记）。"""
    if event.get("native"):
        return True
    if entry is None:
        return False
    etype = event.get("type")
    if etype == "tool_call":
        return not entry.uses_entml_tools
    if etype == "thinking":
        return not entry.uses_entml
    if etype == "answer":
        return not entry.uses_entml_tools
    return False
