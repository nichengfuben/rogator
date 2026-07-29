from __future__ import annotations

"""模型注册表：外键（API）→ 内键（上游）→ entml 思考开关。"""

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
    """上游模型已在列表中，但注册表缺少外键:内键:值 配置。"""


@dataclass
class ModelRegistry:
    entries_in_order: List[ModelRegistryEntry]
    by_external: Dict[str, ModelRegistryEntry]
    by_internal: Dict[str, ModelRegistryEntry]


def _parse_line(line: str) -> Optional[ModelRegistryEntry]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(":", 2)
    if len(parts) != 3:
        return None
    external_id, internal_id, flag = (p.strip() for p in parts)
    if not external_id or not internal_id:
        return None
    lowered = flag.lower()
    if lowered in ("true", "1", "yes"):
        uses_entml = True
    elif lowered in ("false", "0", "no"):
        uses_entml = False
    else:
        return None
    return ModelRegistryEntry(external_id, internal_id, uses_entml)


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
    entry = _MODEL_REGISTRY.by_internal.get(internal_model)
    if entry is None:
        raise ModelNotConfiguredError(
            f"模型 ID {internal_model} 存在于上游列表但未配置，请检查 persist/model_registry.jsonl"
        )
    return entry.uses_entml
