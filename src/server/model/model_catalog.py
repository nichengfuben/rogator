from __future__ import annotations

"""OpenAI /v1/models 元数据（Kimi Code think_efforts 等）。"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

from server.model.model_registry import ModelRegistryEntry
from server.model.model_thinking import always_qwen_thinking

# echotools 挡位；Kimi 选 Off 时通过 off_effort 发 reasoning_effort: none
THINK_EFFORTS: Dict[str, Any] = {
    "support": True,
    "valid_efforts": ["low", "medium", "high", "xhigh", "max", "auto"],
    "default_effort": "medium",
    "off_effort": "none",
}

# 默认 256K（256×1024）；无上游元数据时 per-model 回退值
MODEL_CONTEXT_LENGTH: int = 256 * 1024


def model_context_length() -> int:
    """全局配置的模型上下文（config 未设时与 DEFAULT_MODEL_CONTEXT_LENGTH 一致）。"""
    try:
        from server.config import CONFIG

        return int(CONFIG.model_context_length)
    except Exception:
        return MODEL_CONTEXT_LENGTH


# 上游原生思考、不走 entml；内键 qwen3.8 永远 Thinking
_ALWAYS_THINKING_INTERNAL = frozenset({"qwen3.8-max-preview"})


def model_supports_thinking(registry_entry: ModelRegistryEntry) -> bool:
    internal_id = registry_entry.internal_id
    if internal_id in _ALWAYS_THINKING_INTERNAL:
        return True
    return registry_entry.uses_entml


def _resolve_model_meta(
    model_id: str,
    meta_by_id: Optional[Mapping[str, Any]] = None,
) -> "ModelMeta":
    from server.model.model_meta import ModelMeta, default_model_meta

    if meta_by_id and model_id in meta_by_id:
        raw = meta_by_id[model_id]
        if isinstance(raw, ModelMeta):
            return raw.finalized()
        if isinstance(raw, dict):
            return ModelMeta.from_dict(raw)
    return default_model_meta()


def build_openai_model_entry(
    external_id: str,
    *,
    registry_entry: ModelRegistryEntry,
    meta_by_id: Optional[Mapping[str, Any]] = None,
    created: int = 1700000000,
) -> Dict[str, Any]:
    internal_id = registry_entry.internal_id
    meta = _resolve_model_meta(internal_id, meta_by_id)
    from server.model.model_meta import capabilities_for_api

    entry: Dict[str, Any] = {
        "id": external_id,
        "object": "model",
        "created": created,
        "owned_by": "qwen",
        "context_length": meta.context_length,
        "capabilities": capabilities_for_api(meta.capabilities),
        "modality": list(meta.modality),
    }
    if model_supports_thinking(registry_entry):
        if always_qwen_thinking(internal_id):
            entry["always_thinking"] = True
        elif registry_entry.uses_entml:
            entry["think_efforts"] = dict(THINK_EFFORTS)
    return entry


def build_openai_models_list(
    registry_entries: Sequence[ModelRegistryEntry],
    *,
    meta_by_id: Optional[Mapping[str, Any]] = None,
    created: int = 1700000000,
) -> List[Dict[str, Any]]:
    return [
        build_openai_model_entry(
            entry.external_id,
            registry_entry=entry,
            meta_by_id=meta_by_id,
            created=created,
        )
        for entry in registry_entries
    ]
