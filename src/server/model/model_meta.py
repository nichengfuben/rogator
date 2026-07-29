from __future__ import annotations

"""上游模型元数据解析、默认值与持久化结构。"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from server.formats.constants import (
    DEFAULT_MODEL_CONTEXT_LENGTH,
    DEFAULT_MODEL_MODALITY,
    PERSISTED_MODEL_CAPABILITIES,
)

__all__ = [
    "DEFAULT_MODEL_CONTEXT_LENGTH",
    "ModelMeta",
    "capabilities_for_api",
    "default_model_meta",
    "merge_capabilities",
    "merge_model_meta",
    "normalize_capabilities",
    "parse_upstream_models_payload",
    "read_models_cache_payload",
    "upstream_model_ids",
]


@dataclass
class ModelMeta:
    context_length: int = DEFAULT_MODEL_CONTEXT_LENGTH
    capabilities: Dict[str, bool] = field(
        default_factory=lambda: dict(PERSISTED_MODEL_CAPABILITIES),
    )
    modality: List[str] = field(default_factory=lambda: list(DEFAULT_MODEL_MODALITY))

    def finalized(self) -> "ModelMeta":
        return ModelMeta(
            context_length=self.context_length,
            capabilities=merge_capabilities(self.capabilities),
            modality=list(self.modality) if self.modality else list(DEFAULT_MODEL_MODALITY),
        )

    def to_dict(self) -> Dict[str, Any]:
        meta = self.finalized()
        return {
            "context_length": meta.context_length,
            "capabilities": dict(meta.capabilities),
            "modality": list(meta.modality),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelMeta":
        ctx = raw.get("context_length")
        context_length = DEFAULT_MODEL_CONTEXT_LENGTH
        if isinstance(ctx, int) and ctx > 0:
            context_length = ctx
        caps = normalize_capabilities(raw.get("capabilities"))
        modality_raw = raw.get("modality")
        modality = (
            [str(x) for x in modality_raw if x]
            if isinstance(modality_raw, list) and modality_raw
            else list(DEFAULT_MODEL_MODALITY)
        )
        return cls(
            context_length=context_length,
            capabilities=caps,
            modality=modality,
        ).finalized()


def default_model_meta() -> ModelMeta:
    return ModelMeta().finalized()


_PLATFORM_META_SKIP = frozenset({"thinking", "tools", "native_tools"})


def merge_capabilities(*parts: Mapping[str, bool]) -> Dict[str, bool]:
    """默认能力打底并合并各层；不含 thinking/tools/native_tools（持久化/内存均忽略）。"""
    merged = dict(PERSISTED_MODEL_CAPABILITIES)
    for part in parts:
        for key, value in part.items():
            if isinstance(key, str) and key and key not in _PLATFORM_META_SKIP:
                merged[key] = bool(value)
    return merged


def normalize_capabilities(raw: Any) -> Dict[str, bool]:
    if not isinstance(raw, dict) or not raw:
        return dict(PERSISTED_MODEL_CAPABILITIES)
    overlay = {
        key: bool(value)
        for key, value in raw.items()
        if isinstance(key, str) and key and key not in _PLATFORM_META_SKIP
    }
    return merge_capabilities(overlay)


def capabilities_for_api(stored: Mapping[str, bool]) -> Dict[str, bool]:
    """/v1/models 对外暴露：在持久化能力上现场补 entml 平台能力。"""
    return {
        **dict(stored),
        "thinking": True,
        "tools": True,
        "native_tools": True,
    }


def _upstream_model_blocks(raw: Mapping[str, Any]) -> List[Any]:
    blocks: List[Any] = []
    data = raw.get("data")
    if isinstance(data, list):
        blocks.append(data)
    elif isinstance(data, dict):
        for key in ("data", "models"):
            nested = data.get(key)
            if isinstance(nested, list):
                blocks.append(nested)
    models = raw.get("models")
    if isinstance(models, list):
        blocks.append(models)
    return blocks


def _model_id_from_item(item: Any) -> Optional[str]:
    if isinstance(item, str):
        text = item.strip()
        return text or None
    if not isinstance(item, dict):
        return None
    for key in ("id", "modelId", "model_id", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def parse_upstream_model(item: Mapping[str, Any]) -> Optional[tuple[str, ModelMeta]]:
    model_id = _model_id_from_item(item)
    if not model_id:
        return None
    info = item.get("info")
    meta_block = info.get("meta") if isinstance(info, dict) else {}
    if not isinstance(meta_block, dict):
        meta_block = {}

    ctx = meta_block.get("max_context_length")
    context_length = DEFAULT_MODEL_CONTEXT_LENGTH
    if isinstance(ctx, int) and ctx > 0:
        context_length = ctx

    caps_raw = meta_block.get("capabilities")
    capabilities = (
        normalize_capabilities(caps_raw)
        if isinstance(caps_raw, dict)
        else dict(PERSISTED_MODEL_CAPABILITIES)
    )

    modality_raw = meta_block.get("modality")
    modality = (
        [str(x) for x in modality_raw if x]
        if isinstance(modality_raw, list) and modality_raw
        else list(DEFAULT_MODEL_MODALITY)
    )

    return model_id, ModelMeta(
        context_length=context_length,
        capabilities=capabilities,
        modality=modality,
    ).finalized()


def parse_upstream_models_payload(raw: Any) -> Dict[str, ModelMeta]:
    """从上游 models API 响应提取 id → 元数据（保序）。"""
    if not isinstance(raw, dict):
        return {}
    result: Dict[str, ModelMeta] = {}
    for block in _upstream_model_blocks(raw):
        for item in block:
            if not isinstance(item, dict):
                continue
            parsed = parse_upstream_model(item)
            if parsed is None:
                continue
            model_id, meta = parsed
            result[model_id] = meta
    return result


def upstream_model_ids(raw: Any) -> List[str]:
    return list(parse_upstream_models_payload(raw).keys())


def merge_model_meta(
    model_ids: Iterable[str],
    *layers: Mapping[str, ModelMeta],
) -> Dict[str, ModelMeta]:
    """按 model_ids 顺序合并元数据；能力默认打底、后层覆盖（不含 thinking）。"""
    result: Dict[str, ModelMeta] = {}
    for model_id in model_ids:
        context_length = DEFAULT_MODEL_CONTEXT_LENGTH
        cap_layers: List[Mapping[str, bool]] = []
        modality = list(DEFAULT_MODEL_MODALITY)
        for layer in layers:
            meta = layer.get(model_id)
            if meta is None:
                continue
            if meta.context_length > 0:
                context_length = meta.context_length
            cap_layers.append(meta.capabilities)
            if meta.modality:
                modality = list(meta.modality)
        result[model_id] = ModelMeta(
            context_length=context_length,
            capabilities=(
                merge_capabilities(*cap_layers)
                if cap_layers
                else dict(PERSISTED_MODEL_CAPABILITIES)
            ),
            modality=modality,
        ).finalized()
    return result


def read_models_cache_payload(data: Mapping[str, Any]) -> tuple[List[str], Dict[str, ModelMeta], float]:
    updated_at = float(data.get("updated_at", 0) or 0)
    raw_models = data.get("models", [])
    model_ids: List[str] = []
    meta_by_id: Dict[str, ModelMeta] = {}

    if isinstance(raw_models, list) and raw_models and isinstance(raw_models[0], dict):
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            model_id = _model_id_from_item(item)
            if not model_id:
                continue
            model_ids.append(model_id)
            meta_by_id[model_id] = ModelMeta.from_dict(item)
        return model_ids, meta_by_id, updated_at

    if isinstance(raw_models, list):
        model_ids = [str(m) for m in raw_models if m]

    raw_meta = data.get("meta") or data.get("details") or {}
    if isinstance(raw_meta, dict):
        for model_id, item in raw_meta.items():
            if isinstance(item, dict):
                meta_by_id[str(model_id)] = ModelMeta.from_dict(item)

    return model_ids, meta_by_id, updated_at
