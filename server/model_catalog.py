from __future__ import annotations

"""OpenAI /v1/models 元数据（Kimi Code think_efforts 等）。"""

from typing import Any, Dict, List, Optional

from server.model_thinking import uses_entml_thinking

# echotools 挡位；Kimi 选 Off 时通过 off_effort 发 reasoning_effort: none
THINK_EFFORTS: Dict[str, Any] = {
    "support": True,
    "valid_efforts": ["low", "medium", "high", "xhigh", "max", "auto"],
    "default_effort": "medium",
    "off_effort": "none",
}

# 上游原生思考、不走 entml；qwen3.8 永远 Thinking
_NATIVE_THINKING_MODELS = frozenset({"qwen3.8-max-preview"})
_ALWAYS_THINKING_MODELS = _NATIVE_THINKING_MODELS


def model_supports_thinking(model_id: str) -> bool:
    if model_id in _NATIVE_THINKING_MODELS:
        return True
    return uses_entml_thinking(model_id)


def build_openai_model_entry(model_id: str, *, created: int = 1700000000) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": "qwen",
    }
    if model_supports_thinking(model_id):
        if model_id in _ALWAYS_THINKING_MODELS:
            entry["always_thinking"] = True
        else:
            entry["think_efforts"] = dict(THINK_EFFORTS)
    return entry


def build_openai_models_list(model_ids: List[str]) -> List[Dict[str, Any]]:
    return [build_openai_model_entry(m) for m in model_ids]
