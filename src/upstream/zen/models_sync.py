from __future__ import annotations

"""Zen 注册表同步：把远端 models 列表写入 model_registry.jsonl。

独立为单文件，避免 ``client.py`` 行数超 400 软上限；保持 ``upstream.zen``
子项 ≤7（与原 ``models/registry_sync.py`` 同一职责，但文件平铺）。
"""

import logging
from pathlib import Path
from typing import List, Optional

from server.model.model_registry import (
    MODEL_REGISTRY_FILE,
    load_model_registry,
    reload_model_registry,
)

logger = logging.getLogger("rogator")

# 禁止同步的模型
ZEN_BLOCKED_MODELS = {"qwen-asr"}


def sync_zen_registry(model_ids: List[str], *, registry_path: Optional[Path] = None) -> int:
    """将 zen 上游模型同步到注册表；返回新增数量。"""
    p = registry_path or MODEL_REGISTRY_FILE
    if not p.is_file():
        logger.warning("zen registry sync: model_registry.jsonl not found at %s", p)
        return 0
    valid_ids = [m for m in model_ids if m and m not in ZEN_BLOCKED_MODELS]
    if not valid_ids:
        return 0
    registry = load_model_registry(p)
    existing_internal: set = set(registry.by_internal.keys())
    existing_external: set = set(registry.by_external.keys())
    new_models: List[str] = []
    for model_id in valid_ids:
        if model_id not in existing_internal and model_id not in existing_external:
            new_models.append(model_id)
            existing_internal.add(model_id)
            existing_external.add(model_id)
    if not new_models:
        reload_model_registry(p)
        return 0
    text = p.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    new_lines = [f"{m}:{m}:true:true" for m in new_models]
    p.write_text(text + "\n".join(new_lines) + "\n", encoding="utf-8")
    reload_model_registry(p)
    logger.info("zen registry: added %d model(s) to model_registry.jsonl", len(new_models))
    return len(new_models)