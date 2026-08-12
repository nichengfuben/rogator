from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from server.model.model_registry import MODEL_REGISTRY_FILE, load_model_registry, reload_model_registry
from upstream.cursor.models.identity import external_id_for, is_valid_model_id

logger = logging.getLogger("rogator")


def sync_cursor_registry(model_ids: List[str], *, path: Path | None = None) -> int:
    """将 Cursor 上游模型追加到 ``model_registry.jsonl``（外键=内键，native 响应）。"""
    p = path or MODEL_REGISTRY_FILE
    if not p.is_file():
        return 0

    registry = load_model_registry(p)
    existing_internal = set(registry.by_internal.keys())
    existing_external = set(registry.by_external.keys())
    new_lines: List[str] = []

    for model_id in model_ids:
        if not is_valid_model_id(model_id):
            continue
        external = external_id_for(model_id)
        internal = model_id
        if internal in existing_internal or external in existing_external:
            continue
        new_lines.append(f"{external}:{internal}:false:false")
        existing_internal.add(internal)
        existing_external.add(external)

    if not new_lines:
        return 0

    text = p.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    p.write_text(text + "\n".join(new_lines) + "\n", encoding="utf-8")
    reload_model_registry(p)
    logger.info("Cursor registry: added %d model(s)", len(new_lines))
    return len(new_lines)
