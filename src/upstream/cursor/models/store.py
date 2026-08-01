from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.persist.migrate_util import models_cache_updated_at
from core.persist.paths import models_path
from upstream.cursor.models.identity import is_valid_model_id, meta_for_model

logger = logging.getLogger("rogator")

UPSTREAM = "cursor"


def cache_path(root: Path | None = None) -> Path:
    return models_path(UPSTREAM, root)


def merge_model_lists(*parts: List[str]) -> List[str]:
    seen: set[str] = set()
    merged: List[str] = []
    for part in parts:
        for model_id in part:
            text = str(model_id or "").strip()
            if not text or not is_valid_model_id(text) or text in seen:
                continue
            seen.add(text)
            merged.append(text)
    return merged


def _meta_for(model_id: str) -> Dict[str, Any]:
    return meta_for_model(model_id)


def read_cache(root: Path | None = None) -> Tuple[List[str], Dict[str, Any], float]:
    path = cache_path(root)
    if not path.is_file():
        return [], {}, 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        updated_at = models_cache_updated_at(path, float(data.get("updated_at", 0) or 0))
        raw_models = data.get("models") or []
        model_ids = merge_model_lists([str(m) for m in raw_models if m]) if isinstance(raw_models, list) else []
        meta: Dict[str, Any] = {}
        raw_meta = data.get("meta") or {}
        if isinstance(raw_meta, dict):
            for mid, item in raw_meta.items():
                if isinstance(item, dict) and is_valid_model_id(str(mid)):
                    meta[str(mid)] = {**meta_for_model(str(mid)), **dict(item)}
        for mid in model_ids:
            meta.setdefault(mid, meta_for_model(mid))
        return model_ids, meta, updated_at
    except Exception as exc:
        logger.debug("Cursor models cache read failed: %s", exc)
        return [], {}, 0.0


def write_cache(
    models: List[str],
    meta: Dict[str, Any],
    *,
    root: Path | None = None,
) -> None:
    path = cache_path(root)
    try:
        clean_models = merge_model_lists(models)
        clean_meta = {
            mid: meta.get(mid) or meta_for_model(mid)
            for mid in clean_models
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "models": clean_models,
            "meta": clean_meta,
            "updated_at": int(time.time()),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("Cursor models cache write failed: %s", exc)


def load_merged(
    config_ids: List[str],
    *,
    root: Path | None = None,
) -> Tuple[List[str], Dict[str, Any], float]:
    disk_ids, disk_meta, updated_at = read_cache(root)
    merged = merge_model_lists(config_ids, disk_ids)
    meta = {mid: meta_for_model(mid) for mid in merged}
    for mid, item in disk_meta.items():
        if mid in meta and isinstance(item, dict):
            meta[mid] = {**meta[mid], **item}
    return merged, meta, updated_at
