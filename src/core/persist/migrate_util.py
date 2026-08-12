from __future__ import annotations

"""持久化 JSON 写入工具。"""

import json
from pathlib import Path
from typing import Any

from core.session.io import atomic_write_text


def write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=indent)
    atomic_write_text(path, payload)


def models_cache_updated_at(path: Path, stored: float) -> float:
    """模型缓存时间：存储值无效（<=0）时回退文件 mtime（对齐 qwen 缓存语义）。"""
    try:
        if stored > 0:
            return float(stored)
    except (TypeError, ValueError):
        pass
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0
