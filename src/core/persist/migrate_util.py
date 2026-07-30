from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.session.io import atomic_write_text

logger = logging.getLogger("rogator")

_UTC8 = timezone(timedelta(hours=8))


def format_utc8(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=_UTC8).strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=indent)
    atomic_write_text(path, payload)


def models_cache_updated_at(path: Path, payload_updated_at: float) -> float:
    """优先 JSON ``updated_at``；缺失时回退到文件 mtime。"""
    ts = float(payload_updated_at or 0)
    if ts > 0:
        return ts
    try:
        if path.is_file():
            return path.stat().st_mtime
    except OSError:
        pass
    return 0.0


def archive_file(path: Path) -> None:
    if not path.is_file():
        return
    backup = path.with_suffix(path.suffix + ".bak")
    if backup.is_file():
        backup.unlink()
    shutil.move(str(path), str(backup))
    logger.info("已归档 %s → %s", path, backup)
