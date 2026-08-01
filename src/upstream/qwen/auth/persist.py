from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

from core.persist.migrate_util import archive_file, read_json, write_json
from core.persist.paths import models_path, unified_models_path

logger = logging.getLogger("rogator")

NAME = "qwen"
LOGIN_HISTORY_ENABLED = True
ALLOWS_EMPTY_LOGIN_BUCKET = False


def legacy_session_bucket(data: Dict[str, Any]) -> Dict[str, Any] | None:
    if "sessions" in data and "upstreams" not in data:
        return {
            "sessions": list(data.get("sessions") or []),
            "current_index": int(data.get("current_index") or 0),
            "blocked_accounts": dict(data.get("blocked_accounts") or {}),
            "muted_accounts": dict(data.get("muted_accounts") or {}),
            "updated_at": int(data.get("updated_at") or time.time()),
        }
    return None


def migrate_models(
    root: Path,
    dest: Path,
    unified: Path | None,
    *,
    archive_unified: bool = False,
) -> bool:
    path = unified or unified_models_path(root)
    if not path.is_file():
        return False
    data = read_json(path)
    if not isinstance(data, dict):
        return False
    write_json(dest, data)
    logger.info("已迁移 models [qwen] → %s", dest)
    if archive_unified:
        archive_unified_models(root)
    return True


def archive_unified_models(root: Path) -> None:
    unified = unified_models_path(root)
    if unified.is_file() and models_path("qwen", root).is_file():
        archive_file(unified)
