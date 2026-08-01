from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

from core.persist.migrate_util import models_cache_updated_at, write_json
from core.persist.paths import models_path

logger = logging.getLogger("rogator")

NAME = "deepseek"
LOGIN_HISTORY_ENABLED = True
ALLOWS_EMPTY_LOGIN_BUCKET = True


def read_models_cache(root: Path | None = None) -> Tuple[List[str], float]:
    path = models_path(NAME, root)
    if not path.is_file():
        return [], 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        updated_at = models_cache_updated_at(path, float(data.get("updated_at", 0) or 0))
        raw_models = data.get("models") or []
        if not isinstance(raw_models, list):
            return [], updated_at
        return [str(m).strip() for m in raw_models if m], updated_at
    except Exception as exc:
        logger.debug("DeepSeek models cache read failed: %s", exc)
        return [], 0.0


def write_models_cache(models: List[str], *, root: Path | None = None) -> None:
    path = models_path(NAME, root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {
            "models": list(models),
            "meta": {},
            "updated_at": int(time.time()),
        })
    except Exception as exc:
        logger.debug("DeepSeek models cache write failed: %s", exc)


def session_expiry_ttl() -> float | None:
    try:
        from server.config.app_config import _load_upstream_toml

        raw = _load_upstream_toml("deepseek") or {}
        session = raw.get("session") if isinstance(raw, dict) else None
        if isinstance(session, dict) and session.get("token_ttl_seconds") is not None:
            return float(session["token_ttl_seconds"])
    except Exception:
        pass
    return 3600.0


def migrate_models(
    root: Path,
    dest: Path,
    unified: Path | None,
    *,
    archive_unified: bool = False,
) -> bool:
    try:
        from upstream.deepseek.lib.protocol.consts import MODELS
    except Exception:
        MODELS = []
    payload = {
        "models": list(MODELS),
        "meta": {},
        "updated_at": int(time.time()),
    }
    write_json(dest, payload)
    logger.info("已初始化 models [deepseek] → %s", dest)
    return True
