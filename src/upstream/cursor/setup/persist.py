from __future__ import annotations

import logging
import time
from pathlib import Path

from core.persist.migrate_util import write_json

logger = logging.getLogger("rogator")

NAME = "cursor"
LOGIN_HISTORY_ENABLED = False
ALLOWS_EMPTY_LOGIN_BUCKET = False


def migrate_models(
    root: Path,
    dest: Path,
    unified: Path | None,
    *,
    archive_unified: bool = False,
) -> bool:
    try:
        from upstream.cursor.client import _model_ids_from_config
        from upstream.cursor.models.identity import meta_for_model

        ids = _model_ids_from_config()
        payload = {
            "models": ids,
            "meta": {mid: meta_for_model(mid) for mid in ids},
            "updated_at": int(time.time()),
        }
        write_json(dest, payload)
        logger.info("已初始化 models [cursor] → %s", dest)
        return True
    except Exception as exc:
        logger.debug("cursor models init skipped: %s", exc)
        return False
