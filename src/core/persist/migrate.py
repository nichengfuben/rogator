from __future__ import annotations

"""persist 统一文件 → 按 upstream 分桶迁移。"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.persist.migrate_logins import (
    migrate_login_history_upstream,
    maybe_archive_unified_login_history,
    split_login_history_logins,
)
from core.persist.migrate_util import archive_file, read_json, write_json
from core.persist.paths import (
    KNOWN_UPSTREAMS,
    models_path,
    persist_root,
    sessions_path,
    unified_models_path,
    unified_sessions_path,
)
from core.persist.upstream_persist import call_persist, persist_attr

logger = logging.getLogger("rogator")

__all__ = [
    "migrate_all",
    "migrate_login_history_upstream",
    "migrate_models_upstream",
    "migrate_sessions_upstream",
    "maybe_archive_unified_login_history",
    "maybe_archive_unified_models",
    "maybe_archive_unified_sessions",
    "split_login_history_logins",
]


def _session_bucket_from_unified(data: Dict[str, Any], upstream: str) -> Dict[str, Any] | None:
    ups = data.get("upstreams")
    if isinstance(ups, dict):
        bucket = ups.get(upstream)
        if isinstance(bucket, dict):
            return {
                "sessions": list(bucket.get("sessions") or []),
                "current_index": int(bucket.get("current_index") or 0),
                "blocked_accounts": dict(bucket.get("blocked_accounts") or {}),
                "muted_accounts": dict(bucket.get("muted_accounts") or {}),
                "updated_at": int(data.get("updated_at") or time.time()),
            }
    legacy = call_persist(upstream, "legacy_session_bucket", data, default=None)
    if legacy is not None:
        return legacy
    return None


def _read_existing_session_bucket(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = read_json(path)
    except Exception:
        return None
    if not isinstance(raw, dict) or "sessions" not in raw:
        return None
    return {
        "sessions": list(raw.get("sessions") or []),
        "current_index": int(raw.get("current_index") or 0),
        "blocked_accounts": dict(raw.get("blocked_accounts") or {}),
        "muted_accounts": dict(raw.get("muted_accounts") or {}),
        "updated_at": int(raw.get("updated_at") or 0),
    }


def _load_unified_sessions_source(root: Path) -> Dict[str, Any] | None:
    unified = unified_sessions_path(root)
    if unified.is_file():
        data = read_json(unified)
        return data if isinstance(data, dict) else None
    backup = unified.with_suffix(unified.suffix + ".bak")
    if backup.is_file():
        data = read_json(backup)
        return data if isinstance(data, dict) else None
    return None


def migrate_sessions_upstream(
    upstream: str,
    root: Path,
    *,
    archive_unified: bool = False,
    force: bool = False,
) -> bool:
    dest = sessions_path(upstream, root)
    existing = _read_existing_session_bucket(dest)
    if existing is not None and existing["sessions"] and not force:
        return False

    unified_data = _load_unified_sessions_source(root)
    if unified_data is None:
        return False

    key = upstream.strip().lower()
    bucket = _session_bucket_from_unified(unified_data, key)
    if bucket is None:
        return False
    if not bucket["sessions"] and existing is not None and existing["sessions"]:
        return False

    write_json(dest, bucket, indent=None)
    logger.info(
        "已迁移 sessions [%s] → %s (%d 条)",
        upstream,
        dest,
        len(bucket["sessions"]),
    )

    if archive_unified:
        maybe_archive_unified_sessions(root)
    return True


def maybe_archive_unified_sessions(root: Path) -> None:
    unified = unified_sessions_path(root)
    if not unified.is_file():
        return
    try:
        data = read_json(unified)
    except Exception:
        return
    if not isinstance(data, dict) or "upstreams" not in data:
        return
    ups = data.get("upstreams")
    if not isinstance(ups, dict):
        return
    for upstream in ups:
        if upstream.strip().lower() in KNOWN_UPSTREAMS:
            if not sessions_path(upstream, root).is_file():
                return
    archive_file(unified)


def migrate_models_upstream(
    upstream: str,
    root: Path,
    *,
    archive_unified: bool = False,
) -> bool:
    dest = models_path(upstream, root)
    if dest.is_file():
        return False

    unified = unified_models_path(root)
    unified_path = unified if unified.is_file() else None
    return bool(call_persist(
        upstream.strip().lower(),
        "migrate_models",
        root,
        dest,
        unified_path,
        archive_unified=archive_unified,
        default=False,
    ))


def maybe_archive_unified_models(root: Path) -> None:
    call_persist("qwen", "archive_unified_models", root)


def migrate_all(
    root: Path,
    *,
    upstreams: Iterable[str] = KNOWN_UPSTREAMS,
    archive_unified: bool = True,
) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {
        "login_history": [],
        "sessions": [],
        "models": [],
        "archived": [],
    }
    for upstream in upstreams:
        key = upstream.strip().lower()
        if migrate_login_history_upstream(key, root, archive_unified=False):
            results["login_history"].append(key)
        if migrate_sessions_upstream(key, root, archive_unified=False):
            results["sessions"].append(key)
        if migrate_models_upstream(key, root, archive_unified=False):
            results["models"].append(key)

    if archive_unified:
        for name, archive_fn in (
            ("login_history.json", maybe_archive_unified_login_history),
            ("sessions.json", maybe_archive_unified_sessions),
            ("models.json", maybe_archive_unified_models),
        ):
            before = (persist_root(root) / name).is_file()
            archive_fn(root)
            if before and not (persist_root(root) / name).is_file():
                results["archived"].append(name)
    return results
