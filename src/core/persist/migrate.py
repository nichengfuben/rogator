from __future__ import annotations

"""persist 统一文件 → 按 upstream 分桶迁移。"""

import csv
import json
import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from core.persist.paths import (
    KNOWN_UPSTREAMS,
    login_history_path,
    models_path,
    persist_root,
    sessions_path,
    unified_login_history_path,
    unified_models_path,
    unified_sessions_path,
)
from core.session.io import atomic_write_text

logger = logging.getLogger("rogator")

_UTC8 = timezone(timedelta(hours=8))


def _format_utc8(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=_UTC8).strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=indent)
    atomic_write_text(path, payload)


def _archive(path: Path) -> None:
    if not path.is_file():
        return
    backup = path.with_suffix(path.suffix + ".bak")
    if backup.is_file():
        backup.unlink()
    shutil.move(str(path), str(backup))
    logger.info("已归档 %s → %s", path, backup)


def load_upstream_usernames(upstream: str, root: Path) -> Set[str]:
    csv_path = upstream_dir_accounts_csv(upstream, root)
    if not csv_path.is_file():
        return set()
    out: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            email = (row.get("email") or "").strip()
            if email:
                out.add(email)
    return out


def upstream_dir_accounts_csv(upstream: str, root: Path) -> Path:
    return persist_root(root) / upstream.strip().lower() / "accounts.csv"


def classify_username(
    username: str,
    *,
    qwen_usernames: Set[str],
    deepseek_usernames: Set[str],
    default: str = "qwen",
) -> str:
    if username in deepseek_usernames:
        return "deepseek"
    if username in qwen_usernames:
        return "qwen"
    return default


def _login_entry(value: Any) -> Dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _fill_nested_buckets(
    logins_root: Dict[str, Any],
    buckets: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    for upstream in KNOWN_UPSTREAMS:
        raw = logins_root.get(upstream)
        if not isinstance(raw, dict):
            continue
        for username, entry in raw.items():
            parsed = _login_entry(entry)
            if parsed is not None:
                buckets[upstream][str(username)] = parsed


def _fill_flat_buckets(
    logins_root: Dict[str, Any],
    buckets: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    qwen_usernames: Set[str],
    deepseek_usernames: Set[str],
) -> None:
    for username, entry in logins_root.items():
        parsed = _login_entry(entry)
        if parsed is None:
            continue
        upstream = classify_username(
            str(username),
            qwen_usernames=qwen_usernames,
            deepseek_usernames=deepseek_usernames,
        )
        buckets[upstream][str(username)] = parsed


def split_login_history_logins(
    logins_root: Dict[str, Any],
    *,
    qwen_usernames: Set[str],
    deepseek_usernames: Set[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {
        upstream: {} for upstream in KNOWN_UPSTREAMS
    }
    nested = all(
        isinstance(logins_root.get(up), dict) for up in KNOWN_UPSTREAMS if up in logins_root
    )
    if nested and any(up in logins_root for up in KNOWN_UPSTREAMS):
        _fill_nested_buckets(logins_root, buckets)
        return buckets
    _fill_flat_buckets(
        logins_root,
        buckets,
        qwen_usernames=qwen_usernames,
        deepseek_usernames=deepseek_usernames,
    )
    return buckets


def migrate_login_history_upstream(
    upstream: str,
    root: Path,
    *,
    archive_unified: bool = False,
) -> bool:
    dest = login_history_path(upstream, root)
    if dest.is_file():
        return False

    unified = unified_login_history_path(root)
    if not unified.is_file():
        return False

    data = _read_json(unified)
    if not isinstance(data, dict):
        return False
    logins_root = data.get("logins")
    if not isinstance(logins_root, dict):
        return False

    qwen_usernames = load_upstream_usernames("qwen", root)
    deepseek_usernames = load_upstream_usernames("deepseek", root)
    buckets = split_login_history_logins(
        logins_root,
        qwen_usernames=qwen_usernames,
        deepseek_usernames=deepseek_usernames,
    )
    bucket = buckets.get(upstream.strip().lower(), {})
    if not bucket and upstream != "deepseek":
        return False

    payload = {
        "updated_at": data.get("updated_at") or _format_utc8(time.time()),
        "logins": bucket,
    }
    _write_json(dest, payload)
    logger.info("已迁移登录历史 [%s] → %s (%d 条)", upstream, dest, len(bucket))

    if archive_unified:
        maybe_archive_unified_login_history(root)
    return True


def maybe_archive_unified_login_history(root: Path) -> None:
    unified = unified_login_history_path(root)
    if not unified.is_file():
        return
    if all(login_history_path(up, root).is_file() for up in KNOWN_UPSTREAMS):
        _archive(unified)


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
    if upstream == "qwen" and "sessions" in data and "upstreams" not in data:
        return {
            "sessions": list(data.get("sessions") or []),
            "current_index": int(data.get("current_index") or 0),
            "blocked_accounts": dict(data.get("blocked_accounts") or {}),
            "muted_accounts": dict(data.get("muted_accounts") or {}),
            "updated_at": int(data.get("updated_at") or time.time()),
        }
    return None


def _read_existing_session_bucket(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = _read_json(path)
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
        data = _read_json(unified)
        return data if isinstance(data, dict) else None
    backup = unified.with_suffix(unified.suffix + ".bak")
    if backup.is_file():
        data = _read_json(backup)
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

    bucket = _session_bucket_from_unified(unified_data, upstream.strip().lower())
    if bucket is None:
        return False
    if not bucket["sessions"] and existing is not None and existing["sessions"]:
        return False

    _write_json(dest, bucket, indent=None)
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
        data = _read_json(unified)
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
    _archive(unified)


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
    if upstream == "qwen" and unified.is_file():
        data = _read_json(unified)
        if isinstance(data, dict):
            _write_json(dest, data)
            logger.info("已迁移 models [qwen] → %s", dest)
            if archive_unified:
                maybe_archive_unified_models(root)
            return True

    if upstream == "deepseek":
        try:
            from upstream.deepseek.lib.protocol.consts import MODELS
        except Exception:
            MODELS = []
        payload = {
            "models": list(MODELS),
            "meta": {},
            "updated_at": int(time.time()),
        }
        _write_json(dest, payload)
        logger.info("已初始化 models [deepseek] → %s", dest)
        return True

    return False


def maybe_archive_unified_models(root: Path) -> None:
    unified = unified_models_path(root)
    if not unified.is_file():
        return
    if models_path("qwen", root).is_file():
        _archive(unified)


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
