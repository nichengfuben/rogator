from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, Set

from core.persist.migrate_util import archive_file, format_utc8, read_json, write_json
from core.persist.paths import KNOWN_UPSTREAMS, login_history_path, persist_root, unified_login_history_path
from core.persist.upstream_persist import login_history_enabled_upstreams, persist_attr

logger = logging.getLogger("rogator")


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


def _username_sets(root: Path) -> Dict[str, Set[str]]:
    return {
        name: load_upstream_usernames(name, root)
        for name in login_history_enabled_upstreams()
    }


def classify_username(
    username: str,
    *,
    qwen_usernames: Set[str] | None = None,
    deepseek_usernames: Set[str] | None = None,
    root: Path | None = None,
    default: str = "qwen",
) -> str:
    if qwen_usernames is not None and deepseek_usernames is not None:
        if username in deepseek_usernames:
            return "deepseek"
        if username in qwen_usernames:
            return "qwen"
        return default

    if root is not None:
        for name in login_history_enabled_upstreams():
            if username in load_upstream_usernames(name, root):
                return name
    return default


def _login_entry(value: Any) -> Dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _split_nested_logins(logins_root: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {upstream: {} for upstream in KNOWN_UPSTREAMS}
    for upstream in KNOWN_UPSTREAMS:
        raw = logins_root.get(upstream)
        if not isinstance(raw, dict):
            continue
        for username, entry in raw.items():
            parsed = _login_entry(entry)
            if parsed is not None:
                buckets[upstream][str(username)] = parsed
    return buckets


def _split_flat_logins(
    logins_root: Dict[str, Any],
    *,
    qwen_usernames: Set[str],
    deepseek_usernames: Set[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {upstream: {} for upstream in KNOWN_UPSTREAMS}
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
    return buckets


def split_login_history_logins(
    logins_root: Dict[str, Any],
    *,
    qwen_usernames: Set[str],
    deepseek_usernames: Set[str],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    nested = all(
        isinstance(logins_root.get(up), dict) for up in KNOWN_UPSTREAMS if up in logins_root
    )
    if nested and any(up in logins_root for up in KNOWN_UPSTREAMS):
        return _split_nested_logins(logins_root)
    return _split_flat_logins(
        logins_root,
        qwen_usernames=qwen_usernames,
        deepseek_usernames=deepseek_usernames,
    )


def migrate_login_history_upstream(
    upstream: str,
    root: Path,
    *,
    archive_unified: bool = False,
) -> bool:
    key = upstream.strip().lower()
    if not persist_attr(key, "LOGIN_HISTORY_ENABLED", False):
        return False

    dest = login_history_path(key, root)
    if dest.is_file():
        return False

    unified = unified_login_history_path(root)
    if not unified.is_file():
        return False

    data = read_json(unified)
    if not isinstance(data, dict):
        return False
    logins_root = data.get("logins")
    if not isinstance(logins_root, dict):
        return False

    username_sets = _username_sets(root)
    buckets = split_login_history_logins(
        logins_root,
        qwen_usernames=username_sets.get("qwen", set()),
        deepseek_usernames=username_sets.get("deepseek", set()),
    )
    bucket = buckets.get(key, {})
    if not bucket and not persist_attr(key, "ALLOWS_EMPTY_LOGIN_BUCKET", False):
        return False

    payload = {
        "updated_at": data.get("updated_at") or format_utc8(time.time()),
        "logins": bucket,
    }
    write_json(dest, payload)
    logger.info("已迁移登录历史 [%s] → %s (%d 条)", key, dest, len(bucket))

    if archive_unified:
        maybe_archive_unified_login_history(root)
    return True


def maybe_archive_unified_login_history(root: Path) -> None:
    unified = unified_login_history_path(root)
    if not unified.is_file():
        return
    peers = login_history_enabled_upstreams()
    if peers and all(login_history_path(up, root).is_file() for up in peers):
        archive_file(unified)
