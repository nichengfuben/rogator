from __future__ import annotations

"""Cursor 上游配置（统一 TOML：template/upstream/cursor.toml → configs/cursor.toml）。"""

from typing import Any, Dict, List

_DEFAULT: Dict[str, Any] = {
    "capabilities": {
        "chat": True,
        "vision": True,
        "search": False,
        "count_tokens": False,
        "image_gen": False,
        "tts": False,
    },
    "cursor": {
        "base_url": "https://agentn.global.api5.cursor.sh",
        "client_version": "cli-2026.07.23-e383d2b",
        "default_model": "composer-2.5-fast",
        "request_timeout": 300,
        "heartbeat_interval": 5,
        "timezone": "Asia/Shanghai",
    },
    "starcursor": {
        "base_url": "http://starcursor.airoe.cn",
        "api_keys": [],
        "switch_threshold": 80,
        "usage_threshold": 90.0,
        "poll_interval": 30,
        "status_refresh_interval": 30,
        "request_timeout": 15,
        "max_retry_per_pull": 3,
    },
    "models": {
        "default": "composer-2.5-fast",
        "mapping": {},
        "fallback": [],
    },
}


def load_cursor_upstream_config() -> Dict[str, Any]:
    try:
        from server.config.app_config import _load_upstream_toml
    except Exception:
        return dict(_DEFAULT)
    raw = _load_upstream_toml("cursor") or {}
    out = dict(_DEFAULT)
    for section in ("cursor", "starcursor", "models", "capabilities"):
        src = raw.get(section)
        if isinstance(src, dict):
            merged = dict(out.get(section) or {})
            merged.update(src)
            out[section] = merged
    return out


def starcursor_config() -> Dict[str, Any]:
    return dict(load_cursor_upstream_config().get("starcursor") or {})


def cursor_agent_config() -> Dict[str, Any]:
    return dict(load_cursor_upstream_config().get("cursor") or {})
