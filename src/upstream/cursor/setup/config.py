from __future__ import annotations

"""Cursor 上游配置（统一 TOML：template/upstream/cursor.toml → configs/cursor.toml）。"""

from typing import Any, Dict

# 配置节名：对外用 token_service；仍可读旧节 starcursor（兼容）
_TOKEN_SECTION = "token_service"
_TOKEN_SECTION_LEGACY = "starcursor"

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
    _TOKEN_SECTION: {
        # 自建 Token 拉取服务地址（须自行填写）
        "base_url": "",
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


def _merge_section(out: Dict[str, Any], raw: Dict[str, Any], name: str) -> None:
    src = raw.get(name)
    if isinstance(src, dict):
        merged = dict(out.get(name) or {})
        merged.update(src)
        out[name] = merged


def load_cursor_upstream_config() -> Dict[str, Any]:
    try:
        from server.config.app_config import _load_upstream_toml
    except Exception:
        return dict(_DEFAULT)
    raw = _load_upstream_toml("cursor") or {}
    out = dict(_DEFAULT)
    for section in ("cursor", _TOKEN_SECTION, "models", "capabilities"):
        _merge_section(out, raw, section)
    # 旧节名兼容：仅当新节未被用户写出时并入
    legacy = raw.get(_TOKEN_SECTION_LEGACY)
    if isinstance(legacy, dict):
        user_new = raw.get(_TOKEN_SECTION)
        if not isinstance(user_new, dict) or not user_new:
            merged = dict(out.get(_TOKEN_SECTION) or {})
            merged.update(legacy)
            out[_TOKEN_SECTION] = merged
    return out


def token_service_config() -> Dict[str, Any]:
    """自建 Token 服务配置（拉号 / 用量轮询）。"""
    return dict(load_cursor_upstream_config().get(_TOKEN_SECTION) or {})


def cursor_agent_config() -> Dict[str, Any]:
    return dict(load_cursor_upstream_config().get("cursor") or {})
