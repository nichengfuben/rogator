from __future__ import annotations

"""Cursor 上游配置（统一 TOML：template/upstream/cursor.toml → configs/cursor.toml）。"""

import platform
import sys
from typing import Any, Dict, Optional

_TOKEN_SECTION = "token_service"

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
    return out


def token_service_config() -> Dict[str, Any]:
    """自建 Token 服务配置（拉号 / 用量轮询）。"""
    return dict(load_cursor_upstream_config().get(_TOKEN_SECTION) or {})


def cursor_agent_config() -> Dict[str, Any]:
    return dict(load_cursor_upstream_config().get("cursor") or {})


def _node_arch() -> str:
    """对齐 Node `os.arch()`。"""
    machine = (platform.machine() or "").lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("i386", "i686", "x86"):
        return "ia32"
    return machine or "unknown"


def cli_package_version(client_version: Optional[str] = None) -> str:
    """对齐 `version.ts` 的 `x()`：裸版本号（无 `cli-` 前缀）。"""
    raw = client_version
    if raw is None:
        raw = cursor_agent_config().get("client_version")
    text = str(raw or _DEFAULT["cursor"]["client_version"]).strip()
    if text.startswith("cli-"):
        text = text[4:]
    return text or "unknown"


def cursor_cli_user_agent(client_version: Optional[str] = None) -> str:
    """对齐 `user-agent.ts`：`Cursor-CLI/${version} (${platform} ${arch})`。"""
    return f"Cursor-CLI/{cli_package_version(client_version)} ({sys.platform} {_node_arch()})"
