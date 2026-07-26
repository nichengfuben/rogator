from __future__ import annotations

"""从 config.toml 加载运行时配置。"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:
    import tomli as _toml_loader

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


def _loads_toml(text: str) -> Dict[str, Any]:
    """解析 TOML 文本；3.11+ 用 stdlib tomllib，否则 tomli。"""
    if sys.version_info >= (3, 11):
        return _toml_loader.loads(text)
    return _toml_loader.loads(text.encode("utf-8"))


@dataclass(frozen=True)
class AppConfig:
    port: int = 8932
    host: str = "0.0.0.0"
    prelogin: int = 3
    max_retry_on_error: int = 3
    max_concurrent: int = 8
    max_queue_size: int = 1000
    max_chars: int = 1024000
    request_total_timeout: float = 600.0
    login_timeout: float = 30.0
    prelogin_timeout: float = 120.0


def _deep_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or _CONFIG_PATH
    raw: Dict[str, Any] = {}
    if cfg_path.exists():
        raw = _loads_toml(cfg_path.read_text(encoding="utf-8"))

    return AppConfig(
        port=int(_deep_get(raw, "server", "port", default=8932)),
        host=str(_deep_get(raw, "server", "host", default="0.0.0.0")),
        prelogin=int(_deep_get(raw, "server", "prelogin", default=3)),
        max_retry_on_error=int(_deep_get(raw, "retry", "max_retry_on_error", default=3)),
        max_concurrent=int(_deep_get(raw, "limits", "max_concurrent", default=8)),
        max_queue_size=int(_deep_get(raw, "limits", "max_queue_size", default=1000)),
        max_chars=int(_deep_get(raw, "limits", "max_chars", default=1024000)),
        request_total_timeout=float(_deep_get(raw, "timeout", "request_total", default=600.0)),
        login_timeout=float(_deep_get(raw, "timeout", "login", default=30.0)),
        prelogin_timeout=float(_deep_get(raw, "timeout", "prelogin", default=120.0)),
    )


# 模块级单例，启动时加载
CONFIG: AppConfig = load_config()
