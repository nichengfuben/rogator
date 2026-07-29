from __future__ import annotations

"""从 config/config.toml 加载运行时配置。"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from server.config_files import PROJECT_ROOT, ensure_user_config_file

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:
    import tomli as _toml_loader

LOG_DIR = PROJECT_ROOT / "logs"


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
    qwen_send_max_chars: int = 256_000
    model_context_length: int = 256_000
    send_full_prompt: bool = False
    client_max_body_bytes: int = 32 * 1024 * 1024
    create_chat_timeout: float = 15.0
    request_total_timeout: float = 600.0
    login_timeout: float = 30.0
    prelogin_timeout: float = 120.0
    record_prompt: bool = True
    print_prompt: bool = False
    record_response: bool = False
    log_level: str = "DEBUG"
    log_to_file: bool = True
    log_name: str = "rogator"
    log_color: bool = True
    access_log: bool = True


def resolve_log_path(path: str, *, project_root: Path | None = None) -> Path:
    """将相对路径解析为绝对路径（兼容旧配置）。"""
    root = project_root or PROJECT_ROOT
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p


def _deep_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _build_app_config(raw: Dict[str, Any]) -> AppConfig:
    return AppConfig(
        port=int(_deep_get(raw, "server", "port", default=8932)),
        host=str(_deep_get(raw, "server", "host", default="0.0.0.0")),
        prelogin=int(_deep_get(raw, "server", "prelogin", default=3)),
        max_retry_on_error=int(_deep_get(raw, "retry", "max_retry_on_error", default=3)),
        max_concurrent=int(_deep_get(raw, "limits", "max_concurrent", default=8)),
        max_queue_size=int(_deep_get(raw, "limits", "max_queue_size", default=1000)),
        qwen_send_max_chars=int(_deep_get(raw, "limits", "qwen_send_max_chars", default=256000)),
        model_context_length=int(
            _deep_get(raw, "limits", "model_context_length", default=256000)
        ),
        send_full_prompt=bool(_deep_get(raw, "limits", "send_full_prompt", default=False)),
        client_max_body_bytes=int(
            _deep_get(raw, "limits", "client_max_body_bytes", default=32 * 1024 * 1024)
        ),
        request_total_timeout=float(_deep_get(raw, "timeout", "request_total", default=600.0)),
        create_chat_timeout=float(_deep_get(raw, "timeout", "create_chat", default=15.0)),
        login_timeout=float(_deep_get(raw, "timeout", "login", default=30.0)),
        prelogin_timeout=float(_deep_get(raw, "timeout", "prelogin", default=120.0)),
        record_prompt=bool(_deep_get(raw, "fncall", "record_prompt", default=True)),
        print_prompt=bool(_deep_get(raw, "fncall", "print_prompt", default=False)),
        record_response=bool(_deep_get(raw, "fncall", "record_response", default=False)),
        log_level=str(
            _deep_get(raw, "debug", "level")
            or _deep_get(raw, "logging", "level", default="DEBUG")
        ).upper(),
        log_to_file=bool(
            _deep_get(raw, "debug", "log_to_file")
            if _deep_get(raw, "debug", "log_to_file") is not None
            else _deep_get(raw, "logging", "log_to_file", default=True)
        ),
        log_name=str(
            _deep_get(raw, "debug", "log_name")
            or _deep_get(raw, "logging", "log_name", default="rogator")
        ),
        log_color=bool(
            _deep_get(raw, "debug", "color")
            if _deep_get(raw, "debug", "color") is not None
            else _deep_get(raw, "logging", "color", default=True)
        ),
        access_log=bool(
            _deep_get(raw, "debug", "access_log")
            if _deep_get(raw, "debug", "access_log") is not None
            else _deep_get(raw, "logging", "access_log", default=True)
        ),
    )


def load_config(path: Path | None = None) -> AppConfig:
    """加载配置；缺失或解析失败直接抛错，不做模板合并。"""
    cfg_path = path if path is not None else ensure_user_config_file()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    try:
        raw = _loads_toml(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"无法解析配置文件 {cfg_path}: {exc}") from exc
    return _build_app_config(raw)


# 模块级单例，启动时加载
CONFIG: AppConfig = load_config()
