from __future__ import annotations

"""配置加载：用户 config.toml 覆盖 template/config.toml，不使用代码内置默认值。"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from server.config.files import (
    ensure_user_config_file,
    overlay_user_config,
    template_config_path,
)

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:
    import tomli as _toml_loader

from server.config.files import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"


def _loads_toml(text: str) -> Dict[str, Any]:
    """解析 TOML 文本；3.11+ 用 stdlib tomllib，否则 tomli。"""
    if sys.version_info >= (3, 11):
        return _toml_loader.loads(text)
    return _toml_loader.loads(text.encode("utf-8"))


@dataclass(frozen=True)
class AppConfig:
    port: int
    host: str
    prelogin: int
    max_retry_on_error: int
    max_concurrent: int
    max_queue_size: int
    qwen_send_max_chars: int
    model_context_length: int
    send_full_prompt: bool
    client_max_body_bytes: int
    create_chat_timeout: float
    request_total_timeout: float
    login_timeout: float
    prelogin_timeout: float
    record_prompt: bool
    print_prompt: bool
    record_response: bool
    log_level: str
    log_to_file: bool
    log_name: str
    log_color: bool
    access_log: bool


def resolve_log_path(path: str, *, project_root: Path | None = None) -> Path:
    """将相对路径解析为绝对路径（兼容旧配置）。"""
    root = project_root or PROJECT_ROOT
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p


def _deep_get(data: Dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _require_get(data: Dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    seen: list[str] = []
    for key in keys:
        seen.append(key)
        if not isinstance(cur, dict) or key not in cur:
            dotted = ".".join(seen)
            raise ValueError(
                f"配置项缺失: {dotted}（请检查 template/config.toml 与用户 config.toml）"
            )
        cur = cur[key]
    return cur


def _resolve_log_field(raw: Dict[str, Any], *keys: str) -> Any:
    """读取 debug.*；兼容旧版 logging.*（overlay 后仍可能只写在 logging 节）。"""
    val = _deep_get(raw, "debug", *keys)
    if val is not None:
        return val
    return _require_get(raw, "logging", *keys)


def _build_app_config(raw: Dict[str, Any]) -> AppConfig:
    return AppConfig(
        port=int(_require_get(raw, "server", "port")),
        host=str(_require_get(raw, "server", "host")),
        prelogin=int(_require_get(raw, "server", "prelogin")),
        max_retry_on_error=int(_require_get(raw, "retry", "max_retry_on_error")),
        max_concurrent=int(_require_get(raw, "limits", "max_concurrent")),
        max_queue_size=int(_require_get(raw, "limits", "max_queue_size")),
        qwen_send_max_chars=int(_require_get(raw, "limits", "qwen_send_max_chars")),
        model_context_length=int(_require_get(raw, "limits", "model_context_length")),
        send_full_prompt=bool(_require_get(raw, "limits", "send_full_prompt")),
        client_max_body_bytes=int(_require_get(raw, "limits", "client_max_body_bytes")),
        request_total_timeout=float(_require_get(raw, "timeout", "request_total")),
        create_chat_timeout=float(_require_get(raw, "timeout", "create_chat")),
        login_timeout=float(_require_get(raw, "timeout", "login")),
        prelogin_timeout=float(_require_get(raw, "timeout", "prelogin")),
        record_prompt=bool(_require_get(raw, "fncall", "record_prompt")),
        print_prompt=bool(_require_get(raw, "fncall", "print_prompt")),
        record_response=bool(_require_get(raw, "fncall", "record_response")),
        log_level=str(_resolve_log_field(raw, "level")).upper(),
        log_to_file=bool(_resolve_log_field(raw, "log_to_file")),
        log_name=str(_resolve_log_field(raw, "log_name")),
        log_color=bool(_resolve_log_field(raw, "color")),
        access_log=bool(_resolve_log_field(raw, "access_log")),
    )


def _read_config_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    try:
        return _loads_toml(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"无法解析配置文件 {path}: {exc}") from exc


def load_config(
    path: Path | None = None,
    *,
    template_path: Path | None = None,
) -> AppConfig:
    """加载配置：template 为底，用户 config 覆盖；缺失项取自 template，不用代码默认值。"""
    user_path = path if path is not None else ensure_user_config_file()
    tpl_path = template_path if template_path is not None else template_config_path()
    template_raw = _read_config_file(tpl_path)
    user_raw = _read_config_file(user_path)
    return _build_app_config(overlay_user_config(template_raw, user_raw))


# 模块级单例，启动时加载
CONFIG: AppConfig = load_config()
