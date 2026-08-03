from __future__ import annotations

"""配置加载：用户 config.toml 覆盖 template/config.toml，不使用代码内置默认值。"""

import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterator

from server.config.files import (
    PROJECT_ROOT,
    LEGACY_UPSTREAM_DEFAULTS_NAME,
    USER_CONFIG_DIR,
    USER_UPSTREAM_DIR,
    ensure_user_config_file,
    overlay_user_config,
    template_config_path,
    upstream_config_template_path,
    upstream_template_dir,
    upstream_user_config_path,
)

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:
    import tomli as _toml_loader

LOG_DIR = PROJECT_ROOT / "logs"


def _loads_toml(text: str) -> Dict[str, Any]:
    """解析 TOML 文本；3.11+ 用 stdlib tomllib，否则 tomli（均接受 str）。"""
    return _toml_loader.loads(text)


@dataclass(frozen=True)
class AppConfig:
    port: int
    host: str
    prelogin: int
    login_interval: float
    startup_force_kill_port: bool
    max_retry_on_error: int
    max_concurrent: int
    max_queue_size: int
    qwen_send_max_chars: int
    model_context_length: int
    send_full_prompt: bool
    client_max_body_bytes: int
    create_chat_timeout: float
    models_refresh_interval: float
    shutdown_wait_active_requests: float
    shutdown_total_timeout: float
    shutdown_hard_exit_timeout: float
    record_prompt: bool
    print_prompt: bool
    record_response: bool
    record_sse: bool
    log_level: str
    log_to_file: bool
    log_name: str
    log_color: bool
    access_log: bool
    upstream_enabled: tuple[str, ...]


class LiveConfig:
    """稳定 identity 的配置代理；``from X import CONFIG`` 后仍读到热更新快照。"""

    __slots__ = ("_current",)

    def __init__(self, current: AppConfig) -> None:
        object.__setattr__(self, "_current", current)

    def snapshot(self) -> AppConfig:
        return self._current

    def swap(self, new: AppConfig) -> AppConfig:
        old = self._current
        object.__setattr__(self, "_current", new)
        return old

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current, name)

    def __iter__(self) -> Iterator[str]:
        return (f.name for f in fields(self._current))

    def __repr__(self) -> str:
        return f"LiveConfig({self._current!r})"


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


def _resolve_fncall_record_flags(raw: Dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    record_all = bool(_require_get(raw, "fncall", "record_all"))
    if record_all:
        return True, True, True, True
    return False, False, False, False


def _build_app_config(raw: Dict[str, Any]) -> AppConfig:
    record_prompt, print_prompt, record_response, record_sse = _resolve_fncall_record_flags(raw)
    return AppConfig(
        port=int(_require_get(raw, "server", "port")),
        host=str(_require_get(raw, "server", "host")),
        prelogin=int(_require_get(raw, "server", "prelogin")),
        login_interval=float(_require_get(raw, "server", "login_interval")),
        startup_force_kill_port=bool(_require_get(raw, "server", "startup_force_kill_port")),
        max_retry_on_error=int(_require_get(raw, "retry", "max_retry_on_error")),
        max_concurrent=int(_require_get(raw, "limits", "max_concurrent")),
        max_queue_size=int(_require_get(raw, "limits", "max_queue_size")),
        qwen_send_max_chars=int(_require_get(raw, "limits", "qwen_send_max_chars")),
        model_context_length=int(_require_get(raw, "limits", "model_context_length")),
        send_full_prompt=bool(_require_get(raw, "limits", "send_full_prompt")),
        client_max_body_bytes=int(_require_get(raw, "limits", "client_max_body_bytes")),
        create_chat_timeout=float(_require_get(raw, "timeout", "create_chat")),
        models_refresh_interval=float(_require_get(raw, "models", "refresh_interval")),
        shutdown_wait_active_requests=float(
            _require_get(raw, "shutdown", "wait_active_requests")
        ),
        shutdown_total_timeout=float(_require_get(raw, "shutdown", "total_timeout")),
        shutdown_hard_exit_timeout=float(_require_get(raw, "shutdown", "hard_exit_timeout")),
        record_prompt=record_prompt,
        print_prompt=print_prompt,
        record_response=record_response,
        record_sse=record_sse,
        log_level=str(_resolve_log_field(raw, "level")).upper(),
        log_to_file=bool(_resolve_log_field(raw, "log_to_file")),
        log_name=str(_resolve_log_field(raw, "log_name")),
        log_color=bool(_resolve_log_field(raw, "color")),
        access_log=bool(_resolve_log_field(raw, "access_log")),
        upstream_enabled=_parse_upstream_enabled(
            _require_get(raw, "upstream", "enabled")
        ),
    )


def _parse_upstream_enabled(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError("upstream.enabled 必须为字符串数组")
    names = tuple(
        str(item).strip().lower()
        for item in raw
        if str(item).strip()
    )
    if not names:
        raise ValueError("upstream.enabled 不能为空")
    return names


def _read_config_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    try:
        return _loads_toml(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"无法解析配置文件 {path}: {exc}") from exc


def _overlay_if_exists(
    base: Dict[str, Any],
    path: Path,
) -> Dict[str, Any]:
    if not path.is_file():
        return base
    overlay = _read_config_file(path)
    return overlay_user_config(base, overlay) if base else overlay


def _load_upstream_toml(name: str) -> Dict[str, Any]:
    """template/upstream_config.toml → template/upstream/<name>.toml → config/upstream/<name>/config.toml。"""
    raw: Dict[str, Any] = {}
    raw = _overlay_if_exists(raw, upstream_config_template_path())
    legacy_defaults = USER_CONFIG_DIR / LEGACY_UPSTREAM_DEFAULTS_NAME
    if legacy_defaults.is_file():
        raw = _overlay_if_exists(raw, legacy_defaults)
    raw = _overlay_if_exists(raw, upstream_template_dir() / f"{name}.toml")
    raw = _overlay_if_exists(raw, upstream_user_config_path(name))
    # 兼容旧路径
    raw = _overlay_if_exists(raw, USER_UPSTREAM_DIR / f"{name}.toml")
    raw = _overlay_if_exists(raw, USER_UPSTREAM_DIR / name / f"{name}.toml")
    raw = _overlay_if_exists(raw, USER_CONFIG_DIR / f"{name}.toml")
    legacy_configs = PROJECT_ROOT / "configs" / f"{name}.toml"
    raw = _overlay_if_exists(raw, legacy_configs)
    return raw


def load_config(
    path: Path | None = None,
    *,
    template_path: Path | None = None,
) -> AppConfig:
    """加载配置：template 为底，用户 config 覆盖；上游 configs 合并 Qwen 限流等。"""
    user_path = path if path is not None else ensure_user_config_file()
    tpl_path = template_path if template_path is not None else template_config_path()
    template_raw = _read_config_file(tpl_path)
    user_raw = _read_config_file(user_path)
    merged = overlay_user_config(template_raw, user_raw)
    qwen_raw = _load_upstream_toml("qwen")
    if qwen_raw:
        limits = dict(merged.get("limits") or {})
        q_limits = qwen_raw.get("limits") or {}
        if isinstance(q_limits, dict):
            for key, val in q_limits.items():
                limits[key] = val
        merged["limits"] = limits
    return _build_app_config(merged)


# 模块级单例：identity 稳定，热重载只 swap 内部快照
CONFIG = LiveConfig(load_config())


def get_config() -> AppConfig:
    """当前不可变快照（供 ``dataclasses.replace`` / 对比用）。"""
    return CONFIG.snapshot()
