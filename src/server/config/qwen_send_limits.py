from __future__ import annotations

"""Qwen 按模型 wire 字符发送上限（probe 实测）；未配置则回退全局 limits.qwen_send_max_chars。"""

import sys
from pathlib import Path
from typing import Dict, Final, Optional

from server.config import CONFIG
from server.config.files import upstream_user_dir

if sys.version_info >= (3, 11):
    import tomllib as _toml_loader
else:
    import tomli as _toml_loader

_LIMITS_FILENAME: Final[str] = "model_send_limits.toml"
_limits_cache: Dict[str, int] | None = None
_limits_mtime: float = 0.0


def model_send_limits_path() -> Path:
    return upstream_user_dir("qwen") / _LIMITS_FILENAME


def _read_limits_file(path: Path) -> Dict[str, int]:
    raw = _toml_loader.loads(path.read_text(encoding="utf-8"))
    section = raw.get("models") or {}
    if not isinstance(section, dict):
        return {}
    out: Dict[str, int] = {}
    for key, val in section.items():
        name = str(key).strip()
        if not name:
            continue
        out[name] = int(val)
    return out


def load_model_send_limits(*, force: bool = False) -> Dict[str, int]:
    """读取 model_send_limits.toml；按 mtime 缓存。"""
    global _limits_cache, _limits_mtime
    path = model_send_limits_path()
    if not path.is_file():
        _limits_cache = {}
        _limits_mtime = 0.0
        return {}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _limits_cache or {}
    if not force and _limits_cache is not None and mtime == _limits_mtime:
        return _limits_cache
    _limits_cache = _canonicalize_limit_keys(_read_limits_file(path))
    _limits_mtime = mtime
    return _limits_cache


def invalidate_model_send_limits_cache() -> None:
    global _limits_cache
    _limits_cache = None


def _to_internal_id(model: str) -> str:
    try:
        from server.config.files import MODEL_REGISTRY_FILE
        from server.model.model_registry import load_model_registry

        reg = load_model_registry(MODEL_REGISTRY_FILE)
        if model in reg.by_internal:
            return model
        if model in reg.by_external:
            return reg.by_external[model].internal_id
    except Exception:
        pass
    return model


def _canonicalize_limit_keys(raw: Dict[str, int]) -> Dict[str, int]:
    """加载时把误写的 external_id 映射为 internal_id；存取均以 internal 为准。"""
    try:
        from server.config.files import MODEL_REGISTRY_FILE
        from server.model.model_registry import load_model_registry

        reg = load_model_registry(MODEL_REGISTRY_FILE)
    except Exception:
        return raw
    out: Dict[str, int] = {}
    for key, val in raw.items():
        if key in reg.by_internal:
            out[key] = val
        elif key in reg.by_external:
            out[reg.by_external[key].internal_id] = val
        else:
            out[key] = val
    return out


def resolve_qwen_send_max_chars(
    model: str,
    *,
    fallback: Optional[int] = None,
) -> int:
    """按 internal_id 查表；无条目用 fallback 或 CONFIG.qwen_send_max_chars。"""
    fb = int(fallback if fallback is not None else CONFIG.qwen_send_max_chars)
    limits = load_model_send_limits()
    if not limits:
        return fb
    internal = _to_internal_id(model)
    if internal in limits:
        return limits[internal]
    return fb


def effective_send_max_chars(
    state: object,
    model: Optional[str],
    *,
    fallback: Optional[int] = None,
) -> int:
    """运行时有效上限：PayloadTooLarge 减半 override > 模型表 > 全局 fallback。"""
    if model:
        overrides = getattr(state, "_send_limit_overrides", None) or {}
        if model in overrides:
            return int(overrides[model])
    if model:
        return resolve_qwen_send_max_chars(model, fallback=fallback)
    splitter = getattr(state, "splitter", None)
    if splitter is not None:
        return int(getattr(splitter, "max_chars", 0) or fb_fallback(fallback))
    return fb_fallback(fallback)


def fb_fallback(fallback: Optional[int]) -> int:
    return int(fallback if fallback is not None else CONFIG.qwen_send_max_chars)
