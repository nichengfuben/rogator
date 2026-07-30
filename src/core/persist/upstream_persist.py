from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("rogator")


def load_persist_module(upstream: str) -> Any:
    name = upstream.strip().lower()
    try:
        pkg = importlib.import_module(f"upstream.{name}")
        module_path = getattr(pkg, "PERSIST_MODULE", f"upstream.{name}.persist")
        return importlib.import_module(module_path)
    except ImportError:
        return importlib.import_module(f"upstream.{name}.persist")


def persist_attr(upstream: str, key: str, default: Any = None) -> Any:
    try:
        return getattr(load_persist_module(upstream), key, default)
    except ImportError:
        return default


def call_persist(
    upstream: str,
    fn_name: str,
    *args: Any,
    default: Any = False,
    **kwargs: Any,
) -> Any:
    fn: Optional[Callable[..., Any]] = persist_attr(upstream, fn_name)
    if not callable(fn):
        return default
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.debug("persist.%s(%s) failed: %s", fn_name, upstream, exc)
        return default


def login_history_enabled_upstreams() -> tuple[str, ...]:
    from core.persist.paths import KNOWN_UPSTREAMS

    out: list[str] = []
    for name in KNOWN_UPSTREAMS:
        if persist_attr(name, "LOGIN_HISTORY_ENABLED", False):
            out.append(name)
    return tuple(out)
