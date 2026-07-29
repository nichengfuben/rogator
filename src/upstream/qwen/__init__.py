from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

NAME = "qwen"

_DEFAULT_CAPABILITIES: Dict[str, bool] = {
    "chat": True,
    "vision": True,
    "search": True,
    "tools": True,
    "native_tools": True,
    "count_tokens": True,
    "image_gen": True,
    "tts": True,
}


def _load_capability_overrides() -> Dict[str, bool]:
    try:
        from server.config.app_config import _load_upstream_toml
    except Exception:
        return {}
    raw = _load_upstream_toml("qwen")
    caps = raw.get("capabilities") if isinstance(raw, dict) else None
    if not isinstance(caps, dict):
        return {}
    out: Dict[str, bool] = {}
    for key, val in caps.items():
        if key == "thinking" or key not in _DEFAULT_CAPABILITIES:
            continue
        out[str(key)] = bool(val)
    return out


CAPABILITIES: Dict[str, bool] = {**_DEFAULT_CAPABILITIES, **_load_capability_overrides()}


def create_client(splitter: Any = None) -> Any:
    from upstream.qwen.client import QwenClient

    return QwenClient(splitter)
