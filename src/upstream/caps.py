# -*- coding: utf-8 -*-
from __future__ import annotations

"""?? capabilities?defaults + TOML ?????"""

from typing import Dict, FrozenSet, Mapping


_DEFAULT_SKIP = frozenset({"thinking", "tools", "native_tools"})


def load_capabilities(
    upstream: str,
    defaults: Mapping[str, bool],
    *,
    skip: FrozenSet[str] | None = None,
) -> Dict[str, bool]:
    skip_keys = skip if skip is not None else _DEFAULT_SKIP
    overrides: Dict[str, bool] = {}
    try:
        from server.config.app_config import _load_upstream_toml
    except Exception:
        return dict(defaults)
    raw = _load_upstream_toml(upstream)
    caps = raw.get("capabilities") if isinstance(raw, dict) else None
    if not isinstance(caps, dict):
        return dict(defaults)
    for key, val in caps.items():
        if key in skip_keys or key not in defaults:
            continue
        overrides[str(key)] = bool(val)
    return {**defaults, **overrides}
