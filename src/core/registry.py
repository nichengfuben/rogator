from __future__ import annotations

"""Discover and hold enabled upstream modules."""

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import upstream as upstream_pkg

_CAP_KEYS = frozenset({
    "chat", "vision", "search", "tools", "native_tools",
    "count_tokens", "image_gen", "tts",
})
_PLATFORM_CAPS = frozenset({"thinking", "tools", "native_tools"})


@dataclass
class UpstreamModule:
    name: str
    capabilities: Dict[str, bool]
    create_client: Callable[..., Any]
    module: Any


@dataclass
class UpstreamRegistry:
    modules: Dict[str, UpstreamModule] = field(default_factory=dict)
    enabled: Set[str] = field(default_factory=set)

    def get(self, name: str) -> UpstreamModule:
        return self.modules[name]

    def names(self) -> List[str]:
        return list(self.modules.keys())

    def merged_capabilities(self) -> Dict[str, bool]:
        merged: Dict[str, bool] = {k: False for k in _CAP_KEYS}
        for mod in self.modules.values():
            for key, val in mod.capabilities.items():
                if key in _PLATFORM_CAPS:
                    continue
                if key in _CAP_KEYS and val:
                    merged[key] = True
        for key in _PLATFORM_CAPS:
            merged[key] = True
        return merged


_REGISTRY: Optional[UpstreamRegistry] = None


def _normalize_caps(raw: Dict[str, bool]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for key, val in raw.items():
        if key in _PLATFORM_CAPS:
            continue
        if key in _CAP_KEYS:
            out[key] = bool(val)
    return out


def _load_upstream_enabled() -> Set[str]:
    from server.config import CONFIG

    return set(CONFIG.upstream_enabled)


def load_upstreams() -> UpstreamRegistry:
    global _REGISTRY
    modules: Dict[str, UpstreamModule] = {}
    enabled_set = _load_upstream_enabled()
    for info in pkgutil.iter_modules(upstream_pkg.__path__, upstream_pkg.__name__ + "."):
        if not info.ispkg:
            continue
        name = info.name.rsplit(".", 1)[-1]
        if name.startswith("_"):
            continue
        mod = importlib.import_module(info.name)
        caps = _normalize_caps(dict(getattr(mod, "CAPABILITIES", {}) or {}))
        create = getattr(mod, "create_client", None)
        if not callable(create):
            raise RuntimeError(f"upstream {name} missing create_client()")
        uname = str(getattr(mod, "NAME", name))
        if uname.lower() not in enabled_set:
            continue
        modules[uname] = UpstreamModule(
            name=uname, capabilities=caps, create_client=create, module=mod
        )
    _REGISTRY = UpstreamRegistry(modules=modules, enabled=enabled_set)
    return _REGISTRY


def get_registry() -> UpstreamRegistry:
    if _REGISTRY is None:
        return load_upstreams()
    return _REGISTRY


def get_upstream(name: str) -> UpstreamModule:
    return get_registry().get(name)
