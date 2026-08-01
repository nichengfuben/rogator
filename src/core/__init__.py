from __future__ import annotations

from core.dispatch import select_upstream
from core.registry import get_registry, get_upstream, load_upstreams

__all__ = [
    "get_registry",
    "get_upstream",
    "load_upstreams",
    "select_upstream",
]
