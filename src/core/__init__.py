from __future__ import annotations

from core.dispatch import create_client_for_request, select_upstream
from core.registry import get_registry, get_upstream, load_upstreams

__all__ = [
    "create_client_for_request",
    "get_registry",
    "get_upstream",
    "load_upstreams",
    "select_upstream",
]
