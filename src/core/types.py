from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ChatMessage(TypedDict, total=False):
    role: str
    content: Any


class CompletionChunk(TypedDict, total=False):
    type: str
    content: str
    extra: Dict[str, Any]


class UpstreamInfo(TypedDict, total=False):
    name: str
    capabilities: Dict[str, bool]
    models: List[str]


class UpstreamError(RuntimeError):
    """Generic upstream failure."""


class RateLimitError(UpstreamError):
    """Upstream rate limited."""


class AuthError(UpstreamError):
    """Upstream authentication failed."""


class ModelNotAvailableError(UpstreamError):
    """No upstream has the requested model (and capabilities)."""


class CapabilityError(UpstreamError):
    """No upstream satisfies requested capabilities."""
