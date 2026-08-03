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
    pass


class RateLimitError(UpstreamError):
    pass


class AuthError(UpstreamError):
    pass


class ModelNotAvailableError(UpstreamError):
    pass


class CapabilityError(UpstreamError):
    pass
