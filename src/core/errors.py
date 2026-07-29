from __future__ import annotations

"""Shared upstream / platform errors."""


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
