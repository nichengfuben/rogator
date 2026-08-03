from __future__ import annotations

"""跨 Python 3.8–3.14、Win/Linux/macOS 的小兼容层。"""

import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, TypeVar

T = TypeVar("T")


def removeprefix(text: str, prefix: str) -> str:
    """``str.removeprefix``（3.9+）在 3.8 上的回退。"""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def removesuffix(text: str, suffix: str) -> str:
    """``str.removesuffix``（3.9+）在 3.8 上的回退。"""
    if suffix and text.endswith(suffix):
        return text[: -len(suffix)]
    return text


if sys.version_info >= (3, 10):
    from contextlib import aclosing
else:

    @asynccontextmanager
    async def aclosing(thing: T) -> AsyncIterator[T]:
        try:
            yield thing
        finally:
            aclose = getattr(thing, "aclose", None)
            if aclose is not None:
                await aclose()
