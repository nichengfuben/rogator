from __future__ import annotations

"""阻塞/CPU 工作 offload 到线程池，并用 CapacityLimiter 限流。"""

import asyncio
from functools import partial
from typing import Any, Callable, Optional, TypeVar

from anyio import CapacityLimiter
from anyio import to_thread

T = TypeVar("T")

_fireye_limiter = CapacityLimiter(4)
_io_limiter = CapacityLimiter(8)
_pow_limiter = CapacityLimiter(2)


def fireye_limiter() -> CapacityLimiter:
    return _fireye_limiter


def io_limiter() -> CapacityLimiter:
    return _io_limiter


def pow_limiter() -> CapacityLimiter:
    return _pow_limiter


async def run_blocking(
    func: Callable[..., T],
    /,
    *args: Any,
    limiter: Optional[CapacityLimiter] = None,
    **kwargs: Any,
) -> T:
    cap = limiter or _io_limiter
    async with cap:
        if kwargs:
            bound = partial(func, *args, **kwargs)
            return await to_thread.run_sync(bound)
        return await to_thread.run_sync(func, *args)


async def cancel_task(task: asyncio.Task[Any], *, timeout: float = 30.0) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
