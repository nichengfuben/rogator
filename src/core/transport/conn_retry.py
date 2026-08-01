from __future__ import annotations

"""上游连接级短重试（非换号）；与 server.retry.session_retry 职责分离。"""

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional, TypeVar

from server.formats import UpstreamTimeoutError, as_upstream_connection_error

logger = logging.getLogger("rogator")

T = TypeVar("T")


def reraise_transport_error(
    exc: BaseException,
    *,
    upstream: str,
    timeout_message: str = "",
) -> None:
    """将超时/连接类异常映射为 session_retry 可识别类型后抛出。"""
    if isinstance(exc, asyncio.TimeoutError):
        msg = timeout_message or "{0} upstream timeout".format(upstream)
        raise UpstreamTimeoutError(msg) from exc
    conn_err = as_upstream_connection_error(exc, upstream=upstream)
    if conn_err is not None:
        raise conn_err from exc
    raise exc


async def run_with_connection_retry(
    label: str,
    func: Callable[[], Awaitable[T]],
    *,
    upstream: str,
    attempts: int = 2,
    delay_seconds: float = 0.6,
    transport_owner: Optional[Any] = None,
) -> T:
    """对瞬时连接失败做少量重试，并在重试前 reset transport。"""
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            conn_err = as_upstream_connection_error(exc, upstream=upstream)
            if conn_err is None or attempt >= attempts:
                raise conn_err or exc
            reset = getattr(transport_owner, "reset_http_transport", None)
            if callable(reset):
                await reset()
            logger.warning(
                "%s %s connection failed (retry %d/%d): %s",
                upstream,
                label,
                attempt,
                attempts - 1,
                conn_err.message,
            )
            await asyncio.sleep(delay_seconds * attempt)
    raise RuntimeError("{0} {1} retry exhausted".format(upstream, label))
