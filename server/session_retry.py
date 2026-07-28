from __future__ import annotations

"""请求级换号重试：捕获 TokenExpiredError（含限流）并切换 session。"""

import asyncio
import logging
import re
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, TypeVar

from server.config import CONFIG
from server.formats import PayloadTooLargeError, TokenExpiredError, UpstreamTimeoutError
from server.session_store import mask_username

logger = logging.getLogger("rogator")

T = TypeVar("T")

_RATE_LIMIT_HOURS_RE = re.compile(r'"num"\s*:\s*(\d+)')


def is_retryable_error(exc: BaseException) -> bool:
    return isinstance(exc, (TokenExpiredError, UpstreamTimeoutError))


def _handle_upstream_timeout_retry(
    req_id: str,
    exc: UpstreamTimeoutError,
    *,
    retries: int,
    limit: int,
) -> None:
    if retries > limit:
        logger.error(
            "Retry exhausted for %s after %d attempt(s) (max_retry_on_error=%d): %s",
            req_id, retries, limit, exc,
        )
        raise exc
    logger.warning(
        "Upstream timeout for %s (retry %d/%d): %s",
        req_id, retries, limit, exc,
    )


def parse_rate_limit_block_seconds(message: str) -> float:
    """从限流响应解析封禁时长（秒），默认 24h。"""
    m = _RATE_LIMIT_HOURS_RE.search(message)
    if m:
        return float(int(m.group(1)) * 3600)
    return 86400.0


async def run_with_session_retry(
    req_id: str,
    state: Any,
    func: Callable[[], Awaitable[T]],
    *,
    max_retry: Optional[int] = None,
) -> T:
    """非流式换号重试。"""
    limit = CONFIG.max_retry_on_error if max_retry is None else max_retry
    retries = 0
    last_error: Optional[Exception] = None

    while True:
        try:
            return await func()
        except TokenExpiredError as e:
            last_error = e
            old_name = state.client.current_session_username
            block_seconds = parse_rate_limit_block_seconds(str(e))
            if old_name and "Rate limited" in str(e):
                state.client.block_account(old_name, block_seconds)
            new_session = await state.client.switch_to_next(exclude_username=old_name)
            retries += 1
            if new_session is None or retries > limit:
                logger.error(
                    "Retry exhausted for %s after %d attempt(s) (max_retry_on_error=%d): %s",
                    req_id, retries, limit, e,
                )
                raise
            logger.warning(
                "Session error, switching account for %s (retry %d/%d): "
                "old=%s error_type=%s new=%s",
                req_id, retries, limit,
                mask_username(old_name or ""), type(e).__name__,
                mask_username(new_session.username),
            )
        except PayloadTooLargeError as e:
            if CONFIG.send_full_prompt:
                raise
            if state.splitter.max_chars <= 50000:
                raise
            state.splitter.max_chars = max(50000, state.splitter.max_chars // 2)
            retries += 1
            if retries > 1:
                raise
            logger.warning(
                "Payload too large for %s, reducing send limit to %d and retrying: %s",
                req_id, state.splitter.max_chars, e,
            )
        except UpstreamTimeoutError as e:
            retries += 1
            _handle_upstream_timeout_retry(req_id, e, retries=retries, limit=limit)
        except Exception:
            raise


async def _close_async_generator(
    agen: Optional[AsyncGenerator[dict, None]],
) -> None:
    """显式关闭嵌套 async generator，避免 shutdown/断连时 aclose 未 await。"""
    if agen is None:
        return
    aclose = getattr(agen, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except (GeneratorExit, asyncio.CancelledError):
        raise
    except StopAsyncIteration:
        return
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "already running" in msg or "cannot reuse" in msg:
            return
        raise
    except Exception:
        logger.debug("async generator aclose failed", exc_info=True)


async def stream_with_session_retry(
    req_id: str,
    state: Any,
    make_stream: Callable[[], AsyncGenerator[dict, None]],
    *,
    max_retry: Optional[int] = None,
) -> AsyncGenerator[dict, None]:
    """流式换号重试：失败则整段重开。"""
    limit = CONFIG.max_retry_on_error if max_retry is None else max_retry
    retries = 0
    inner: Optional[AsyncGenerator[dict, None]] = None

    try:
        while True:
            inner = make_stream()
            try:
                async for event in inner:
                    yield event
                return
            except asyncio.CancelledError:
                await _close_async_generator(inner)
                inner = None
                raise
            except TokenExpiredError as e:
                await _close_async_generator(inner)
                inner = None
                old_name = state.client.current_session_username
                block_seconds = parse_rate_limit_block_seconds(str(e))
                if old_name and "Rate limited" in str(e):
                    state.client.block_account(old_name, block_seconds)
                new_session = await state.client.switch_to_next(exclude_username=old_name)
                retries += 1
                if new_session is None or retries > limit:
                    logger.error(
                        "Retry exhausted for %s after %d attempt(s) (max_retry_on_error=%d): %s",
                        req_id, retries, limit, e,
                    )
                    raise
                logger.warning(
                    "Session error, switching account for %s (retry %d/%d): "
                    "old=%s error_type=%s new=%s",
                    req_id, retries, limit,
                    mask_username(old_name or ""), type(e).__name__,
                    mask_username(new_session.username),
                )
            except PayloadTooLargeError as e:
                await _close_async_generator(inner)
                inner = None
                if CONFIG.send_full_prompt:
                    raise
                if state.splitter.max_chars <= 50000:
                    raise
                state.splitter.max_chars = max(50000, state.splitter.max_chars // 2)
                retries += 1
                if retries > 1:
                    raise
                logger.warning(
                    "Payload too large for %s, reducing send limit to %d and retrying: %s",
                    req_id, state.splitter.max_chars, e,
                )
            except UpstreamTimeoutError as e:
                await _close_async_generator(inner)
                inner = None
                retries += 1
                _handle_upstream_timeout_retry(req_id, e, retries=retries, limit=limit)
            except Exception:
                await _close_async_generator(inner)
                inner = None
                raise
    finally:
        await _close_async_generator(inner)
