from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Optional, TypeVar

from server.config import CONFIG
from server.formats import (
    BaxiaSmBlockedError,
    PayloadTooLargeError,
    TokenExpiredError,
    UpstreamConnectionError,
    UpstreamTimeoutError,
    UpstreamWafBlockedError,
    UpstreamChatNotFoundError,
)
from upstream.qwen.chat.store import mask_username

if TYPE_CHECKING:
    from state import AppState

logger = logging.getLogger("rogator")

T = TypeVar("T")

_RATE_LIMIT_HOURS_RE = re.compile(r'"num"\s*:\s*(\d+)')


async def _reset_client_transport(client: Optional[Any]) -> None:
    reset = getattr(client, "reset_http_transport", None)
    if callable(reset):
        await reset()


def is_retryable_error(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            TokenExpiredError,
            BaxiaSmBlockedError,
            UpstreamWafBlockedError,
            UpstreamTimeoutError,
            UpstreamConnectionError,
            UpstreamChatNotFoundError,
        ),
    )


def _is_shutting_down(state: Any) -> bool:
    from state import AppState

    return isinstance(state, AppState) and state.is_shutting_down


def _raise_if_shutting_down(state: Any) -> None:
    if _is_shutting_down(state):
        raise asyncio.CancelledError("Shutting down")


def _log_retry_exhausted(req_id: str, retries: int, limit: int, exc: BaseException) -> None:
    msg = "Retry exhausted for %s after %d attempt(s) (max_retry_on_error=%d): %s"
    args = (req_id, retries, limit, exc)
    if limit == 0:
        logger.warning(msg, *args)
    elif isinstance(exc, BaxiaSmBlockedError):
        logger.debug(msg, *args)
    else:
        logger.error(msg, *args)


def _handle_upstream_timeout_retry(
    req_id: str,
    exc: UpstreamTimeoutError,
    *,
    retries: int,
    limit: int,
    state: Any | None = None,
) -> None:
    if state is not None and _is_shutting_down(state):
        raise asyncio.CancelledError("Shutting down") from exc
    if retries > limit:
        _log_retry_exhausted(req_id, retries, limit, exc)
        raise exc
    logger.warning(
        "Upstream timeout for %s (retry %d/%d): %s",
        req_id, retries, limit, exc,
    )


def _handle_upstream_connection_retry(
    req_id: str,
    exc: UpstreamConnectionError,
    *,
    retries: int,
    limit: int,
    state: Any | None = None,
) -> None:
    if state is not None and _is_shutting_down(state):
        raise asyncio.CancelledError("Shutting down") from exc
    if retries > limit:
        _log_retry_exhausted(req_id, retries, limit, exc)
        raise exc
    logger.warning(
        "Upstream connection failed for %s (retry %d/%d): %s",
        req_id, retries, limit, exc,
    )


def parse_rate_limit_block_seconds(message: str) -> float:
    """从限流响应解析封禁时长（秒），默认 24h。"""
    m = _RATE_LIMIT_HOURS_RE.search(message)
    if m:
        return float(int(m.group(1)) * 3600)
    return 86400.0


async def _switch_session_after_account_error(
    req_id: str,
    state: Any,
    exc: TokenExpiredError | BaxiaSmBlockedError | UpstreamWafBlockedError,
    *,
    retries: int,
    limit: int,
    client: Any | None = None,
) -> None:
    retry_client = client if client is not None else state.client
    switch = getattr(retry_client, "switch_to_next", None)
    if not callable(switch):
        if _is_shutting_down(state):
            raise asyncio.CancelledError("Shutting down") from exc
        _log_retry_exhausted(req_id, retries, limit, exc)
        raise exc
    old_name = getattr(retry_client, "current_session_username", None)
    block = getattr(retry_client, "block_account", None)
    block_seconds = parse_rate_limit_block_seconds(str(exc))
    if (
        isinstance(exc, TokenExpiredError)
        and old_name
        and block
        and callable(block)
        and "Rate limited" in str(exc)
    ):
        block(old_name, block_seconds)
    new_session = await switch(exclude_username=old_name)
    if new_session is None or retries > limit:
        if _is_shutting_down(state):
            raise asyncio.CancelledError("Shutting down") from exc
        _log_retry_exhausted(req_id, retries, limit, exc)
        raise exc
    log_fn = logger.debug if isinstance(exc, BaxiaSmBlockedError) else logger.warning
    log_fn(
        "Session error, switching account for %s (retry %d/%d): "
        "old=%s error_type=%s new=%s",
        req_id, retries, limit,
        mask_username(old_name or ""), type(exc).__name__,
        mask_username(new_session.username),
    )


def _shrink_payload_limit_or_raise(
    state: Any,
    req_id: str,
    exc: PayloadTooLargeError,
    retries: int,
    *,
    model: str | None = None,
) -> None:
    if CONFIG.send_full_prompt:
        raise exc
    from server.config.qwen_send_limits import effective_send_max_chars

    current = effective_send_max_chars(state, model)
    if current <= 50000:
        raise exc
    new_limit = max(50000, current // 2)
    if model:
        overrides = getattr(state, "_send_limit_overrides", None)
        if overrides is None:
            state._send_limit_overrides = {}
            overrides = state._send_limit_overrides
        overrides[model] = new_limit
    elif getattr(state, "splitter", None) is not None:
        state.splitter.max_chars = new_limit
    if retries > 1:
        raise exc
    logger.warning(
        "Payload too large for %s, reducing send limit to %d and retrying: %s",
        req_id, new_limit, exc,
    )


async def _handle_chat_not_found_retry(
    req_id: str,
    state: Any,
    exc: UpstreamChatNotFoundError,
    *,
    retries: int,
    limit: int,
    client: Any | None,
) -> int:
    """CHAT_NOT_FOUND 业务层重试：旧号作废后换号，不 reset transport。

    必须先 mark_invalid 再 switch：池选号按"最少在途+随机"，若旧号仍有效，
    重试可能又租回同一会话，导致 create_chat 重建后依旧 CHAT_NOT_FOUND。
    """
    retry_client = client if client is not None else state.client
    old_name = getattr(retry_client, "current_session_username", None)
    invalidate = getattr(retry_client, "mark_invalid_current", None)
    if callable(invalidate):
        invalidate()
    switch = getattr(retry_client, "switch_to_next", None)
    if callable(switch):
        new_session = await switch(exclude_username=old_name)
        if new_session is not None and retries <= limit:
            logger.warning(
                "CHAT_NOT_FOUND for %s (retry %d/%d), invalidated old and switched "
                "session: old=%s new=%s",
                req_id, retries, limit,
                mask_username(old_name or ""),
                mask_username(new_session.username),
            )
            return retries
    if retries <= limit:
        logger.warning(
            "CHAT_NOT_FOUND for %s (retry %d/%d), invalidated session %s",
            req_id, retries, limit,
            mask_username(old_name or ""),
        )
        return retries
    _log_retry_exhausted(req_id, retries, limit, exc)
    raise exc


async def _dispatch_session_retry(
    req_id: str,
    state: Any,
    exc: BaseException,
    *,
    retries: int,
    limit: int,
    client: Any | None,
    model: str | None = None,
) -> int:
    """分派可重试异常；返回更新后的 retries。不可重试则 raise。"""
    _raise_if_shutting_down(state)
    retries += 1
    if isinstance(exc, (TokenExpiredError, BaxiaSmBlockedError, UpstreamWafBlockedError)):
        if isinstance(exc, BaxiaSmBlockedError):
            try:
                from upstream.qwen.auth.crypto import reset_baxia_runtime

                reset_baxia_runtime()
            except Exception:
                pass
        await _switch_session_after_account_error(
            req_id, state, exc, retries=retries, limit=limit, client=client,
        )
        return retries
    if isinstance(exc, PayloadTooLargeError):
        _shrink_payload_limit_or_raise(state, req_id, exc, retries, model=model)
        return retries
    if isinstance(exc, UpstreamTimeoutError):
        await _reset_client_transport(client)
        _handle_upstream_timeout_retry(req_id, exc, retries=retries, limit=limit, state=state)
        return retries
    if isinstance(exc, UpstreamConnectionError):
        await _reset_client_transport(client)
        _handle_upstream_connection_retry(req_id, exc, retries=retries, limit=limit, state=state)
        return retries
    if isinstance(exc, UpstreamChatNotFoundError):
        return await _handle_chat_not_found_retry(
            req_id, state, exc, retries=retries, limit=limit, client=client,
        )
    raise exc


async def run_with_session_retry(
    req_id: str,
    state: Any,
    func: Callable[[], Awaitable[T]],
    *,
    max_retry: Optional[int] = None,
    client: Any | None = None,
) -> T:
    """非流式换号重试。"""
    limit = CONFIG.max_retry_on_error if max_retry is None else max_retry
    retries = 0
    retryable = (
        TokenExpiredError,
        BaxiaSmBlockedError,
        UpstreamWafBlockedError,
        PayloadTooLargeError,
        UpstreamTimeoutError,
        UpstreamConnectionError,
        UpstreamChatNotFoundError,
    )

    while True:
        try:
            return await func()
        except retryable as exc:
            retries = await _dispatch_session_retry(
                req_id, state, exc, retries=retries, limit=limit, client=client,
            )
        except Exception:
            raise


