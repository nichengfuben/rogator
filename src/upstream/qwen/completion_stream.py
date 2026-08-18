from __future__ import annotations

"""Qwen chat completion SSE 流与 OpenAI 上游适配。"""

import asyncio
import re
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp

from echotools.base.logger import get_logger

from handlers.chat_request import apply_prompt_budget, prepare_injected_messages
from upstream.qwen.auth.http import map_connection_error
from upstream.qwen.chat.chat import (
    abort_upstream_on_cancel,
    handle_chat_error,
)
from upstream.qwen.chat.routes import BASE_URL, CHAT_PATH
from upstream.qwen.chat.upload.upstream_api import (
    reconnect_sse_events_with_retry,
    update_user_settings,
)
from core.transport.http import upstream_timeout
from server.formats import (
    BaxiaSmBlockedError,
    REQUEST_TOTAL_TIMEOUT,
    TokenExpiredError,
    UpstreamChatNotFoundError,
    UpstreamTimeoutError,
)
from server.model.model_thinking import ThinkingRoute
from upstream.qwen.chat.sse import iter_sse_events

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = get_logger("rogator")


async def _iter_qwen_sse_or_reconnect(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
    resp: aiohttp.ClientResponse,
    response_id_box: List[str],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    try:
        async for event in iter_sse_events(
            client, resp, session, response_id_out=response_id_box,
        ):
            yield event
    except UpstreamTimeoutError:
        rid = response_id_box[0] if response_id_box else ""
        if not rid:
            raise
        async for event in reconnect_sse_events_with_retry(
            client, session, chat_id, rid, cookies=cookies,
        ):
            yield event


def _note_sse_stats(stats: Dict[str, Any], event: Dict[str, Any]) -> None:
    now_ms = int(time.time() * 1000)
    if not stats["first_chunk_ms"]:
        stats["first_chunk_ms"] = now_ms
    stats["end_chunk_ms"] = now_ms
    content = event.get("content")
    if isinstance(content, str):
        stats["total_chars"] += len(content)


async def _stream_one_chat_post(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: Any,
    response_id_box: List[str],
    stats: Dict[str, Any],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    http = await client._ensure_http_session()
    async with http.post(url, json=payload, headers=headers, timeout=timeout) as resp:
        absorb_fn = getattr(client, "absorb_cookies_for_session", None)
        if callable(absorb_fn):
            absorb_fn(session, resp, binding=cookies)
        elif cookies is not None:
            from upstream.qwen.auth.crypto import absorb_response_cookies
            absorb_response_cookies(cookies, resp)
        if resp.status != 200:
            await handle_chat_error(client, resp, session)
        async for event in _iter_qwen_sse_or_reconnect(
            client, session, chat_id, resp, response_id_box, cookies=cookies,
        ):
            _note_sse_stats(stats, event)
            yield event


def _reraise_or_prepare_retry(exc: Exception, attempt: int) -> Exception:
    conn_err = map_connection_error(exc)
    if conn_err is None or attempt >= 2:
        if conn_err is not None:
            raise conn_err from exc
        raise
    return exc


async def _iter_post_chat_sse_attempts(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: Any,
    response_id_box: List[str],
    stats: Dict[str, Any],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    for attempt in range(1, 3):
        try:
            async for event in _stream_one_chat_post(
                client, session, chat_id, url, payload, headers, timeout,
                response_id_box, stats, cookies=cookies,
            ):
                yield event
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _reraise_or_prepare_retry(exc, attempt)
            await client.reset_http_transport()
            await asyncio.sleep(0.6 * attempt)


async def _report_post_chat_stream_stats(
    client: "QwenClient",
    session: "QwenSession",
    *,
    chat_id: str,
    model: str,
    request_id: str,
    response_id_box: List[str],
    stats: Dict[str, Any],
) -> None:
    from upstream.qwen.auth.report import report_streaming_statistics

    await report_streaming_statistics(
        client,
        session,
        chat_id=chat_id,
        model=model,
        request_id=request_id,
        response_id=response_id_box[0] if response_id_box else "",
        api_start_ms=stats["api_start_ms"],
        first_chunk_ms=stats["first_chunk_ms"],
        end_chunk_ms=stats["end_chunk_ms"] or int(time.time() * 1000),
        total_chars=stats["total_chars"],
        is_error=stats["is_error"],
    )


async def _post_chat_sse(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    response_id_box: List[str],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    from upstream.qwen.auth.report import report_completions_request_id

    url = f"{BASE_URL}{CHAT_PATH}?chat_id={chat_id}"
    timeout = upstream_timeout(REQUEST_TOTAL_TIMEOUT)
    request_id = str(headers.get("X-Request-Id") or headers.get("x-request-id") or "")
    model = str(payload.get("model") or "")
    await report_completions_request_id(
        client, session, request_id=request_id, chat_id=chat_id,
    )
    stats: Dict[str, Any] = {
        "api_start_ms": int(time.time() * 1000),
        "first_chunk_ms": 0,
        "end_chunk_ms": 0,
        "total_chars": 0,
        "is_error": False,
    }
    try:
        async for event in _iter_post_chat_sse_attempts(
            client, session, chat_id, url, payload, headers, timeout,
            response_id_box, stats, cookies=cookies,
        ):
            yield event
    except Exception:
        stats["is_error"] = True
        raise
    finally:
        await _report_post_chat_stream_stats(
            client, session, chat_id=chat_id, model=model, request_id=request_id,
            response_id_box=response_id_box, stats=stats,
        )


async def chat_completion_stream(
    client: "QwenClient",
    session: "QwenSession",
    chat_id: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    cookies: Optional[Dict[str, str]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    response_id_box: List[str] = []
    cancelled = False
    baxia_sm_retry = False
    try:
        async for event in _post_chat_sse(
            client, session, chat_id, payload, headers, response_id_box,
            cookies=cookies,
        ):
            yield event
    except BaxiaSmBlockedError:
        baxia_sm_retry = True
        raise
    except (asyncio.CancelledError, GeneratorExit):
        cancelled = True
        response_id = response_id_box[0] if response_id_box else ""
        await abort_upstream_on_cancel(client, session, chat_id, response_id)
        raise
    except Exception as exc:
        conn_err = map_connection_error(exc)
        if conn_err is not None:
            raise conn_err from exc
        raise
    finally:
        if not cancelled and not baxia_sm_retry:
            await client.cleanup_chat(session, chat_id)


_URL_RE = re.compile(r"https?://[^\s<>\"']+")


def _extract_page_urls(text: str) -> List[str]:
    if not text:
        return []
    seen: set[str] = set()
    urls: List[str] = []
    for match in _URL_RE.findall(text):
        url = match.rstrip(".,;:)")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


async def _upload_base64_images(client, session, image_uris: List[str]) -> List[Any]:
    files: List[Any] = []
    for uri in image_uris:
        try:
            _, image_obj = await client.upload_file_from_base64(session, uri)
            files.append(image_obj)
        except Exception as e:
            logger.warning("Image upload failed: %s", e)
    return files


async def _upload_remote_media(client, session, media_urls: List[str]) -> List[Any]:
    files: List[Any] = []
    for media_url in media_urls:
        try:
            _, media_obj = await client.upload_file_from_url(session, media_url)
            files.append(media_obj)
        except Exception as e:
            logger.debug("Remote media upload failed: %s", e)
    return files


async def _upload_text_attachment(
    client, session, filename: Optional[str], file_bytes: Optional[bytes],
) -> List[Any]:
    if not filename or not file_bytes:
        return []
    try:
        _, file_obj = await client.upload_file(session, file_bytes, filename)
        return [file_obj]
    except Exception as e:
        logger.warning("Upload failed: %s, sending truncated text without attachment", e)
        return []


async def _collect_uploaded_files(
    client, session, messages, image_uris, media_urls, filename, file_bytes, send_text: str,
) -> List[Any]:
    from upstream.qwen.chat.upload.upstream_api import parse_urls

    files: List[Any] = []
    files.extend(await _upload_base64_images(client, session, image_uris))
    files.extend(await _upload_remote_media(client, session, media_urls))
    files.extend(await _upload_text_attachment(client, session, filename, file_bytes))
    page_urls = _extract_page_urls(send_text)
    if page_urls:
        try:
            files.extend(await parse_urls(client, session, page_urls))
        except Exception as exc:
            logger.debug("parse_urls failed: %s", exc)
    return files


async def _prepare_stream(
    state,
    client,
    session,
    messages,
    model,
    tools,
    req_id,
    protocol_options=None,
    *,
    prompt_api: str = "openai",
) -> Tuple[List, List, ThinkingRoute]:
    injected, full_content, route = prepare_injected_messages(
        state, messages, tools, req_id, model, protocol_options, prompt_api,
    )

    image_uris = client.extract_base64_images(messages)
    media_urls = client.extract_remote_media_urls(messages)
    final_messages, send_text, filename, file_bytes = apply_prompt_budget(
        state, injected, full_content, use_file_split=True, model=model,
    )
    files = await _collect_uploaded_files(
        client, session, messages, image_uris, media_urls, filename, file_bytes, send_text,
    )
    final_messages[0]["content"] = send_text
    return final_messages, files, route


async def _retry_after_function_role(
    client: "QwenClient",
    session: "QwenSession",
    model: str,
    final_messages: List[Dict[str, Any]],
    uploaded_files: List[Any],
    route: ThinkingRoute,
    cookies: Dict[str, str],
    old_chat_id: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """function role 触发时：update settings → cleanup → recreate → restream。"""
    if old_chat_id:
        await abort_upstream_on_cancel(client, session, old_chat_id, "")
    try:
        ok = await update_user_settings(client, session)
        if not ok:
            raise RuntimeError("settings update failed")
        new_chat_id = await client.create_chat(session, model, cookies=cookies)
    except Exception:
        logger.warning("function role retry: setup failed, fallback")
        yield {"type": "thinking", "content": ""}
        raise asyncio.CancelledError("qwen function role detected")
    try:
        async for event in client.chat_completion(
            session, new_chat_id, final_messages, model, uploaded_files,
            qwen_thinking_enabled=route.qwen_native_enabled,
            qwen_thinking_mode=route.qwen_native_mode,
            cookies=cookies,
        ):
            if isinstance(event, dict) and event.get("type") == "_qwen_function_role":
                yield {"type": "thinking", "content": ""}
                raise asyncio.CancelledError("qwen function role retry failed")
            yield event
    except (asyncio.CancelledError, GeneratorExit):
        if new_chat_id:
            await abort_upstream_on_cancel(client, session, new_chat_id, "")
        raise
    except Exception:
        logger.warning("function role retry: restream failed, fallback")
        yield {"type": "thinking", "content": ""}
        raise asyncio.CancelledError("qwen function role retry failed")


async def _stream_openai_chat_inner(
    client: "QwenClient",
    session: "QwenSession",
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    state: Any,
    *,
    protocol_options: Optional[Dict[str, Any]] = None,
    prompt_api: str = "openai",
    files: Optional[List[Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """stream_openai_chat 的实际逻辑，剥离 async with 以降低嵌套深度。"""
    final_messages, uploaded_files, route = await _prepare_stream(
        state, client, session, messages, model, tools, req_id, protocol_options,
        prompt_api=prompt_api,
    )
    if files is not None:
        uploaded_files = files
    send_text = final_messages[0].get("content") or ""
    yield {"type": "prompt_meta", "prompt_chars": len(send_text)}
    chat_id = ""
    thinking_mode = route.qwen_native_mode or "Fast"
    cookies = client.begin_chat_cookies(session, thinking_mode=thinking_mode)
    try:
        chat_id = await client.create_chat(session, model, cookies=cookies)
        async for event in client.chat_completion(
            session, chat_id, final_messages, model, uploaded_files,
            qwen_thinking_enabled=route.qwen_native_enabled,
            qwen_thinking_mode=route.qwen_native_mode,
            cookies=cookies,
        ):
            if isinstance(event, dict) and event.get("type") == "_qwen_function_role":
                async for retry_event in _retry_after_function_role(
                    client, session, model, final_messages, uploaded_files,
                    route, cookies, chat_id,
                ):
                    yield retry_event
                return
            yield event
    except UpstreamChatNotFoundError:
        if chat_id:
            await client.cleanup_chat(session, chat_id)
        raise
    except (asyncio.CancelledError, GeneratorExit):
        if chat_id:
            await abort_upstream_on_cancel(client, session, chat_id, "")
        raise


async def stream_openai_chat(
    state: Any,
    client: Any,
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]],
    req_id: str,
    *,
    protocol_options: Optional[Dict[str, Any]] = None,
    prompt_api: str = "openai",
    files: Optional[List[Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    async with client.lease_valid_session() as session:
        if not session:
            raise TokenExpiredError("No valid session available")
        async for event in _stream_openai_chat_inner(
            client, session, messages, model, tools, req_id, state,
            protocol_options=protocol_options, prompt_api=prompt_api, files=files,
        ):
            yield event
