from __future__ import annotations

"""DeepSeek 客户端单次请求的上下文、HIF/PoW、payload 构造辅助。"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp

from upstream.deepseek.lib.adapter.helpers.file_upload import resolve_model_type
from upstream.deepseek.lib.adapter.helpers.pmtutil import build_prompt, translate_chunk
from upstream.deepseek.lib.guard.pow import get_pow_response
from upstream.deepseek.lib.protocol.headers import build_headers
from upstream.deepseek.lib.session.sessapi import create_session


def build_request_context(
    candidate: Any,
    messages: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:

    token = candidate.meta.get("token", "")
    username = candidate.meta.get("identifier", "")
    prompt = build_prompt(messages)
    return {
        "token": token,
        "username": username,
        "prompt": prompt,
        "model_type": resolve_model_type(model),
    }


async def acquire_hif_and_pow(
    hif_managers: Dict[str, Any],
    pow_client: Any,
    session: aiohttp.ClientSession,
    username: str,
    token: str,
    *,
    target_path: str = "/api/v0/chat/completion",
) -> Tuple[str, str, str]:

    mgr = hif_managers.get(username)
    hif_leim = ""
    hif_dliq = ""
    if mgr is not None:
        hif_leim, hif_dliq = await mgr.ensure_valid()

    pow_resp = ""
    if pow_client.available:
        pow_resp = await get_pow_response(session, token, pow_client, target_path)
    return hif_leim, hif_dliq, pow_resp


async def prepare_session(
    session: aiohttp.ClientSession,
    token: str,
    hif_leim: str,
    hif_dliq: str,
    pow_resp: str,
) -> Tuple[Any, Dict[str, str]]:

    session_id = await create_session(session, token)
    req_headers = build_headers(
        token=token,
        session_id=session_id,
        hif_leim=hif_leim,
        hif_dliq=hif_dliq,
        pow_response=pow_resp,
    )
    return session_id, req_headers


def build_chat_payload(
    ctx: Dict[str, Any],
    session_id: Any,
    *,
    ref_file_ids: Optional[List[str]] = None,
    thinking_enabled: bool = False,
    search_enabled: bool = False,
) -> Dict[str, Any]:

    return {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": ctx["model_type"],
        "prompt": ctx["prompt"],
        "ref_file_ids": list(ref_file_ids or []),
        "thinking_enabled": bool(thinking_enabled),
        "search_enabled": bool(search_enabled),
        "action": None,
        "preempt": False,
    }


def build_post_kwargs(
    req_headers: Dict[str, str],
    payload: Dict[str, Any],
    proxy_override: Any,
    get_proxy_kwarg: Any,
) -> Dict[str, Any]:

    post_kw: Dict[str, Any] = {
        "headers": req_headers,
        "json": payload,
        "timeout": aiohttp.ClientTimeout(total=600),
        "ssl": False,
    }
    if proxy_override is not None:
        post_kw["proxy"] = get_proxy_kwarg()
    return post_kw


async def prepare_request(
    session: aiohttp.ClientSession,
    ctx: Dict[str, Any],
    hif_leim: str,
    hif_dliq: str,
    pow_resp: str,
    proxy_override: Any,
    get_proxy_kwarg: Any,
    stream_parser_cls: Any,
) -> Tuple[Any, Dict[str, Any], Any]:

    session_id, req_headers = await prepare_session(
        session, ctx["token"], hif_leim, hif_dliq, pow_resp
    )
    payload = build_chat_payload(ctx, session_id)
    parser = stream_parser_cls(include_thinking=False)
    post_kw = build_post_kwargs(req_headers, payload, proxy_override, get_proxy_kwarg)
    return session_id, post_kw, parser


async def prepare_full_request(
    session: aiohttp.ClientSession,
    hif_managers: Dict[str, Any],
    pow_client: Any,
    candidate: Any,
    messages: List[Dict[str, Any]],
    model: str,
    proxy_override: Any,
    get_proxy_kwarg: Any,
    parser_cls: Any,
    *,
    ref_file_ids: Optional[List[str]] = None,
    thinking_enabled: bool = False,
    search_enabled: bool = False,
    include_thinking: bool = False,
) -> Tuple[Dict[str, Any], str, str, Any, Dict[str, Any], Any]:

    ctx = build_request_context(candidate, messages, model)
    token = ctx["token"]
    username = ctx["username"]

    hif_leim, hif_dliq, pow_resp = await acquire_hif_and_pow(
        hif_managers, pow_client, session, username, token
    )

    session_id, req_headers = await prepare_session(
        session, token, hif_leim, hif_dliq, pow_resp
    )
    payload = build_chat_payload(
        ctx,
        session_id,
        ref_file_ids=ref_file_ids,
        thinking_enabled=thinking_enabled,
        search_enabled=search_enabled,
    )
    post_kw = build_post_kwargs(req_headers, payload, proxy_override, get_proxy_kwarg)
    parser = parser_cls(include_thinking=include_thinking)
    return ctx, session_id, hif_leim, hif_dliq, post_kw, parser


async def stream_initial_response(
    session: aiohttp.ClientSession,
    url: str,
    post_kw: Dict[str, Any],
    parser: Any,
    parse_sse_stream: Any,
    state: Dict[str, bool],
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """发起初次请求并流式解析响应。"""
    async with session.post(url, **post_kw) as resp:
        if resp.status != 200:
            raise Exception("聊天失败 HTTP {}".format(resp.status))

        async for chunk in parse_sse_stream(resp, parser):
            if chunk.get("needs_continue"):
                state["needs_continue"] = True
            elif chunk.get("type") not in ("event", "status"):
                translated = translate_chunk(chunk)
                if translated is not None:
                    yield translated
