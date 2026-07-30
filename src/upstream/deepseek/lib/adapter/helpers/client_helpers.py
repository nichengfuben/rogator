from __future__ import annotations

"""DeepSeek 客户端请求构造辅助模块。

职责：
    承载单次请求生命周期中的上下文提取、HIF/PoW 获取、会话与请求头
    准备、payload/post 参数构造、初次响应流式解析等纯函数，供
    ``client.py`` 中的 :class:`DeepseekClient` facade 调用。拆分自
    ``client.py``，不改变任何现有行为。
"""

from typing import Any, AsyncGenerator, Dict, List, Tuple, Union

import aiohttp

from upstream.deepseek.lib.protocol.headers import build_headers
from upstream.deepseek.lib.protocol.payload import make_stream_id
from upstream.deepseek.lib.adapter.helpers.pmtutil import build_prompt, translate_chunk
from upstream.deepseek.lib.guard.pow import get_pow_response
from upstream.deepseek.lib.runtime.session.sessapi import create_session


def build_request_context(
    candidate: Any,
    messages: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    """从候选项与参数中提取本次请求所需的凭证与设置。

    Args:
        candidate: 候选项（含 token）。
        messages: 消息列表。
        model: 模型名。

    Returns:
        包含 token / username / prompt / model_type 的字典。
    """
    token = candidate.meta.get("token", "")
    username = candidate.meta.get("identifier", "")
    prompt = build_prompt(messages)
    return {
        "token": token,
        "username": username,
        "prompt": prompt,
        "model_type": "default",
    }


async def acquire_hif_and_pow(
    hif_managers: Dict[str, Any],
    pow_client: Any,
    session: aiohttp.ClientSession,
    username: str,
    token: str,
) -> Tuple[str, str, str]:
    """获取本次请求所需的 HIF 令牌与 PoW 响应。

    Args:
        hif_managers: username -> HifTokenManager 映射。
        pow_client: WasmPow 实例。
        session: 共享的 aiohttp ClientSession。
        username: 账号用户名。
        token: 账号 token。

    Returns:
        (hif_leim, hif_dliq, pow_response) 三元组。
    """
    mgr = hif_managers.get(username)
    hif_leim = ""
    hif_dliq = ""
    if mgr is not None:
        hif_leim, hif_dliq = await mgr.ensure_valid()

    pow_resp = ""
    if pow_client.available:
        pow_resp = await get_pow_response(
            session, token, pow_client, "/api/v0/chat/completion"
        )
    return hif_leim, hif_dliq, pow_resp


async def prepare_session(
    session: aiohttp.ClientSession,
    token: str,
    hif_leim: str,
    hif_dliq: str,
    pow_resp: str,
) -> Tuple[Any, Dict[str, str]]:
    """创建会话并构建本次请求所需的请求头。

    Args:
        session: 共享的 aiohttp ClientSession。
        token: 账号 token。
        hif_leim: HIF leim 令牌。
        hif_dliq: HIF dliq 令牌。
        pow_resp: PoW 响应。

    Returns:
        (session_id, req_headers) 二元组。
    """
    session_id = await create_session(session, token)
    req_headers = build_headers(
        token=token,
        session_id=session_id,
        hif_leim=hif_leim,
        hif_dliq=hif_dliq,
        pow_response=pow_resp,
    )
    return session_id, req_headers


def build_chat_payload(ctx: Dict[str, Any], session_id: Any) -> Dict[str, Any]:
    """根据请求上下文与会话 id 构建请求体。

    Args:
        ctx: ``build_request_context`` 返回的上下文字典。
        session_id: 已创建的会话 id。

    Returns:
        请求体字典。
    """
    return {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": ctx["model_type"],
        "prompt": ctx["prompt"],
        "ref_file_ids": [],
        "thinking_enabled": False,
        "search_enabled": False,
        "preempt": False,
        "client_stream_id": make_stream_id(),
    }


def build_post_kwargs(
    req_headers: Dict[str, str],
    payload: Dict[str, Any],
    proxy_override: Any,
    get_proxy_kwarg: Any,
) -> Dict[str, Any]:
    """构建传递给 ``session.post`` 的关键字参数。

    Args:
        req_headers: 请求头。
        payload: 请求体。
        proxy_override: 代理覆盖开关（None/True/False）。
        get_proxy_kwarg: 用于取得实际 proxy 值的可调用对象。

    Returns:
        ``session.post`` 关键字参数字典。
    """
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
    """准备本次请求所需的会话、请求体/请求头 kwargs 与流解析器。

    Args:
        session: 共享的 aiohttp ClientSession。
        ctx: ``build_request_context`` 返回的上下文字典。
        hif_leim: HIF leim 令牌。
        hif_dliq: HIF dliq 令牌。
        pow_resp: PoW 响应。
        proxy_override: 代理覆盖开关（None/True/False）。
        get_proxy_kwarg: 用于取得实际 proxy 值的可调用对象。
        stream_parser_cls: ``StreamParser`` 类。

    Returns:
        (session_id, post_kw, parser) 三元组。
    """
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
) -> Tuple[Dict[str, Any], str, str, Any, Dict[str, Any], Any]:
    """整合上下文提取、HIF/PoW 获取、会话准备与 post 参数构造。

    Returns:
        (ctx, session_id, hif_leim, hif_dliq, post_kw, parser) 六元组。
    """
    ctx = build_request_context(candidate, messages, model)
    token = ctx["token"]
    username = ctx["username"]

    hif_leim, hif_dliq, pow_resp = await acquire_hif_and_pow(
        hif_managers, pow_client, session, username, token
    )

    session_id, req_headers = await prepare_session(
        session, token, hif_leim, hif_dliq, pow_resp
    )
    payload = build_chat_payload(ctx, session_id)
    post_kw = build_post_kwargs(req_headers, payload, proxy_override, get_proxy_kwarg)
    parser = parser_cls(include_thinking=False)
    return ctx, session_id, hif_leim, hif_dliq, post_kw, parser


async def stream_initial_response(
    session: aiohttp.ClientSession,
    url: str,
    post_kw: Dict[str, Any],
    parser: Any,
    parse_sse_stream: Any,
    state: Dict[str, bool],
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """发起初次请求并流式解析响应。

    Args:
        session: 共享的 aiohttp ClientSession。
        url: 请求 url。
        post_kw: ``session.post`` 关键字参数。
        parser: 流解析器。
        parse_sse_stream: 用于解析 SSE 流的可调用对象（bound method）。
        state: 用于回传 needs_continue 标记的可变字典。

    Yields:
        str（文本增量）或 dict（thinking/usage）。
    """
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
