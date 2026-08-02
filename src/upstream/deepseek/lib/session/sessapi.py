from __future__ import annotations

# src/platforms/deepseek/core/sessionapi.py
"""DeepSeek 会话管理 API 封装（create / history / stop / feedback 等）"""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

from upstream.deepseek.lib.protocol.consts import DEFAULT_HOST
from upstream.deepseek.lib.protocol.headers import build_headers

logger = logging.getLogger(__name__)


async def create_session(
    session: aiohttp.ClientSession,
    token: str,
) -> str:
    """创建新的聊天会话。

    Args:
        session: aiohttp ClientSession。
        token: Bearer 令牌。

    Returns:
        新会话 ID。

    Raises:
        Exception: 创建失败时抛出。
    """
    headers = build_headers(token)
    async with session.post(
        "https://{}/api/v0/chat_session/create".format(DEFAULT_HOST),
        headers=headers,
        json={},
        timeout=aiohttp.ClientTimeout(total=30),
        ssl=False,
    ) as resp:
        if resp.status != 200:
            raise Exception("会话创建失败 HTTP {}".format(resp.status))
        data = await resp.json()
        biz_data = data.get("data", {}).get("biz_data", {})
        session_id = (
            biz_data.get("chat_session", {}).get("id")
            or biz_data.get("id", "")
        )
        if not session_id:
            raise Exception("会话 ID 为空: {}".format(data))
        return str(session_id)


async def get_session_list(
    session: aiohttp.ClientSession,
    token: str,
    *,
    pinned: bool = False,
    count: int = 0,
) -> List[Dict[str, Any]]:
    """获取历史会话列表（对齐 HAR：lte_cursor.pinned=false）。"""
    headers = build_headers(token)
    params: Dict[str, Any] = {"lte_cursor.pinned": str(pinned).lower()}
    if count > 0:
        params["count"] = count
    async with session.get(
        "https://{}/api/v0/chat_session/fetch_page".format(DEFAULT_HOST),
        headers=headers,
        params=params,
        timeout=aiohttp.ClientTimeout(total=30),
        ssl=False,
    ) as resp:
        if resp.status != 200:
            logger.warning("获取会话列表失败 HTTP %d", resp.status)
            return []
        data = await resp.json()
        return (
            data.get("data", {}).get("biz_data", {}).get("chat_sessions", [])
        )


async def get_history_messages(
    session: aiohttp.ClientSession,
    token: str,
    chat_session_id: str,
) -> List[Dict[str, Any]]:
    """获取对话历史消息。

    Args:
        session: aiohttp ClientSession。
        token: Bearer 令牌。
        chat_session_id: 会话 ID。

    Returns:
        消息字典列表。
    """
    headers = build_headers(token)
    async with session.get(
        "https://{}/api/v0/chat/history_messages".format(DEFAULT_HOST),
        headers=headers,
        params={"chat_session_id": chat_session_id},
        timeout=aiohttp.ClientTimeout(total=30),
        ssl=False,
    ) as resp:
        if resp.status != 200:
            logger.warning("获取历史消息失败 HTTP %d", resp.status)
            return []
        data = await resp.json()
        return (
            data.get("data", {}).get("biz_data", {}).get("chat_messages", [])
        )


async def stop_stream(
    session: aiohttp.ClientSession,
    token: str,
    chat_session_id: str,
    message_id: str,
) -> bool:
    """停止流式生成。

    Args:
        session: aiohttp ClientSession。
        token: Bearer 令牌。
        chat_session_id: 会话 ID。
        message_id: 消息 ID。

    Returns:
        是否成功。
    """
    headers = build_headers(token)
    try:
        async with session.post(
            "https://{}/api/v0/chat/stop_stream".format(DEFAULT_HOST),
            headers=headers,
            json={"chat_session_id": chat_session_id, "message_id": message_id},
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=False,
        ) as resp:
            if resp.status != 200:
                return False
            data = await resp.json()
            return data.get("data", {}).get("biz_code", -1) == 0
    except Exception as exc:
        logger.warning("stop_stream 失败: %s", exc)
        return False


async def message_feedback(
    session: aiohttp.ClientSession,
    token: str,
    chat_session_id: str,
    message_id: str,
    feedback_type: str,
    feedback_tag: str = "",
    description: str = "",
) -> bool:
    """反馈消息。

    Args:
        session: aiohttp ClientSession。
        token: Bearer 令牌。
        chat_session_id: 会话 ID。
        message_id: 消息 ID。
        feedback_type: GOOD 或 BAD。
        feedback_tag: 标签（可选）。
        description: 描述（可选）。

    Returns:
        是否成功。
    """
    headers = build_headers(token)
    payload: Dict[str, Any] = {
        "chat_session_id": chat_session_id,
        "message_id": message_id,
        "feedback_type": feedback_type,
    }
    if feedback_tag:
        payload["feedback_tag"] = feedback_tag
    if description:
        payload["description"] = description
    try:
        async with session.post(
            "https://{}/api/v0/chat/message_feedback".format(DEFAULT_HOST),
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=False,
        ) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("message_feedback 失败: %s", exc)
        return False


async def update_session_title(
    session: aiohttp.ClientSession,
    token: str,
    chat_session_id: str,
    title: str,
) -> bool:
    """更新会话标题。

    Args:
        session: aiohttp ClientSession。
        token: Bearer 令牌。
        chat_session_id: 会话 ID。
        title: 新标题。

    Returns:
        是否成功。
    """
    headers = build_headers(token)
    try:
        async with session.post(
            "https://{}/api/v0/chat_session/update_title".format(DEFAULT_HOST),
            headers=headers,
            json={"chat_session_id": chat_session_id, "title": title},
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=False,
        ) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("update_session_title 失败: %s", exc)
        return False


async def delete_session(
    session: aiohttp.ClientSession,
    token: str,
    chat_session_id: str,
) -> bool:
    """删除单个会话。

    Args:
        session: aiohttp ClientSession。
        token: Bearer 令牌。
        chat_session_id: 会话 ID。

    Returns:
        是否成功。
    """
    headers = build_headers(token)
    try:
        async with session.post(
            "https://{}/api/v0/chat_session/delete".format(DEFAULT_HOST),
            headers=headers,
            json={"chat_session_id": chat_session_id},
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=False,
        ) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("delete_session 失败: %s", exc)
        return False


async def delete_all_sessions(
    session: aiohttp.ClientSession,
    token: str,
) -> bool:
    """删除所有会话。

    Args:
        session: aiohttp ClientSession。
        token: Bearer 令牌。

    Returns:
        是否成功。
    """
    headers = build_headers(token)
    try:
        async with session.post(
            "https://{}/api/v0/chat_session/delete_all".format(DEFAULT_HOST),
            headers=headers,
            json={},
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=False,
        ) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("delete_all_sessions 失败: %s", exc)
        return False


async def update_pinned(
    session: aiohttp.ClientSession,
    token: str,
    chat_session_id: str,
    pinned: bool,
) -> bool:
    """置顶或取消置顶会话。

    Args:
        session: aiohttp ClientSession。
        token: Bearer 令牌。
        chat_session_id: 会话 ID。
        pinned: 是否置顶。

    Returns:
        是否成功。
    """
    headers = build_headers(token)
    try:
        async with session.post(
            "https://{}/api/v0/chat_session/update_pinned".format(DEFAULT_HOST),
            headers=headers,
            json={"chat_session_id": chat_session_id, "pinned": pinned},
            timeout=aiohttp.ClientTimeout(total=15),
            ssl=False,
        ) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("update_pinned 失败: %s", exc)
        return False
