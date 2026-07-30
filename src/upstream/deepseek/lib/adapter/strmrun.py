from __future__ import annotations

# src/platforms/deepseek/core/adapter/streamrun.py
"""DeepSeek 流式请求辅助逻辑——SSE 解析、截断续写循环、usage 统计"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Union

import aiohttp

from upstream.deepseek.lib.protocol.consts import DEFAULT_HOST, MAX_CONTINUE
from upstream.deepseek.lib.protocol.headers import build_headers
from upstream.deepseek.lib.adapter.helpers.pmtutil import translate_chunk
from upstream.deepseek.lib.biz_error import raise_if_user_muted
from upstream.deepseek.lib.runtime.stream.strmpars import StreamParser

logger = logging.getLogger(__name__)


class _StreamRunMixin:
    """承载 SSE 解析、续写循环与 usage 统计逻辑的混入类。"""

    def _build_continue_request(
        self,
        session_id: str,
        token: str,
        hif_leim: str,
        hif_dliq: str,
        message_id: str,
    ) -> tuple:
        """构造 continue（截断续写）请求所需的 headers 与 payload。

        Args:
            session_id: 会话 ID。
            token: 账号 token。
            hif_leim: HIF leim 令牌。
            hif_dliq: HIF dliq 令牌。
            message_id: 需要续写的消息 ID。

        Returns:
            (headers, payload) 二元组。
        """
        cont_headers = build_headers(
            token=token,
            session_id=session_id,
            hif_leim=hif_leim,
            hif_dliq=hif_dliq,
        )
        cont_payload = {
            "chat_session_id": session_id,
            "message_id": message_id,
            "fallback_to_resume": True,
        }
        return cont_headers, cont_payload

    async def _consume_continue_response(
        self,
        cont_resp: Any,
        parser: StreamParser,
        continue_flag: List[bool],
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """消费一次 continue 响应，并 yield 翻译后的增量内容。

        Args:
            cont_resp: aiohttp 响应对象。
            parser: 当前请求复用的 StreamParser。
            continue_flag: 单元素列表，用于向调用方回传是否需要再次续写。

        Yields:
            str（文本增量）或 dict（thinking）。
        """
        async for chunk in self._parse_sse_stream(cont_resp, parser):
            if chunk.get("needs_continue"):
                continue_flag[0] = True
            elif chunk.get("type") not in ("event", "status"):
                translated = translate_chunk(chunk)
                if translated is not None:
                    yield translated

    async def _run_continue_loop(
        self,
        parser: StreamParser,
        session_id: str,
        token: str,
        hif_leim: str,
        hif_dliq: str,
        needs_continue: bool,
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """处理 continue（截断续写）循环。

        Args:
            parser: 当前请求复用的 StreamParser。
            session_id: 会话 ID。
            token: 账号 token。
            hif_leim: HIF leim 令牌。
            hif_dliq: HIF dliq 令牌。
            needs_continue: 首个响应结束后是否需要续写。

        Yields:
            str（文本增量）或 dict（thinking）。
        """
        continue_count = 0
        while needs_continue and continue_count < MAX_CONTINUE:
            continue_count += 1
            mid = parser.message_id
            if mid is None:
                break
            await asyncio.sleep(0.1)

            cont_headers, cont_payload = self._build_continue_request(
                session_id, token, hif_leim, hif_dliq, mid
            )
            parser.begin_stream(is_continuation=True)
            async with self._session.post(
                "https://{}/api/v0/chat/continue".format(DEFAULT_HOST),
                headers=cont_headers,
                json=cont_payload,
                timeout=aiohttp.ClientTimeout(total=600),
                ssl=False,
            ) as cont_resp:
                if cont_resp.status != 200:
                    break
                needs_continue = False
                continue_flag = [False]
                async for translated in self._consume_continue_response(
                    cont_resp, parser, continue_flag
                ):
                    yield translated
                needs_continue = continue_flag[0] or parser.should_continue

    def _compute_usage(
        self, parser: StreamParser, prompt: str
    ) -> List[Dict[str, Any]]:
        """根据累积内容估算 token 用量。

        Args:
            parser: 已完成本次请求解析的 StreamParser。
            prompt: 本次请求的提示词。

        Returns:
            仅包含一个 usage 字典的列表，便于以 for 循环 yield。
        """
        content_len = len(parser.accumulated_content)
        think_len = len(parser.accumulated_thinking)
        total_chars = content_len + think_len
        prompt_tokens = max(len(prompt) // 3, 1)
        completion_tokens = max(total_chars // 3, 0)
        return [
            {
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            }
        ]

    @staticmethod
    def _parse_line_if_nonempty(
        parser: StreamParser, line: str
    ) -> Any:
        """解析一行 SSE 数据；空行返回 ``None``。"""
        if not line.strip():
            return None
        return parser.parse_line(line)

    async def _parse_sse_stream(
        self,
        resp: Any,
        parser: StreamParser,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """解析 SSE 流式响应。

        Args:
            resp: aiohttp 响应对象。
            parser: StreamParser 实例。

        Yields:
            解析后的 chunk 字典。
        """
        buf = ""
        async for raw_chunk in resp.content.iter_chunked(4096):
            if not raw_chunk:
                continue
            buf += raw_chunk.decode("utf-8", errors="ignore")
            lines = buf.split("\n")
            buf = lines[-1]
            for line in lines[:-1]:
                raise_if_user_muted(line)
                result = self._parse_line_if_nonempty(parser, line)
                if result is not None:
                    yield result
        # 处理剩余缓冲区
        if buf.strip():
            raise_if_user_muted(buf)
            result = parser.parse_line(buf)
            if result is not None:
                yield result
