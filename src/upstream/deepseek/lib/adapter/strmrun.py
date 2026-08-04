from __future__ import annotations

# src/platforms/deepseek/core/adapter/streamrun.py
"""DeepSeek 流式请求辅助逻辑——SSE 解析、截断续写循环、usage 统计"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Union

import aiohttp

from server.records.sse_record import append_sse_bytes_async
from upstream.deepseek.lib.adapter.helpers.biz_error import raise_if_user_muted
from upstream.deepseek.lib.adapter.helpers.pmtutil import translate_chunk
from upstream.deepseek.lib.protocol.consts import DEFAULT_HOST, MAX_CONTINUE
from upstream.deepseek.lib.protocol.headers import build_headers
from upstream.deepseek.lib.stream.strmpars import StreamParser

logger = logging.getLogger(__name__)


class _StreamRunMixin:
    def _build_continue_request(
        self,
        session_id: str,
        token: str,
        hif_leim: str,
        hif_dliq: str,
        message_id: str,
    ) -> tuple:

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

    def _compute_usage(self, parser: StreamParser, prompt: str) -> List[Dict[str, Any]]:

        prompt_tokens = max(len(prompt) // 3, 1)
        upstream_total = parser.accumulated_token_usage
        if upstream_total > 0:
            completion_tokens = max(upstream_total - prompt_tokens, 0)
            return [
                {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": upstream_total,
                    }
                }
            ]
        content_len = len(parser.accumulated_content)
        think_len = len(parser.accumulated_thinking)
        total_chars = content_len + think_len
        completion_tokens = (total_chars + 2) // 3 if total_chars > 0 else 0
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
    def _parse_line_if_nonempty(parser: StreamParser, line: str) -> Any:

        if not line.strip():
            return None
        return parser.parse_line(line)

    async def _parse_sse_stream(
        self,
        resp: Any,
        parser: StreamParser,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """解析 SSE 流式响应。"""
        buf = ""
        async for raw_chunk in resp.content.iter_chunked(4096):
            if not raw_chunk:
                continue
            await append_sse_bytes_async(raw_chunk)
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
