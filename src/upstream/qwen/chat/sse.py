from __future__ import annotations

"""SSE 流错误检测、live 事件迭代与字节行缓冲。

将原 ``sse_buffer.py`` 合并进本文件，减少 ``src/upstream/qwen/chat`` 子项数；
``ByteLineBuffer`` 仅在本模块内部使用，外部测试改从 ``upstream.qwen.chat.sse``
导入同名对象。
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Iterator, List, Optional

import aiohttp

from upstream.qwen.chat.upload.parse import SseEventAssembler, parse_sse_event, parse_sse_line
from server.formats import (
    BaxiaSmBlockedError,
    DataInspectionFailedError,
    TokenExpiredError,
    UpstreamChatNotFoundError,
    UpstreamTimeoutError,
    UpstreamWafBlockedError,
)
from server.records.sse_record import append_sse_bytes_async

if TYPE_CHECKING:
    from upstream.qwen.client import QwenClient
    from upstream.qwen.chat.store import QwenSession

logger = logging.getLogger("rogator")

_BAXIA_SM_MARKERS: frozenset[str] = frozenset(
    {"RGV587", "FAIL_SYS", "FAIL_SYS_USER_VALIDATE", "RGV587_ERROR::SM"}
)
_UPSTREAM_RATE_LIMIT_CODES: frozenset[str] = frozenset(
    {"RateLimited", "ParallelLimited", "quotaLimited", "Too_Many_Requests", "quota_limit"}
)


class ByteLineBuffer:
    """TCP chunk 流切出独立行；用 bytearray 避免 s += chunk 的 O(n^2) 行为。"""

    __slots__ = ("_buf", "_chunk_hits", "_max_pending")

    def __init__(self, *, max_pending: int = 1 << 20) -> None:
        # 1 MiB 上限：防止上游误发不带换行的巨大单行让缓冲区膨胀；
        # 超过即抛 ``BufferError``，让上层走错误路径。
        self._buf = bytearray()
        self._chunk_hits = 0
        self._max_pending = max_pending

    def feed(self, chunk: bytes) -> List[bytes]:
        if not chunk:
            return []
        self._chunk_hits += 1
        self._buf.extend(chunk)
        if len(self._buf) > self._max_pending:
            raise BufferError(
                f"SSE 行缓冲超过 {self._max_pending} 字节，无换行"
            )
        nl = self._buf.find(b"\n")
        if nl < 0:
            return []
        lines: List[bytes] = []
        while nl >= 0:
            line = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            lines.append(line)
            nl = self._buf.find(b"\n")
        return lines

    def flush(self) -> bytes:
        if not self._buf:
            return b""
        tail = bytes(self._buf)
        self._buf.clear()
        return tail

    def pending(self) -> int:
        return len(self._buf)

    @property
    def chunk_hits(self) -> int:
        return self._chunk_hits


def iter_byte_lines(chunks: Iterator[bytes]) -> Iterator[bytes]:
    """对迭代器产出的 bytes 块，按 ``\\n`` 切出整行；最后一段无换行的尾部单独 yield。"""
    buf = ByteLineBuffer()
    for chunk in chunks:
        for line in buf.feed(chunk):
            yield line
    tail = buf.flush()
    if tail:
        yield tail


def _is_baxia_sm_block(message: str, *, punish_url: str = "") -> bool:
    if punish_url and any(marker in message for marker in _BAXIA_SM_MARKERS):
        return True
    return "RGV587_ERROR::SM" in message or "FAIL_SYS_USER_VALIDATE" in message


def _raise_for_success_false(
    client: "QwenClient",
    session: "QwenSession",
    obj: Dict[str, Any],
) -> None:
    from upstream.qwen.chat.chat import raise_qwen_session_error

    msg = json.dumps(obj, ensure_ascii=False)
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    code = str(data.get("code") or "")
    if code in _UPSTREAM_RATE_LIMIT_CODES:
        client._invalidate_session(session)
        logger.warning("Session %s upstream rate limited (%s)", session.username[:6], code)
        raise TokenExpiredError(f"Rate limited: {msg[:200]}")
    if code == "CHAT_NOT_FOUND":
        raise UpstreamChatNotFoundError(f"Qwen chat not found: {msg[:200]}", upstream="qwen")
    raise_qwen_session_error(client, session, msg)
    raise RuntimeError(f"Qwen API error: {msg}")


def raise_sse_inline_error(
    client: "QwenClient",
    session: "QwenSession",
    line: str,
) -> None:
    """HTTP 200 但 body 为 Baxia/业务错误 JSON 时抛出可重试或 WAF 异常。"""
    stripped = line.strip()
    if not stripped.startswith("{"):
        return
    try:
        obj = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return
    if not isinstance(obj, dict):
        return

    if "success" in obj:
        if obj.get("success", True):
            return
        _raise_for_success_false(client, session, obj)

    ret = obj.get("ret")
    if not isinstance(ret, list) or not ret:
        return
    msg = " ".join(str(part) for part in ret if part)
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    punish_url = str(data.get("url") or "")
    if _is_baxia_sm_block(msg, punish_url=punish_url):
        logger.debug(
            "Baxia SM blocked [%s]: %s",
            session.username[:6],
            msg[:160],
        )
        raise BaxiaSmBlockedError(msg[:200])
    if punish_url or "FAIL_SYS" in msg or "RGV587" in msg:
        raise UpstreamWafBlockedError(
            f"Qwen Baxia blocked: {msg[:200]}",
            upstream="qwen",
        )
    raise RuntimeError(f"Qwen upstream error: {msg[:200]}")


def _check_sse_error_line(client: "QwenClient", line: str, session: "QwenSession") -> None:
    raise_sse_inline_error(client, session, line)


def _track_response_id(
    event: Dict[str, Any],
    response_id_out: Optional[list],
) -> None:
    if response_id_out is None:
        return
    rid = event.get("response_id")
    if rid and event.get("type") in (
        "response_created",
        "response_stopped",
        "response_info",
    ):
        response_id_out[:] = [str(rid)]


def _event_from_sse_data(
    client: "QwenClient",
    session: "QwenSession",
    data_str: str,
    response_id_out: Optional[list],
) -> Optional[Dict[str, Any]]:
    if not data_str or data_str == "[DONE]":
        return None
    event = parse_sse_event(data_str)
    if event:
        _track_response_id(event, response_id_out)
        # 检测顶层 error.code 是否为可重试限流错误（如 quota_limit）
        err = event.get("error")
        if isinstance(err, dict):
            _raise_for_sse_error_event(client, session, err)
        return event
    # parse_sse_event 无法解析时（如 Qwen error 事件无 choices），直接检查原始 JSON
    try:
        obj = json.loads(data_str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    err = obj.get("error")
    if isinstance(err, dict):
        _raise_for_sse_error_event(client, session, err)
    return None


def _raise_for_sse_error_event(
    client: "QwenClient",
    session: "QwenSession",
    err: Dict[str, Any],
) -> None:
    """顶层 error 帧分类：限流抛可重试 TokenExpired，内容安全拦截抛非重试业务错误。"""
    err_code = str(err.get("code") or "")
    if err_code == "data_inspection_failed":
        logger.info(
            "Session %s Qwen data inspection failed (stage=%s): %s",
            session.username[:6], err.get("stage"), err.get("details"),
        )
        raise DataInspectionFailedError(
            str(err.get("details") or "content inspection failed"),
            code=err_code,
            stage=str(err.get("stage") or ""),
        )
    if err_code in _UPSTREAM_RATE_LIMIT_CODES:
        logger.warning(
            "Session %s upstream quota limited via SSE error event (%s)",
            session.username[:6], err_code,
        )
        raise TokenExpiredError(f"Rate limited (SSE error): {err_code}")


def _dispatch_assembled_sse_line(
    client: "QwenClient",
    session: "QwenSession",
    line: str,
    assembler: SseEventAssembler,
    response_id_out: Optional[list],
) -> Optional[Dict[str, Any]]:
    payload = assembler.feed_line(line)
    if payload is None:
        if line and not line.startswith("data:") and not line.startswith(":"):
            _check_sse_error_line(client, line, session)
        return None
    return _event_from_sse_data(client, session, payload, response_id_out)


async def iter_sse_events(
    client: "QwenClient",
    resp: aiohttp.ClientResponse,
    session: "QwenSession",
    *,
    response_id_out: Optional[list] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """逐行解析 SSE；对齐前端 kT/_T 组帧 + TCP chunk 行缓冲。"""
    buf = ByteLineBuffer()
    assembler = SseEventAssembler()
    try:
        async for raw in resp.content:
            await append_sse_bytes_async(raw)
            for raw_line in buf.feed(raw):
                line = raw_line.rstrip(b"\r").decode("utf-8", errors="replace")
                event = _dispatch_assembled_sse_line(
                    client, session, line, assembler, response_id_out,
                )
                if event:
                    yield event
        tail = buf.flush()
        if tail:
            line = tail.rstrip(b"\r").decode("utf-8", errors="replace")
            event = _dispatch_assembled_sse_line(
                client, session, line, assembler, response_id_out,
            )
            if event:
                yield event
        eof_payload = assembler.flush_eof()
        if eof_payload:
            event = _event_from_sse_data(
                client, session, eof_payload, response_id_out,
            )
            if event:
                yield event
    except asyncio.TimeoutError as e:
        raise UpstreamTimeoutError("Upstream SSE read timed out") from e