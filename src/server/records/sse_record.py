from __future__ import annotations

"""上游 SSE 原始流实时落盘（parse 前）。"""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from echotools.base.logger import get_logger

from server.config import CONFIG, LOG_DIR

__all__ = [
    "SseStreamRecorder",
    "append_sse_bytes",
    "record_sse_stream",
    "sse_dump_dir",
]

logger = get_logger("rogator")

_SSE_SUBDIR = "sse"
_active_recorder: ContextVar[Optional["SseStreamRecorder"]] = ContextVar(
    "rogator_sse_recorder",
    default=None,
)


def sse_dump_dir() -> Path:
    return LOG_DIR / _SSE_SUBDIR


class SseStreamRecorder:
    """按 TCP chunk 追加写入 logs/sse/{req_id}.sse。"""

    __slots__ = ("req_id", "_path", "_handle", "_bytes", "_enabled")

    def __init__(self, req_id: str) -> None:
        self.req_id = req_id
        self._path = sse_dump_dir() / f"{req_id}.sse"
        self._handle: Optional[BinaryIO] = None
        self._bytes = 0
        self._enabled = bool(CONFIG.record_sse)

    def write(self, data: bytes) -> None:
        if not self._enabled or not data:
            return
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("ab")
            logger.debug("SSE 落盘开始 req_id=%s path=%s", self.req_id, self._path)
        self._handle.write(data)
        self._handle.flush()
        self._bytes += len(data)

    def close(self) -> None:
        if not self._enabled or self._handle is None:
            return
        self._handle.close()
        self._handle = None
        logger.info(
            "record sse req_id=%s bytes=%d path=%s",
            self.req_id,
            self._bytes,
            self._path,
        )


def append_sse_bytes(data: bytes) -> None:
    """iter_sse_events 每收到一块 raw 时调用。"""
    rec = _active_recorder.get(None)
    if rec is not None:
        rec.write(data)


@contextmanager
def record_sse_stream(req_id: str) -> Iterator[None]:
    """请求级 SSE 录制上下文（与 prompts/responses 共用 req_id）。"""
    if not CONFIG.record_sse:
        yield
        return
    recorder = SseStreamRecorder(req_id)
    token = _active_recorder.set(recorder)
    try:
        yield
    finally:
        try:
            _active_recorder.reset(token)
        except ValueError:
            # async generator aclose 可能在不同 Context 触发；仅清理 recorder
            _active_recorder.set(None)
        recorder.close()
