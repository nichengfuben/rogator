from __future__ import annotations

"""上游模型 thinking/answer 落盘（parse_sse 后、entml 解析前）。"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List

from echotools.base.logger import get_logger

from server.config import CONFIG, LOG_DIR

__all__ = ["RawResponseRecorder", "record_raw_response", "response_dump_dir"]

logger = get_logger("rogator")

_RESPONSES_SUBDIR = "responses"


def response_dump_dir() -> Path:
    return LOG_DIR / _RESPONSES_SUBDIR


class RawResponseRecorder:
    """按上游事件顺序累积 thinking/answer 原文（不经 entml 解析）。"""

    __slots__ = ("req_id", "_parts", "_enabled")

    def __init__(self, req_id: str) -> None:
        self.req_id = req_id
        self._parts: List[str] = []
        self._enabled = bool(CONFIG.record_response)

    def ingest_event(self, event: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        etype = event.get("type")
        if etype not in ("thinking", "answer"):
            return
        content = event.get("content", "")
        if content:
            self._parts.append(content)

    @property
    def raw_text(self) -> str:
        return "".join(self._parts)

    def finalize(self) -> None:
        if not self._enabled:
            return
        text = self.raw_text
        if not text:
            return
        dump_dir = response_dump_dir()
        dump_dir.mkdir(parents=True, exist_ok=True)
        path = dump_dir / f"{self.req_id}.txt"
        path.write_text(text, encoding="utf-8")
        logger.info(
            "record response req_id=%s chars=%d path=%s",
            self.req_id,
            len(text),
            path,
        )


@contextmanager
def record_raw_response(req_id: str) -> Iterator[RawResponseRecorder]:
    """请求结束时写入 logs/responses/{req_id}.txt（与 prompts/{req_id}.txt 对齐）。"""
    recorder = RawResponseRecorder(req_id)
    try:
        yield recorder
    finally:
        recorder.finalize()
