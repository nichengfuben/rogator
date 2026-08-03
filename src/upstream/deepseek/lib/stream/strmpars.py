from __future__ import annotations

"""DeepSeek 流式响应解析器"""

import json
import re
from typing import Any, Dict, Optional, Tuple

from upstream.deepseek.lib.stream.fragchunk import _FragmentChunkMixin
from upstream.deepseek.lib.stream.fraghndl import _FragmentHandlerMixin
from upstream.deepseek.lib.stream.srchres import SearchResult

# 重新导出 SearchResult，保持 `from .streamparser import SearchResult` 等旧引用路径可用
__all__ = ["SearchResult", "StreamParser", "parse_sse_line"]


class StreamParser(_FragmentHandlerMixin, _FragmentChunkMixin):
    def __init__(self, include_thinking: bool = False) -> None:
        self._inc: bool = include_thinking
        self._content: str = ""
        self._think: str = ""
        self._msg_id: Optional[int] = None
        self._parent_id: Optional[int] = None
        self._status: str = "WIP"
        self._is_think: bool = False
        self._think_started: bool = False
        self._search: Dict[int, SearchResult] = {}
        self._cite_buf: str = ""
        self._first_frag: bool = False
        self._skip_first_response_frag: bool = False
        self._tok_usage: int = 0
        self._cur_event: Optional[str] = None
        self._stream_closed: bool = False
        self._close_click_behavior: Optional[str] = None
        self._close_auto_resume: Optional[bool] = None
        self._should_continue: bool = False

    @property
    def status(self) -> str:
        return self._status

    @property
    def message_id(self) -> Optional[int]:
        return self._msg_id

    @property
    def accumulated_content(self) -> str:
        return self._content

    @property
    def accumulated_thinking(self) -> str:
        return self._think

    @property
    def accumulated_token_usage(self) -> int:
        return self._tok_usage

    @property
    def should_continue(self) -> bool:

        if self._should_continue:
            return True
        if self._status in ("INCOMPLETE", "TIMEOUT"):
            return True
        if self._close_click_behavior == "retry":
            return True
        if self._close_auto_resume is not None and self._status != "FINISHED":
            return True
        if self._status == "WIP" and not self._stream_closed:
            return True
        return False

    def begin_stream(self, is_continuation: bool = False) -> None:

        self._cur_event = None
        self._stream_closed = False
        self._close_click_behavior = None
        self._close_auto_resume = None
        self._should_continue = False
        # 每次新流开始重置首片段相关状态
        self._first_frag = False
        self._is_think = False
        self._think_started = False
        self._skip_first_response_frag = is_continuation

    def _replace_citations(self, text: str) -> str:

        # 统一匹配 [citation:N] 和 [reference:N]
        def _rep(m: Any) -> str:
            i = int(m.group(1))
            if i in self._search:
                return "[" + self._search[i].url + "]" + str(i)
            return m.group(0)

        return re.sub(r"\[(?:citation|reference):(\d+)\]", _rep, text)

    def _proc_cite(self, chunk: str) -> Tuple[str, str]:

        self._cite_buf += chunk
        result = ""
        buf = self._cite_buf
        while buf:
            m = re.search(r"\[(?:citation|reference):(\d+)\]", buf)
            if m:
                result += buf[: m.start()]
                i = int(m.group(1))
                result += (
                    "[" + self._search[i].url + "]" + str(i)
                    if i in self._search
                    else m.group(0)
                )
                buf = buf[m.end() :]
            else:
                # 检查是否有不完整的引用标记（citation 或 reference 都匹配）
                inc = re.search(
                    r"\[(?:c(?:i(?:t(?:a(?:t(?:i(?:o(?:n)?)?)?)?)?)?)?|"
                    r"r(?:e(?:f(?:e(?:r(?:e(?:n(?:c(?:e)?)?)?)?)?)?)?)?)?"
                    r":?\d*\]?$",
                    buf,
                )
                if inc:
                    result += buf[: inc.start()]
                    self._cite_buf = buf[inc.start() :]
                    return result, self._cite_buf
                result += buf
                buf = ""
        self._cite_buf = ""
        return result, ""

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:

        line = line.strip()
        if not line:
            return None

        if line.startswith("event:"):
            ev = line[6:].strip()
            self._cur_event = ev
            if ev in ("finish", "close"):
                self._stream_closed = True
                # 刷新引用缓冲区
                if self._cite_buf:
                    rem = self._cite_buf
                    self._cite_buf = ""
                    if self._is_think and self._inc:
                        self._is_think = False
                        return {"type": "thinking", "content": rem}
                    return {"type": "content", "content": rem}
                if self._is_think:
                    self._is_think = False
                return {"type": "event", "event": ev}
            return None

        if not line.startswith("data:"):
            return None

        raw = line[5:].strip()
        if not raw:
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        return self._proc(data)


def parse_sse_line(
    data_str: str,
    parser: Optional[StreamParser] = None,
) -> Optional[Dict[str, Any]]:
    """解析单行 SSE 数据，委托给 ``StreamParser``。"""
    if parser is None:
        return None
    if not data_str.strip():
        return None
    return parser.parse_line(data_str)
