from __future__ import annotations

"""DeepSeek 流式解析——fragment 内容/搜索/chunk 处理混入类。"""

from typing import Any, Dict, List, Optional

from upstream.deepseek.lib.stream.srchres import SearchResult


class _FragmentChunkMixin:
    """封装 fragment 正文/思考/搜索/chunk 增量处理，供 StreamParser 组合。"""

    def _extract_search(self, results: List[Any]) -> None:
        for r in results:
            if isinstance(r, dict) and "cite_index" in r:
                self._search[r["cite_index"]] = SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("snippet", ""),
                    cite_index=r["cite_index"],
                )

    def _extract_references(self, references: List[Dict[str, Any]]) -> None:
        for ref in references:
            if not isinstance(ref, dict):
                continue
            ref_id = ref.get("id")
            ref_type = ref.get("type")
            if ref_id is not None:
                _ = ref_type

    def _handle_frag(self, frag: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ft = frag.get("type", "RESPONSE")
        content = frag.get("content")
        skip_result = self._maybe_skip_first_response_frag(ft, content)
        if skip_result is not None:
            return skip_result
        if ft == "THINK":
            return self._handle_think_frag(content)
        if ft == "RESPONSE":
            return self._handle_response_frag(content)
        if ft == "SEARCH" and "results" in frag:
            self._extract_search(frag["results"])
        return None

    def _maybe_skip_first_response_frag(
        self, ft: str, content: Any,
    ) -> Optional[Dict[str, Any]]:
        is_first_frag = not self._first_frag
        if not is_first_frag:
            return None
        self._first_frag = True
        if self._skip_first_response_frag and ft == "RESPONSE":
            self._skip_first_response_frag = False
            if content:
                if self._content and str(content).startswith(self._content):
                    self._content = str(content)
                elif not self._content:
                    self._content = str(content)
            return None
        self._skip_first_response_frag = False
        return None

    def _handle_think_frag(self, content: Any) -> Optional[Dict[str, Any]]:
        self._is_think = True
        self._think_started = True
        if not content:
            return None
        self._think += content
        if not self._inc:
            return None
        pc, _ = self._proc_cite(content)
        return {"type": "thinking", "content": pc}

    def _handle_response_frag(self, content: Any) -> Optional[Dict[str, Any]]:
        if self._is_think:
            self._is_think = False
        if not content:
            return None
        self._content += content
        pc, _ = self._proc_cite(content)
        return {"type": "content", "content": pc}

    def _handle_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        if self._is_think:
            self._think += chunk
            if self._inc:
                pc, _ = self._proc_cite(chunk)
                return {"type": "thinking", "content": pc} if pc else None
            return None
        self._content += chunk
        pc, _ = self._proc_cite(chunk)
        return {"type": "content", "content": pc} if pc else None
