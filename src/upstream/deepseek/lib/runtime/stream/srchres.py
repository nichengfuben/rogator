from __future__ import annotations

"""DeepSeek 流式解析——搜索结果数据结构。"""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """搜索结果条目。"""

    url: str
    title: str
    snippet: str
    cite_index: int
