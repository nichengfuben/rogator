from __future__ import annotations

"""tool_call_id 规范化：Cursor 偶发把 call-id 与 fc_ 用换行拼在一起。"""

from typing import Iterable, List


def normalize_tool_call_id(raw: object) -> str:
    """对外 OpenAI id：取首行，去掉空白。"""
    s = str(raw or "").replace("\r\n", "\n").strip()
    if not s:
        return ""
    return s.split("\n", 1)[0].strip() or s


def tool_call_id_aliases(raw: object) -> List[str]:
    """查找用：首行 / 各行 / 原文（去首尾空白）。"""
    s = str(raw or "").replace("\r\n", "\n").strip()
    if not s:
        return []
    out: List[str] = []
    for part in s.split("\n"):
        p = part.strip()
        if p and p not in out:
            out.append(p)
    if s not in out:
        out.append(s)
    return out


def expand_tool_call_ids(ids: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in ids or []:
        for a in tool_call_id_aliases(raw):
            if a not in seen:
                seen.add(a)
                out.append(a)
    return out
