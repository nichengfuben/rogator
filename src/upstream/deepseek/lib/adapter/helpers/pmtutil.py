from __future__ import annotations

"""DeepSeek prompt 拼装与 chunk 协议转换。"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


def _message_text(content: Any) -> str:
    if not isinstance(content, list):
        return str(content)
    return "\n".join(
        p.get("text", "")
        for p in content
        if isinstance(p, dict) and p.get("type") == "text"
    )


def _format_message(role: str, content: str) -> Optional[str]:
    if role == "system":
        return "系统指令: {}".format(content)
    if role == "user":
        return "用户: {}".format(content)
    if role == "assistant":
        return "助手: {}".format(content)
    if role == "tool":
        return "工具结果: {}".format(content)
    return None


def build_prompt(messages: List[Dict[str, Any]]) -> str:

    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        formatted = _format_message(role, _message_text(m.get("content", "")))
        if formatted is not None:
            parts.append(formatted)
    return "\n\n".join(parts)


def translate_chunk(chunk: Dict[str, Any]) -> Optional[Union[str, Dict[str, Any]]]:

    t = chunk.get("type")
    if t == "content":
        content = chunk.get("content", "")
        return content if content else None
    if t == "thinking":
        content = chunk.get("content", "")
        return {"thinking": content} if content else None
    return None


@dataclass
class Account:
    """DeepSeek 账号凭证。"""

    username: str
    password: str
    token: str = ""
    user_id: str = ""
    device_id: str = ""
    context_length: Optional[int] = None
    session_cookie: str = ""
    hif_leim: str = ""
    hif_dliq: str = ""
    hif_expire: float = 0.0
