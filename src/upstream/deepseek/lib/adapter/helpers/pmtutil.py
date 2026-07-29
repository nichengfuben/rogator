from __future__ import annotations

"""DeepSeek 提示词/账号/chunk 转换工具模块。

从 ``client.py`` 拆分而来，承载与 HTTP 会话生命周期无关的纯函数与数据结构：
prompt 拼装、chunk 协议转换、模型能力判断、账号凭证 dataclass。
"""

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
    """将 OpenAI 格式消息列表转为 DeepSeek 单 prompt 字符串。

    Args:
        messages: OpenAI 格式消息列表。

    Returns:
        拼接后的提示文本。
    """
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        formatted = _format_message(role, _message_text(m.get("content", "")))
        if formatted is not None:
            parts.append(formatted)
    return "\n\n".join(parts)


def translate_chunk(chunk: Dict[str, Any]) -> Optional[Union[str, Dict[str, Any]]]:
    """将内部 chunk 转换为 yield 协议格式。

    Args:
        chunk: 内部 chunk 字典。

    Returns:
        str（正文增量）、dict（thinking/usage）或 None。
    """
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
    context_length: Optional[int] = None
    session_cookie: str = ""
    hif_leim: str = ""
    hif_dliq: str = ""
    hif_expire: float = 0.0
