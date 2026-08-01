from __future__ import annotations

"""DeepSeek 提示词/账号/chunk 转换工具模块。

从 ``client.py`` 拆分而来，承载与 HTTP 会话生命周期无关的纯函数与数据结构：
prompt 拼装、chunk 协议转换、模型能力判断、账号凭证 dataclass。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text") or "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


def build_prompt(messages: List[Dict[str, Any]]) -> str:
    """将消息转为 DeepSeek 单 prompt（与 Qwen entml 路径一致）。

    Rogator 经 ``inject_fncall`` 后通常只发一条 user，正文已是完整 entml prompt。
    不再拼接「用户: / 系统指令: / 助手:」等角色前缀。
    """
    if not messages:
        return ""
    # 与 Qwen ``chat_completion`` 一致：优先取首条 content
    first = _message_text(messages[0].get("content", "")).strip()
    if first:
        return first
    parts: List[str] = []
    for m in messages:
        text = _message_text(m.get("content", "")).strip()
        if text:
            parts.append(text)
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
