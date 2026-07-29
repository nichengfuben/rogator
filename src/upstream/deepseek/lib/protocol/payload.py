


import secrets
from datetime import datetime
from typing import Any, Dict, Optional

def make_stream_id() -> str:
    """生成流式请求 ID。

    Returns:
        格式为 YYYYMMDD-{8位十六进制} 的字符串。
    """
    return "{}-{}".format(
        datetime.now().strftime("%Y%m%d"),
        secrets.token_hex(8),
    )


def build_payload(
    session_id: str,
    prompt: str,
    model: str,
    *,
    stream_id: Optional[str] = None,
) -> Dict[str, Any]:
    """构建 DeepSeek ``/api/v0/chat/completion`` 请求体。

    Args:
        session_id: 会话 ID。
        prompt: 提示文本。
        model: 模型名。
        stream_id: 客户端流 ID（可选，自动生成）。

    Returns:
        请求体字典。
    """
    return {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": "default",
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": False,
        "search_enabled": False,
        "preempt": False,
        "client_stream_id": stream_id if stream_id is not None else make_stream_id(),
    }

__all__ = [
    "build_payload",
    "make_stream_id",
]
