

# src/platforms/deepseek/core/headers.py
"""DeepSeek 请求头构建工具"""

from typing import Dict, Optional

from upstream.deepseek.lib.protocol.consts import COMMON_HEADERS, DEFAULT_HOST


def build_headers(
    token: str,
    session_id: str = "",
    hif_leim: str = "",
    hif_dliq: str = "",
    pow_response: str = "",
) -> Dict[str, str]:
    """构建完整请求头。

    Args:
        token: Bearer 令牌。
        session_id: 对话会话 ID（可选），用于构建 Referer。
        hif_leim: x-hif-leim 令牌（可选）。
        hif_dliq: x-hif-dliq 令牌（可选）。
        pow_response: x-ds-pow-response（可选）。

    Returns:
        请求头字典。
    """
    h: Dict[str, str] = {
        **COMMON_HEADERS,
        "authorization": "Bearer {}".format(token),
        "origin": "https://{}".format(DEFAULT_HOST),
    }
    if session_id:
        h["referer"] = "https://{}/a/chat/s/{}".format(DEFAULT_HOST, session_id)
    else:
        h["referer"] = "https://{}/".format(DEFAULT_HOST)

    if hif_leim:
        h["x-hif-leim"] = hif_leim
    if hif_dliq:
        h["x-hif-dliq"] = hif_dliq
    if pow_response:
        h["x-ds-pow-response"] = pow_response
    return h


def build_basic_headers(token: str = "") -> Dict[str, str]:
    """构建最简请求头（无 session / hif / pow）。

    Args:
        token: Bearer 令牌（可空）。

    Returns:
        基础请求头字典。
    """
    h: Dict[str, str] = {
        **COMMON_HEADERS,
        "origin": "https://{}".format(DEFAULT_HOST),
        "referer": "https://{}/".format(DEFAULT_HOST),
    }
    if token:
        h["authorization"] = "Bearer {}".format(token)
    return h

__all__ = [
    "build_headers",
    "build_basic_headers",
]

# 注：``payloads.py`` 依赖本模块（导入 build_headers），因此本文件不得
# 重导出它的符号，否则会形成循环导入。
