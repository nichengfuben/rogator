from __future__ import annotations

"""send limit 运行时解析：PayloadTooLarge 减半 override > splitter > 全局 fallback。"""

from typing import Optional

from server.config import CONFIG


def effective_send_max_chars(
    state: object,
    model: Optional[str],
    *,
    fallback: Optional[int] = None,
) -> int:
    """运行时有效上限：PayloadTooLarge 减半 override > splitter > 全局 fallback。"""
    if model:
        overrides = getattr(state, "_send_limit_overrides", None) or {}
        if model in overrides:
            return int(overrides[model])
    splitter = getattr(state, "splitter", None)
    if splitter is not None:
        return int(getattr(splitter, "max_chars", 0) or fb_fallback(fallback))
    return fb_fallback(fallback)


def fb_fallback(fallback: Optional[int]) -> int:
    return int(fallback if fallback is not None else CONFIG.qwen_send_max_chars)
