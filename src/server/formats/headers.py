from __future__ import annotations

"""统一伪造 Cloudflare 边缘响应头，避免客户端识别出 Python aiohttp 服务。"""

import secrets
import time
from typing import Dict

# 抓包自 api.anthropic.com 的 Cloudflare 边缘头；CF-RAY 为真实边缘 Ray ID。
# 结构：前 8 位为 32 位 tick 时间戳（十六进制），后 8 位随机，-LAX 机场码。
_CF_RAY_SUFFIX = "-LAX"
# tick 速率约 6/s（抓包样本：8s 间隔差值 47 ≈ 6 ticks/s）。偏移锚定在真实抓包
# 样本 2026-08-12T07:45:43Z = 0xA29DDCB7（该时刻 epoch 秒 1786520743 × 6），
# 使本地值与真实边缘同量级；32 位回绕（约 8 年后）与真实设计一致
_CF_RAY_TS_OFFSET = 0xA29DDCB7 - 1786520743 * 6


def _cf_ray() -> str:
    tick = int(time.time() * 6) + _CF_RAY_TS_OFFSET
    # token_hex(4) = 8 个 hex 字符（后 8 位）
    return f"{tick & 0xFFFFFFFF:08x}{secrets.token_hex(4)}{_CF_RAY_SUFFIX}"


def cloudflare_headers() -> Dict[str, str]:
    # 对齐真实 Cloudflare 边缘响应顺序和字段
    return {
        "Server": "cloudflare",
        "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
        "X-Robots-Tag": "none",
        "server-timing": "x-originResponse;dur=",
        "cf-cache-status": "DYNAMIC",
        "CF-RAY": _cf_ray(),
        "Connection": "keep-alive",
    }


__all__ = ["cloudflare_headers"]
