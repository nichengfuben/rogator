"""DeepSeek 上游 SSE 业务错误解析。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


class DeepSeekUserMutedError(Exception):
    """账号被 DeepSeek 静音（``biz_msg: user is muted``）。"""

    def __init__(
        self,
        *,
        biz_msg: str = "user is muted",
        mute_until: Optional[float] = None,
    ) -> None:
        self.biz_msg = biz_msg
        self.mute_until = mute_until
        super().__init__(biz_msg)


class DeepSeekWafChallengeError(DeepSeekUserMutedError):
    """CloudFront WAF 人机挑战（HTTP 202 + ``x-amzn-waf-action: challenge``）。"""

    def __init__(self) -> None:
        super().__init__(biz_msg="waf challenge")


def raise_if_waf_challenge(status: int, headers: Any) -> None:
    """若响应为 WAF challenge，抛出 :class:`DeepSeekWafChallengeError`。"""
    action = ""
    if headers is not None:
        get = getattr(headers, "get", None)
        if callable(get):
            action = str(get("x-amzn-waf-action") or get("X-Amzn-Waf-Action") or "")
    if status == 202 and action.lower() == "challenge":
        raise DeepSeekWafChallengeError()


class DeepSeekAccountsExhaustedError(RuntimeError):
    """所有 DeepSeek 账号均 mute 或不可用。"""


def parse_biz_error_from_line(line: str) -> Optional[Dict[str, Any]]:
    """从 SSE 行解析 ``biz_code`` / ``biz_msg``（非标准 event/data 帧）。"""
    raw = line.strip()
    if raw.startswith("data:"):
        raw = raw[5:].strip()
    if not raw.startswith("{"):
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    data = obj.get("data")
    if not isinstance(data, dict):
        return None
    biz_msg = data.get("biz_msg")
    biz_code = data.get("biz_code")
    if biz_msg is None and biz_code is None:
        return None
    biz_data = data.get("biz_data") if isinstance(data.get("biz_data"), dict) else {}
    mute_until = biz_data.get("mute_until")
    return {
        "biz_code": biz_code,
        "biz_msg": str(biz_msg or ""),
        "mute_until": float(mute_until) if mute_until is not None else None,
    }


def raise_if_user_muted(line: str) -> None:
    """若该行表示账号 mute，抛出 :class:`DeepSeekUserMutedError`。"""
    biz = parse_biz_error_from_line(line)
    if not biz:
        return
    msg = str(biz.get("biz_msg") or "").lower()
    if "muted" not in msg and biz.get("biz_code") != 5:
        return
    raise DeepSeekUserMutedError(
        biz_msg=str(biz.get("biz_msg") or "user is muted"),
        mute_until=biz.get("mute_until"),
    )
