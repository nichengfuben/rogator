from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

_MAX_RESULT = 100_000
_META_FIELDS = frozenset({
    "id", "execId", "spanContext", "acceptHookAdditionalContexts",
    "requestContextArgs", "hookAdditionalContexts",
})


def truncate(text: str, limit: int = _MAX_RESULT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated, {len(text)}B total]"


def tool_type(msg: Dict[str, Any]) -> Optional[str]:
    for key in msg:
        if key not in _META_FIELDS:
            return key
    return None


def base_msg(msg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": msg.get("id", 0)}
    exec_id = msg.get("execId")
    if exec_id:
        out["execId"] = exec_id
    return out


def elapsed_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def finish(base: Dict[str, Any], start: float, field: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    result[field] = payload
    result["localExecutionTimeMs"] = elapsed_ms(start)
    return result
