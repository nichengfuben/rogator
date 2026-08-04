from __future__ import annotations

"""Baxia 会话态与 token 组装（对齐 FE：umid 稳定、bx-ua 按 reqUrl 刷新）。"""

import os
import logging
import threading
from typing import Dict, Final, Literal, Optional, Tuple

from upstream.qwen.auth.crypto import (
    generate_bxua,
    generate_fingerprint,
    validate_bxumidtoken,
)
from upstream.qwen.chat.routes import BAXIA_SDK_VERSION

logger = logging.getLogger("rogator")

BAXIA_UA_PATH_MARKERS: Final[Tuple[str, ...]] = (
    "/api/chat/completions",
    "/api/chats/new",
    "/api/chat/completed",
    "/api/v1/chats",
    "/api/v1/chats/all/tags",
    "/api/task/suggestions/completions",
    "/api/v1/tasks/status",
    "/api/v1/files/getstsToken",
    "/api/task/title/completions",
    "/api/task/tags/completions",
    "/api/parse_url",
    "/api/v2/chats",
    "/api/v2/chat/completions",
    "/api/v2/task/suggestions/completions",
    "/api/v2/files/getstsToken",
    "/api/v2/community",
    "/api/v2/tts/completions",
    "/api/v2/files/getfilelink",
    "/api/v2/files/parse",
    "/api/v2/files/parse/status",
    "/api/v2/evaluations/feedback",
)
BaxiaMode = Literal["full", "version", "none"]

_runtime_lock = threading.Lock()
_runtime_fp: str = ""
_runtime_umid: str = ""


def path_needs_baxia_ua(path: str) -> bool:
    return any(marker in path for marker in BAXIA_UA_PATH_MARKERS)


def resolve_baxia_mode(path: str = "", *, explicit: Optional[BaxiaMode] = None) -> BaxiaMode:
    if explicit is not None:
        return explicit
    if not path:
        return "full"
    if path_needs_baxia_ua(path):
        return "full"
    return "version"


def ensure_baxia_runtime(*, fingerprint_override: str = "") -> Tuple[str, str]:
    global _runtime_fp, _runtime_umid
    with _runtime_lock:
        if fingerprint_override.strip():
            _runtime_fp = fingerprint_override.strip()
        elif not _runtime_fp:
            _runtime_fp = generate_fingerprint()
        if not _runtime_umid:
            env_umid = os.environ.get("QWEN_BX_UMIDTOKEN", "").strip()
            if env_umid:
                _runtime_umid = env_umid
            else:
                from upstream.qwen.auth.fireye.umid import get_umid_token

                _runtime_umid = get_umid_token(_runtime_fp)
        return _runtime_fp, _runtime_umid


def reset_baxia_runtime() -> None:
    global _runtime_fp, _runtime_umid
    with _runtime_lock:
        _runtime_fp = ""
        _runtime_umid = ""
    try:
        from upstream.qwen.auth.fireye import reset_session

        reset_session()
    except Exception:
        pass


def get_baxia_tokens(
    *,
    fingerprint_override: str = "",
    req_url: str = "",
) -> Dict[str, str]:
    fingerprint, umid = ensure_baxia_runtime(
        fingerprint_override=fingerprint_override,
    )
    bx_ua = ""
    try:
        from upstream.qwen.auth.fireye import (
            bind_fingerprint,
            get_fy_token,
            get_uid_token,
            resolve_baxia_req_url,
        )

        bind_fingerprint(fingerprint, umid=umid)
        cand = get_fy_token(req_url or resolve_baxia_req_url(), fingerprint=fingerprint)
        if cand.startswith("231!") and len(cand) > 100:
            bx_ua = cand
        fy_umid = get_uid_token(fingerprint=fingerprint, bx_ua=bx_ua).strip()
        if fy_umid and validate_bxumidtoken(fy_umid):
            umid = fy_umid
            with _runtime_lock:
                _runtime_umid = umid
    except Exception as exc:
        logger.debug("fireye token fallback: %s", exc)
    if not bx_ua:
        bx_ua = generate_bxua(fingerprint)
    return {
        "bxV": BAXIA_SDK_VERSION,
        "bxUa": bx_ua,
        "bxUmidToken": umid,
        "fingerprint": fingerprint,
    }


__all__ = [
    "BAXIA_UA_PATH_MARKERS",
    "BaxiaMode",
    "ensure_baxia_runtime",
    "get_baxia_tokens",
    "path_needs_baxia_ua",
    "reset_baxia_runtime",
    "resolve_baxia_mode",
]
