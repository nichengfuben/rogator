from __future__ import annotations

"""纯 Python fireye 入口：get_fy_token / get_uid_token。"""

import threading
from typing import Dict, Final
from urllib.parse import urljoin

from upstream.qwen.auth.fireye.codec import wrap_token
from upstream.qwen.auth.fireye.env import FingerprintEnv, default_env
from upstream.qwen.auth.fireye.payload import build_fy_payload
from upstream.qwen.auth.fireye.session import FireyeSession, get_session, reset_session
from upstream.qwen.auth.fireye.umid import get_umid_token

from upstream.qwen.chat.routes import CHAT_ORIGIN, CHAT_PATH

_DEFAULT_ORIGIN: Final[str] = "https://chat.qwen.ai/"
_lock = threading.Lock()


def _ensure_fingerprint(fp: str, sess: FireyeSession) -> str:
    cand = fp.strip() or sess.fingerprint
    if cand:
        sess.fingerprint = cand
        return cand
    from upstream.qwen.auth.crypto import build_fingerprint

    sess.fingerprint = build_fingerprint()
    return sess.fingerprint


def resolve_baxia_req_url(
    api_path: str = "",
    *,
    chat_id: str = "",
    query: str = "",
) -> str:
    """对齐 FE getFYToken({ reqUrl })：完整 URL 含 path/query。"""
    path = (api_path or "").strip() or CHAT_PATH
    if not path.startswith("/"):
        path = "/" + path.lstrip("/")
    url = f"{CHAT_ORIGIN}{path}"
    qparts: list[str] = []
    if query.strip():
        qparts.append(query.strip().lstrip("?"))
    if chat_id.strip() and "chat_id=" not in url and "completions" in path:
        qparts.append(f"chat_id={chat_id.strip()}")
    if qparts:
        url = url + "?" + "&".join(qparts)
    return url


def _normalize_url(req_url: str) -> str:
    raw = (req_url or "").strip()
    if not raw:
        return urljoin(_DEFAULT_ORIGIN, "api/v2/chat/completions")
    if raw.startswith("/"):
        return urljoin(_DEFAULT_ORIGIN, raw.lstrip("/"))
    return raw


def get_fy_token(
    req_url: str = "",
    *,
    fingerprint: str = "",
    env: FingerprintEnv | None = None,
) -> str:
    with _lock:
        sess = get_session()
        fp = _ensure_fingerprint(fingerprint, sess)
        profile = env or sess.env
        seq = sess.bump_seq()
        raw = build_fy_payload(
            fingerprint=fp,
            req_url=_normalize_url(req_url),
            env=profile,
            seq=seq,
            sess=sess,
        )
        return wrap_token(raw)


def get_uid_token(
    req_url: str = "",
    *,
    fingerprint: str = "",
    bx_ua: str = "",
) -> str:
    with _lock:
        sess = get_session()
        fp = _ensure_fingerprint(fingerprint, sess)
        if sess.umid:
            return sess.umid
        sess.umid = get_umid_token(fp, bx_ua=bx_ua)
        return sess.umid


def bind_fingerprint(fingerprint: str, *, umid: str = "") -> None:
    with _lock:
        sess = get_session()
        sess.fingerprint = fingerprint.strip()
        if umid.strip():
            sess.umid = umid.strip()


def request_tokens(
    req_url: str = "",
    *,
    fingerprint: str = "",
) -> Dict[str, str]:
    fp = fingerprint.strip()
    if fp:
        bind_fingerprint(fp)
    bx_ua = get_fy_token(req_url, fingerprint=fp)
    bx_umid = get_uid_token(fingerprint=fp, bx_ua=bx_ua)
    sess = get_session()
    return {
        "bxUa": bx_ua,
        "bxUmidToken": bx_umid,
        "bxV": "2.5.37",
        "fingerprint": sess.fingerprint,
    }


__all__ = [
    "bind_fingerprint",
    "get_fy_token",
    "get_uid_token",
    "request_tokens",
    "reset_session",
    "resolve_baxia_req_url",
]
