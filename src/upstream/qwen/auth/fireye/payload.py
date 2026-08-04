from __future__ import annotations

"""bx-ua 二进制布局：HAR 对齐的 header / session / context / body。"""

import hashlib
import hmac
import re
import secrets
import struct
import time
from typing import Final, List, Pattern, Tuple

from upstream.qwen.auth.fireye.codec import expected_magic
from upstream.qwen.auth.fireye.env import FingerprintEnv
from upstream.qwen.auth.fireye.session import FireyeSession

_HEADER_LEN: Final[int] = 12
_BLOCK_LEN: Final[int] = 16
_MIN_BODY: Final[int] = 1100
_MAX_BODY: Final[int] = 1260
_CHAT_ID_RE: Final[Pattern[str]] = re.compile(
    r"(?:chat[_-]?id=|/c/)([0-9a-f-]{36})",
    re.IGNORECASE,
)


def _derive_block(seed: bytes, label: str, url: str) -> bytes:
    msg = f"{label}|{url}".encode()
    return hmac.new(seed, msg, hashlib.sha256).digest()[:_BLOCK_LEN]


def _extract_chat_id(url: str) -> str:
    m = _CHAT_ID_RE.search(url)
    return m.group(1) if m else ""


def _build_header(seq: int, ts_ms: int) -> bytes:
    # 字节 5 恒为 0x94；HAR/Node 样本均符合。
    buf = bytearray(_HEADER_LEN)
    buf[0:4] = secrets.token_bytes(4)
    buf[4] = (seq >> 8) & 0xFF
    buf[5] = expected_magic()
    buf[6:8] = struct.pack(">H", seq & 0xFFFF)
    tick = (ts_ms // 1000) & 0xFFFFFFFF
    buf[8:12] = struct.pack(">I", tick ^ 0xE8E80000)
    return bytes(buf)


def _expand_tables(seed: bytes) -> Tuple[bytes, bytes, bytes, bytes]:
    out: List[bytes] = []
    for idx in range(4):
        block = bytearray(256)
        for pos in range(256):
            block[pos] = hmac.new(
                seed,
                f"tbl{idx}:{pos}".encode(),
                hashlib.sha256,
            ).digest()[0]
        out.append(bytes(block))
    return out[0], out[1], out[2], out[3]


def _xor_round(data: bytearray, tables: Tuple[bytes, bytes, bytes, bytes]) -> None:
    # 对齐 fireyejs case 60：四表按 4 字节块异或。
    t0, t1, t2, t3 = tables
    for off in range(0, len(data) - 3, 4):
        b0, b1, b2, b3 = data[off], data[off + 1], data[off + 2], data[off + 3]
        data[off] = (b0 ^ t0[b0]) ^ t3[b3]
        data[off + 1] = (b1 ^ t1[b1]) ^ t0[b0]
        data[off + 2] = (b2 ^ t2[b2]) ^ t1[b1]
        data[off + 3] = (b3 ^ t3[b3]) ^ t2[b2]


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(
            hmac.new(
                key,
                nonce + struct.pack(">I", counter),
                hashlib.sha256,
            ).digest()
        )
        counter += 1
    return bytes(out[:length])


def _target_length(url: str, seed: bytes) -> int:
    span = _MAX_BODY - _MIN_BODY
    bias = seed[0] % span
    base = _MIN_BODY + bias
    if "completions" in url:
        base += 80
    if "chats/new" in url:
        base += 50
    return min(base, _MAX_BODY)


def _session_scope(url: str) -> str:
    """HAR：同会话内 session 块稳定；completions 按 chat_id 变。"""
    chat_id = _extract_chat_id(url)
    if chat_id and "completions" in url:
        return f"chat:{chat_id}"
    return "session"


def _context_scope(url: str) -> str:
    """HAR：context 块在 session 内稳定；completions 按完整 URL 变。"""
    if "completions" in url:
        return url
    return "session"


def build_fy_payload(
    *,
    fingerprint: str,
    req_url: str,
    env: FingerprintEnv,
    seq: int,
    sess: FireyeSession | None = None,
) -> bytes:
    ts_ms = int(time.time() * 1000)
    seed = env.digest(fingerprint)
    sess_scope = _session_scope(req_url)
    ctx_scope = _context_scope(req_url)
    if sess is not None and sess_scope == "session" and sess.session_block:
        session_block = sess.session_block
    else:
        session_block = _derive_block(seed, "session", sess_scope)
        if sess is not None and sess_scope == "session":
            sess.session_block = session_block
    if sess is not None and ctx_scope == "session" and sess.context_block:
        context_block = sess.context_block
    else:
        context_block = _derive_block(seed, "context", ctx_scope)
        if sess is not None and ctx_scope == "session":
            sess.context_block = context_block
    body_len = _target_length(req_url, seed)
    plain = bytearray(body_len)
    meta = f"{fingerprint}|{req_url}|{seq}|{ts_ms}".encode()
    plain[: len(meta)] = meta
    fill = _keystream(seed, b"fill", body_len - len(meta))
    plain[len(meta) :] = fill
    tables = _expand_tables(hmac.new(seed, b"round", hashlib.sha256).digest())
    _xor_round(plain, tables)
    stream = _keystream(
        hmac.new(seed, b"body", hashlib.sha256).digest(),
        struct.pack(">Q", ts_ms) + session_block,
        body_len,
    )
    body = bytes(a ^ b for a, b in zip(plain, stream))
    return (
        _build_header(seq, ts_ms)
        + session_block
        + context_block
        + body
    )
