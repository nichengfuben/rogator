from __future__ import annotations

"""231! 前缀与 fireyejs 同款标准 Base64（+/，非 url-safe）。"""

import base64
from typing import Final

FIREYE_VERSION: Final[int] = 231
TOKEN_PREFIX: Final[str] = f"{FIREYE_VERSION}!"
_MAGIC_BYTE: Final[int] = 0x94


def b64_encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def b64_decode(text: str) -> bytes:
    return base64.b64decode(text)


def wrap_token(raw: bytes) -> str:
    return TOKEN_PREFIX + b64_encode(raw)


def unwrap_token(token: str) -> bytes:
    if not token.startswith(TOKEN_PREFIX):
        raise ValueError("bx-ua 缺少 231! 前缀")
    return b64_decode(token[len(TOKEN_PREFIX) :])


def magic_byte_offset() -> int:
    return 5


def expected_magic() -> int:
    return _MAGIC_BYTE
