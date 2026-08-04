from __future__ import annotations

"""fireye 会话状态：指纹、umid、请求序号。"""

import threading
from dataclasses import dataclass, field

from upstream.qwen.auth.fireye.env import FingerprintEnv, default_env


@dataclass
class FireyeSession:
    fingerprint: str = ""
    umid: str = ""
    seq: int = 0
    env: FingerprintEnv = field(default_factory=default_env)
    session_block: bytes = b""
    context_block: bytes = b""

    def bump_seq(self) -> int:
        self.seq = (self.seq + 1) & 0xFFFF
        return self.seq

    def reset(self) -> None:
        self.fingerprint = ""
        self.umid = ""
        self.seq = 0
        self.session_block = b""
        self.context_block = b""


_lock = threading.Lock()
_session = FireyeSession()


def get_session() -> FireyeSession:
    return _session


def reset_session() -> None:
    global _session
    with _lock:
        _session = FireyeSession()
