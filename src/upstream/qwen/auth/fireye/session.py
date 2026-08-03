from __future__ import annotations

"""fireye 会话状态：指纹、umid、请求序号。"""

import threading
from dataclasses import dataclass, field

from upstream.qwen.auth.fireye.env import BrowserEnv, default_env


@dataclass
class FireyeSession:
    fingerprint: str = ""
    umid: str = ""
    seq: int = 0
    env: BrowserEnv = field(default_factory=default_env)

    def bump_seq(self) -> int:
        self.seq = (self.seq + 1) & 0xFFFF
        return self.seq

    def reset(self) -> None:
        self.fingerprint = ""
        self.umid = ""
        self.seq = 0


_lock = threading.Lock()
_session = FireyeSession()


def get_session() -> FireyeSession:
    return _session


def reset_session() -> None:
    global _session
    with _lock:
        _session = FireyeSession()
