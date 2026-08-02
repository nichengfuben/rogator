from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


def load_use_proxy() -> bool:
    """Standalone Rogator: proxy optional; default off."""
    return False


@dataclass
class Candidate:
    """轻量候选项（本地轻量替代 core.dispatch.cand）。"""

    id: str
    platform: str
    resource_id: str
    models: list
    context_length: Optional[int] = None
    meta: dict = field(default_factory=dict)
    chat: bool = False
    completions: bool = False
    responses: bool = False
    thinking: bool = False
    search: bool = False
    tools: bool = False
    continuation: bool = False
    vision: bool = False

    def __init__(
        self,
        *,
        id: str,
        platform: str,
        resource_id: str,
        models: list,
        context_length: Optional[int] = None,
        meta: Optional[dict] = None,
        **caps: bool,
    ) -> None:
        self.id = id
        self.platform = platform
        self.resource_id = resource_id
        self.models = list(models)
        self.context_length = context_length
        self.meta = dict(meta) if meta else {}
        for key, value in caps.items():
            setattr(self, key, value)


def make_candidate_id(platform: str, suffix: str) -> str:
    """生成候选项 ID。"""
    raw = "{}:{}".format(platform, suffix)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
