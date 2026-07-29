from __future__ import annotations

from typing import Any, Dict, Optional


def build_usage_dict() -> Dict[str, int]:
    """构建默认 usage 字典（所有值为 0）。"""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def extract_usage_from_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从上游 SSE 事件中取出原始 usage 对象。"""
    if event.get("type") == "usage":
        raw = event.get("data")
        return raw if isinstance(raw, dict) else None
    raw = event.get("usage")
    return raw if isinstance(raw, dict) else None


def normalize_upstream_usage(raw: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """将 Qwen 上游 usage 解析为 OpenAI 计数：prompt/completion/total_tokens。"""
    base = build_usage_dict()
    if not raw:
        return base

    prompt = raw.get("prompt_tokens")
    if prompt is None:
        prompt = raw.get("input_tokens", 0)
    completion = raw.get("completion_tokens")
    if completion is None:
        completion = raw.get("output_tokens", 0)

    try:
        prompt_i = max(0, int(prompt or 0))
        completion_i = max(0, int(completion or 0))
    except (TypeError, ValueError):
        return base

    total = raw.get("total_tokens")
    if total is None:
        total_i = prompt_i + completion_i
    else:
        try:
            total_i = max(0, int(total))
        except (TypeError, ValueError):
            total_i = prompt_i + completion_i

    return {
        "prompt_tokens": prompt_i,
        "completion_tokens": completion_i,
        "total_tokens": total_i,
    }


def extract_cached_tokens(raw: Optional[Dict[str, Any]]) -> int:
    """从上游 usage 提取 cached prompt tokens。"""
    if not raw:
        return 0
    details = raw.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return 0
    try:
        return max(0, int(details.get("cached_tokens") or 0))
    except (TypeError, ValueError):
        return 0


def should_emit_anthropic_message_start(event: Dict[str, Any], message_started: bool) -> bool:
    """Anthropic 流式：拿到上游 usage 后再发 message_start（跳过 response.created）。"""
    if message_started:
        return False
    if event.get("type") == "response_created":
        return False
    return extract_usage_from_event(event) is not None


class UpstreamUsageTracker:
    """从上游 SSE 累积 token 用量（Qwen 为递增快照，取最后一次）。"""

    __slots__ = ("_usage", "_cached_tokens", "_seen")

    def __init__(self) -> None:
        self._usage = build_usage_dict()
        self._cached_tokens = 0
        self._seen = False

    def ingest_event(self, event: Dict[str, Any]) -> None:
        raw = extract_usage_from_event(event)
        if not raw:
            return
        self._seen = True
        self._usage = normalize_upstream_usage(raw)
        self._cached_tokens = extract_cached_tokens(raw)

    @property
    def has_usage(self) -> bool:
        return self._seen

    @property
    def openai_usage(self) -> Dict[str, Any]:
        result: Dict[str, Any] = dict(self._usage)
        if self._cached_tokens:
            result["prompt_tokens_details"] = {"cached_tokens": self._cached_tokens}
        return result

    @property
    def anthropic_message_start_usage(self) -> Dict[str, int]:
        """Anthropic message_start：上游 input/output 快照。"""
        out = self._usage["completion_tokens"]
        return {
            "input_tokens": self._usage["prompt_tokens"],
            "output_tokens": max(1, out) if self._seen else 0,
        }

    @property
    def anthropic_message_delta_usage(self) -> Dict[str, int]:
        """Anthropic message_delta：官方仅含累计 output_tokens（上游）。"""
        return {"output_tokens": self._usage["completion_tokens"]}

    def openai_stream_usage(self) -> Optional[Dict[str, Any]]:
        """流式 finish chunk：无上游用量时不伪造零值。"""
        if not self._seen:
            return None
        return self.openai_usage
