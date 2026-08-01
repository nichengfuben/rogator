from __future__ import annotations

import json
from typing import Any, Dict, Optional

from echotools.logger import get_logger

from server.model.token_estimate import estimate_stream_tokens_from_char_count

logger = get_logger("rogator")


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


def anthropic_cache_usage_fields(raw: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Anthropic message_start：映射 Qwen cached_tokens → cache_read_input_tokens。"""
    cached = extract_cached_tokens(raw)
    if not cached:
        return {}
    return {"cache_read_input_tokens": cached}


def should_emit_anthropic_message_start(event: Dict[str, Any], message_started: bool) -> bool:
    """Anthropic 流式：首次见到上游 usage 时发 message_start（此前缓冲 content）。"""
    if message_started:
        return False
    if event.get("type") == "response_created":
        return False
    return extract_usage_from_event(event) is not None


class UpstreamUsageTracker:
    """从上游 SSE 累积 token 用量（Qwen 为递增快照，取最后一次）。"""

    __slots__ = (
        "_usage",
        "_cached_tokens",
        "_seen",
        "_last_raw_usage",
        "_first_raw_usage",
        "_estimated_input",
        "_output_chars",
    )

    def __init__(self) -> None:
        self._usage = build_usage_dict()
        self._cached_tokens = 0
        self._seen = False
        self._last_raw_usage: Optional[Dict[str, Any]] = None
        self._first_raw_usage: Optional[Dict[str, Any]] = None
        self._estimated_input = 0
        self._output_chars = 0

    def set_estimated_input_from_prompt_chars(self, prompt_chars: int) -> None:
        """流式：inject 后实际发送 prompt 字符 // 4，供客户端实时显示 input。"""
        self._estimated_input = estimate_stream_tokens_from_char_count(prompt_chars)

    def add_output_chars(self, char_count: int) -> None:
        """流式：累计已生成字符，上游无 usage 时用 // 4 估算 output。"""
        if char_count > 0:
            self._output_chars += char_count

    def ingest_event(self, event: Dict[str, Any]) -> None:
        raw = extract_usage_from_event(event)
        if not raw:
            return
        if not self._seen:
            self._first_raw_usage = dict(raw)
        self._seen = True
        self._last_raw_usage = dict(raw)
        self._usage = normalize_upstream_usage(raw)
        self._cached_tokens = extract_cached_tokens(raw)

    def ingest_upstream_event(self, event: Dict[str, Any]) -> Optional[str]:
        """流式摄取：prompt_meta / usage / thinking|answer 计字；返回 etype。"""
        etype = event.get("type")
        etype_s = etype if isinstance(etype, str) else None
        if etype_s == "prompt_meta":
            self.set_estimated_input_from_prompt_chars(int(event.get("prompt_chars") or 0))
            return etype_s
        self.ingest_event(event)
        content = event.get("content", "")
        if content and etype_s in ("thinking", "answer"):
            self.add_output_chars(len(content))
        return etype_s

    @property
    def last_raw_usage(self) -> Optional[Dict[str, Any]]:
        """chat.qwen.ai 上游 usage 原始 JSON（最后一次快照）。"""
        return self._last_raw_usage

    @property
    def first_raw_usage(self) -> Optional[Dict[str, Any]]:
        """首次见到的上游 usage（用于 Anthropic message_start input）。"""
        return self._first_raw_usage

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
        """Anthropic message_start：有上游用上游 input，否则 inject prompt // 4；output 固定 0。"""
        raw = self._first_raw_usage or self._last_raw_usage
        if raw:
            prompt = normalize_upstream_usage(raw)["prompt_tokens"]
        else:
            prompt = self._estimated_input
        usage: Dict[str, int] = {
            "input_tokens": prompt,
            "output_tokens": 0,
        }
        if raw:
            usage.update(anthropic_cache_usage_fields(raw))
        return usage

    @property
    def anthropic_message_delta_usage(self) -> Dict[str, int]:
        """Anthropic message_delta：有上游用上游 output，否则已生成字符 // 4。"""
        if self._seen:
            return {"output_tokens": self._usage["completion_tokens"]}
        return {"output_tokens": estimate_stream_tokens_from_char_count(self._output_chars)}

    def openai_stream_usage(self) -> Optional[Dict[str, Any]]:
        """流式 usage：上游到达后用真实值，否则 prompt/output 均用 // 4 估算。"""
        if self._seen:
            return self.openai_usage
        if not self._estimated_input and not self._output_chars:
            return None
        prompt = self._estimated_input
        completion = estimate_stream_tokens_from_char_count(self._output_chars)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }


def log_qwen_upstream_usage(req_id: str, tracker: UpstreamUsageTracker) -> None:
    """响应结束后 DEBUG 打印 chat.qwen.ai 上游 usage 完整 JSON。"""
    raw = tracker.last_raw_usage
    if raw is None:
        logger.debug("chat.qwen.ai usage req=%s (none)", req_id)
        return
    logger.debug(
        "chat.qwen.ai usage req=%s %s",
        req_id,
        json.dumps(raw, ensure_ascii=False, sort_keys=True),
    )


def openai_stream_include_usage(body: Optional[Dict[str, Any]]) -> bool:
    """流式是否按 OpenAI 官方 include_usage 协议发 usage chunk。

    适配器默认开启（kimi-code-cli 等客户端依赖流式 usage）；
    显式 ``stream_options.include_usage=false`` 可关闭。
    """
    if not body:
        return True
    opts = body.get("stream_options")
    if not isinstance(opts, dict):
        return True
    if "include_usage" in opts:
        return bool(opts["include_usage"])
    return True
