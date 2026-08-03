from __future__ import annotations

"""fncall prompt 注入与流式 parser delta 共享辅助。"""

from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Final,
    Iterator,
    List,
    Optional,
    Tuple,
)

from echotools.base.logger import get_logger
from echotools.exec.fncall.prompt.inject import inject_fncall as _echotools_inject
from echotools.exec.fncall.tool_id import fix_tool_call_id
from echotools.exec.protocol.base import ToolProtocol

from server.config import CONFIG, LOG_DIR

__all__ = [
    "STREAM_CHUNK_SIZE",
    "advance_partial_buffer",
    "emit_parser_stream_deltas",
    "finalize_parser_tool_calls",
    "inject_fncall_for_request",
    "iter_parser_stream_deltas",
    "iter_text_chunks",
    "prompt_dump_dir",
    "reconcile_pending_tool_index",
    "resolve_streamed_tool_calls",
    "take_parser_final_delta",
]

logger = get_logger("rogator")

_PROMPTS_SUBDIR = "prompts"
STREAM_CHUNK_SIZE: Final[int] = 20


def prompt_dump_dir() -> Path:
    return LOG_DIR / _PROMPTS_SUBDIR


def _should_dump_prompt() -> bool:
    return bool(CONFIG.record_prompt or CONFIG.print_prompt)


def _dump_prompt(prompt: str, req_id: str) -> None:
    dump_dir = prompt_dump_dir()
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / f"{req_id}.txt"
    path.write_text(prompt, encoding="utf-8")
    logger.debug("prompt 已写入 %s", path)


def inject_fncall_for_request(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    protocol: ToolProtocol,
    *,
    req_id: str,
    api: str,
    model: str,
    lang: str = "zh",
    user_system_prompt: str = "",
    loop_detection_threshold: int = 3,
    protocol_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    injected = _echotools_inject(
        messages=messages,
        tools=tools,
        protocol=protocol,
        lang=lang,
        user_system_prompt=user_system_prompt,
        loop_detection_threshold=loop_detection_threshold,
        dump_prompt=False,
        dump_dir=None,
        protocol_options=protocol_options,
    )
    prompt = injected[0]["content"]
    if _should_dump_prompt():
        _dump_prompt(prompt, req_id)
    logger.info(
        "inject prompt api=%s req_id=%s model=%s chars=%d tools=%d dump_dir=%s",
        api,
        req_id,
        model,
        len(prompt),
        len(tools or []),
        str(prompt_dump_dir()) if _should_dump_prompt() else None,
    )
    return injected


# ==== entml 流式解析共享 ====


def iter_text_chunks(text: str, size: int = STREAM_CHUNK_SIZE) -> Iterator[str]:
    """按固定大小切分文本；空串不产出。"""
    if not text:
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


def iter_parser_stream_deltas(parser) -> Iterator[Tuple[str, str]]:
    """取出 parser 中全部待发 stream delta（跳过空 partial_json）。"""
    while True:
        delta_info = parser.consume_stream_delta()
        if not delta_info:
            break
        name, partial_json = delta_info
        if not partial_json:
            continue
        yield name, partial_json


async def emit_parser_stream_deltas(
    parser,
    on_delta: Callable[[str, str], Awaitable[bool]],
) -> bool:
    """对每个 (name, partial_json) 调用 on_delta；返回 False 表示中止。"""
    for name, partial_json in iter_parser_stream_deltas(parser):
        if not await on_delta(name, partial_json):
            return False
    return True


def finalize_parser_tool_calls(
    parser,
    *,
    warn: Optional[Callable[..., None]] = None,
    warn_prefix: str = "stream parser.finalize failed",
) -> Tuple[str, List[Dict[str, Any]]]:
    """调用 parser.finalize 并规范化 tool_call id。"""
    final_text = parser.partial_text
    try:
        final_text, parsed_calls = parser.finalize()
        return final_text, [fix_tool_call_id(tc) for tc in parsed_calls]
    except Exception as e:
        if warn is not None:
            warn("%s: %s", warn_prefix, e)
        return final_text, []


def resolve_streamed_tool_calls(
    parsed_calls: List[Dict[str, Any]],
    streamed_tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return parsed_calls or streamed_tool_calls


def reconcile_pending_tool_index(
    pending: int,
    all_tool_calls: List[Dict[str, Any]],
    stream_tool_blocks_sent: int,
) -> int:
    """用已流式发出的 tool 块数校正 pending 索引/计数。"""
    if all_tool_calls and stream_tool_blocks_sent:
        return max(pending, min(len(all_tool_calls), stream_tool_blocks_sent))
    return pending


def take_parser_final_delta(parser) -> Optional[Tuple[str, str]]:
    """流结束时补全未闭合 invoke 的最终 delta；已 closed 或无内容则 None。"""
    if parser.streaming_invoke_closed:
        return None
    final_delta = parser.complete_stream_delta_if_needed()
    if not final_delta:
        return None
    name, piece = final_delta
    if not piece:
        return None
    return name, piece


def advance_partial_buffer(last_len: int, current: str) -> Tuple[str, int]:
    """返回相对 last_len 的新增片段与新长度。"""
    if len(current) <= last_len:
        return "", last_len
    return current[last_len:], len(current)
