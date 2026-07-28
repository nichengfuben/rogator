"""Rogator entml 矩阵：加载 echotools 语料，模拟 OAI/ANT 流式与非流式路径。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_QWEN_ROOT = Path(__file__).resolve().parents[1]
_TESTS_DIR = Path(__file__).resolve().parent
for path in (_QWEN_ROOT, _TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fixtures.simulated_llm_tool_responses import (  # noqa: E402
    SimulatedCase,
    iter_cases_with_tools,
    tools_for_case,
)
from echotools.exec.fncall import get_protocol  # noqa: E402
from echotools.exec.fncall.parsers.stream import FncallStreamParser  # noqa: E402

CHUNK_SIZES = [1, 5, 17, 64, 9999]


def batch_parse(text: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _, calls = get_protocol("entml").parse(text, tools)
    return calls


def merged_stream_json_per_invoke(
    text: str,
    tools: List[Dict[str, Any]],
    chunk_size: int,
) -> List[str]:
    """与 handler 一致：每轮 feed 后 drain 全部 consume。"""
    parser = FncallStreamParser(protocol=get_protocol("entml"), tools=tools)
    merged_list: List[str] = []
    current_name: Optional[str] = None
    current = ""
    step = max(1, min(chunk_size, len(text) or 1))
    for i in range(0, len(text), step):
        merged_at_start = len(merged_list)
        ready = parser.feed(text[i : i + step])
        while True:
            delta = parser.consume_stream_delta()
            if not delta:
                break
            name, piece = delta
            if current_name is not None and name != current_name:
                merged_list.append(current)
                current = ""
            current_name = name
            current += piece
        if ready:
            if current:
                merged_list.append(current)
                current = ""
                current_name = None
            streamed = len(merged_list) - merged_at_start
            for tc in ready[streamed:]:
                merged_list.append(tc["function"]["arguments"])
    comp = parser.complete_stream_delta_if_needed()
    if comp:
        name, piece = comp
        if current_name is not None and name != current_name:
            merged_list.append(current)
            current = ""
        current_name = name
        current += piece
    parser.finalize()
    if current:
        merged_list.append(current)
    return merged_list


def simulate_anthropic_wire_json(
    text: str,
    tools: List[Dict[str, Any]],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """模拟 anthro handler：drain delta + ready 批补。"""
    parser = FncallStreamParser(protocol=get_protocol("entml"), tools=tools)
    wire: List[str] = []
    cur = ""
    cur_name: Optional[str] = None
    stream_open = False
    step = max(1, min(chunk_size, len(text) or 1))

    def _flush_block() -> None:
        nonlocal cur, cur_name, stream_open
        if cur:
            wire.append(cur)
        cur = ""
        cur_name = None
        stream_open = False

    for i in range(0, len(text), step):
        wire_at_start = len(wire)
        ready = parser.feed(text[i : i + step])
        while True:
            d = parser.consume_stream_delta()
            if not d:
                break
            name, piece = d
            if stream_open and cur_name is not None and name != cur_name:
                _flush_block()
            if not stream_open:
                stream_open = True
                cur_name = name
            cur += piece
        if ready:
            if stream_open:
                _flush_block()
            streamed = len(wire) - wire_at_start
            for tc in ready[streamed:]:
                wire.append(tc["function"]["arguments"])

    parser.finalize()
    late = parser.get_ready_tool_calls()
    if stream_open:
        _flush_block()
    elif late:
        for tc in late:
            wire.append(tc["function"]["arguments"])
    elif not wire:
        pass

    all_calls = batch_parse(text, tools)
    return wire, all_calls


def simulate_openai_wire_json(
    text: str,
    tools: List[Dict[str, Any]],
    chunk_size: int,
) -> Tuple[Dict[int, str], List[Dict[str, Any]]]:
    parser = FncallStreamParser(protocol=get_protocol("entml"), tools=tools)
    by_index: Dict[int, str] = {}
    tool_index = 0
    stream_name: Optional[str] = None
    step = max(1, min(chunk_size, len(text) or 1))

    for i in range(0, len(text), step):
        idx_at_start = tool_index
        ready = parser.feed(text[i : i + step])
        while True:
            d = parser.consume_stream_delta()
            if not d:
                break
            name, piece = d
            if stream_name is not None and name != stream_name:
                tool_index += 1
                stream_name = None
            if stream_name is None:
                stream_name = name
            by_index[tool_index] = by_index.get(tool_index, "") + piece
        if ready:
            if stream_name is not None:
                tool_index += len(ready)
                stream_name = None
            else:
                streamed = tool_index - idx_at_start
                for tc in ready[streamed:]:
                    by_index[tool_index] = tc["function"]["arguments"]
                    tool_index += 1

    parser.finalize()
    late = parser.get_ready_tool_calls()
    if late:
        if stream_name is not None:
            tool_index += len(late)
        else:
            for tc in late:
                by_index[tool_index] = tc["function"]["arguments"]
                tool_index += 1

    return by_index, batch_parse(text, tools)


def assert_args_equal(
    parsed: Dict[str, Any],
    expected: Dict[str, Any],
    *,
    case_id: str,
) -> None:
    assert parsed == expected, f"{case_id}: {parsed!r} != {expected!r}"
    for key, exp in expected.items():
        assert type(parsed[key]) is type(exp), (
            f"{case_id}.{key}: type {type(parsed[key]).__name__} != {type(exp).__name__}"
        )
