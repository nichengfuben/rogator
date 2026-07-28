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
    """与 handler 一致：分块 feed 后取各 invoke 的流式 arguments 快照。"""
    parser = FncallStreamParser(protocol=get_protocol("entml"), tools=tools)
    step = max(1, min(chunk_size, len(text) or 1))
    for i in range(0, len(text), step):
        parser.feed(text[i : i + step])
        while parser.consume_stream_delta():
            pass
    comp = parser.complete_stream_delta_if_needed()
    if comp:
        parser._track_stream_delta(comp[0], comp[1])
        while parser.consume_stream_delta():
            pass
    parser.finalize()
    return [snap for snap in parser.stream_invoke_argument_snapshots() if snap]


def simulate_anthropic_wire_json(
    text: str,
    tools: List[Dict[str, Any]],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """模拟 anthro handler：drain delta，最终以各 slot 快照为 wire JSON。"""
    parser = FncallStreamParser(protocol=get_protocol("entml"), tools=tools)
    step = max(1, min(chunk_size, len(text) or 1))
    for i in range(0, len(text), step):
        parser.feed(text[i : i + step])
        while parser.consume_stream_delta():
            pass
    comp = parser.complete_stream_delta_if_needed()
    if comp:
        parser._track_stream_delta(comp[0], comp[1])
        while parser.consume_stream_delta():
            pass
    parser.finalize()
    wire = [snap for snap in parser.stream_invoke_argument_snapshots() if snap]
    all_calls = batch_parse(text, tools)
    return wire, all_calls


def simulate_openai_wire_json(
    text: str,
    tools: List[Dict[str, Any]],
    chunk_size: int,
) -> Tuple[Dict[int, str], List[Dict[str, Any]]]:
    parser = FncallStreamParser(protocol=get_protocol("entml"), tools=tools)
    step = max(1, min(chunk_size, len(text) or 1))
    for i in range(0, len(text), step):
        parser.feed(text[i : i + step])
        while parser.consume_stream_delta():
            pass
    comp = parser.complete_stream_delta_if_needed()
    if comp:
        parser._track_stream_delta(comp[0], comp[1])
        while parser.consume_stream_delta():
            pass
    parser.finalize()
    snaps = [snap for snap in parser.stream_invoke_argument_snapshots() if snap]
    by_index = {idx: snap for idx, snap in enumerate(snaps)}
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
