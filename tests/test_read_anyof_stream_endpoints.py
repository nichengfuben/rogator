"""Read + anyOf integer 流式 wire：ANT / OAI 端点 json 累积须与 batch 一致。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TESTS = _ROOT / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from echotools.exec.fncall import get_protocol
from echotools.exec.fncall.parsers.stream import FncallStreamParser

from rogator_entml_harness import (
    simulate_anthropic_wire_json,
    simulate_openai_wire_json,
)

READ_TEXT = (
    '<entml:invoke name="Read">\n'
    '<entml:parameter name="path">X:/Project/Local/DeepSeek/core/guard/pow.py</entml:parameter>\n'
    '<entml:parameter name="line_offset">143</entml:parameter>\n'
    '<entml:parameter name="n_lines">15</entml:parameter>\n'
    "</entml:invoke>"
)

READ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line_offset": {
                        "anyOf": [
                            {"type": "integer", "minimum": 1},
                            {"type": "integer", "maximum": -1},
                        ],
                    },
                    "n_lines": {"type": "integer", "exclusiveMinimum": 0},
                },
                "required": ["path"],
            },
        },
    }
]

EXPECTED = {
    "path": "X:/Project/Local/DeepSeek/core/guard/pow.py",
    "line_offset": 143,
    "n_lines": 15,
}


def _batch_args(text: str) -> dict:
    _, calls = get_protocol("entml").parse(text, READ_TOOLS)
    return json.loads(calls[0]["function"]["arguments"])


def _accumulate_stream_deltas(text: str, chunk: int) -> str:
    """与 ANT/OAI handler 相同：累加 ``consume_stream_delta`` 的 partial_json。"""
    parser = FncallStreamParser(protocol=get_protocol("entml"), tools=READ_TOOLS)
    buf = ""
    for i in range(0, len(text), chunk):
        ready = parser.feed(text[i : i + chunk])
        while True:
            delta = parser.consume_stream_delta()
            if not delta:
                break
            buf += delta[1]
        if ready and buf:
            exp = ready[0]["function"]["arguments"]
            if exp.startswith(buf) and len(exp) > len(buf):
                buf += exp[len(buf) :]
    if not parser.streaming_invoke_closed:
        comp = parser.complete_stream_delta_if_needed()
        if comp:
            buf += comp[1]
    parser.finalize()
    return buf


@pytest.mark.parametrize("chunk", [1, 17, 64])
def test_anthropic_wire_read_anyof_matches_batch(chunk: int) -> None:
    batch = _batch_args(READ_TEXT)
    assert batch == EXPECTED
    wire, _calls = simulate_anthropic_wire_json(READ_TEXT, READ_TOOLS, chunk)
    assert len(wire) == 1
    assert json.loads(wire[0]) == batch
    buf = _accumulate_stream_deltas(READ_TEXT, chunk)
    assert json.loads(buf) == batch


@pytest.mark.parametrize("chunk", [1, 17, 64])
def test_openai_wire_read_anyof_matches_batch(chunk: int) -> None:
    batch = _batch_args(READ_TEXT)
    by_idx, _calls = simulate_openai_wire_json(READ_TEXT, READ_TOOLS, chunk)
    assert len(by_idx) == 1
    assert json.loads(by_idx[0]) == batch
    buf = _accumulate_stream_deltas(READ_TEXT, chunk)
    assert json.loads(buf) == batch
