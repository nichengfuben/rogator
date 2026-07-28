"""Anthropic 流式 tool_use：同一 invoke 多 chunk 时 JSON 必须完整。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_QWEN = Path(__file__).resolve().parents[1]
if str(_QWEN) not in sys.path:
    sys.path.insert(0, str(_QWEN))

_ECHO = _QWEN.parent.parent / "PyPi" / "echotools-sdk" / "src"
if _ECHO.is_dir():
    sys.path.insert(0, str(_ECHO))

from echotools.exec.fncall import get_protocol
from echotools.exec.fncall.parsers.stream import FncallStreamParser

BASH_TEXT = (
    '<entml:invoke name="Bash">\n'
    '<entml:parameter name="command">echo "Bash tool is working"</entml:parameter>\n'
    "</entml:invoke>"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def _simulate_anthro_wire(text: str, chunk: int) -> tuple[list[str], list[dict]]:
    """模拟修复后的 anthro 逻辑：tool_use 块内不重复 content_block_stop。"""
    parser = FncallStreamParser(protocol=get_protocol("entml"), tools=TOOLS)
    events: list[str] = []
    stream_tool = None
    block_type = None

    def stop(idx: int) -> None:
        events.append(f"stop:{idx}")

    def json_delta(idx: int, piece: str) -> None:
        events.append(f"json:{idx}:{piece}")

    for i in range(0, len(text), chunk):
        ready = parser.feed(text[i : i + chunk])
        while True:
            d = parser.consume_stream_delta()
            if not d:
                break
            name, partial = d
            if block_type is not None and block_type != "tool_use":
                stop(block_type)  # simplified
                block_type = None
            if stream_tool is not None and stream_tool["name"] != name:
                stop(stream_tool["idx"])
                stream_tool = None
            if stream_tool is None:
                stream_tool = {"name": name, "idx": len(events), "buf": ""}
                events.append(f"start:{name}")
                block_type = "tool_use"
            stream_tool["buf"] += partial
            json_delta(stream_tool["idx"], partial)
        if ready:
            if stream_tool is not None:
                exp = ready[0]["function"]["arguments"]
                if exp.startswith(stream_tool["buf"]) and len(exp) > len(stream_tool["buf"]):
                    tail = exp[len(stream_tool["buf"]) :]
                    stream_tool["buf"] += tail
                    json_delta(stream_tool["idx"], tail)
                stop(stream_tool["idx"])
                stream_tool = None
                block_type = None
            ready = []

    parser.finalize()
    return events, parser.get_ready_tool_calls() or []


@pytest.mark.parametrize("chunk", [1, 3, 5, 20])
def test_bash_invoke_single_stop_and_valid_json(chunk: int) -> None:
    events, _ = _simulate_anthro_wire(BASH_TEXT, chunk)
    stops = [e for e in events if e.startswith("stop:")]
    assert len(stops) == 1, f"expected one content_block_stop, got {stops!r}"
    merged = ""
    for e in events:
        if e.startswith("json:"):
            merged += e.split(":", 2)[-1]
    json.loads(merged)
    assert "Bash tool is working" in merged


def _simulate_truncated_anthro_wire(text: str, chunk: int) -> str:
    """上游截断、无 </entml:invoke>：force_close 后 wire JSON 必须可解析。"""
    parser = FncallStreamParser(protocol=get_protocol("entml"), tools=TOOLS)
    merged = ""
    for i in range(0, len(text), chunk):
        parser.feed(text[i : i + chunk])
        while True:
            d = parser.consume_stream_delta()
            if not d:
                break
            merged += d[1]
    if not parser.streaming_invoke_closed:
        comp = parser.complete_stream_delta_if_needed()
        if comp:
            merged += comp[1]
    parser.finalize()
    return merged


@pytest.mark.parametrize("chunk", [1, 17, 64])
def test_truncated_large_bash_force_close_valid_json(chunk: int) -> None:
    """回归：~7KiB Bash 在 </entml:invoke> 前截断时 client 累积 JSON 必须合法。"""
    cmd = 'python -c "import base64;' + ("X" * 6800)
    truncated = (
        '<entml:invoke name="Bash">\n'
        f'<entml:parameter name="command">{cmd}'
    )
    assert len(truncated) > 6800
    merged = _simulate_truncated_anthro_wire(truncated, chunk)
    assert 6800 < len(merged) < 7200
    obj = json.loads(merged)
    assert obj["command"].startswith('python -c "import base64;')
    assert len(obj["command"]) > 6800
