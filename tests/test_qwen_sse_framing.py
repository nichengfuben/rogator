from __future__ import annotations

from upstream.qwen.chat.sse import SseEventAssembler, parse_sse_event


def test_assembler_dispatches_on_blank_line() -> None:
    asm = SseEventAssembler()
    assert asm.feed_line('data: {"response.created":{"response_id":"r1"}}') is None
    payload = asm.feed_line("")
    assert payload is not None
    event = parse_sse_event(payload)
    assert event is not None
    assert event.get("type") == "response_created"
    assert event.get("response_id") == "r1"


def test_assembler_skips_keepalive_phase() -> None:
    payload = '{"choices":[{"delta":{"phase":"KeepAlive","content":"ping"}}]}'
    assert parse_sse_event(payload) is None
