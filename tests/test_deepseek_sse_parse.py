from __future__ import annotations

"""DeepSeek SSE 行解析：response/fragments APPEND 与 token usage。"""

from pathlib import Path

from upstream.deepseek.lib.adapter.strmrun import _StreamRunMixin
from upstream.deepseek.lib.stream.strmpars import StreamParser

_SSE_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "logs"
    / "sse"
    / "req-1785690996-1be628e91e78.sse"
)


def _parse_sse_file(path: Path, *, include_thinking: bool = False) -> StreamParser:
    parser = StreamParser(include_thinking=include_thinking)
    parser.begin_stream(is_continuation=False)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parser.parse_line(line)
    return parser


def test_response_fragments_append_includes_nihao() -> None:
    if not _SSE_FIXTURE.is_file():
        # 本地无落盘样本时跳过（CI 不依赖 logs/）
        import pytest

        pytest.skip(f"missing fixture {_SSE_FIXTURE}")

    parser = _parse_sse_file(_SSE_FIXTURE)
    assert parser.accumulated_content == "你好。有什么可以帮你的？"


def test_response_fragments_append_line_parses_content() -> None:
    parser = StreamParser(include_thinking=False)
    parser.begin_stream(is_continuation=False)
    line = (
        'data: {"p":"response/fragments","o":"APPEND",'
        '"v":[{"id":3,"type":"RESPONSE","content":"你好","references":[],"stage_id":1}]}'
    )
    result = parser.parse_line(line)
    assert result == {"type": "content", "content": "你好"}
    assert parser.accumulated_content == "你好"


def test_content_delta_period_counts_as_token_in_tracker() -> None:
    from server.formats import UpstreamUsageTracker

    parser = StreamParser(include_thinking=False)
    parser.begin_stream(is_continuation=False)
    parser.parse_line(
        'data: {"p":"response/fragments","o":"APPEND",'
        '"v":[{"id":3,"type":"RESPONSE","content":"你好","references":[],"stage_id":1}]}'
    )
    parser.parse_line('data: {"p":"response/fragments/-1/content","v":"。"}')
    assert parser.accumulated_content == "你好。"

    tracker = UpstreamUsageTracker()
    tracker.set_estimated_input_from_prompt_chars(100)
    tracker.ingest_upstream_event({"type": "answer", "content": "。"})
    usage = tracker.openai_stream_usage()
    assert usage is not None
    assert usage["completion_tokens"] == 1


def test_compute_usage_prefers_upstream_accumulated_token_usage() -> None:
    parser = StreamParser(include_thinking=False)
    parser.begin_stream(is_continuation=False)
    parser.parse_line(
        'data: {"p":"response","o":"BATCH","v":[{"p":"accumulated_token_usage","v":40134}]}'
    )
    usage_chunks = _StreamRunMixin._compute_usage(None, parser, "用户: hi")  # noqa: SLF001
    usage = usage_chunks[0]["usage"]
    assert usage["total_tokens"] == 40134
    assert usage["prompt_tokens"] >= 1
    assert usage["completion_tokens"] == 40134 - usage["prompt_tokens"]
