from __future__ import annotations

"""ByteLineBuffer 单元测试：覆盖跨 chunk 半行、尾部空 chunk、累积单行上限。"""

import pytest

from upstream.qwen.chat.sse import ByteLineBuffer, iter_byte_lines


def test_feed_single_chunk_with_newlines() -> None:
    buf = ByteLineBuffer()
    assert buf.feed(b"data: a\ndata: b\n") == [b"data: a", b"data: b"]
    assert buf.feed(b"") == []


def test_feed_split_across_chunks() -> None:
    buf = ByteLineBuffer()
    assert buf.feed(b"data: par") == []
    assert buf.feed(b"tial\nnext") == [b"data: partial"]
    assert buf.pending() == len(b"next")
    tail = buf.flush()
    assert tail == b"next"


def test_feed_carriage_return_stripped_upstream() -> None:
    """行内含 ``\\r\\n`` 时按字节切行，调用方在拿到 bytes 后自行 rstrip('\\r')。"""
    buf = ByteLineBuffer()
    lines = buf.feed(b"data: a\r\ndata: b\r\n")
    assert lines == [b"data: a\r", b"data: b\r"]


def test_feed_over_max_pending_raises() -> None:
    buf = ByteLineBuffer(max_pending=4)
    with pytest.raises(BufferError):
        buf.feed(b"aaaaa")


def test_feed_flush_clears_pending() -> None:
    buf = ByteLineBuffer()
    buf.feed(b"tail-no-newline")
    assert buf.flush() == b"tail-no-newline"
    assert buf.pending() == 0
    assert buf.flush() == b""


def test_iter_byte_lines_combines_partials() -> None:
    chunks = iter([b"a\nb", b"\nc\n", b"d"])
    out = list(iter_byte_lines(chunks))
    assert out == [b"a", b"b", b"c", b"d"]


def test_pending_grows_only_with_partial_line() -> None:
    buf = ByteLineBuffer()
    buf.feed(b"data: x\ndata:")
    assert buf.pending() == len(b"data:")


def test_chunk_hits_counter() -> None:
    buf = ByteLineBuffer()
    buf.feed(b"a\n")
    buf.feed(b"b\n")
    buf.feed(b"")
    assert buf.chunk_hits == 2