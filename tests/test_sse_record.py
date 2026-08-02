from __future__ import annotations

"""logs/sse/{req_id}.sse 实时落盘测试。"""

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from server.config import get_config
from server.records.sse_record import (
    SseStreamRecorder,
    append_sse_bytes,
    record_sse_stream,
    sse_dump_dir,
)


def test_sse_dump_dir_under_project_root() -> None:
    assert sse_dump_dir().name == "sse"
    assert sse_dump_dir().parent.name == "logs"


def test_record_sse_disabled_no_file(tmp_path, monkeypatch) -> None:
    dump = tmp_path / "sse"
    monkeypatch.setattr("server.records.sse_record.sse_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "server.records.sse_record.CONFIG",
        replace(get_config(), record_sse=False),
    )
    recorder = SseStreamRecorder("req_off")
    recorder.write(b"data: {}\n\n")
    recorder.close()
    assert list(dump.glob("*.sse")) == []


def test_record_sse_writes_incrementally(
    tmp_path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rogator")
    dump = tmp_path / "sse"
    monkeypatch.setattr("server.records.sse_record.sse_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "server.records.sse_record.CONFIG",
        replace(get_config(), record_sse=True),
    )
    recorder = SseStreamRecorder("req_sse")
    recorder.write(b"data: chunk1\n\n")
    path = dump / "req_sse.sse"
    assert path.is_file()
    assert path.read_bytes() == b"data: chunk1\n\n"
    recorder.write(b"data: chunk2\n\n")
    assert path.read_bytes() == b"data: chunk1\n\ndata: chunk2\n\n"
    recorder.close()
    assert "record sse req_id=req_sse" in caplog.text


def test_record_sse_stream_context_and_append(tmp_path, monkeypatch) -> None:
    dump = tmp_path / "sse"
    monkeypatch.setattr("server.records.sse_record.sse_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "server.records.sse_record.CONFIG",
        replace(get_config(), record_sse=True),
    )
    with record_sse_stream("req_ctx"):
        append_sse_bytes(b"data: live\n\n")
    assert (dump / "req_ctx.sse").read_bytes() == b"data: live\n\n"


def test_load_config_record_sse_flag(tmp_path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[fncall]\nrecord_sse = true\n", encoding="utf-8")
    tpl_path = Path(__file__).resolve().parents[1] / "template" / "config.toml"
    from server.config import load_config

    cfg = load_config(cfg_path, template_path=tpl_path)
    assert cfg.record_sse is True


@pytest.mark.asyncio
async def test_record_sse_survives_async_generator_aclose(tmp_path, monkeypatch) -> None:
    """换号重试 aclose 内层 generator 时，ContextVar reset 不得抛 ValueError。"""
    dump = tmp_path / "sse"
    monkeypatch.setattr("server.records.sse_record.sse_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "server.records.sse_record.CONFIG",
        replace(get_config(), record_sse=True),
    )

    async def _inner():
        append_sse_bytes(b"data: partial\n\n")
        yield 1
        append_sse_bytes(b"data: more\n\n")

    with record_sse_stream("req_aclose"):
        gen = _inner()
        await gen.asend(None)
        await gen.aclose()

    assert (dump / "req_aclose.sse").read_bytes() == b"data: partial\n\n"


@pytest.mark.asyncio
async def test_deepseek_parse_sse_stream_appends_raw_bytes(tmp_path, monkeypatch) -> None:
    from upstream.deepseek.lib.adapter.strmrun import _StreamRunMixin
    from upstream.deepseek.lib.stream.strmpars import StreamParser

    class _FakeContent:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        async def iter_chunked(self, _n: int):
            for chunk in self._chunks:
                yield chunk

    class _FakeResp:
        def __init__(self, chunks: list[bytes]) -> None:
            self.content = _FakeContent(chunks)

    raw = b"event: ready\ndata: {}\n\n"
    dump = tmp_path / "sse"
    monkeypatch.setattr("server.records.sse_record.sse_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "server.records.sse_record.CONFIG",
        replace(get_config(), record_sse=True),
    )

    parser = StreamParser(include_thinking=False)
    mixin = _StreamRunMixin()
    with record_sse_stream("req_ds"):
        async for _ in mixin._parse_sse_stream(_FakeResp([raw]), parser):
            pass

    assert (dump / "req_ds.sse").read_bytes() == raw
