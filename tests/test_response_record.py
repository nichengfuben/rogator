from __future__ import annotations

"""上游模型 thinking/answer 落盘（logs/responses/{req_id}.txt）。"""

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from server.config import get_config
from server.records.response_record import RawResponseRecorder, record_raw_response, response_dump_dir


def test_response_dump_dir_under_project_root() -> None:
    assert response_dump_dir().name == "responses"
    assert response_dump_dir().parent.name == "logs"


def test_record_response_disabled_no_file(tmp_path, monkeypatch) -> None:
    dump = tmp_path / "responses"
    monkeypatch.setattr("server.records.response_record.response_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "server.records.response_record.CONFIG",
        replace(get_config(), record_response=False),
    )
    recorder = RawResponseRecorder("req_off")
    recorder.ingest_event({"type": "answer", "content": "hello"})
    recorder.finalize()
    assert list(dump.glob("*.txt")) == []


def test_record_response_writes_req_id_file(
    tmp_path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rogator")
    dump = tmp_path / "responses"
    monkeypatch.setattr("server.records.response_record.response_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "server.records.response_record.CONFIG",
        replace(get_config(), record_response=True),
    )
    recorder = RawResponseRecorder("req_abc")
    recorder.ingest_event({"type": "thinking", "content": "plan "})
    recorder.ingest_event({"type": "answer", "content": "done"})
    recorder.ingest_event({"type": "usage", "data": {"input_tokens": 1}})
    recorder.finalize()
    path = dump / "req_abc.txt"
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == "plan done"
    assert "record response req_id=req_abc" in caplog.text


def test_record_raw_response_context_manager_finalizes(tmp_path, monkeypatch) -> None:
    dump = tmp_path / "responses"
    monkeypatch.setattr("server.records.response_record.response_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "server.records.response_record.CONFIG",
        replace(get_config(), record_response=True),
    )
    with record_raw_response("req_ctx") as recorder:
        recorder.ingest_event({"type": "answer", "content": "x"})
    assert (dump / "req_ctx.txt").read_text(encoding="utf-8") == "x"


def test_prompt_and_response_share_req_id(
    tmp_path,
    monkeypatch,
) -> None:
    """prompts/{req_id}.txt 与 responses/{req_id}.txt 使用同一 req_id。"""
    from handlers.fncall_inject import inject_fncall_for_request, prompt_dump_dir
    from echotools.fncall import get_protocol

    prompts = tmp_path / "prompts"
    responses = tmp_path / "responses"
    monkeypatch.setattr("handlers.fncall_inject.prompt_dump_dir", lambda: prompts)
    monkeypatch.setattr("server.records.response_record.response_dump_dir", lambda: responses)
    monkeypatch.setattr(
        "handlers.fncall_inject.CONFIG",
        replace(get_config(), record_prompt=True, print_prompt=False, record_response=True),
    )
    monkeypatch.setattr(
        "server.records.response_record.CONFIG",
        replace(get_config(), record_prompt=True, print_prompt=False, record_response=True),
    )

    req_id = "req-pair-001"
    inject_fncall_for_request(
        [{"role": "user", "content": "hi"}],
        [],
        get_protocol("entml"),
        req_id=req_id,
        api="openai",
        model="test",
    )
    with record_raw_response(req_id) as recorder:
        recorder.ingest_event({"type": "thinking", "content": "t"})
        recorder.ingest_event({"type": "answer", "content": "a"})

    assert (prompts / f"{req_id}.txt").is_file()
    assert (responses / f"{req_id}.txt").is_file()
    assert (responses / f"{req_id}.txt").read_text(encoding="utf-8") == "ta"


def test_load_config_record_response_flag(tmp_path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[fncall]\nrecord_response = true\n", encoding="utf-8")
    tpl_path = Path(__file__).resolve().parents[1] / "template" / "config.toml"
    from server.config import load_config

    cfg = load_config(cfg_path, template_path=tpl_path)
    assert cfg.record_response is True
