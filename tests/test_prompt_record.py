from __future__ import annotations

"""inject 后 prompt 落盘（logs/prompts/{req_id}.txt）与 debug 日志配置。"""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import pytest

from echotools.fncall import get_protocol
from handlers.fncall_inject import inject_fncall_for_request, prompt_dump_dir
from server.config import CONFIG


@pytest.fixture
def sample_messages() -> List[Dict[str, Any]]:
    return [{"role": "user", "content": "杭州天气怎么样？"}]


@pytest.fixture
def sample_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def test_prompt_dump_disabled_no_file(
    sample_messages: List[Dict[str, Any]],
    sample_tools: List[Dict[str, Any]],
    tmp_path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rogator")
    dump = tmp_path / "prompts"
    monkeypatch.setattr("handlers.fncall_inject.prompt_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "handlers.fncall_inject.CONFIG",
        replace(CONFIG, record_prompt=False, print_prompt=False),
    )
    injected = inject_fncall_for_request(
        sample_messages,
        sample_tools,
        get_protocol("entml"),
        req_id="req_off",
        api="openai",
        model="test-model",
    )
    assert injected[0]["content"]
    assert list(dump.glob("*.txt")) == []
    assert "inject prompt api=openai req_id=req_off" in caplog.text
    assert "dump_dir=None" in caplog.text


def test_record_prompt_writes_req_id_file(
    sample_messages: List[Dict[str, Any]],
    sample_tools: List[Dict[str, Any]],
    tmp_path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="rogator")
    dump = tmp_path / "prompts"
    monkeypatch.setattr("handlers.fncall_inject.prompt_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "handlers.fncall_inject.CONFIG",
        replace(CONFIG, record_prompt=True, print_prompt=False),
    )
    injected = inject_fncall_for_request(
        sample_messages,
        sample_tools,
        get_protocol("entml"),
        req_id="req_oai",
        api="openai",
        model="qwen-test",
    )
    prompt = injected[0]["content"]
    path = dump / "req_oai.txt"
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == prompt
    assert "dump_dir=" in caplog.text


def test_print_prompt_also_dumps_file(
    sample_messages: List[Dict[str, Any]],
    sample_tools: List[Dict[str, Any]],
    tmp_path,
    monkeypatch,
) -> None:
    dump = tmp_path / "prompts"
    monkeypatch.setattr("handlers.fncall_inject.prompt_dump_dir", lambda: dump)
    monkeypatch.setattr(
        "handlers.fncall_inject.CONFIG",
        replace(CONFIG, record_prompt=False, print_prompt=True),
    )
    inject_fncall_for_request(
        sample_messages,
        sample_tools,
        get_protocol("entml"),
        req_id="req_print",
        api="openai",
        model="qwen-test",
    )
    assert len(list(dump.glob("req_print.txt"))) == 1


def test_load_config_fncall_flags(tmp_path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[fncall]\nrecord_prompt = true\nprint_prompt = true\n",
        encoding="utf-8",
    )
    tpl_path = Path(__file__).resolve().parents[1] / "template" / "config.toml"
    from server.config import load_config

    cfg = load_config(cfg_path, template_path=tpl_path)
    assert cfg.record_prompt is True
    assert cfg.print_prompt is True


def test_prompt_dump_dir_under_project_root() -> None:
    assert prompt_dump_dir().name == "prompts"
    assert prompt_dump_dir().parent.name == "logs"


def test_load_config_debug_log_name(tmp_path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[debug]\nlevel = "INFO"\nlog_name = "rogator-test"\n',
        encoding="utf-8",
    )
    tpl_path = Path(__file__).resolve().parents[1] / "template" / "config.toml"
    from server.config import load_config

    cfg = load_config(cfg_path, template_path=tpl_path)
    assert cfg.log_level == "INFO"
    assert cfg.log_name == "rogator-test"
    assert cfg.access_log is True


def test_load_config_access_log_flag(tmp_path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[debug]\naccess_log = false\n", encoding="utf-8")
    tpl_path = Path(__file__).resolve().parents[1] / "template" / "config.toml"
    from server.config import load_config

    cfg = load_config(cfg_path, template_path=tpl_path)
    assert cfg.access_log is False


def test_resolve_log_file_path_format(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    from server.config import LOG_DIR
    from server.config.logging_setup import resolve_log_file_path

    monkeypatch.setattr("server.config.logging_setup.LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(
        "server.config.logging_setup.CONFIG",
        replace(CONFIG, log_to_file=True, log_name="rogator"),
    )
    path = resolve_log_file_path()
    assert path is not None
    assert path.parent.name == "logs"
    assert path.name.startswith("rogator-")
    assert path.name.endswith(".log")
    assert len(path.name) == len("rogator-YYYYMMDD-HHMMSS.log")


def test_setup_logging_writes_file(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    log_dir = tmp_path / "logs"
    monkeypatch.setattr("server.config.logging_setup.LOG_DIR", log_dir)
    monkeypatch.setattr(
        "server.config.logging_setup.CONFIG",
        replace(
            CONFIG,
            log_to_file=True,
            log_name="rogator",
            log_level="INFO",
        ),
    )
    from server.config.logging_setup import setup_logging

    path = setup_logging()
    assert path is not None
    assert path.name.startswith("rogator-")
    import logging

    logging.getLogger("rogator").info("logging smoke test")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert path.is_file()
    assert "logging smoke test" in path.read_text(encoding="utf-8")
