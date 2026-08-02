from __future__ import annotations

"""DeepSeek 文件上传单元测试。"""

from typing import Any, Dict, List

import pytest

from upstream.deepseek.lib.adapter.helpers.client_helpers import build_chat_payload
from upstream.deepseek.lib.adapter.helpers.file_collect import (
    collect_message_attachments,
    extract_base64_images,
)
from upstream.deepseek.lib.adapter.helpers.file_upload import (
    is_parse_error_status,
    is_parse_success_status,
    resolve_model_type,
    wait_files_ready,
)
from upstream.deepseek.lib.protocol.consts import MODEL_PRO, MODEL_VISION


def test_resolve_model_type() -> None:
    assert resolve_model_type(MODEL_VISION) == "vision"
    assert resolve_model_type(MODEL_PRO) == "default"


def test_parse_status_helpers() -> None:
    assert is_parse_success_status("SUCCESS")
    assert not is_parse_success_status("PARSING")
    assert is_parse_error_status("FAILED")
    assert is_parse_error_status("CONTENT_FILTER")


def test_extract_base64_image_from_messages() -> None:
    tiny_png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": tiny_png}}]}]
    items = extract_base64_images(messages)
    assert len(items) == 1
    data, name = items[0]
    assert data.startswith(b"\x89PNG")
    assert name.endswith(".png")


def test_collect_message_attachments_with_splitter() -> None:
    messages: List[Dict[str, Any]] = [{"role": "user", "content": "hi"}]
    items = collect_message_attachments(
        messages, filename="note.txt", file_bytes=b"hello",
    )
    assert len(items) == 1
    assert items[0][1] == "note.txt"


def test_build_chat_payload_with_ref_files() -> None:
    payload = build_chat_payload(
        {"model_type": "vision", "prompt": "describe"},
        "sess-9",
        ref_file_ids=["f1", "f2"],
        thinking_enabled=False,
        search_enabled=False,
    )
    assert payload["ref_file_ids"] == ["f1", "f2"]
    assert payload["model_type"] == "vision"
    assert payload["thinking_enabled"] is False


class _FakeResp:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.status = 200
        self._payload = payload

    async def json(self) -> Dict[str, Any]:
        return self._payload

    async def __aenter__(self) -> "_FakeResp":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def get(self, *args: Any, **kwargs: Any) -> _FakeResp:
        self.calls += 1
        payload = self._responses.pop(0)
        return _FakeResp(payload)


@pytest.mark.asyncio
async def test_wait_files_ready_polls_until_success() -> None:
    pending = {
        "code": 0,
        "data": {
            "biz_code": 0,
            "biz_data": {"files": [{"id": "abc", "status": "PARSING"}]},
        },
    }
    done = {
        "code": 0,
        "data": {
            "biz_code": 0,
            "biz_data": {"files": [{"id": "abc", "status": "SUCCESS"}]},
        },
    }
    session = _FakeSession([pending, done])
    records = await wait_files_ready(
        session,  # type: ignore[arg-type]
        "token",
        ["abc"],
        poll_interval=0.01,
        max_attempts=5,
    )
    assert session.calls == 2
    assert records[0]["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_wait_files_ready_raises_on_failed_status() -> None:
    failed = {
        "code": 0,
        "data": {
            "biz_code": 0,
            "biz_data": {"files": [{"id": "abc", "status": "FAILED"}]},
        },
    }
    session = _FakeSession([failed])
    with pytest.raises(RuntimeError, match="parse failed"):
        await wait_files_ready(
            session,  # type: ignore[arg-type]
            "token",
            ["abc"],
            poll_interval=0.01,
            max_attempts=2,
        )
