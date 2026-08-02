from __future__ import annotations

"""DeepSeek client/settings 对齐测试。"""

from upstream.deepseek.lib.user.settingsapi import collect_setting_ids
from upstream.deepseek.lib.protocol.consts import COMMON_HEADERS


def test_common_headers_match_har_baseline() -> None:
    assert COMMON_HEADERS["x-client-version"] == "2.3.0"
    assert COMMON_HEADERS["x-client-bundle-id"] == "com.deepseek.chat"
    assert "153" in COMMON_HEADERS["user-agent"]


def test_collect_setting_ids_from_biz_settings() -> None:
    settings = {
        "sse_auto_resume_timeout": {"id": 714289219, "value": 3000},
        "chat_hcaptcha": {"id": 58457202, "value": True},
        "nested": {"value": 1},
    }
    ids = collect_setting_ids(settings)
    assert 714289219 in ids
    assert 58457202 in ids
    assert len(ids) == 2


def test_completion_payload_shape() -> None:
    from upstream.deepseek.lib.adapter.helpers.client_helpers import build_chat_payload

    payload = build_chat_payload(
        {"model_type": "default", "prompt": "hi"},
        "sess-1",
    )
    assert payload["action"] is None
    assert payload["prompt"] == "hi"
    assert "client_stream_id" not in payload
