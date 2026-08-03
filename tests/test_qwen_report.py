from __future__ import annotations

"""Qwen aplus / users/status 上报单元测试。"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upstream.qwen.auth.report import (
    _base_typarms,
    _spm_for_path,
    report_chat_generation,
    report_completions_request_id,
    report_generation_create_return,
    report_streaming_statistics,
    report_user_status,
)
from upstream.qwen.chat.routes import APP_VERSION, BAXIA_SDK_VERSION, USER_AGENT
from upstream.qwen.chat.upload.payload import build_new_chat_payload
from core.session.accounts import Account
from core.session.store import PlatformSession


def _session() -> PlatformSession:
    return PlatformSession(
        account=Account(username="u@test.com", password="pw"),
        token="tok",
        user_id="uid-123",
        upstream="qwen",
    )


class TestHarAlignmentConstants(unittest.TestCase):
    def test_versions_match_har(self) -> None:
        self.assertEqual(APP_VERSION, "0.2.81")
        self.assertEqual(BAXIA_SDK_VERSION, "2.5.37")
        self.assertIn("Chrome/153", USER_AGENT)

    def test_new_chat_payload_matches_har_shape(self) -> None:
        payload = build_new_chat_payload("qwen3.8-max")
        self.assertEqual(payload["chatId"], "")
        self.assertEqual(payload["chat_mode"], "normal")
        self.assertEqual(payload["chat_type"], "t2t")
        self.assertEqual(payload["models"], ["qwen3.8-max"])
        self.assertIn("timestamp", payload)
        self.assertNotIn("title", payload)


class TestReportHelpers(unittest.TestCase):
    def test_spm_paths(self) -> None:
        self.assertEqual(_spm_for_path("/"), "a2ty_o01.29997169")
        self.assertEqual(_spm_for_path("/c/new-chat"), "a2ty_o01.29997170")
        self.assertEqual(_spm_for_path("/c/abc"), "a2ty_o01.29997173")

    def test_typarms_uid(self) -> None:
        parms = _base_typarms(_session())
        self.assertEqual(parms["typarm2"], "uid-123")
        self.assertEqual(parms["cdn_version"], APP_VERSION)


class TestReportFireAndForget(unittest.IsolatedAsyncioTestCase):
    async def test_report_calls_are_silent(self) -> None:
        client = MagicMock()
        client._ensure_http_session = AsyncMock()
        session = _session()
        with patch(
            "upstream.qwen.auth.report._silent_request",
            new_callable=AsyncMock,
        ) as silent:
            await report_user_status(client, session, page_path="/")
            await report_chat_generation(client, session)
            await report_generation_create_return(client, session, "chat-1")
            await report_completions_request_id(
                client, session, request_id="req-1", chat_id="chat-1",
            )
            await report_streaming_statistics(
                client,
                session,
                chat_id="chat-1",
                model="qwen3.8-max",
                request_id="req-1",
                response_id="resp-1",
                api_start_ms=1000,
                first_chunk_ms=1100,
                end_chunk_ms=2000,
            )
        self.assertGreaterEqual(silent.await_count, 5)


if __name__ == "__main__":
    unittest.main()
