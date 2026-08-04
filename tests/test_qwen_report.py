from __future__ import annotations

"""Qwen aplus / users/status 上报单元测试。"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upstream.qwen.auth.report import (
    _base_typarms,
    _spm_for_path,
    report_chat_generation,
    report_clk_generate_mode,
    report_completions_request_id,
    report_create_chat_sequence,
    report_file_parse_success,
    report_file_upload_finish,
    report_file_upload_oss_token_time,
    report_file_upload_start,
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
        self.assertEqual(payload["chat_mode"], "local")
        self.assertEqual(payload["chat_type"], "t2t")
        self.assertEqual(payload["models"], ["qwen3.8-max"])
        self.assertIn("timestamp", payload)
        self.assertNotIn("title", payload)


class TestReportHelpers(unittest.TestCase):
    def test_spm_paths(self) -> None:
        self.assertEqual(_spm_for_path("/"), "a2ty_o01.29997169")
        self.assertEqual(_spm_for_path("/c/new-chat"), "a2ty_o01.29997170")
        self.assertEqual(_spm_for_path("/c/local"), "a2ty_o01.29997173")

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
            "upstream.qwen.auth.report.chat.silent_request",
            new_callable=AsyncMock,
        ) as silent_chat, patch(
            "upstream.qwen.auth.report.core.silent_request",
            new_callable=AsyncMock,
        ) as silent_core:
            await report_user_status(client, session, page_path="/c/local")
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
        silent = silent_chat
        self.assertGreaterEqual(
            silent_chat.await_count + silent_core.await_count, 5,
        )
        status_headers = silent.await_args_list[0].kwargs.get("headers") or {}
        self.assertIn("bx-v", status_headers)
        self.assertNotIn("bx-ua", status_headers)
        self.assertNotIn("Authorization", status_headers)
        local_urls = [
            (c.kwargs.get("params") or {}).get("_p_url", "")
            for c in (silent_chat.await_args_list + silent_core.await_args_list)
            if c.args and len(c.args) >= 3 and "tongyi-sg" in str(c.args[2])
        ]
        self.assertTrue(any("/c/local" in u for u in local_urls))

    async def test_create_chat_sequence_covers_clk_and_pv(self) -> None:
        client = MagicMock()
        session = _session()
        with patch(
            "upstream.qwen.auth.report.chat.silent_request",
            new_callable=AsyncMock,
        ) as silent_chat, patch(
            "upstream.qwen.auth.report.core.silent_request",
            new_callable=AsyncMock,
        ) as silent_core:
            await report_create_chat_sequence(client, session)
            await report_clk_generate_mode(client, session)
        silent_calls = silent_chat.await_args_list + silent_core.await_args_list
        # aes goes through core.silent_request (report_aes_events); tongyi/v.gif/status via chat
        urls = [c.args[2] for c in silent_calls if len(c.args) >= 3]
        self.assertTrue(any("clkGenerateMode" in u for u in urls))
        self.assertTrue(any("chatGeneration" in u for u in urls))
        self.assertTrue(any(u.endswith("/v.gif") for u in urls))
        self.assertTrue(any(u.endswith("/aes.1.1") for u in urls))
        self.assertTrue(any("users/status" in u for u in urls))
        gen_calls = [
            c for c in silent_calls
            if len(c.args) >= 3 and "chatGeneration" in str(c.args[2])
        ]
        self.assertTrue(gen_calls)
        gokey = (gen_calls[0].kwargs.get("params") or {}).get("gokey", "")
        self.assertIn("send_type=click", gokey)
        self.assertIn("msg_type=t2t", gokey)

    async def test_file_upload_reports_cover_aes_and_tongyi(self) -> None:
        client = MagicMock()
        session = _session()
        with patch(
            "upstream.qwen.auth.report.core.silent_request",
            new_callable=AsyncMock,
        ) as silent:
            await report_file_upload_oss_token_time(
                client,
                session,
                filename="a.txt",
                filesize=100,
                content_type="text/plain",
                start_ms=1.0,
                end_ms=2.0,
            )
            await report_file_upload_start(
                client,
                session,
                filename="a.txt",
                filesize=100,
                content_type="text/plain",
                start_ms=2.0,
            )
            await report_file_upload_finish(
                client,
                session,
                filename="a.txt",
                filesize=100,
                content_type="text/plain",
                upload_start_ms=2.0,
                upload_end_ms=5.0,
                all_elapsed_ms=4,
            )
            await report_file_parse_success(
                client,
                session,
                file_id="fid-1",
                filename="a.txt",
                filesize=100,
                content_type="text/plain",
            )
        self.assertEqual(silent.await_count, 6)
        urls = [c.args[2] for c in silent.await_args_list]
        self.assertEqual(sum(1 for u in urls if u.endswith("/aes.1.1")), 4)
        self.assertTrue(
            any("tongyi-sg.qwen_chat.FileUpload-AllTime" in u for u in urls)
        )
        self.assertTrue(
            any("tongyi-sg.qwen_chat.filePaseSuccess" in u for u in urls)
        )
        aes_bodies = [
            c.kwargs.get("data") or ""
            for c in silent.await_args_list
            if c.args[2].endswith("/aes.1.1")
        ]
        joined = "\n".join(str(b) for b in aes_bodies)
        for needle in (
            "FileUpload-ossTokenTime",
            "FileUpload-startUpload",
            "FileUpload-finishUpload",
            "FileUpload-AllTime",
            "filePaseSuccess",
        ):
            self.assertIn(needle, joined)


if __name__ == "__main__":
    unittest.main()
