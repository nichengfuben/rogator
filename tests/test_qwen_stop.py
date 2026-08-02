from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from upstream.qwen.chat.chat import abort_upstream_on_cancel, iter_sse_events
from upstream.qwen.chat.routes import STOP_CHAT_PATH
from upstream.qwen.chat.upload.payload import build_stop_payload
from upstream.qwen.client import QwenClient


class TestBuildStopPayload(unittest.TestCase):
    def test_chat_id_only(self) -> None:
        self.assertEqual(build_stop_payload("abc"), {"chat_id": "abc"})

    def test_with_response_id(self) -> None:
        self.assertEqual(
            build_stop_payload("abc", "resp-1"),
            {"chat_id": "abc", "response_id": "resp-1"},
        )

    def test_stop_path_matches_provider(self) -> None:
        self.assertEqual(STOP_CHAT_PATH, "/api/v2/chat/completions/stop")


class TestAbortUpstreamOnCancel(unittest.IsolatedAsyncioTestCase):
    async def test_stop_then_delete(self) -> None:
        client = MagicMock()
        session = MagicMock(token="tok", username="user01")
        stop = AsyncMock(return_value=True)
        delete = AsyncMock(return_value=True)
        with patch(
            "upstream.qwen.chat.chat.stop_upstream_generation", stop,
        ), patch(
            "upstream.qwen.chat.chat.delete_upstream_chat", delete,
        ):
            await abort_upstream_on_cancel(client, session, "chat-1", "resp-9")
        stop.assert_awaited_once_with(client, session, "chat-1", "resp-9")
        delete.assert_awaited_once_with(client, session, "chat-1")


class TestIterSseEventsResponseId(unittest.IsolatedAsyncioTestCase):
    async def test_captures_response_created(self) -> None:
        client = MagicMock()
        session = MagicMock()

        async def _lines():
            yield b'data: {"response.created":{"response_id":"rid-42"}}\n'
            yield b"data: [DONE]\n"

        resp = MagicMock()
        resp.content = _lines()
        box: list[str] = []
        async for _ in iter_sse_events(client, resp, session, response_id_out=box):
            pass
        self.assertEqual(box, ["rid-42"])


class TestChatCompletionCancel(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_aborts_upstream(self) -> None:
        client = QwenClient(MagicMock())
        session = MagicMock(token="tok", username="user01")
        abort = AsyncMock()
        cleanup = AsyncMock()

        async def _fake_iter(*_a, **_k):
            yield {"type": "answer", "content": "hi"}
            raise asyncio.CancelledError()

        with patch.object(client, "_ensure_http_session", AsyncMock()), patch(
            "upstream.qwen.client.iter_sse_events", _fake_iter,
        ), patch(
            "upstream.qwen.client.abort_upstream_on_cancel", abort,
        ), patch.object(client, "cleanup_chat", cleanup):
            resp_cm = MagicMock()
            resp_cm.__aenter__ = AsyncMock(return_value=MagicMock(status=200))
            resp_cm.__aexit__ = AsyncMock(return_value=False)
            http = MagicMock()
            http.post = MagicMock(return_value=resp_cm)
            client._ensure_http_session.return_value = http

            gen = client.chat_completion(session, "chat-x", [{"content": "hello"}])
            with self.assertRaises(asyncio.CancelledError):
                async for _ in gen:
                    pass

        abort.assert_awaited_once()
        cleanup.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
