from __future__ import annotations

"""ProxyToggleManager 单元测试：持久化、防抖、并发安全。"""

import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from upstream.qwen.media.proxy_toggle import ProxyToggleManager


class TestProxyToggleManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.mgr = ProxyToggleManager()
        self.test_persist = Path("persist/qwen/test_proxy_toggle.json")

    def tearDown(self) -> None:
        if self.test_persist.exists():
            self.test_persist.unlink()

    async def test_initialize_no_proxy_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
                os.environ.pop(k, None)
            await self.mgr.initialize()
        self.assertFalse(self.mgr.enabled)

    async def test_initialize_probe_fails(self) -> None:
        with patch.dict(os.environ, {"HTTP_PROXY": "http://bad:8080"}):
            with patch(
                "upstream.qwen.media.proxy_toggle._probe_proxy_alive",
                new_callable=AsyncMock,
                return_value=False,
            ):
                await self.mgr.initialize()
        self.assertFalse(self.mgr.enabled)

    async def test_initialize_probe_ok_reads_persist(self) -> None:
        self.test_persist.parent.mkdir(parents=True, exist_ok=True)
        self.test_persist.write_text(
            json.dumps({"enabled": 1}), encoding="utf-8",
        )
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}):
            with patch(
                "upstream.qwen.media.proxy_toggle._probe_proxy_alive",
                new_callable=AsyncMock,
                return_value=True,
            ):
                with patch(
                    "upstream.qwen.media.proxy_toggle._PERSIST_PATH",
                    self.test_persist,
                ):
                    await self.mgr.initialize()
        self.assertTrue(self.mgr.enabled)

    async def test_initialize_no_persist_defaults_disabled(self) -> None:
        if self.test_persist.exists():
            self.test_persist.unlink()
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}):
            with patch(
                "upstream.qwen.media.proxy_toggle._probe_proxy_alive",
                new_callable=AsyncMock,
                return_value=True,
            ):
                with patch(
                    "upstream.qwen.media.proxy_toggle._PERSIST_PATH",
                    self.test_persist,
                ):
                    await self.mgr.initialize()
        self.assertFalse(self.mgr.enabled)
        self.assertTrue(self.test_persist.exists())
        data = json.loads(self.test_persist.read_text(encoding="utf-8"))
        self.assertEqual(data["enabled"], 0)

    async def test_sm_block_toggles_state(self) -> None:
        self.mgr._enabled = False
        self.mgr._initialized = True
        result = await self.mgr.on_sm_block("req-001")
        self.assertTrue(result)
        self.assertTrue(self.mgr.enabled)

    async def test_sm_block_same_task_no_double_toggle(self) -> None:
        self.mgr._enabled = False
        self.mgr._initialized = True
        r1 = await self.mgr.on_sm_block("req-002")
        r2 = await self.mgr.on_sm_block("req-002")
        self.assertTrue(r1)
        self.assertTrue(r2)
        self.assertTrue(self.mgr.enabled)

    async def test_sm_block_different_tasks_toggle_each(self) -> None:
        self.mgr._enabled = False
        self.mgr._initialized = True
        r1 = await self.mgr.on_sm_block("req-003")
        r2 = await self.mgr.on_sm_block("req-004")
        self.assertTrue(r1)
        self.assertFalse(r2)
        self.assertFalse(self.mgr.enabled)

    async def test_release_task_allows_re_toggle(self) -> None:
        self.mgr._enabled = False
        self.mgr._initialized = True
        await self.mgr.on_sm_block("req-005")
        self.mgr.release_task("req-005")
        result = await self.mgr.on_sm_block("req-005")
        self.assertFalse(result)
        self.assertFalse(self.mgr.enabled)

    async def test_concurrent_sm_blocks_single_toggle(self) -> None:
        self.mgr._enabled = False
        self.mgr._initialized = True
        results = await asyncio.gather(
            self.mgr.on_sm_block("req-concurrent"),
            self.mgr.on_sm_block("req-concurrent"),
            self.mgr.on_sm_block("req-concurrent"),
        )
        self.assertEqual(results.count(True), 3)
        self.assertTrue(self.mgr.enabled)

    async def test_persist_written_on_toggle(self) -> None:
        with patch(
            "upstream.qwen.media.proxy_toggle._PERSIST_PATH", self.test_persist,
        ):
            self.mgr._enabled = False
            self.mgr._initialized = True
            await self.mgr.on_sm_block("req-persist")
        self.assertTrue(self.test_persist.exists())
        data = json.loads(self.test_persist.read_text(encoding="utf-8"))
        self.assertEqual(data["enabled"], 1)

    async def test_has_proxy_env_detection(self) -> None:
        from upstream.qwen.media.proxy_toggle import _has_proxy_env

        with patch.dict(os.environ, {}, clear=True):
            for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
                os.environ.pop(k, None)
            self.assertFalse(_has_proxy_env())
        with patch.dict(os.environ, {"HTTP_PROXY": "http://p:8080"}):
            self.assertTrue(_has_proxy_env())


class TestChatCompletionStreamSmBlock(unittest.IsolatedAsyncioTestCase):
    """验证 chat_completion_stream 遇到 BaxiaSmBlockedError 时触发代理切换。"""

    async def _make_mock_client(self):
        from unittest.mock import AsyncMock, MagicMock
        client = MagicMock()
        client.cleanup_chat = AsyncMock(return_value=True)
        return client

    async def test_sm_block_with_req_id_triggers_toggle(self):
        from server.formats import BaxiaSmBlockedError
        from upstream.qwen.completion_stream import chat_completion_stream

        mock_toggle = AsyncMock()
        mock_toggle.enabled = True

        async def _raise_sm(*a, **kw):
            raise BaxiaSmBlockedError("SM blocked")
            yield  # noqa: make it an async generator

        with patch(
            "upstream.qwen.completion_stream._post_chat_sse", side_effect=_raise_sm,
        ), patch(
            "upstream.qwen.media.proxy_toggle.get_proxy_toggle", return_value=mock_toggle,
        ):
            client = await self._make_mock_client()
            with self.assertRaises(BaxiaSmBlockedError):
                async for _ in chat_completion_stream(
                    client, None, "chat-1", {}, {}, req_id="req-sm-1",
                ):
                    pass
        mock_toggle.on_sm_block.assert_awaited_once_with("req-sm-1")

    async def test_sm_block_without_req_id_skips_toggle(self):
        from server.formats import BaxiaSmBlockedError
        from upstream.qwen.completion_stream import chat_completion_stream

        mock_toggle = AsyncMock()
        mock_toggle.enabled = True

        async def _raise_sm(*a, **kw):
            raise BaxiaSmBlockedError("SM blocked")
            yield

        with patch(
            "upstream.qwen.completion_stream._post_chat_sse", side_effect=_raise_sm,
        ), patch(
            "upstream.qwen.media.proxy_toggle.get_proxy_toggle", return_value=mock_toggle,
        ):
            client = await self._make_mock_client()
            with self.assertRaises(BaxiaSmBlockedError):
                async for _ in chat_completion_stream(
                    client, None, "chat-2", {}, {}, req_id="",
                ):
                    pass
        mock_toggle.on_sm_block.assert_not_awaited()

    async def test_sm_block_toggles_real_manager_state(self):
        """用真实 ProxyToggleManager 验证 SM block 触发后状态自动翻转。"""
        from server.formats import BaxiaSmBlockedError
        from upstream.qwen.completion_stream import chat_completion_stream

        real_mgr = ProxyToggleManager()
        real_mgr._enabled = True
        real_mgr._initialized = True

        test_persist = Path("persist/qwen/test_sm_toggle_real.json")
        try:
            async def _raise_sm(*a, **kw):
                raise BaxiaSmBlockedError("SM blocked")
                yield

            with patch(
                "upstream.qwen.completion_stream._post_chat_sse",
                side_effect=_raise_sm,
            ), patch(
                "upstream.qwen.media.proxy_toggle.get_proxy_toggle",
                return_value=real_mgr,
            ), patch(
                "upstream.qwen.media.proxy_toggle._PERSIST_PATH",
                test_persist,
            ):
                client = await self._make_mock_client()
                self.assertTrue(real_mgr.enabled)
                with self.assertRaises(BaxiaSmBlockedError):
                    async for _ in chat_completion_stream(
                        client, None, "chat-real", {}, {},
                        req_id="req-real-toggle",
                    ):
                        pass
            self.assertFalse(real_mgr.enabled)
            self.assertTrue(test_persist.exists())
            data = json.loads(test_persist.read_text(encoding="utf-8"))
            self.assertEqual(data["enabled"], 0)
        finally:
            if test_persist.exists():
                test_persist.unlink()


if __name__ == "__main__":
    unittest.main()

