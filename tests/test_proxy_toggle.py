from __future__ import annotations

"""ProxyToggleManager 单元测试：持久化、防抖、并发安全。"""

import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from upstream.qwen.proxy_toggle import ProxyToggleManager


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
                "upstream.qwen.proxy_toggle._probe_proxy_alive",
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
                "upstream.qwen.proxy_toggle._probe_proxy_alive",
                new_callable=AsyncMock,
                return_value=True,
            ):
                with patch(
                    "upstream.qwen.proxy_toggle._PERSIST_PATH",
                    self.test_persist,
                ):
                    await self.mgr.initialize()
        self.assertTrue(self.mgr.enabled)

    async def test_initialize_no_persist_defaults_disabled(self) -> None:
        if self.test_persist.exists():
            self.test_persist.unlink()
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}):
            with patch(
                "upstream.qwen.proxy_toggle._probe_proxy_alive",
                new_callable=AsyncMock,
                return_value=True,
            ):
                with patch(
                    "upstream.qwen.proxy_toggle._PERSIST_PATH",
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
            "upstream.qwen.proxy_toggle._PERSIST_PATH", self.test_persist,
        ):
            self.mgr._enabled = False
            self.mgr._initialized = True
            await self.mgr.on_sm_block("req-persist")
        self.assertTrue(self.test_persist.exists())
        data = json.loads(self.test_persist.read_text(encoding="utf-8"))
        self.assertEqual(data["enabled"], 1)

    async def test_has_proxy_env_detection(self) -> None:
        from upstream.qwen.proxy_toggle import _has_proxy_env

        with patch.dict(os.environ, {}, clear=True):
            for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
                os.environ.pop(k, None)
            self.assertFalse(_has_proxy_env())
        with patch.dict(os.environ, {"HTTP_PROXY": "http://p:8080"}):
            self.assertTrue(_has_proxy_env())


if __name__ == "__main__":
    unittest.main()
