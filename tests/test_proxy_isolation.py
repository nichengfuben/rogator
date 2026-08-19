from __future__ import annotations

"""验证 HTTP_PROXY 等代理环境变量仅对 QwenClient 生效。"""

import os
import unittest
from unittest.mock import patch

import aiohttp

from core.transport.http import close_shared_connector
from core.transport.owned import HttpTransportMixin
from server.retry.http_client import client_session


class _DefaultClient(HttpTransportMixin):
    """模拟非 Qwen 上游（Zen/DeepSeek/Ollama）。"""

    def __init__(self) -> None:
        self._init_http_transport()


class _QwenLikeClient(HttpTransportMixin):
    """模拟 QwenClient 覆盖 _client_session_kwargs。"""

    def __init__(self) -> None:
        self._init_http_transport()

    def _client_session_kwargs(self) -> dict:
        return {"use_env_proxy": True}


class TestProxyIsolation(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        await close_shared_connector()

    async def test_default_client_session_ignores_proxy_env(self) -> None:
        """默认 client_session 不读取 HTTP_PROXY。"""
        with patch.dict(os.environ, {"HTTP_PROXY": "http://evil-proxy:8080"}):
            session = client_session()
            try:
                self.assertFalse(session.trust_env)
            finally:
                await session.close()

    async def test_explicit_use_env_proxy_enables_trust_env(self) -> None:
        """use_env_proxy=True 时 trust_env 为 True。"""
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}):
            session = client_session(use_env_proxy=True)
            try:
                self.assertTrue(session.trust_env)
            finally:
                await session.close()

    async def test_default_mixin_client_no_proxy(self) -> None:
        """非 Qwen 上游通过 HttpTransportMixin 创建的 session 不走代理。"""
        client = _DefaultClient()
        session = await client.ensure_http_session()
        self.assertFalse(session.trust_env)
        await client.close_http_transport()

    async def test_qwen_like_mixin_client_uses_proxy(self) -> None:
        """QwenClient 覆盖 _client_session_kwargs 后 session 走代理。"""
        client = _QwenLikeClient()
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}):
            session = await client.ensure_http_session()
            self.assertTrue(session.trust_env)
        await client.close_http_transport()

    async def test_reset_preserves_proxy_setting_for_qwen(self) -> None:
        """QwenClient reset 后重建的 session 仍然启用代理。"""

        class _RecreateClient(_QwenLikeClient):
            def _should_recreate_http_on_reset(self) -> bool:
                return True

        client = _RecreateClient()
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy:8080"}):
            await client.ensure_http_session()
            await client.reset_http_transport()
            new_session = await client.ensure_http_session()
            self.assertTrue(new_session.trust_env)
        await client.close_http_transport()

    async def test_socks_proxy_only_with_use_env_proxy(self) -> None:
        """SOCKS 代理连接器仅在 use_env_proxy=True 时创建。"""
        with patch.dict(os.environ, {"HTTP_PROXY": "socks5://localhost:1080"}):
            no_proxy_session = client_session()
            try:
                self.assertIsInstance(
                    no_proxy_session.connector, aiohttp.TCPConnector
                )
            finally:
                await no_proxy_session.close()

            with_proxy_session = client_session(use_env_proxy=True)
            try:
                try:
                    from aiohttp_socks import ProxyConnector

                    self.assertIsInstance(
                        with_proxy_session.connector, ProxyConnector
                    )
                except ImportError:
                    self.assertIsInstance(
                        with_proxy_session.connector, aiohttp.TCPConnector
                    )
            finally:
                await with_proxy_session.close()


if __name__ == "__main__":
    unittest.main()
