from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from server.retry.http_client import active_proxy_url, client_session, init_http_proxy_from_env, sync_proxy_env


class TestHttpProxyEnv(unittest.TestCase):
    def test_sync_proxy_env_mirrors_case(self) -> None:
        env = {}
        with patch.dict(os.environ, env, clear=True):
            os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
            sync_proxy_env()
            self.assertEqual(os.environ.get("http_proxy"), "http://127.0.0.1:10808")

    def test_active_proxy_prefers_https(self) -> None:
        with patch.dict(os.environ, {
            "HTTP_PROXY": "http://127.0.0.1:7890",
            "HTTPS_PROXY": "http://127.0.0.1:10808",
        }, clear=True):
            self.assertEqual(active_proxy_url(), "http://127.0.0.1:10808")

    def test_client_session_uses_trust_env_for_http_proxy(self) -> None:
        import asyncio

        async def _run() -> None:
            with patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:10808"}, clear=True):
                session = client_session()
                try:
                    self.assertTrue(getattr(session, "_trust_env", False))
                finally:
                    await session.close()

        asyncio.run(_run())

    def test_init_logs_when_proxy_set(self) -> None:
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:10808"}, clear=True):
            url = init_http_proxy_from_env()
        self.assertEqual(url, "http://127.0.0.1:10808")


if __name__ == "__main__":
    unittest.main()
