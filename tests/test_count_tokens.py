from __future__ import annotations

"""count_tokens 端点：响应结构 {"input_tokens": number} 与官方对齐（纯 application/json）。"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from handlers.platform_handlers import count_tokens_handler


def _fake_state() -> SimpleNamespace:
    return SimpleNamespace(model="qwen3-8-max-preview", protocol="entml")


class TestCountTokens(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/messages/count_tokens", count_tokens_handler)
        return app

    async def test_count_tokens_response_shape(self) -> None:
        payload = json.dumps({
            "model": "qwen3-8-max-preview",
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 1024,
        })
        with patch("handlers.platform_handlers.get_state", return_value=_fake_state()), \
                patch("handlers.platform_handlers.resolve_handler_model") as mock_resolve, \
                patch(
                    "handlers.platform_handlers.estimate_anthropic_injected_input_tokens",
                    return_value=15234,
                ):
            mock_resolve.return_value = SimpleNamespace()
            resp = await self.client.post(
                "/v1/messages/count_tokens",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers.get("Content-Type"), "application/json")
        body = await resp.json()
        self.assertEqual(body, {"input_tokens": 15234})
        self.assertIsInstance(body["input_tokens"], int)

    async def test_count_tokens_invalid_json_returns_400(self) -> None:
        resp = await self.client.post(
            "/v1/messages/count_tokens",
            data="{not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status, 400)
        self.assertEqual(resp.headers.get("Content-Type"), "application/json")
        self.assertEqual(resp.headers.get("Server"), "cloudflare")
        self.assertRegex(resp.headers.get("CF-RAY", ""), r"^[0-9a-f]{16}-LAX$")


if __name__ == "__main__":
    unittest.main()
