from __future__ import annotations

""" /api/hello 存活探针。"""

import unittest

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from handlers.platform_handlers import api_hello_handler


class TestApiHello(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/api/hello", api_hello_handler)
        return app

    async def test_get_hello(self) -> None:
        resp = await self.client.get("/api/hello")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.headers.get("Content-Type"), "application/json")
        self.assertEqual(await resp.json(), {"message": "hello"})

    async def test_get_hello_headers_aligned(self) -> None:
        resp = await self.client.get("/api/hello")
        headers = resp.headers
        self.assertEqual(headers.get("Server"), "cloudflare")
        self.assertEqual(headers.get("server-timing"), "x-originResponse;dur=")
        self.assertEqual(headers.get("X-Robots-Tag"), "none")
        self.assertEqual(
            headers.get("Content-Security-Policy"),
            "default-src 'none'; frame-ancestors 'none'",
        )
        self.assertEqual(headers.get("cf-cache-status"), "DYNAMIC")
        self.assertRegex(headers.get("CF-RAY", ""), r"^[0-9a-f]{16}-LAX$")

    async def test_head_hello(self) -> None:
        resp = await self.client.head("/api/hello")
        self.assertEqual(resp.status, 200)
        self.assertEqual(await resp.text(), "")


if __name__ == "__main__":
    unittest.main()
