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
        self.assertTrue(resp.headers.get("Content-Type", "").startswith("text/plain"))
        self.assertEqual(await resp.text(), "hello")

    async def test_head_hello(self) -> None:
        resp = await self.client.head("/api/hello")
        self.assertEqual(resp.status, 200)
        self.assertEqual(await resp.text(), "")


if __name__ == "__main__":
    unittest.main()
