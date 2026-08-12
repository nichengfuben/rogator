from __future__ import annotations

"""UA 分流：claude-code/ 前缀的 /v1/models 返回 Anthropic 官方模型列表格式。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from handlers.platform_handlers import _CLAUDE_CODE_MAX_TOKENS, list_models_handler
from server.model.model_meta import ModelMeta
from server.model.model_registry import get_model_registry


def _fake_state() -> SimpleNamespace:
    registry = get_model_registry()
    entries = registry.entries_in_order[:2]
    return SimpleNamespace(
        _models=[e.internal_id for e in entries],
        models_fetch_timestamp=lambda: 1754640000.0,
        merged_model_meta=lambda: {
            e.internal_id: ModelMeta(context_length=200000 + i * 1000)
            for i, e in enumerate(entries)
        },
        owner_of_model=lambda internal_id: "qwen",
    )


class TestListModelsUaRouting(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/models", list_models_handler)
        return app

    async def _fetch(self, query: str = "", ua: str = "claude-code/2.1.220") -> dict:
        headers = {} if ua == "" else {"User-Agent": ua}
        with patch("handlers.platform_handlers.get_state", return_value=_fake_state()):
            resp = await self.client.get(f"/v1/models{query}", headers=headers)
        self.assertEqual(resp.status, 200)
        return await resp.json()

    async def test_claude_code_ua_gets_anthropic_format(self) -> None:
        payload = await self._fetch()
        data = payload["data"]
        self.assertTrue(data)
        self.assertIs(payload["has_more"], False)
        self.assertEqual(payload["first_id"], data[0]["id"])
        self.assertEqual(payload["last_id"], data[-1]["id"])
        for item in data:
            self.assertEqual(
                list(item.keys()),
                [
                    "id",
                    "max_input_tokens",
                    "max_tokens",
                    "created_at",
                    "display_name",
                    "type",
                ],
            )
            self.assertEqual(item["type"], "model")
            self.assertEqual(item["max_tokens"], _CLAUDE_CODE_MAX_TOKENS)
            self.assertNotIn("capabilities", item)
            self.assertNotIn("modality", item)

    async def test_claude_code_ua_limit_truncates(self) -> None:
        total = len((await self._fetch())["data"])
        self.assertGreater(total, 1)
        payload = await self._fetch("?limit=1")
        self.assertEqual(len(payload["data"]), 1)
        self.assertIs(payload["has_more"], True)
        self.assertEqual(payload["first_id"], payload["data"][0]["id"])
        self.assertEqual(payload["last_id"], payload["data"][0]["id"])

    async def test_claude_code_ua_large_limit_no_truncate(self) -> None:
        total = len((await self._fetch())["data"])
        payload = await self._fetch("?limit=1000")
        self.assertEqual(len(payload["data"]), total)
        self.assertIs(payload["has_more"], False)

    async def test_claude_code_ua_invalid_limit_ignored(self) -> None:
        total = len((await self._fetch())["data"])
        payload = await self._fetch("?limit=-1")
        self.assertEqual(len(payload["data"]), total)
        self.assertIs(payload["has_more"], False)

    async def test_other_ua_limit_truncates(self) -> None:
        payload = await self._fetch("?limit=1", ua="openai-python/1.30.0")
        self.assertEqual(len(payload["data"]), 1)
        self.assertEqual(payload["object"], "list")

    async def test_other_ua_keeps_openai_format(self) -> None:
        payload = await self._fetch(ua="openai-python/1.30.0")
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["object"], "model")
        self.assertIn("updated_at", payload)

    async def test_no_ua_keeps_openai_format(self) -> None:
        payload = await self._fetch(ua="")
        self.assertEqual(payload["object"], "list")


if __name__ == "__main__":
    unittest.main()
