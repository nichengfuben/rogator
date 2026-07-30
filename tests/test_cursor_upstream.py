from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from upstream.cursor.auth_store import auth_path, get_access_token, write_auth
from upstream.cursor.client import _merge_model_inventory, _model_ids_from_config
from upstream.cursor.models_store import load_merged, read_cache, write_cache
from upstream.cursor.converter import messages_to_cursor_history, split_prompt_and_history
from upstream.cursor.token_service import KeyPool, is_limit_reached, parse_usage


class TestCursorAuthStore(unittest.TestCase):
    def test_write_and_read_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(write_auth("tok123", "ref456", root=root))
            self.assertEqual(get_access_token(root=root), "tok123")
            text = auth_path(root).read_text(encoding="utf-8")
            self.assertIn("access_token", text)
            self.assertIn("ref456", text)


class TestCursorModelsStore(unittest.TestCase):
    def test_read_write_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_cache(
                ["composer-2.5-fast", "composer-2.5"],
                {"composer-2.5-fast": {"id": "composer-2.5-fast", "owned_by": "cursor"}},
                root=root,
            )
            ids, meta, updated_at = read_cache(root=root)
            self.assertEqual(ids, ["composer-2.5-fast", "composer-2.5"])
            self.assertIn("composer-2.5-fast", meta)
            self.assertGreater(updated_at, 0)

    def test_load_merged_prefers_config_then_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_cache(["remote-model"], {}, root=root)
            models, _, _ = load_merged(["composer-2.5-fast"], root=root)
            self.assertEqual(models[0], "composer-2.5-fast")
            self.assertIn("remote-model", models)


class TestCursorModelInventory(unittest.TestCase):
    def test_merge_api_and_mapping(self) -> None:
        ids, meta = _merge_model_inventory([
            {"modelId": "composer-2.5-fast", "displayName": "Fast"},
            {"modelId": "composer-2.5", "displayName": "Standard"},
        ])
        self.assertIn("composer-2.5-fast", ids)
        self.assertIn("composer-2.5", ids)
        self.assertIn("composer-2.5-fast", meta)

    def test_config_fallback_ids(self) -> None:
        with patch("upstream.cursor.client.load_cursor_upstream_config", return_value={
            "models": {"default": "composer-2.5-fast", "mapping": {"gpt-4": "composer-2.5-fast"}, "fallback": ["legacy-model"]},
            "cursor": {},
        }):
            ids = _model_ids_from_config()
        self.assertIn("gpt-4", ids)
        self.assertIn("legacy-model", ids)


class TestCursorConfig(unittest.TestCase):
    def test_default_starcursor_section(self) -> None:
        with patch("server.config.app_config._load_upstream_toml", return_value={}):
            from upstream.cursor.config import load_cursor_upstream_config
            cfg = load_cursor_upstream_config()
        sc = cfg["starcursor"]
        self.assertIn("base_url", sc)
        self.assertIn("poll_interval", sc)
        self.assertEqual(sc["poll_interval"], 30)


class TestCursorConverter(unittest.TestCase):
    def test_split_prompt_and_history(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]
        prompt, history = split_prompt_and_history(messages)
        self.assertEqual(prompt, "again")
        self.assertEqual(len(history), 2)
        cursor_hist = messages_to_cursor_history(messages[:-1])
        self.assertEqual(history, cursor_hist)


class TestCursorTokenHelpers(unittest.TestCase):
    def test_key_pool_rotates(self) -> None:
        pool = KeyPool(["k1", "k2"], threshold=80, refresh_interval=60)
        first = pool.current
        assert first is not None
        pool.switch_next()
        second = pool.current
        assert second is not None
        self.assertNotEqual(first.key, second.key)

    def test_usage_limit(self) -> None:
        u = parse_usage({
            "individualUsage": {"plan": {
                "autoPercentUsed": 96,
                "apiPercentUsed": 94,
                "breakdown": {"total": 100},
            }},
        })
        self.assertTrue(is_limit_reached(u, threshold=95.0))


class TestCursorClientStartup(unittest.IsolatedAsyncioTestCase):
    async def test_startup_without_keys_warns(self) -> None:
        from upstream.cursor.client import CursorClient

        with patch("upstream.cursor.client.starcursor_config", return_value={"api_keys": [], "poll_interval": 30}):
            with patch("upstream.cursor.client.get_access_token", return_value=None):
                with patch("upstream.cursor.client.CursorClient.fetch_models", return_value=[]):
                    client = CursorClient(None)
                    await client.startup()
        self.assertTrue(client._startup_done)


if __name__ == "__main__":
    unittest.main()
