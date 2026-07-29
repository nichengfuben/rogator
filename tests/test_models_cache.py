from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from upstream.qwen.account import ModelsFetchMixin, merge_model_lists
from server.formats import DEFAULT_MODELS
from server.model.model_meta import DEFAULT_MODEL_CONTEXT_LENGTH, ModelMeta, default_model_meta


class _ModelsClient(ModelsFetchMixin):
    def __init__(self) -> None:
        self._models: list[str] = list(DEFAULT_MODELS)
        self._model_meta: dict[str, ModelMeta] = {}
        self._models_fetch_time: float = 0.0

    async def _ensure_cleanup(self) -> None:
        return

    async def get_valid_session(self):
        return None


class TestMergeModelLists(unittest.TestCase):
    def test_default_plus_remote_dedupes(self) -> None:
        merged = merge_model_lists(
            ["a", "b"],
            ["b", "c"],
            ["c", "d"],
        )
        self.assertEqual(merged, ["a", "b", "c", "d"])

    def test_default_always_first(self) -> None:
        merged = merge_model_lists(list(DEFAULT_MODELS), ["extra-model"])
        self.assertEqual(merged[0], DEFAULT_MODELS[0])
        self.assertIn("extra-model", merged)


class TestModelsRefreshDue(unittest.TestCase):
    def test_no_timestamp_means_due(self) -> None:
        client = _ModelsClient()
        self.assertTrue(client.models_refresh_due(3600.0))

    def test_fresh_cache_not_due(self) -> None:
        client = _ModelsClient()
        client._models_fetch_time = time.time() - 100
        self.assertFalse(client.models_refresh_due(3600.0))

    def test_stale_cache_is_due(self) -> None:
        client = _ModelsClient()
        client._models_fetch_time = time.time() - 7200
        self.assertTrue(client.models_refresh_due(3600.0))

    def test_zero_interval_always_due(self) -> None:
        client = _ModelsClient()
        client._models_fetch_time = time.time()
        self.assertTrue(client.models_refresh_due(0))


class TestLoadModelsCache(unittest.TestCase):
    def test_loads_updated_at_and_merges_default(self) -> None:
        client = _ModelsClient()
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "models.json"
            payload = {
                "models": ["upstream-only"],
                "meta": {
                    "upstream-only": {
                        "context_length": 131072,
                        "capabilities": {"chat": True, "vision": True},
                        "modality": ["text", "image"],
                    },
                },
                "updated_at": 1_700_000_000,
            }
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("upstream.qwen.account.MODELS_CACHE_FILE", str(cache_path)):
                models = client.load_models_cache()
        self.assertIn("upstream-only", models)
        self.assertEqual(models[0], DEFAULT_MODELS[0])
        self.assertEqual(client._models_fetch_time, 1_700_000_000.0)
        meta = client._model_meta["upstream-only"]
        self.assertEqual(meta.context_length, 131072)
        self.assertTrue(meta.capabilities.get("vision"))

    def test_default_meta_for_models_without_disk_entry(self) -> None:
        client = _ModelsClient()
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "missing.json"
            with patch("upstream.qwen.account.MODELS_CACHE_FILE", str(cache_path)):
                client.load_models_cache()
        meta = client._model_meta[DEFAULT_MODELS[0]]
        self.assertEqual(meta.context_length, DEFAULT_MODEL_CONTEXT_LENGTH)
        self.assertTrue(meta.capabilities.get("vision"))
        self.assertNotIn("thinking", meta.capabilities)


if __name__ == "__main__":
    unittest.main()
