from __future__ import annotations

import unittest

from server.model.model_catalog import build_openai_model_entry
from server.model.model_registry import (
    get_model_registry,
    list_external_models,
    resolve_request_model,
)
from server.model.platform_models import QWEN_ASR_EXTERNAL_ID, QWEN_ASR_INTERNAL_ID


class TestQwenAsrPlatformModel(unittest.TestCase):
    def test_list_includes_qwen_asr_without_upstream_inventory(self) -> None:
        ids = list_external_models(["qwen3.7-max"])
        self.assertIn(QWEN_ASR_EXTERNAL_ID, ids)

    def test_resolve_qwen_asr_without_upstream(self) -> None:
        entry = resolve_request_model(QWEN_ASR_EXTERNAL_ID, ["qwen3.7-max"])
        self.assertEqual(entry.internal_id, QWEN_ASR_INTERNAL_ID)

    def test_models_api_entry_shape(self) -> None:
        reg = get_model_registry()
        entry = reg.by_external[QWEN_ASR_EXTERNAL_ID]
        payload = build_openai_model_entry(
            QWEN_ASR_EXTERNAL_ID,
            registry_entry=entry,
            owned_by="qwen",
        )
        self.assertEqual(payload["id"], QWEN_ASR_EXTERNAL_ID)
        self.assertEqual(payload["modality"], ["audio"])
        self.assertTrue(payload["capabilities"]["asr"])
        self.assertTrue(payload["capabilities"]["transcription"])
        self.assertNotIn("thinking", payload["capabilities"])
        self.assertNotIn("tools", payload["capabilities"])


if __name__ == "__main__":
    unittest.main()
