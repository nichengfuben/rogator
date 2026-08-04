from __future__ import annotations

import unittest

from upstream.qwen.chat.routes import PERSISTED_MODEL_CAPABILITIES
from server.model.model_catalog import (
    MODEL_CONTEXT_LENGTH,
    build_openai_model_entry,
    model_supports_thinking,
)
from server.model.model_registry import ModelRegistryEntry
from server.model.model_meta import (
    DEFAULT_MODEL_CONTEXT_LENGTH,
    ModelMeta,
    capabilities_for_api,
    merge_capabilities,
    merge_model_meta,
    normalize_capabilities,
    parse_upstream_models_payload,
)


class TestParseUpstreamModelsPayload(unittest.TestCase):
    def test_nested_data_data_with_meta(self) -> None:
        payload = {
            "success": True,
            "data": {
                "data": [
                    {
                        "id": "qwen3.7-plus",
                        "info": {
                            "meta": {
                                "max_context_length": 1_000_000,
                                "capabilities": {
                                    "vision": True,
                                    "document": True,
                                    "thinking": False,
                                },
                                "modality": ["text", "image", "video"],
                            },
                        },
                    },
                ],
            },
        }
        meta = parse_upstream_models_payload(payload)
        self.assertIn("qwen3.7-plus", meta)
        entry = meta["qwen3.7-plus"]
        self.assertEqual(entry.context_length, 1_000_000)
        self.assertTrue(entry.capabilities["vision"])
        self.assertTrue(entry.capabilities["chat"])
        self.assertTrue(entry.capabilities["audio"])
        self.assertNotIn("thinking", entry.capabilities)
        self.assertEqual(entry.modality, ["text", "image", "video"])

    def test_partial_upstream_caps_keep_default_baseline(self) -> None:
        payload = {
            "data": {
                "data": [
                    {
                        "id": "qwen3.7-max",
                        "info": {
                            "meta": {
                                "capabilities": {
                                    "document": True,
                                    "audio": False,
                                },
                            },
                        },
                    },
                ],
            },
        }
        caps = parse_upstream_models_payload(payload)["qwen3.7-max"].capabilities
        self.assertTrue(caps["vision"])
        self.assertTrue(caps["video"])
        self.assertNotIn("thinking", caps)
        self.assertFalse(caps["audio"])

    def test_missing_meta_uses_defaults(self) -> None:
        payload = {"data": {"data": [{"id": "plain-model"}]}}
        meta = parse_upstream_models_payload(payload)["plain-model"]
        self.assertEqual(meta.context_length, DEFAULT_MODEL_CONTEXT_LENGTH)
        self.assertEqual(meta.capabilities, PERSISTED_MODEL_CAPABILITIES)


class TestStoredCapabilities(unittest.TestCase):
    def test_thinking_ignored_in_merge(self) -> None:
        caps = merge_capabilities({"thinking": False, "vision": False})
        self.assertNotIn("thinking", caps)
        self.assertFalse(caps["vision"])

    def test_normalize_starts_from_default_baseline(self) -> None:
        caps = normalize_capabilities({"citations": True})
        self.assertTrue(caps["chat"])
        self.assertTrue(caps["vision"])
        self.assertTrue(caps["citations"])
        self.assertNotIn("thinking", caps)

    def test_to_dict_omits_thinking(self) -> None:
        payload = ModelMeta(
            capabilities={"vision": True, "tools": True, "native_tools": True},
        ).to_dict()
        self.assertNotIn("thinking", payload["capabilities"])
        self.assertNotIn("tools", payload["capabilities"])
        self.assertNotIn("native_tools", payload["capabilities"])

    def test_capabilities_for_api_adds_thinking(self) -> None:
        api = capabilities_for_api({"vision": True})
        self.assertTrue(api["thinking"])
        self.assertTrue(api["tools"])
        self.assertTrue(api["native_tools"])
        self.assertTrue(api["vision"])


class TestBuildOpenAIModelEntry(unittest.TestCase):
    def _entry(
        self,
        external: str,
        internal: str,
        *,
        uses_entml: bool = True,
        uses_entml_tools: bool = True,
    ) -> ModelRegistryEntry:
        return ModelRegistryEntry(external, internal, uses_entml, uses_entml_tools)

    def test_exposes_finalized_capabilities(self) -> None:
        meta = {
            "demo-internal": ModelMeta(
                context_length=512000,
                capabilities={"chat": True, "vision": True},
                modality=["text", "image"],
            ),
        }
        entry = build_openai_model_entry(
            "demo-external",
            registry_entry=self._entry("demo-external", "demo-internal"),
            meta_by_id=meta,
            created=1_700_000_111,
        )
        self.assertEqual(entry["created"], 1_700_000_111)
        self.assertEqual(entry["context_length"], 512000)
        self.assertTrue(entry["capabilities"]["vision"])
        self.assertTrue(entry["capabilities"]["audio"])
        self.assertTrue(entry["capabilities"]["thinking"])
        self.assertTrue(entry["capabilities"]["tools"])
        self.assertTrue(entry["capabilities"]["native_tools"])
        self.assertEqual(entry["modality"], ["text", "image"])

    def test_default_context_is_256k_tokens(self) -> None:
        entry = build_openai_model_entry(
            "unknown-external",
            registry_entry=self._entry("unknown-external", "unknown-internal"),
            created=123,
        )
        self.assertEqual(entry["context_length"], MODEL_CONTEXT_LENGTH)
        self.assertTrue(entry["capabilities"]["vision"])
        self.assertTrue(entry["capabilities"]["thinking"])
        self.assertTrue(entry["capabilities"]["tools"])
        self.assertTrue(entry["capabilities"]["native_tools"])

    def test_think_efforts_preserved(self) -> None:
        reg = self._entry("qwen3-7-max", "qwen3.7-max", uses_entml=True)
        self.assertTrue(model_supports_thinking(reg))
        entry = build_openai_model_entry("qwen3-7-max", registry_entry=reg, created=123)
        self.assertIn("think_efforts", entry)


class TestMergeModelMeta(unittest.TestCase):
    def test_remote_overrides_disk_context(self) -> None:
        disk = {"m1": ModelMeta(context_length=1000)}
        remote = {"m1": ModelMeta(context_length=2000, capabilities={"vision": False})}
        merged = merge_model_meta(["m1"], disk, remote)
        self.assertEqual(merged["m1"].context_length, 2000)
        self.assertFalse(merged["m1"].capabilities["vision"])
        self.assertTrue(merged["m1"].capabilities["chat"])
        self.assertNotIn("thinking", merged["m1"].capabilities)


if __name__ == "__main__":
    unittest.main()
