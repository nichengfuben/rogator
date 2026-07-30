from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from server.model.model_registry import (
    ModelInternalIdError,
    ModelNotConfiguredError,
    ModelNotFoundError,
    is_native_upstream_event,
    load_model_registry,
    reload_model_registry,
    resolve_request_model,
)


class TestModelRegistry(unittest.TestCase):
    def test_parse_external_internal_entml_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text(
                "qwen3-7-max:qwen3.7-max:true:true\n"
                "qwen3-8-max-preview:qwen3.8-max-preview:false:true\n"
                "cursor-fast:composer-2.5-fast:false:false\n",
                encoding="utf-8",
            )
            reg = load_model_registry(p)
            self.assertTrue(reg.by_external["qwen3-7-max"].uses_entml)
            self.assertTrue(reg.by_external["qwen3-7-max"].uses_entml_tools)
            self.assertFalse(reg.by_external["qwen3-8-max-preview"].uses_entml)
            self.assertTrue(reg.by_external["qwen3-8-max-preview"].uses_entml_tools)
            self.assertFalse(reg.by_external["cursor-fast"].uses_entml_tools)

    def test_legacy_three_field_defaults_tools_to_thinking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text("legacy:legacy-internal:false\n", encoding="utf-8")
            reg = load_model_registry(p)
            entry = reg.by_external["legacy"]
            self.assertFalse(entry.uses_entml)
            self.assertFalse(entry.uses_entml_tools)

    def test_resolve_external_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text("qwen3-5-omni-flash:qwen3.5-omni-flash:true:true\n", encoding="utf-8")
            reload_model_registry(p)
            entry = resolve_request_model(
                "qwen3-5-omni-flash",
                ["qwen3.5-omni-flash"],
            )
            self.assertEqual(entry.internal_id, "qwen3.5-omni-flash")
            reload_model_registry()

    def test_reject_internal_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text("qwen3-5-omni-flash:qwen3.5-omni-flash:true:true\n", encoding="utf-8")
            reload_model_registry(p)
            with self.assertRaises(ModelInternalIdError):
                resolve_request_model("qwen3.5-omni-flash", ["qwen3.5-omni-flash"])
            reload_model_registry()

    def test_unconfigured_upstream_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text("qwen3-7-max:qwen3.7-max:true:true\n", encoding="utf-8")
            reload_model_registry(p)
            with self.assertRaises(ModelNotConfiguredError):
                resolve_request_model("qwen3.5-omni-flash", ["qwen3.5-omni-flash"])
            reload_model_registry()

    def test_unknown_model(self) -> None:
        with self.assertRaises(ModelNotFoundError):
            resolve_request_model("no-such-model", ["qwen3.7-max"])

    def test_cursor_external_maps_to_internal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text(
                "cursor-composer-2-5-fast:composer-2.5-fast:false:false\n"
                "gpt-4:gpt-4:false:false\n",
                encoding="utf-8",
            )
            reload_model_registry(p)
            entry = resolve_request_model(
                "cursor-composer-2-5-fast",
                ["composer-2.5-fast"],
            )
            self.assertEqual(entry.internal_id, "composer-2.5-fast")
            self.assertFalse(entry.uses_entml)
            self.assertFalse(entry.uses_entml_tools)
            alias = resolve_request_model("gpt-4", ["gpt-4"])
            self.assertEqual(alias.internal_id, "gpt-4")
            reload_model_registry()

    def test_is_native_upstream_event_from_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text("cursor-fast:composer-2.5-fast:false:false\n", encoding="utf-8")
            reload_model_registry(p)
            entry = resolve_request_model("cursor-fast", ["composer-2.5-fast"])
            self.assertTrue(is_native_upstream_event(entry, {"type": "tool_call"}))
            self.assertTrue(is_native_upstream_event(entry, {"type": "thinking"}))
            self.assertTrue(is_native_upstream_event(entry, {"type": "answer", "content": "hi"}))
            reload_model_registry()


if __name__ == "__main__":
    unittest.main()
