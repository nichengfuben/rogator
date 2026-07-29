from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from server.model.model_registry import (
    ModelInternalIdError,
    ModelNotConfiguredError,
    ModelNotFoundError,
    load_model_registry,
    reload_model_registry,
    resolve_request_model,
)


class TestModelRegistry(unittest.TestCase):
    def test_parse_external_internal_entml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text(
                "qwen3-7-max:qwen3.7-max:true\n"
                "qwen3-8-max-preview:qwen3.8-max-preview:false\n",
                encoding="utf-8",
            )
            reg = load_model_registry(p)
            self.assertTrue(reg.by_external["qwen3-7-max"].uses_entml)
            self.assertFalse(reg.by_external["qwen3-8-max-preview"].uses_entml)

    def test_resolve_external_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text("qwen3-5-omni-flash:qwen3.5-omni-flash:true\n", encoding="utf-8")
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
            p.write_text("qwen3-5-omni-flash:qwen3.5-omni-flash:true\n", encoding="utf-8")
            reload_model_registry(p)
            with self.assertRaises(ModelInternalIdError):
                resolve_request_model("qwen3.5-omni-flash", ["qwen3.5-omni-flash"])
            reload_model_registry()

    def test_unconfigured_upstream_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "registry.jsonl"
            p.write_text("qwen3-7-max:qwen3.7-max:true\n", encoding="utf-8")
            reload_model_registry(p)
            with self.assertRaises(ModelNotConfiguredError):
                resolve_request_model("qwen3.5-omni-flash", ["qwen3.5-omni-flash"])
            reload_model_registry()

    def test_unknown_model(self) -> None:
        with self.assertRaises(ModelNotFoundError):
            resolve_request_model("no-such-model", ["qwen3.7-max"])


if __name__ == "__main__":
    unittest.main()
