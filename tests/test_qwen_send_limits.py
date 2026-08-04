"""Qwen 按模型发送上限与 prompt split 覆盖。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)
import path_setup  # noqa: F401

from state import LongTextSplitter
from server.config import qwen_send_limits as sl


class TestQwenSendLimits(unittest.TestCase):
    def setUp(self) -> None:
        sl.invalidate_model_send_limits_cache()

    def test_resolve_by_internal_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_send_limits.toml"
            path.write_text(
                '[models]\n"qwen3.7-max" = 126922\n',
                encoding="utf-8",
            )
            with unittest.mock.patch.object(sl, "model_send_limits_path", return_value=path):
                self.assertEqual(sl.resolve_qwen_send_max_chars("qwen3.7-max", fallback=999), 126922)
                self.assertEqual(sl.resolve_qwen_send_max_chars("unknown-model", fallback=999), 999)

    def test_external_key_in_toml_maps_to_internal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_send_limits.toml"
            path.write_text(
                '[models]\n"qwen3-7-max" = 126922\n',
                encoding="utf-8",
            )
            with unittest.mock.patch.object(sl, "model_send_limits_path", return_value=path):
                sl.invalidate_model_send_limits_cache()
                self.assertEqual(sl.resolve_qwen_send_max_chars("qwen3.7-max", fallback=999), 126922)

    def test_external_model_arg_resolves_via_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_send_limits.toml"
            path.write_text(
                '[models]\n"qwen3.7-max" = 126922\n',
                encoding="utf-8",
            )
            with unittest.mock.patch.object(sl, "model_send_limits_path", return_value=path):
                sl.invalidate_model_send_limits_cache()
                self.assertEqual(sl.resolve_qwen_send_max_chars("qwen3-7-max", fallback=999), 126922)

    def test_effective_uses_runtime_override(self) -> None:
        state = MagicMock()
        state._send_limit_overrides = {"qwen3.7-max": 80000}
        with unittest.mock.patch.object(
            sl, "resolve_qwen_send_max_chars", return_value=126922,
        ):
            self.assertEqual(sl.effective_send_max_chars(state, "qwen3.7-max"), 80000)

    def test_split_respects_max_chars_override(self) -> None:
        splitter = LongTextSplitter(max_chars=1_000_000, send_full_prompt=False)
        full = "a" * 200
        send, name, blob = splitter.split(full, max_chars=50)
        self.assertEqual(len(send), 50)
        self.assertIsNotNone(blob)
        self.assertEqual(len(blob), 150)


if __name__ == "__main__":
    unittest.main()
