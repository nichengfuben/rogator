"""send limit 运行时解析：PayloadTooLarge 减半 override > splitter > 全局 fallback。"""

from __future__ import annotations

import sys
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


class TestEffectiveSendMaxChars(unittest.TestCase):
    def test_runtime_override_takes_priority(self) -> None:
        state = MagicMock()
        state._send_limit_overrides = {"deepseek-v4-pro": 80000}
        val = sl.effective_send_max_chars(state, "deepseek-v4-pro")
        self.assertEqual(val, 80000)

    def test_splitter_fallback_when_no_model(self) -> None:
        state = MagicMock()
        state._send_limit_overrides = {}
        state.splitter.max_chars = 123456
        val = sl.effective_send_max_chars(state, None)
        self.assertEqual(val, 123456)

    def test_fb_fallback_when_no_splitter(self) -> None:
        state = MagicMock(spec=[])
        val = sl.effective_send_max_chars(state, None, fallback=99999)
        self.assertEqual(val, 99999)

    def test_model_no_override_uses_splitter(self) -> None:
        state = MagicMock()
        state._send_limit_overrides = {}
        state.splitter.max_chars = 222222
        val = sl.effective_send_max_chars(state, "unknown-model")
        self.assertEqual(val, 222222)


class TestSplitRespectsMaxChars(unittest.TestCase):
    def test_split_respects_max_chars_override(self) -> None:
        splitter = LongTextSplitter(max_chars=1_000_000, send_full_prompt=False)
        full = "a" * 200
        send, name, blob = splitter.split(full, max_chars=50)
        self.assertEqual(len(send), 50)
        self.assertIsNotNone(blob)
        self.assertEqual(len(blob), 150)


if __name__ == "__main__":
    unittest.main()
