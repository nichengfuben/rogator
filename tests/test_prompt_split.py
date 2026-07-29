"""LongTextSplitter：inject 后超限 → 尾部 send、前缀附件。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT):
    s = str(entry)
    if s not in sys.path:
        sys.path.insert(0, s)
import path_setup  # noqa: F401

from state import LongTextSplitter


class TestLongTextSplitter(unittest.TestCase):
    def test_under_limit_no_attachment(self) -> None:
        splitter = LongTextSplitter(max_chars=100)
        send, name, blob = splitter.split("abcdefghij")
        self.assertEqual(send, "abcdefghij")
        self.assertIsNone(name)
        self.assertIsNone(blob)

    def test_over_limit_tail_send_prefix_attachment(self) -> None:
        splitter = LongTextSplitter(max_chars=5)
        full = "ABCDEFGHIJ"
        send, name, blob = splitter.split(full)
        self.assertEqual(send, "FGHIJ")
        self.assertIsNotNone(name)
        self.assertIsNotNone(blob)
        self.assertEqual(blob.decode("utf-8"), "ABCDE")
        self.assertEqual(blob.decode("utf-8") + send, full)

    def test_lengths_sum_to_full(self) -> None:
        splitter = LongTextSplitter(max_chars=150_000)
        full = "x" * 205_116
        send, _, blob = splitter.split(full)
        self.assertEqual(len(send), 150_000)
        self.assertEqual(len(blob), 55_116)
        self.assertEqual(blob.decode("utf-8") + send, full)

    def test_send_full_prompt_never_splits(self) -> None:
        splitter = LongTextSplitter(max_chars=5, send_full_prompt=True)
        full = "ABCDEFGHIJ" * 1000
        send, name, blob = splitter.split(full)
        self.assertEqual(send, full)
        self.assertIsNone(name)
        self.assertIsNone(blob)


if __name__ == "__main__":
    unittest.main()
