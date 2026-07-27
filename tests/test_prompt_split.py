"""LongTextSplitter：inject 后超限 → 尾部 send、前缀附件。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

if __name__ == "__main__":
    unittest.main()
