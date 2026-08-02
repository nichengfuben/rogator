from __future__ import annotations

"""core.compat 跨版本工具测试。"""

import unittest

from core.transport.compat import removeprefix, removesuffix


class TestCompat(unittest.TestCase):
    def test_removeprefix(self) -> None:
        self.assertEqual(removeprefix("https://example.com", "https://"), "example.com")
        self.assertEqual(removeprefix("http://example.com", "https://"), "http://example.com")

    def test_removesuffix(self) -> None:
        self.assertEqual(removesuffix("foo.txt", ".txt"), "foo")
        self.assertEqual(removesuffix("foo", ".txt"), "foo")


if __name__ == "__main__":
    unittest.main()
