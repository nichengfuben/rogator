from __future__ import annotations

"""echotools 导入与版本自检。"""

import unittest
from unittest.mock import patch

from path_setup import ensure_echotools_importable


class TestEchotoolsBootstrap(unittest.TestCase):
    def test_importable_on_current_install(self) -> None:
        ensure_echotools_importable()

    def test_rejects_broken_242(self) -> None:
        class _FakeEchotools:
            __version__ = "2.4.2"

        with patch.dict("sys.modules", {"echotools": _FakeEchotools()}):
            with self.assertRaises(RuntimeError) as ctx:
                ensure_echotools_importable()
            self.assertIn("2.4.2", str(ctx.exception))

    def test_rejects_below_245(self) -> None:
        class _FakeEchotools:
            __version__ = "2.4.4"

        with patch.dict("sys.modules", {"echotools": _FakeEchotools()}):
            with self.assertRaises(RuntimeError) as ctx:
                ensure_echotools_importable()
            self.assertIn("2.4.5", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
