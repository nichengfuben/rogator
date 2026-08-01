from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from server.config.logging_setup import (
    setup_logging,
    shutdown_logging,
    resolve_access_log,
    silence_hpack_debug,
)


class TestLoggingShutdown(unittest.TestCase):
    def tearDown(self) -> None:
        shutdown_logging()

    def test_shutdown_closes_handlers_and_clears_registry(self) -> None:
        setup_logging("INFO")
        root = logging.getLogger()
        self.assertGreater(len(root.handlers), 0)

        shutdown_logging()

        self.assertEqual(root.handlers, [])
        handler_list = getattr(logging, "_handlerList", None)
        if handler_list is not None:
            self.assertEqual(handler_list, [])

    def test_shutdown_idempotent(self) -> None:
        setup_logging("INFO")
        shutdown_logging()
        shutdown_logging()

    def test_shutdown_without_handler_list(self) -> None:
        setup_logging("INFO")
        with patch.object(logging, "_handlerList", None):
            shutdown_logging()
        self.assertEqual(logging.getLogger().handlers, [])

    def test_shutdown_swallows_close_errors(self) -> None:
        setup_logging("INFO")
        root = logging.getLogger()
        handler = root.handlers[0]

        def _boom() -> None:
            raise OSError("close failed")

        with patch.object(handler, "close", side_effect=_boom):
            shutdown_logging()
        self.assertEqual(root.handlers, [])


class TestAccessLog(unittest.TestCase):
    def test_resolve_access_log_enabled(self) -> None:
        setup_logging("INFO")
        access = resolve_access_log(True)
        self.assertIsNotNone(access)
        self.assertEqual(access.name, "aiohttp.access")
        self.assertTrue(access.propagate)
        self.assertEqual(access.handlers, [])

    def test_resolve_access_log_disabled(self) -> None:
        self.assertIsNone(resolve_access_log(False))


class TestHpackSilence(unittest.TestCase):
    def test_silence_hpack_debug_noop(self) -> None:
        import hpack.hpack as hpack_mod
        import hpack.table as table_mod

        silence_hpack_debug()
        # Must not emit even when logger level is DEBUG
        with self.assertLogs(level="DEBUG") as cm:
            logging.getLogger("rogator").debug("probe")
            hpack_mod.log.debug("Adding %s=%s", b"k", b"v")
            table_mod.log.debug("Resizing header table to %d from %d", 0, 4096)
        self.assertTrue(any("probe" in r.getMessage() for r in cm.records))
        self.assertFalse(any("Adding" in r.getMessage() for r in cm.records))
        self.assertFalse(any("Resizing" in r.getMessage() for r in cm.records))


if __name__ == "__main__":
    unittest.main()
