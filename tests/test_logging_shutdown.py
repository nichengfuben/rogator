from __future__ import annotations

import logging
import unittest
from unittest.mock import patch

from server.config.logging_setup import (
    setup_logging,
    shutdown_logging,
    resolve_access_log,
    _DowngradePortKillSuccessFilter,
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

    def test_port_kill_success_log_downgraded(self) -> None:
        setup_logging("INFO")
        record = logging.LogRecord(
            name="echotools.exec.process.port",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="已终止占用端口的进程: %s",
            args=(1234,),
            exc_info=None,
        )
        _DowngradePortKillSuccessFilter().filter(record)
        self.assertEqual(record.levelno, logging.DEBUG)
        self.assertEqual(record.levelname, "DEBUG")
    def test_resolve_access_log_enabled(self) -> None:
        setup_logging("INFO")
        access = resolve_access_log(True)
        self.assertIsNotNone(access)
        self.assertEqual(access.name, "aiohttp.access")
        self.assertTrue(access.propagate)
        self.assertEqual(access.handlers, [])

    def test_resolve_access_log_disabled(self) -> None:
        self.assertIsNone(resolve_access_log(False))


if __name__ == "__main__":
    unittest.main()
