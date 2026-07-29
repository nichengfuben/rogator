from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock

from server.config.shutdown import (
    _request_shutdown_once,
    reset_shutdown_signal_state_for_tests,
)


class TestShutdownSignals(unittest.TestCase):
    def setUp(self) -> None:
        reset_shutdown_signal_state_for_tests()

    def tearDown(self) -> None:
        reset_shutdown_signal_state_for_tests()

    def test_second_interrupt_raises_system_exit(self) -> None:
        state = MagicMock()
        state.shutdown_event = asyncio.Event()
        _request_shutdown_once(state, source="Interrupt")
        with self.assertRaises(SystemExit) as ctx:
            _request_shutdown_once(state, source="Interrupt")
        self.assertEqual(ctx.exception.code, 130)

    def test_repeat_while_event_set_raises_system_exit(self) -> None:
        state = MagicMock()
        state.shutdown_event = asyncio.Event()
        state.shutdown_event.set()
        _request_shutdown_once(state, source="Interrupt")
        with self.assertRaises(SystemExit):
            _request_shutdown_once(state, source="Interrupt")


if __name__ == "__main__":
    unittest.main()
