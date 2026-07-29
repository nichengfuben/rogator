from __future__ import annotations

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from server.config.shutdown import (
    _request_shutdown_once,
    reset_shutdown_signal_state_for_tests,
)


class TestShutdownSignals(unittest.TestCase):
    def setUp(self) -> None:
        reset_shutdown_signal_state_for_tests()

    def tearDown(self) -> None:
        reset_shutdown_signal_state_for_tests()

    def test_first_interrupt_sets_event(self) -> None:
        state = MagicMock()
        state.shutdown_event = asyncio.Event()
        with patch("server.config.shutdown.logger") as log:
            _request_shutdown_once(state, source="Interrupt")
        self.assertTrue(state.shutdown_event.is_set())
        log.info.assert_called_once()

    def test_repeat_interrupt_is_idempotent(self) -> None:
        state = MagicMock()
        state.shutdown_event = asyncio.Event()
        with patch("server.config.shutdown.logger") as log:
            _request_shutdown_once(state, source="Interrupt")
            _request_shutdown_once(state, source="Interrupt")
            _request_shutdown_once(state, source="Interrupt")
        self.assertTrue(state.shutdown_event.is_set())
        log.info.assert_called_once()
        log.warning.assert_not_called()

    def test_repeat_while_event_already_set_is_noop(self) -> None:
        state = MagicMock()
        state.shutdown_event = asyncio.Event()
        state.shutdown_event.set()
        with patch("server.config.shutdown.logger") as log:
            _request_shutdown_once(state, source="Interrupt")
        log.info.assert_not_called()
        log.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
