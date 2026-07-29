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

    def test_repeat_interrupt_forces_exit(self) -> None:
        state = MagicMock()
        state.shutdown_event = asyncio.Event()
        with patch("server.config.shutdown.logger") as log:
            with patch("server.config.shutdown.os._exit") as force_exit:
                force_exit.side_effect = SystemExit(130)
                _request_shutdown_once(state, source="Interrupt")
                with self.assertRaises(SystemExit):
                    _request_shutdown_once(state, source="Interrupt")
                force_exit.assert_called_once_with(130)
        self.assertTrue(state.shutdown_event.is_set())
        self.assertEqual(log.info.call_count, 1)
        log.warning.assert_called_once()

    def test_repeat_while_event_already_set_forces_exit(self) -> None:
        state = MagicMock()
        state.shutdown_event = asyncio.Event()
        state.shutdown_event.set()
        with patch("server.config.shutdown.logger") as log:
            with patch("server.config.shutdown.os._exit") as force_exit:
                force_exit.side_effect = SystemExit(130)
                with self.assertRaises(SystemExit):
                    _request_shutdown_once(state, source="Interrupt")
                force_exit.assert_called_once_with(130)
        log.info.assert_not_called()
        log.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
