from __future__ import annotations

"""startup_force_kill_port 单元测试。"""

import unittest
from unittest.mock import MagicMock, patch

from echotools.exec.process.port import PortReleaseResult

from server.config.startup_port import ensure_listen_port


class TestStartupPort(unittest.TestCase):
    def test_free_port_no_kill(self) -> None:
        with patch("server.config.startup_port._can_bind", return_value=True):
            with patch("server.config.startup_port.ensure_port_available") as mock_kill:
                ensure_listen_port("0.0.0.0", 8932, force_kill=False)
                mock_kill.assert_not_called()

    def test_occupied_without_force_kill_exits(self) -> None:
        with patch("server.config.startup_port._can_bind", return_value=False):
            with self.assertRaises(SystemExit) as ctx:
                ensure_listen_port("0.0.0.0", 8932, force_kill=False)
            self.assertEqual(ctx.exception.code, 1)

    def test_occupied_with_force_kill_released(self) -> None:
        bind = MagicMock(side_effect=[False, True])
        result = PortReleaseResult(
            port=8932,
            occupied=True,
            released=True,
            pids=[1234],
            detail="force killed processes",
        )
        with patch("server.config.startup_port._can_bind", bind):
            with patch(
                "server.config.startup_port.ensure_port_available",
                return_value=result,
            ) as mock_kill:
                ensure_listen_port("0.0.0.0", 8932, force_kill=True)
                mock_kill.assert_called_once_with(8932, True)


if __name__ == "__main__":
    unittest.main()
