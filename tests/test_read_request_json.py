from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.formats import ClientDisconnectedError, read_request_json


class TestReadRequestJson(unittest.IsolatedAsyncioTestCase):
    async def test_empty_body_allowed(self) -> None:
        request = MagicMock()
        request.can_read_body = False
        request.headers = {"Content-Length": "0"}
        self.assertEqual(await read_request_json(request), {})

    async def test_empty_body_with_content_length_raises(self) -> None:
        request = MagicMock()
        request.can_read_body = False
        request.headers = {"Content-Length": "42"}
        with pytest.raises(Exception):  # web.HTTPBadRequest
            await read_request_json(request)

    async def test_connection_reset(self) -> None:
        request = MagicMock()
        request.can_read_body = True
        request.json = AsyncMock(
            side_effect=ConnectionResetError(10054, "远程主机强迫关闭了一个现有的连接。"),
        )
        with pytest.raises(ClientDisconnectedError):
            await read_request_json(request)

    async def test_returns_dict(self) -> None:
        request = MagicMock()
        request.can_read_body = True
        request.json = AsyncMock(return_value={"messages": []})
        self.assertEqual(await read_request_json(request), {"messages": []})


if __name__ == "__main__":
    unittest.main()
