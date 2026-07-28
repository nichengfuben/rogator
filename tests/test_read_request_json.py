from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from server.formats import ClientDisconnectedError, read_request_json


@pytest.mark.asyncio
async def test_read_request_json_empty_body_allowed() -> None:
    request = MagicMock()
    request.can_read_body = False
    assert await read_request_json(request) == {}


@pytest.mark.asyncio
async def test_read_request_json_connection_reset() -> None:
    request = MagicMock()
    request.can_read_body = True
    request.json = AsyncMock(
        side_effect=ConnectionResetError(10054, "远程主机强迫关闭了一个现有的连接。"),
    )
    with pytest.raises(ClientDisconnectedError):
        await read_request_json(request)


@pytest.mark.asyncio
async def test_read_request_json_returns_dict() -> None:
    request = MagicMock()
    request.can_read_body = True
    request.json = AsyncMock(return_value={"messages": []})
    assert await read_request_json(request) == {"messages": []}
