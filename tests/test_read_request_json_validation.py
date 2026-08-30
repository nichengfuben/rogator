"""``read_request_json`` Content-Length 与 body 不一致时返 4xx。"""

from __future__ import annotations

import pytest
from aiohttp import web

from server.formats.errors import read_request_json


class _NoBodyRequest:
    """绕过 aiohttp ``can_read_body`` property，只透传 headers 给函数。"""

    def __init__(self, headers) -> None:
        self.headers = headers
        self.can_read_body = False


async def test_content_length_nonzero_with_empty_body_raises_400() -> None:
    request = _NoBodyRequest({"Content-Length": "10"})
    with pytest.raises(web.HTTPBadRequest) as excinfo:
        await read_request_json(request)
    assert excinfo.value.status == 400


async def test_content_length_zero_returns_empty_dict() -> None:
    request = _NoBodyRequest({"Content-Length": "0"})
    body = await read_request_json(request)
    assert body == {}