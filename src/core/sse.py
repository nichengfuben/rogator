from __future__ import annotations

"""Generic SSE frame parsing (data: lines → payload string)."""

from typing import AsyncIterator, Optional

import aiohttp


async def iter_sse_data_lines(
    resp: aiohttp.ClientResponse,
) -> AsyncIterator[str]:
    """Yield raw SSE ``data:`` payloads (without the ``data:`` prefix)."""
    buffer = ""
    async for chunk in resp.content.iter_any():
        buffer += chunk.decode("utf-8", errors="ignore")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                yield line[5:].lstrip()


def sse_done(payload: Optional[str]) -> bool:
    return payload is None or payload == "[DONE]"
