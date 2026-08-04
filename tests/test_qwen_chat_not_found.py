from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from server.formats import UpstreamChatNotFoundError
from upstream.qwen.chat.chat import raise_sse_inline_error


class TestChatNotFoundError(unittest.TestCase):
    def test_sse_inline_chat_not_found(self) -> None:
        client = MagicMock()
        session = MagicMock(username="user01")
        line = json.dumps({
            "success": False,
            "data": {"code": "CHAT_NOT_FOUND", "details": "chat missing"},
        })
        with self.assertRaises(UpstreamChatNotFoundError):
            raise_sse_inline_error(client, session, line)


if __name__ == "__main__":
    unittest.main()
