from __future__ import annotations

import unittest

from handlers.api_errors import handler_error_response
from server.formats import UpstreamUnavailableError


class TestUpstreamUnavailableError(unittest.TestCase):
    def test_handler_maps_to_503(self) -> None:
        exc = UpstreamUnavailableError("DeepSeek 无可用会话", upstream="deepseek")
        resp = handler_error_response(exc, label="OpenAI non-stream")
        self.assertEqual(resp.status, 503)
        self.assertIn(b"DeepSeek", resp.body)


if __name__ == "__main__":
    unittest.main()
