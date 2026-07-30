from __future__ import annotations

import unittest

from handlers.api_errors import handler_error_response
from server.formats import UpstreamConnectionError, UpstreamUnavailableError
from server.retry.session_retry import is_retryable_error


class TestUpstreamUnavailableError(unittest.TestCase):
    def test_handler_maps_to_503(self) -> None:
        exc = UpstreamUnavailableError("DeepSeek 无可用会话", upstream="deepseek")
        resp = handler_error_response(exc, label="OpenAI non-stream")
        self.assertEqual(resp.status, 503)
        self.assertIn(b"DeepSeek", resp.body)

    def test_connection_error_is_retryable(self) -> None:
        exc = UpstreamConnectionError("Qwen 连接失败", upstream="qwen")
        self.assertTrue(is_retryable_error(exc))


if __name__ == "__main__":
    unittest.main()
