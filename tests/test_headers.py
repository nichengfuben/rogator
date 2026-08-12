from __future__ import annotations

"""统一 Cloudflare 伪装响应头：全站端点一致，CF-RAY 每次随机。"""

import re
import unittest

from server.formats.headers import cloudflare_headers


class TestCloudflareHeaders(unittest.TestCase):
    def test_fixed_fields(self) -> None:
        headers = cloudflare_headers()
        self.assertEqual(headers["Server"], "cloudflare")
        self.assertEqual(headers["server-timing"], "x-originResponse;dur=")
        self.assertEqual(headers["X-Robots-Tag"], "none")
        self.assertEqual(
            headers["Content-Security-Policy"],
            "default-src 'none'; frame-ancestors 'none'",
        )
        self.assertEqual(headers["cf-cache-status"], "DYNAMIC")

    def test_cf_ray_random_per_call(self) -> None:
        first = cloudflare_headers()["CF-RAY"]
        second = cloudflare_headers()["CF-RAY"]
        self.assertRegex(first, r"^[0-9a-f]{16}-LAX$")
        self.assertRegex(second, r"^[0-9a-f]{16}-LAX$")
        self.assertNotEqual(first, second)

    def test_cf_ray_ts_advances_at_6_per_second(self) -> None:
        from unittest.mock import patch

        with patch("server.formats.headers.time.time", side_effect=[1_000_000_000.0, 1_000_000_008.0]):
            first = cloudflare_headers()["CF-RAY"]
            second = cloudflare_headers()["CF-RAY"]
        delta = int(second[:8], 16) - int(first[:8], 16)
        # 抓包样本：8s 间隔差值 47 ≈ 6 ticks/s；8s 应为 48 ticks
        self.assertEqual(delta, 48)


if __name__ == "__main__":
    unittest.main()
