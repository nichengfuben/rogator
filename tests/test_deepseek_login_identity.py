from __future__ import annotations

"""DeepSeek 邮箱/手机号登录 identity 测试。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.session.accounts import accounts_for_upstream
from upstream.deepseek.lib.user.userapi import build_login_payload


class TestDeepSeekLoginPayload(unittest.TestCase):
    def test_email_login_payload(self) -> None:
        payload = build_login_payload(
            "user@example.com", "secret", "did-1",
        )
        self.assertEqual(payload["email"], "user@example.com")
        self.assertEqual(payload["mobile"], "")
        self.assertEqual(payload["area_code"], "")
        self.assertEqual(payload["password"], "secret")
        self.assertEqual(payload["device_id"], "did-1")
        self.assertEqual(payload["os"], "web")

    def test_mobile_login_payload(self) -> None:
        payload = build_login_payload("13800138000", "secret", "did-2")
        self.assertEqual(payload["email"], "")
        self.assertEqual(payload["mobile"], "13800138000")
        self.assertEqual(payload["area_code"], "+86")

    def test_mobile_with_plus_prefix(self) -> None:
        payload = build_login_payload("+86 138-0013-8000", "secret", "did-3")
        self.assertEqual(payload["mobile"], "13800138000")
        self.assertEqual(payload["area_code"], "+86")

    def test_mobile_custom_area_code(self) -> None:
        payload = build_login_payload(
            "912345678", "secret", "did-4", area_code="+1",
        )
        self.assertEqual(payload["mobile"], "912345678")
        self.assertEqual(payload["area_code"], "+1")


class TestDeepSeekAccountsCsv(unittest.TestCase):
    def test_phone_column_loads_as_username(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_csv = root / "deepseek" / "accounts.csv"
            ds_csv.parent.mkdir(parents=True)
            ds_csv.write_text(
                "phone,password,area_code\n"
                "13800138000,pw1,+86\n",
                encoding="utf-8",
            )
            with patch.dict(
                "core.session.accounts._UPSTREAM_CSV",
                {"deepseek": ds_csv},
                clear=False,
            ), patch("core.session.accounts._ROOT_CSV", root / "missing.csv"):
                pool = accounts_for_upstream("deepseek")
            self.assertEqual(len(pool), 1)
            self.assertEqual(pool[0].username, "13800138000")
            self.assertEqual(pool[0].area_code, "+86")

    def test_qwen_still_prefers_email_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qwen_csv = root / "qwen" / "accounts.csv"
            qwen_csv.parent.mkdir(parents=True)
            qwen_csv.write_text(
                "email,password\n"
                "qwen@example.com,pw\n",
                encoding="utf-8",
            )
            with patch.dict(
                "core.session.accounts._UPSTREAM_CSV",
                {"qwen": qwen_csv},
                clear=False,
            ), patch("core.session.accounts._ROOT_CSV", root / "missing.csv"):
                pool = accounts_for_upstream("qwen")
            self.assertEqual(pool[0].username, "qwen@example.com")


if __name__ == "__main__":
    unittest.main()
