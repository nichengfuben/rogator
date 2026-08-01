from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.session.accounts import accounts_csv_path, accounts_for_upstream


class TestAccountsForUpstream(unittest.TestCase):
    def test_separate_csv_per_upstream(self) -> None:
        self.assertEqual(
            str(accounts_csv_path("qwen")).replace("\\", "/").split("/")[-2:],
            ["qwen", "accounts.csv"],
        )
        self.assertEqual(
            str(accounts_csv_path("deepseek")).replace("\\", "/").split("/")[-2:],
            ["deepseek", "accounts.csv"],
        )

    def test_deepseek_pool_isolated_from_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qwen_csv = root / "qwen" / "accounts.csv"
            ds_csv = root / "deepseek" / "accounts.csv"
            qwen_csv.parent.mkdir(parents=True)
            ds_csv.parent.mkdir(parents=True)
            qwen_csv.write_text(
                "email,password\n"
                "qwen-only@example.com,pw1\n"
                "qwen-second@example.com,pw2\n",
                encoding="utf-8",
            )
            ds_csv.write_text(
                "email,password\n"
                "deepseek-only@example.com,pw3\n",
                encoding="utf-8",
            )
            fake_paths = {
                "qwen": qwen_csv,
                "deepseek": ds_csv,
            }
            with patch.dict(
                "core.session.accounts._UPSTREAM_CSV",
                fake_paths,
                clear=False,
            ), patch(
                "core.session.accounts._ROOT_CSV",
                root / "missing-root.csv",
            ):
                qwen_pool = accounts_for_upstream("qwen")
                ds_pool = accounts_for_upstream("deepseek")

            self.assertEqual(len(qwen_pool), 2)
            self.assertEqual(len(ds_pool), 1)
            qwen_names = {a.username for a in qwen_pool}
            ds_names = {a.username for a in ds_pool}
            self.assertIn("qwen-only@example.com", qwen_names)
            self.assertIn("deepseek-only@example.com", ds_names)
            self.assertNotIn("qwen-only@example.com", ds_names)
            self.assertNotIn("deepseek-only@example.com", qwen_names)


if __name__ == "__main__":
    unittest.main()
