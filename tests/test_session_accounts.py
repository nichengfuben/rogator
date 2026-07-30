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
                "a@mailto.plus,pw1\n"
                "b@mailto.plus,pw2\n"
                "shared@example.com,pw3\n",
                encoding="utf-8",
            )
            ds_csv.write_text(
                "email,password\n"
                "jntdn@outlook.com,dspw1\n"
                "entdn@outlook.com,dspw2\n",
                encoding="utf-8",
            )
            mapping = {"qwen": qwen_csv, "deepseek": ds_csv}
            with patch("core.session.accounts._UPSTREAM_CSV", mapping):
                with patch("core.session.accounts._ROOT_CSV", root / "missing.csv"):
                    qwen_pool = accounts_for_upstream("qwen")
                    ds_pool = accounts_for_upstream("deepseek")
            self.assertGreater(len(qwen_pool), len(ds_pool))
            ds_names = {a.username for a in ds_pool}
            self.assertEqual(ds_names, {"jntdn@outlook.com", "entdn@outlook.com"})
            sample_qwen_only = next(
                a for a in qwen_pool if a.username.endswith("@mailto.plus")
            )
            self.assertNotIn(sample_qwen_only.username, ds_names)


if __name__ == "__main__":
    unittest.main()
