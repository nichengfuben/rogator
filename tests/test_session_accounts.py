from __future__ import annotations

import unittest

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
        qwen_pool = accounts_for_upstream("qwen")
        ds_pool = accounts_for_upstream("deepseek")
        self.assertGreater(len(qwen_pool), len(ds_pool))
        ds_names = {a.username for a in ds_pool}
        self.assertIn("jntdn@outlook.com", ds_names)
        self.assertIn("entdn@outlook.com", ds_names)
        sample_qwen_only = next(
            a for a in qwen_pool if a.username.endswith("@mailto.plus")
        )
        self.assertNotIn(sample_qwen_only.username, ds_names)


if __name__ == "__main__":
    unittest.main()
