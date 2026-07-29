from __future__ import annotations

"""登录历史持久化与选号策略测试。"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from accounts import Account
from server.client.login_history import LoginHistoryStore, format_utc8


def _accounts(n: int, *, prefix: str = "u") -> list[Account]:
    return [
        Account(username=f"{prefix}{i}@test.com", password="pw")
        for i in range(n)
    ]


class TestLoginHistoryStore(unittest.TestCase):
    def test_record_flush_persists_utc8_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "login_history.json"
            with patch("server.client.login_history.LOGIN_HISTORY_FILE", str(path)):
                store = LoginHistoryStore()
                ts = 1722239280.0
                store.record("a@test.com", at=ts)
                store.flush()
                self.assertEqual(store.last_login_unix("a@test.com"), ts)
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    data["logins"]["a@test.com"]["at_utc8"],
                    format_utc8(ts),
                )

    def test_flush_skips_when_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "login_history.json"
            with patch("server.client.login_history.LOGIN_HISTORY_FILE", str(path)):
                store = LoginHistoryStore()
                store.flush()
                self.assertFalse(path.exists())


class TestPickAccount(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = patch("server.client.login_history.LOGIN_HISTORY_FILE", "/nonexistent")
        self._patch.start()
        self.store = LoginHistoryStore()

    def tearDown(self) -> None:
        self._patch.stop()

    def _pick(self, accounts: list[Account], *, eligible=None):
        if eligible is None:
            eligible = lambda a: True  # noqa: E731
        return self.store.pick_account(accounts, eligible=eligible)

    def test_prefers_never_logged(self) -> None:
        accounts = _accounts(3)
        self.store.record(accounts[0].username, at=100.0)
        self.store.record(accounts[1].username, at=200.0)
        seen = {self._pick(accounts).username for _ in range(20)}
        self.assertEqual(seen, {accounts[2].username})

    def test_picks_from_stale_pool_when_many_accounts(self) -> None:
        accounts = _accounts(25)
        for i, acc in enumerate(accounts):
            self.store.record(acc.username, at=float(i + 1))
        stale_usernames = {a.username for a in accounts[:20]}
        seen = {self._pick(accounts).username for _ in range(50)}
        self.assertTrue(seen.issubset(stale_usernames))
        self.assertTrue(len(seen) > 1)

    def test_picks_from_all_when_few_accounts(self) -> None:
        accounts = _accounts(5)
        for i, acc in enumerate(accounts):
            self.store.record(acc.username, at=float(i + 1))
        seen = {self._pick(accounts).username for _ in range(40)}
        self.assertEqual(seen, {a.username for a in accounts})

    def test_respects_eligible_predicate(self) -> None:
        accounts = _accounts(2)
        blocked = {accounts[1].username}
        picked = self._pick(
            accounts,
            eligible=lambda a: a.username not in blocked,
        )
        self.assertIsNotNone(picked)
        self.assertEqual(picked.username, accounts[0].username)


if __name__ == "__main__":
    unittest.main()
