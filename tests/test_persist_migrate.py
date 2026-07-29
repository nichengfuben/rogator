from __future__ import annotations

"""persist 分桶迁移测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from core.persist.migrate import (
    migrate_all,
    split_login_history_logins,
)
from core.persist.paths import (
    login_history_path,
    models_path,
    sessions_path,
    unified_login_history_path,
    unified_models_path,
    unified_sessions_path,
)
from core.session.store import load_upstream_sessions, save_upstream_sessions, PlatformSession
from core.session.accounts import Account
from server.records.login_history import LoginHistoryStore


class TestSplitLoginHistory(unittest.TestCase):
    def test_flat_logins_split_by_account_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qwen_csv = root / "persist" / "qwen" / "accounts.csv"
            ds_csv = root / "persist" / "deepseek" / "accounts.csv"
            qwen_csv.parent.mkdir(parents=True)
            ds_csv.parent.mkdir(parents=True)
            qwen_csv.write_text("email,password\na@test.com,pw\n", encoding="utf-8")
            ds_csv.write_text("email,password\nb@test.com,pw\n", encoding="utf-8")

            logins = {
                "a@test.com": {"at_unix": 1.0, "at_utc8": "x"},
                "b@test.com": {"at_unix": 2.0, "at_utc8": "y"},
                "unknown@test.com": {"at_unix": 3.0, "at_utc8": "z"},
            }
            buckets = split_login_history_logins(
                logins,
                qwen_usernames={"a@test.com"},
                deepseek_usernames={"b@test.com"},
            )
            self.assertIn("a@test.com", buckets["qwen"])
            self.assertIn("b@test.com", buckets["deepseek"])
            self.assertIn("unknown@test.com", buckets["qwen"])


class TestPersistMigrate(unittest.TestCase):
    def test_migrate_all_splits_unified_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persist = root / "persist"
            (persist / "qwen").mkdir(parents=True)
            (persist / "deepseek").mkdir(parents=True)
            (persist / "qwen" / "accounts.csv").write_text(
                "email,password\nqwen@test.com,pw\n", encoding="utf-8"
            )
            (persist / "deepseek" / "accounts.csv").write_text(
                "email,password\nds@test.com,pw\n", encoding="utf-8"
            )

            unified_login_history_path(root).write_text(
                json.dumps(
                    {
                        "updated_at": "2026-01-01 00:00:00",
                        "logins": {
                            "qwen@test.com": {"at_unix": 1.0, "at_utc8": "x"},
                            "ds@test.com": {"at_unix": 2.0, "at_utc8": "y"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            unified_sessions_path(root).write_text(
                json.dumps(
                    {
                        "upstreams": {
                            "qwen": {
                                "sessions": [
                                    {
                                        "username": "qwen@test.com",
                                        "password": "pw",
                                        "token": "t",
                                        "user_id": "u",
                                        "upstream": "qwen",
                                        "login_time": 1.0,
                                        "is_valid": True,
                                    }
                                ],
                                "current_index": 0,
                                "blocked_accounts": {},
                            },
                            "deepseek": {
                                "sessions": [
                                    {
                                        "username": "ds@test.com",
                                        "password": "pw",
                                        "token": "t2",
                                        "user_id": "u2",
                                        "upstream": "deepseek",
                                        "login_time": 2.0,
                                        "is_valid": True,
                                    }
                                ],
                                "current_index": 0,
                                "blocked_accounts": {},
                            },
                        },
                        "updated_at": 99,
                    }
                ),
                encoding="utf-8",
            )
            unified_models_path(root).write_text(
                json.dumps({"models": ["qwen-model"], "meta": {}, "updated_at": 1}),
                encoding="utf-8",
            )

            results = migrate_all(root, archive_unified=True)
            self.assertIn("qwen", results["login_history"])
            self.assertIn("deepseek", results["login_history"])
            self.assertIn("qwen", results["sessions"])
            self.assertIn("deepseek", results["sessions"])
            self.assertIn("qwen", results["models"])
            self.assertIn("deepseek", results["models"])
            self.assertIn("login_history.json", results["archived"])
            self.assertIn("sessions.json", results["archived"])
            self.assertIn("models.json", results["archived"])

            qwen_login = json.loads(login_history_path("qwen", root).read_text(encoding="utf-8"))
            ds_login = json.loads(login_history_path("deepseek", root).read_text(encoding="utf-8"))
            self.assertIn("qwen@test.com", qwen_login["logins"])
            self.assertIn("ds@test.com", ds_login["logins"])

            qwen_models = json.loads(models_path("qwen", root).read_text(encoding="utf-8"))
            self.assertEqual(qwen_models["models"], ["qwen-model"])
            self.assertTrue(models_path("deepseek", root).is_file())

    def test_sessions_migrates_when_dest_exists_but_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "persist" / "qwen").mkdir(parents=True)
            sessions_path("qwen", root).write_text(
                json.dumps(
                    {
                        "sessions": [],
                        "current_index": 0,
                        "blocked_accounts": {},
                        "updated_at": 1,
                    }
                ),
                encoding="utf-8",
            )
            unified_sessions_path(root).write_text(
                json.dumps(
                    {
                        "upstreams": {
                            "qwen": {
                                "sessions": [
                                    {
                                        "username": "q@test.com",
                                        "password": "pw",
                                        "token": "t",
                                        "user_id": "u",
                                        "upstream": "qwen",
                                        "login_time": 1.0,
                                        "is_valid": True,
                                    }
                                ],
                                "current_index": 0,
                                "blocked_accounts": {},
                            }
                        },
                        "updated_at": 2,
                    }
                ),
                encoding="utf-8",
            )
            from core.persist.migrate import migrate_sessions_upstream

            self.assertTrue(migrate_sessions_upstream("qwen", root))
            restored = json.loads(sessions_path("qwen", root).read_text(encoding="utf-8"))
            self.assertEqual(len(restored["sessions"]), 1)


class TestPerUpstreamSessionsIO(unittest.TestCase):
    def test_save_load_isolated_by_upstream(self) -> None:
        from tests.test_session_cleanup import _make_jwt

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qwen_file = root / "persist" / "qwen" / "sessions.json"
            ds_file = root / "persist" / "deepseek" / "sessions.json"
            qwen_file.parent.mkdir(parents=True)
            ds_file.parent.mkdir(parents=True)

            import time
            from unittest.mock import patch

            import core.session.store as store

            def _sessions_file(upstream: str) -> Path:
                return qwen_file if upstream == "qwen" else ds_file

            old_migrated = set(store._migrated_upstreams)
            try:
                store._migrated_upstreams.clear()
                with patch.object(store, "sessions_file", side_effect=_sessions_file):
                    qwen_sess = PlatformSession(
                        account=Account(username="q@test.com", password="pw"),
                        token=_make_jwt(time.time() + 3600),
                        user_id="u",
                        upstream="qwen",
                    )
                    save_upstream_sessions("qwen", [qwen_sess])
                    qwen_loaded, _ = load_upstream_sessions("qwen")
                    ds_loaded, _ = load_upstream_sessions("deepseek")
                self.assertEqual(len(qwen_loaded), 1)
                self.assertEqual(qwen_loaded[0].username, "q@test.com")
                self.assertEqual(len(ds_loaded), 0)
            finally:
                store._migrated_upstreams.clear()
                store._migrated_upstreams.update(old_migrated)


class TestLoginHistoryMigrationIntegration(unittest.TestCase):
    def test_load_triggers_migration_from_unified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import server.records.login_history as lh_mod

            qwen_path = root / "persist" / "qwen" / "login_history.json"
            qwen_path.parent.mkdir(parents=True)
            (root / "persist" / "deepseek").mkdir(parents=True)
            (root / "persist" / "qwen" / "accounts.csv").write_text(
                "email,password\na@test.com,pw\n", encoding="utf-8"
            )
            unified_login_history_path(root).write_text(
                json.dumps(
                    {
                        "updated_at": "2026-01-01",
                        "logins": {"a@test.com": {"at_unix": 100.0, "at_utc8": "x"}},
                    }
                ),
                encoding="utf-8",
            )

            old_root = lh_mod.PROJECT_ROOT
            old_migrated = set(lh_mod._migrated_upstreams)
            try:
                lh_mod.PROJECT_ROOT = root
                lh_mod._migrated_upstreams.clear()
                store = LoginHistoryStore("qwen")
                self.assertEqual(store.last_login_unix("a@test.com"), 100.0)
                self.assertTrue(qwen_path.is_file())
            finally:
                lh_mod.PROJECT_ROOT = old_root
                lh_mod._migrated_upstreams.clear()
                lh_mod._migrated_upstreams.update(old_migrated)


if __name__ == "__main__":
    unittest.main()
