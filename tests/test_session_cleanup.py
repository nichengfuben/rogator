from __future__ import annotations

"""session 过期与清理测试（基于 JWT exp - 30s）。"""

import base64
import json
import time
import unittest
from unittest.mock import MagicMock

from accounts import Account
from server.session_store import QwenSession, clean_expired
from server.qwen_client import QwenClient


def _make_jwt(exp: float) -> str:
    """构造仅含 exp 的伪 JWT（不校验签名）。"""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": int(exp)}, separators=(",", ":")).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.sig"


def _session(exp_offset: float, valid: bool = True, name: str | None = None) -> QwenSession:
    """exp_offset: 相对 now 的秒数；正=未来，负=已过期。"""
    now = time.time()
    exp = now + exp_offset
    label = name or f"user{int(exp_offset)}@test.com"
    return QwenSession(
        account=Account(username=label, password="pw"),
        token=_make_jwt(exp),
        user_id="uid",
        login_time=now,
        is_valid=valid,
    )


class TestSessionExpiry(unittest.TestCase):
    def test_is_expired_by_jwt_exp(self) -> None:
        # 1 小时后过期 → 未到清理窗口
        fresh = _session(3600)
        # 已过期
        expired = _session(-10)
        self.assertFalse(fresh.is_expired())
        self.assertTrue(expired.is_expired())

    def test_is_expired_early_by_30_seconds(self) -> None:
        # exp = now + 20 → 落在 30s 提前清理窗口内
        near = _session(20)
        self.assertTrue(near.is_expired())
        # exp = now + 60 → 仍安全
        safe = _session(60)
        self.assertFalse(safe.is_expired())

    def test_invalid_token_treated_as_expired(self) -> None:
        s = QwenSession(
            account=Account(username="bad@test.com", password="pw"),
            token="not-a-jwt",
            user_id="uid",
            login_time=time.time(),
            is_valid=True,
        )
        self.assertTrue(s.is_expired())

    def test_clean_expired_removes_old_and_invalid(self) -> None:
        sessions = [
            _session(3600, name="a@test.com"),
            _session(-10, name="b@test.com"),
            _session(3600, valid=False, name="c@test.com"),
        ]
        kept, removed = clean_expired(sessions)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].username, "a@test.com")
        self.assertEqual(len(removed), 2)

    def test_prune_expired_on_get_valid_session(self) -> None:
        client = QwenClient(MagicMock())
        client._sessions = [
            _session(-5, name="old@test.com"),
            _session(3600, name="ok@test.com"),
        ]
        client._current_index = 0

        import asyncio
        session = asyncio.run(client.get_valid_session())

        self.assertEqual(len(client._sessions), 1)
        self.assertIsNotNone(session)
        self.assertFalse(session.is_expired())
        self.assertEqual(session.username, "ok@test.com")

    def test_current_session_hides_expired(self) -> None:
        client = QwenClient(MagicMock())
        client._sessions = [_session(-1, name="gone@test.com")]
        client._current_index = 0
        self.assertIsNone(client.current_session)

    def test_example_exp_cleanup_target(self) -> None:
        # 设计示例：exp=1785033999 → 清理阈值 1785033969
        exp = 1785033999
        s = QwenSession(
            account=Account(username="ex@test.com", password="pw"),
            token=_make_jwt(exp),
            user_id="uid",
            login_time=exp - 3600,
            is_valid=True,
        )
        # 把“现在”钉在阈值两侧验证
        real_time = time.time
        try:
            time.time = lambda: exp - 31  # type: ignore[assignment]
            self.assertFalse(s.is_expired())
            time.time = lambda: exp - 30  # type: ignore[assignment]
            self.assertTrue(s.is_expired())
        finally:
            time.time = real_time  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
