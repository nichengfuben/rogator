from __future__ import annotations

"""并发调度与 session 锁范围测试。"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from accounts import Account
from server.client.qwen_client import QwenClient
from server.client.session_store import QwenSession, SessionStoreMeta
from state import AppState, QueueFullError, RequestScheduler, tracked_request
from tests.test_session_cleanup import _make_jwt


def _session(name: str, valid: bool = True) -> QwenSession:
    return QwenSession(
        account=Account(username=f"{name}@test.com", password="pw"),
        token=_make_jwt(time.time() + 3600),
        user_id=name,
        login_time=time.time(),
        is_valid=valid,
    )


class TestRequestSchedulerSlots(unittest.IsolatedAsyncioTestCase):
    async def test_acquire_release_pending(self) -> None:
        sched = RequestScheduler(max_concurrent=2, max_queue=10)
        await sched.acquire_slot()
        self.assertEqual(sched.pending, 1)
        await sched.acquire_slot()
        self.assertEqual(sched.pending, 2)
        await sched.release_slot()
        self.assertEqual(sched.pending, 1)
        await sched.release_slot()
        self.assertEqual(sched.pending, 0)

    async def test_acquire_respects_max_queue(self) -> None:
        import state as state_mod

        old = state_mod.MAX_QUEUE_SIZE
        state_mod.MAX_QUEUE_SIZE = 1
        try:
            sched = RequestScheduler(max_concurrent=-1, max_queue=1)
            await sched.acquire_slot()
            with self.assertRaises(QueueFullError):
                await sched.acquire_slot()
            await sched.release_slot()
        finally:
            state_mod.MAX_QUEUE_SIZE = old


class TestTrackedRequest(unittest.IsolatedAsyncioTestCase):
    async def test_tracked_request_registers_and_releases(self) -> None:
        state = AppState()
        req_id = "req-track-1"
        async with tracked_request(state, req_id):
            self.assertEqual(state.scheduler.pending, 1)
            self.assertEqual(state.tracker.count, 1)
        self.assertEqual(state.scheduler.pending, 0)
        self.assertEqual(state.tracker.count, 0)


class TestSessionConcurrency(unittest.IsolatedAsyncioTestCase):
    async def test_switch_excludes_failed_username(self) -> None:
        client = QwenClient(MagicMock())
        client._sessions = [
            _session("a"),
            _session("b"),
        ]
        client._current_index = 0
        client._save_meta = MagicMock(return_value=[])

        with patch("server.client.account.random.choice", return_value=client._sessions[1]):
            new = await client.switch_to_next(exclude_username="a@test.com")
        self.assertIsNotNone(new)
        self.assertEqual(new.username, "b@test.com")
        self.assertEqual(client._current_index, 1)

    async def test_concurrent_login_not_blocked_by_lock(self) -> None:
        client = QwenClient(MagicMock())
        client._sessions = []
        client._save_meta = MagicMock(return_value=[])
        client._ensure_cleanup = AsyncMock()
        accounts = [
            Account(username="a@test.com", password="pw"),
            Account(username="b@test.com", password="pw"),
        ]
        overlap = asyncio.Event()
        active = 0
        peak = 0

        async def _slow_login(account: Account) -> QwenSession:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await overlap.wait()
            active -= 1
            return _session(account.username.split("@")[0])

        client.login_account = AsyncMock(side_effect=_slow_login)

        async def _switch() -> None:
            with patch("server.client.account.ACCOUNTS", accounts):
                await client.switch_to_next()

        with patch("server.client.account.ACCOUNTS", accounts):
            task_a = asyncio.create_task(_switch())
            task_b = asyncio.create_task(_switch())
            await asyncio.sleep(0.05)
            overlap.set()
            await asyncio.gather(task_a, task_b)

        self.assertGreaterEqual(peak, 2)

    async def test_get_valid_session_skips_blocking_prelogin(self) -> None:
        empty_meta = SessionStoreMeta()
        with patch("server.client.qwen_client.load_session_store", return_value=([], empty_meta)):
            client = QwenClient(MagicMock())
        client._sessions = [_session("ok")]
        client._prelogin_target = 5
        client.ensure_prelogin = AsyncMock()
        client._ensure_cleanup = AsyncMock()

        session = await client.get_valid_session()
        self.assertIsNotNone(session)
        client.ensure_prelogin.assert_not_called()


if __name__ == "__main__":
    unittest.main()
