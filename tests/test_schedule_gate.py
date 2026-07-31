from __future__ import annotations

"""LinUCB ScheduleGate 单测：硬约束、更新、落盘、gated_tick。"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server.schedule.features import LOGIN_DIM, login_reward, poll_reward
from server.schedule.gate import ScheduleGate, reset_gates_for_tests
from server.schedule.lin_ucb import ARM_ACT, ARM_SKIP, LinUCB
from server.schedule.loops import gated_tick


class TestLinUCB(unittest.TestCase):
    def test_update_and_prefer_high_reward_arm(self) -> None:
        m = LinUCB(3, alpha=0.1, lam=1.0)
        x = [1.0, 0.0, 1.0]
        for _ in range(20):
            m.update(ARM_SKIP, x, 0.95)
            m.update(ARM_ACT, x, 0.05)
        self.assertEqual(m.select(x), ARM_SKIP)


class TestScheduleGate(unittest.TestCase):
    def setUp(self) -> None:
        reset_gates_for_tests()

    def test_force_skip_and_force_act(self) -> None:
        g = ScheduleGate("t_force", LOGIN_DIM, persist=False, max_consecutive_skips=99)
        x = [0.5] * LOGIN_DIM
        self.assertFalse(g.decide(x, force_skip=True))
        self.assertTrue(g.decide(x, force_act=True, force_skip=True))

    def test_max_consecutive_skips_forces_act(self) -> None:
        g = ScheduleGate("t_skip", LOGIN_DIM, persist=False, max_consecutive_skips=2)
        x = [0.5] * LOGIN_DIM
        g.consecutive_skips = 2
        self.assertTrue(g.decide(x))

    def test_persist_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("server.schedule.gate.persist_root", return_value=root):
                g = ScheduleGate("t_persist", LOGIN_DIM, persist=True, max_consecutive_skips=5)
                x = [0.1] * LOGIN_DIM
                g.update(False, x, 0.8)
                g.extra["usage_pct"] = 42.0
                g._save()
                reset_gates_for_tests()
                g2 = ScheduleGate("t_persist", LOGIN_DIM, persist=True, max_consecutive_skips=5)
                self.assertEqual(g2.consecutive_skips, 1)
                self.assertAlmostEqual(g2.extra.get("usage_pct", 0.0), 42.0)


class TestRewards(unittest.TestCase):
    def test_login_idle_full_skip_high(self) -> None:
        self.assertGreater(login_reward(acted=False, logged=0, need=0, fill=1.0), 0.8)

    def test_poll_safe_skip_high(self) -> None:
        r = poll_reward(acted=False, ok=True, usage_pct=20.0, threshold=90.0, usage_changed=False)
        self.assertGreater(r, 0.8)


class TestGatedTick(unittest.IsolatedAsyncioTestCase):
    async def test_skip_does_not_call_act(self) -> None:
        g = ScheduleGate("t_tick", LOGIN_DIM, persist=False, max_consecutive_skips=99)
        calls = {"n": 0}

        async def act() -> float:
            calls["n"] += 1
            return 1.0

        x = [0.5] * LOGIN_DIM
        acted = await gated_tick(
            g, x, enabled=True, force_act=False, force_skip=True, act=act, skip_reward=lambda: 0.9
        )
        self.assertFalse(acted)
        self.assertEqual(calls["n"], 0)

    async def test_disabled_always_acts(self) -> None:
        g = ScheduleGate("t_off", LOGIN_DIM, persist=False, max_consecutive_skips=99)
        calls = {"n": 0}

        async def act() -> float:
            calls["n"] += 1
            return 0.5

        await gated_tick(
            g,
            [0.5] * LOGIN_DIM,
            enabled=False,
            force_act=False,
            force_skip=True,
            act=act,
            skip_reward=lambda: 0.9,
        )
        self.assertEqual(calls["n"], 1)


class TestIntervalLoop(unittest.IsolatedAsyncioTestCase):
    async def test_stops_on_shutdown(self) -> None:
        from server.schedule.loops import interval_loop

        ev = asyncio.Event()
        n = {"c": 0}

        async def tick() -> None:
            n["c"] += 1
            if n["c"] >= 2:
                ev.set()

        task = asyncio.create_task(interval_loop(ev, 0.01, tick))
        await asyncio.wait_for(ev.wait(), timeout=2.0)
        await asyncio.wait_for(task, timeout=2.0)
        self.assertGreaterEqual(n["c"], 2)


if __name__ == "__main__":
    unittest.main()
