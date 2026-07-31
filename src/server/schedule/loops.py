from __future__ import annotations

"""共享 interval / gated tick，供登录维护与 Cursor 轮询复用。"""

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from server.schedule.gate import ScheduleGate

logger = logging.getLogger("rogator")

TickFn = Callable[[], Awaitable[None]]
ActFn = Callable[[], Awaitable[float]]
RewardFn = Callable[[], float]
FeaturesFn = Callable[[], list[float]]
HardFn = Callable[[], tuple[bool, bool]]  # force_act, force_skip


async def interval_loop(
    shutdown_event: asyncio.Event,
    interval: float,
    tick: TickFn,
    *,
    min_interval: float = 0.0,
) -> None:
    wait = max(min_interval, float(interval))
    while not shutdown_event.is_set():
        try:
            await tick()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("schedule tick: %s", exc)
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=wait)
        except asyncio.TimeoutError:
            continue


async def gated_tick(
    gate: ScheduleGate,
    features: list[float],
    *,
    enabled: bool,
    force_act: bool,
    force_skip: bool,
    act: ActFn,
    skip_reward: RewardFn,
) -> bool:
    """执行一次门控决策；返回是否 act。"""
    if not enabled:
        await act()
        return True
    do_act = gate.decide(features, force_act=force_act, force_skip=force_skip)
    if do_act:
        reward = await act()
        gate.update(True, features, float(reward))
        return True
    gate.update(False, features, float(skip_reward()))
    return False


def make_gated_tick(
    *,
    gate: ScheduleGate,
    enabled: bool,
    features_fn: FeaturesFn,
    hard_fn: HardFn,
    act: ActFn,
    skip_reward: RewardFn,
    before: Optional[TickFn] = None,
) -> TickFn:
    async def _tick() -> None:
        if before is not None:
            await before()
        force_act, force_skip = hard_fn()
        await gated_tick(
            gate,
            features_fn(),
            enabled=enabled,
            force_act=force_act,
            force_skip=force_skip,
            act=act,
            skip_reward=skip_reward,
        )

    return _tick
