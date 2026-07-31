from __future__ import annotations

"""登录 / 轮询共享特征（stability 方案 + 需求维）。"""

import math
import time


def clamp01(v: float) -> float:
    return 0.0 if v <= 0 else 1.0 if v >= 1 else float(v)


def hour_cycle(now: float | None = None) -> tuple[float, float]:
    lt = time.localtime(now if now is not None else time.time())
    frac = (lt.tm_hour + lt.tm_min / 60.0) / 24.0
    return math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)


def login_context(
    *,
    valid: int,
    target: int,
    eligible: int,
    pool_size: int,
    muted: int,
    blocked: int,
    sec_since_login: float,
    success_ema: float,
    fail_ema: float,
    request_pressure: float,
    consecutive_skips: int,
) -> list[float]:
    """固定 14 维；改维须 bump gate persist schema。"""
    t = max(1, target)
    fill = valid / t
    need = max(0, target - valid) / t
    hs, hc = hour_cycle()
    return [
        clamp01(fill),
        clamp01(need),
        1.0 if valid == 0 else 0.0,
        1.0 if fill < 0.25 else 0.0,
        clamp01(eligible / max(1, pool_size)),
        clamp01(muted / max(1, pool_size)),
        clamp01(blocked / max(1, pool_size)),
        clamp01(sec_since_login / 600.0),
        clamp01(success_ema),
        clamp01(fail_ema),
        clamp01(request_pressure),
        clamp01(consecutive_skips / 8.0),
        hs,
        hc,
        1.0,
    ]


LOGIN_DIM = 15


def poll_context(
    *,
    has_token: bool,
    usage_pct: float,
    threshold: float,
    sec_since_poll: float,
    last_poll_ok: bool,
    keys_left: int,
    keys_total: int,
    request_pressure: float,
    consecutive_skips: int,
) -> list[float]:
    """固定 12 维。"""
    near = 1.0 if usage_pct >= threshold - 10 else 0.0
    critical = 1.0 if usage_pct >= threshold - 5 else 0.0
    hs, hc = hour_cycle()
    return [
        1.0 if has_token else 0.0,
        clamp01(usage_pct / 100.0),
        near,
        critical,
        clamp01(sec_since_poll / 300.0),
        1.0 if last_poll_ok else 0.0,
        clamp01(keys_left / max(1, keys_total)),
        clamp01(request_pressure),
        clamp01(consecutive_skips / 6.0),
        hs,
        hc,
        1.0,
    ]


POLL_DIM = 12


def login_reward(*, acted: bool, logged: int, need: int, fill: float) -> float:
    if acted:
        if need <= 0:
            return 0.05
        return clamp01(logged / max(1, need))
    return clamp01(fill)


def poll_reward(
    *,
    acted: bool,
    ok: bool,
    usage_pct: float,
    threshold: float,
    usage_changed: bool,
) -> float:
    safe = usage_pct < threshold * 0.55
    near = usage_pct >= threshold - 10
    if acted:
        if not ok:
            return 0.1
        if near or usage_changed:
            return 0.9
        return 0.45 if safe else 0.7
    if near:
        return 0.15
    return 0.95 if safe else 0.5


def request_pressure_from_state() -> float:
    try:
        from handlers import get_state

        state = get_state()
        pending = float(getattr(state.scheduler, "pending", 0) or 0)
        active = float(getattr(state.tracker, "count", 0) or 0)
        return clamp01((pending + active) / 32.0)
    except Exception:
        return 0.0
