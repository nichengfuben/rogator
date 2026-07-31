from __future__ import annotations

"""从 client 组装 login/poll 门控上下文（保持 pool/cursor 函数短）。"""

from typing import Any

from server.config import CONFIG
from server.schedule.features import (
    LOGIN_DIM,
    POLL_DIM,
    login_context,
    login_reward,
    poll_context,
    poll_reward,
    request_pressure_from_state,
)
from server.schedule.gate import ScheduleGate, get_gate


def schedule_enabled() -> bool:
    return bool(getattr(CONFIG, "schedule_enabled", True))


def login_gate(upstream: str) -> ScheduleGate:
    return get_gate(
        f"login:{upstream}",
        LOGIN_DIM,
        max_consecutive_skips=int(CONFIG.schedule_max_consecutive_skips_login),
        alpha=float(CONFIG.schedule_alpha),
        lam=float(CONFIG.schedule_lambda),
        persist=bool(CONFIG.schedule_persist),
    )


def poll_gate() -> ScheduleGate:
    return get_gate(
        "poll:cursor",
        POLL_DIM,
        max_consecutive_skips=int(CONFIG.schedule_max_consecutive_skips_poll),
        alpha=float(CONFIG.schedule_alpha),
        lam=float(CONFIG.schedule_lambda),
        persist=bool(CONFIG.schedule_persist),
    )


def build_login_features(client: Any, gate: ScheduleGate, need: int, target: int) -> list[float]:
    from core.session.store import valid_session_count

    pool = client._pool_accounts()
    pool_n = len(pool) or 1
    muted = sum(1 for a in pool if client._is_account_muted(a.username))
    blocked = sum(1 for a in pool if client._is_account_blocked(a.username))
    eligible = sum(
        1
        for a in pool
        if a.username not in client._active_usernames()
        and not client._is_account_blocked(a.username)
        and not client._is_account_muted(a.username)
    )
    valid = valid_session_count(client._sessions)
    return login_context(
        valid=valid,
        target=target,
        eligible=eligible,
        pool_size=pool_n,
        muted=muted,
        blocked=blocked,
        sec_since_login=gate.sec_since_act(),
        success_ema=gate.success_ema,
        fail_ema=gate.fail_ema,
        request_pressure=request_pressure_from_state(),
        consecutive_skips=gate.consecutive_skips,
    )


def login_hard(valid: int, need: int) -> tuple[bool, bool]:
    """force_act, force_skip。"""
    if valid == 0 and need > 0:
        return True, False
    if need <= 0:
        return False, True
    return False, False


def reward_login(acted: bool, logged: int, need: int, fill: float) -> float:
    return login_reward(acted=acted, logged=logged, need=need, fill=fill)


def build_poll_features(client: Any, gate: ScheduleGate) -> list[float]:
    tokens = client._tokens
    cfg = tokens._cfg
    threshold = float(cfg.get("usage_threshold", 90.0))
    usage = float(gate.extra.get("usage_pct", 0.0))
    keys = list(tokens._pool.all())
    return poll_context(
        has_token=bool(tokens.current_token()),
        usage_pct=usage,
        threshold=threshold,
        sec_since_poll=gate.sec_since_act(),
        last_poll_ok=bool(gate.extra.get("last_poll_ok", 1.0)),
        keys_left=sum(1 for k in keys if getattr(k, "is_active", True)),
        keys_total=max(1, len(keys)),
        request_pressure=request_pressure_from_state(),
        consecutive_skips=gate.consecutive_skips,
    )


def poll_hard(client: Any, gate: ScheduleGate) -> tuple[bool, bool]:
    tokens = client._tokens
    threshold = float(tokens._cfg.get("usage_threshold", 90.0))
    usage = float(gate.extra.get("usage_pct", 0.0))
    if not tokens.current_token():
        return True, False
    if usage >= threshold - 5:
        return True, False
    return False, False


def reward_poll(ok: bool, gate: ScheduleGate, prev_usage: float, new_usage: float, threshold: float) -> float:
    changed = abs(new_usage - prev_usage) >= 1.0
    return poll_reward(
        acted=True,
        ok=ok,
        usage_pct=new_usage,
        threshold=threshold,
        usage_changed=changed,
    )


def skip_reward_poll(gate: ScheduleGate, threshold: float) -> float:
    usage = float(gate.extra.get("usage_pct", 0.0))
    return poll_reward(
        acted=False,
        ok=True,
        usage_pct=usage,
        threshold=threshold,
        usage_changed=False,
    )
