from __future__ import annotations

"""ScheduleGate：硬约束 + LinUCB + 可选落盘。"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from core.persist.paths import persist_root
from server.schedule.lin_ucb import ARM_ACT, ARM_SKIP, LinUCB

logger = logging.getLogger("rogator")

_GATES: Dict[str, "ScheduleGate"] = {}


def bandit_path(gate_id: str) -> Path:
    # Windows：路径中的 ``:`` 会被当成 NTFS ADS（写出空文件 login / poll）
    safe = (
        str(gate_id)
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    return persist_root() / "bandit" / f"{safe}.json"


class ScheduleGate:
    def __init__(
        self,
        gate_id: str,
        dim: int,
        *,
        alpha: float = 0.55,
        lam: float = 1.0,
        max_consecutive_skips: int = 5,
        persist: bool = True,
    ) -> None:
        self.gate_id = gate_id
        self.dim = dim
        self.max_consecutive_skips = max_consecutive_skips
        self.persist = persist
        self.consecutive_skips = 0
        self.last_act_at = 0.0
        self.success_ema = 0.5
        self.fail_ema = 0.05
        self.extra: Dict[str, float] = {}
        self.model = LinUCB(dim, alpha=alpha, lam=lam)
        if persist:
            self._load()

    def decide(self, x: list[float], *, force_act: bool = False, force_skip: bool = False) -> bool:
        if force_skip:
            return False
        if force_act or self.consecutive_skips >= self.max_consecutive_skips:
            return True
        return self.model.select(x) == ARM_ACT

    def update(self, acted: bool, x: list[float], reward: float) -> None:
        arm = ARM_ACT if acted else ARM_SKIP
        self.model.update(arm, x, reward)
        if acted:
            self.consecutive_skips = 0
            self.last_act_at = time.time()
        else:
            self.consecutive_skips += 1
        if self.persist:
            self._save()

    def note_outcome(self, success: bool) -> None:
        if success:
            self.success_ema = 0.9 * self.success_ema + 0.1
            self.fail_ema = 0.9 * self.fail_ema
        else:
            self.fail_ema = 0.9 * self.fail_ema + 0.1
            self.success_ema = 0.9 * self.success_ema

    def sec_since_act(self) -> float:
        if self.last_act_at <= 0:
            return 1e9
        return max(0.0, time.time() - self.last_act_at)

    def _load(self) -> None:
        path = bandit_path(self.gate_id)
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("bandit load %s: %s", self.gate_id, exc)
            return
        self._apply_state(raw)

    def _apply_state(self, raw: Dict[str, Any]) -> None:
        if int(raw.get("dim", self.dim)) != self.dim:
            return
        self.consecutive_skips = int(raw.get("consecutive_skips", 0))
        self.last_act_at = float(raw.get("last_act_at", 0.0))
        self.success_ema = float(raw.get("success_ema", self.success_ema))
        self.fail_ema = float(raw.get("fail_ema", self.fail_ema))
        extra = raw.get("extra")
        if isinstance(extra, dict):
            self.extra = {str(k): float(v) for k, v in extra.items()}
        model_raw = raw.get("model")
        if isinstance(model_raw, dict):
            self.model = LinUCB.from_dict(
                model_raw, dim=self.dim, alpha=self.model.alpha, lam=self.model.lam
            )

    def _save(self) -> None:
        path = bandit_path(self.gate_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dim": self.dim,
                "consecutive_skips": self.consecutive_skips,
                "last_act_at": self.last_act_at,
                "success_ema": self.success_ema,
                "fail_ema": self.fail_ema,
                "extra": self.extra,
                "model": self.model.to_dict(),
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            logger.warning("bandit save %s: %s", self.gate_id, exc)


def get_gate(
    gate_id: str,
    dim: int,
    *,
    max_consecutive_skips: int,
    alpha: float = 0.55,
    lam: float = 1.0,
    persist: bool = True,
) -> ScheduleGate:
    g = _GATES.get(gate_id)
    if g is not None and g.dim == dim:
        return g
    g = ScheduleGate(
        gate_id,
        dim,
        alpha=alpha,
        lam=lam,
        max_consecutive_skips=max_consecutive_skips,
        persist=persist,
    )
    _GATES[gate_id] = g
    return g


def reset_gates_for_tests() -> None:
    _GATES.clear()
