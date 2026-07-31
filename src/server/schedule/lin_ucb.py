from __future__ import annotations

"""Contextual LinUCB：两臂（act / skip）线性模型。"""

import math
from typing import Any, Dict, List, Sequence

ARM_ACT, ARM_SKIP = 1, 0


def _eye(n: int, scale: float = 1.0) -> List[List[float]]:
    return [[scale if i == j else 0.0 for j in range(n)] for i in range(n)]


def _cholesky(a: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                l[i][j] = math.sqrt(max(a[i][i] - s, 1e-12))
            else:
                l[i][j] = (a[i][j] - s) / l[j][j]
    return l


def _solve_lower(l: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    y = [0.0] * len(b)
    for i in range(len(b)):
        y[i] = (b[i] - sum(l[i][j] * y[j] for j in range(i))) / l[i][i]
    return y


def _solve_upper(l: Sequence[Sequence[float]], y: Sequence[float]) -> List[float]:
    n = len(y)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(l[j][i] * x[j] for j in range(i + 1, n))) / l[i][i]
    return x


def _solve_spd(a: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    l = _cholesky(a)
    return _solve_upper(l, _solve_lower(l, b))


class _Arm:
    def __init__(self, dim: int, lam: float) -> None:
        self.a = _eye(dim, lam)
        self.b = [0.0] * dim

    def mu(self) -> List[float]:
        return _solve_spd(self.a, self.b)

    def update(self, x: Sequence[float], r: float) -> None:
        n = len(x)
        for i in range(n):
            for j in range(n):
                self.a[i][j] += x[i] * x[j]
            self.b[i] += r * x[i]

    def to_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, dim: int, lam: float, raw: Dict[str, Any] | None) -> "_Arm":
        arm = cls(dim, lam)
        if not raw:
            return arm
        a, b = raw.get("a"), raw.get("b")
        if isinstance(a, list) and len(a) == dim and isinstance(b, list) and len(b) == dim:
            arm.a = [[float(v) for v in row] for row in a]
            arm.b = [float(v) for v in b]
        return arm


class LinUCB:
    """挑 arm 最大化 x·μ + α √(x·A^{-1}x)。"""

    def __init__(self, dim: int, *, alpha: float = 0.55, lam: float = 1.0) -> None:
        self.dim = dim
        self.alpha = alpha
        self.lam = lam
        self.arms = {ARM_ACT: _Arm(dim, lam), ARM_SKIP: _Arm(dim, lam)}

    def select(self, x: Sequence[float]) -> int:
        best, best_v = ARM_ACT, float("-inf")
        for arm, model in self.arms.items():
            mu = model.mu()
            z = _solve_spd(model.a, x)
            bonus = math.sqrt(max(sum(x[i] * z[i] for i in range(self.dim)), 0.0))
            val = sum(mu[i] * x[i] for i in range(self.dim)) + self.alpha * bonus
            if val > best_v:
                best_v, best = val, arm
        return best

    def update(self, arm: int, x: Sequence[float], reward: float) -> None:
        r = 0.0 if reward < 0 else 1.0 if reward > 1 else reward
        self.arms[arm].update(x, r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "alpha": self.alpha,
            "lam": self.lam,
            "act": self.arms[ARM_ACT].to_dict(),
            "skip": self.arms[ARM_SKIP].to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], *, dim: int, alpha: float, lam: float) -> "LinUCB":
        model = cls(dim, alpha=alpha, lam=lam)
        if int(raw.get("dim", dim)) != dim:
            return model
        model.arms[ARM_ACT] = _Arm.from_dict(dim, lam, raw.get("act"))
        model.arms[ARM_SKIP] = _Arm.from_dict(dim, lam, raw.get("skip"))
        return model
