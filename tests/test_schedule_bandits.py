from __future__ import annotations

"""三种上下文 bandit 在「登录/不登录」「轮询/不轮询」上的极限对比仿真。

算法：
- LinUCB：线性上下文 UCB
- LinTS：Linear Thompson Sampling
- TTPS：Top-Two Thompson Sampling（TAS 族）

长期稳定性目标：
- 有流量时：session / 用量信息可用（availability）
- 无流量时：不要把 32 账号池刷满，也不要空转轮询（load / waste）
- 突发与昼夜：能跟上，但不长期高负载空转

本文件只做算法极限与维度消融，不接入生产调度。
"""

import math
import random
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# 小矩阵工具（仅测试仿真）
# ---------------------------------------------------------------------------


def _eye(n: int, scale: float = 1.0) -> List[List[float]]:
    return [[scale if i == j else 0.0 for j in range(n)] for i in range(n)]


def _cholesky(a: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(a)
    l = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                v = max(a[i][i] - s, 1e-12)
                l[i][j] = math.sqrt(v)
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


def _gauss(rng: random.Random) -> float:
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _outer_add(a: List[List[float]], x: Sequence[float]) -> None:
    n = len(x)
    for i in range(n):
        for j in range(n):
            a[i][j] += x[i] * x[j]


def _clamp01(v: float) -> float:
    return 0.0 if v <= 0 else 1.0 if v >= 1 else v


# ---------------------------------------------------------------------------
# 算法
# ---------------------------------------------------------------------------

ARM_ACT, ARM_SKIP = 1, 0


class _LinArm:
    def __init__(self, dim: int, lam: float = 1.0) -> None:
        self.a = _eye(dim, lam)
        self.b = [0.0] * dim

    def mu(self) -> List[float]:
        return _solve_spd(self.a, self.b)

    def update(self, x: Sequence[float], r: float) -> None:
        _outer_add(self.a, x)
        for i in range(len(x)):
            self.b[i] += r * x[i]


class LinUCB:
    name = "LinUCB"

    def __init__(self, dim: int, *, alpha: float = 0.6, lam: float = 1.0, seed: int = 0) -> None:
        self.dim = dim
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.arms = {ARM_ACT: _LinArm(dim, lam), ARM_SKIP: _LinArm(dim, lam)}

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
        self.arms[arm].update(x, max(0.0, min(1.0, reward)))


class LinTS:
    name = "LinTS"

    def __init__(self, dim: int, *, v: float = 0.35, lam: float = 1.0, seed: int = 0) -> None:
        self.dim = dim
        self.v = v
        self.rng = random.Random(seed)
        self.arms = {ARM_ACT: _LinArm(dim, lam), ARM_SKIP: _LinArm(dim, lam)}

    def _sample(self, model: _LinArm) -> List[float]:
        mu = model.mu()
        l = _cholesky(model.a)
        z = [_gauss(self.rng) for _ in range(self.dim)]
        u = _solve_upper(l, z)
        s = math.sqrt(max(self.v, 1e-9))
        return [mu[i] + s * u[i] for i in range(self.dim)]

    def select(self, x: Sequence[float]) -> int:
        best, best_v = ARM_ACT, float("-inf")
        for arm, model in self.arms.items():
            theta = self._sample(model)
            val = sum(theta[i] * x[i] for i in range(self.dim))
            if val > best_v:
                best_v, best = val, arm
        return best

    def update(self, arm: int, x: Sequence[float], reward: float) -> None:
        self.arms[arm].update(x, max(0.0, min(1.0, reward)))


class TTPS(LinTS):
    """Top-Two Thompson Sampling（TAS 族）。"""

    name = "TTPS"

    def __init__(self, dim: int, *, beta: float = 0.55, v: float = 0.35, lam: float = 1.0, seed: int = 0) -> None:
        super().__init__(dim, v=v, lam=lam, seed=seed)
        self.beta = beta

    def select(self, x: Sequence[float]) -> int:
        scores1 = {
            arm: sum(self._sample(model)[i] * x[i] for i in range(self.dim))
            for arm, model in self.arms.items()
        }
        i_star = max(scores1, key=scores1.get)
        if self.rng.random() < self.beta:
            return i_star
        for _ in range(8):
            scores2 = {
                arm: sum(self._sample(model)[i] * x[i] for i in range(self.dim))
                for arm, model in self.arms.items()
            }
            j = max(scores2, key=scores2.get)
            if j != i_star:
                return j
        return i_star


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------


@dataclass
class LoginWorld:
    """账号池补登 vs 请求消耗 session。"""

    pool_target: int = 32
    session_ttl: int = 40
    login_cost: float = 1.0
    mute_prob: float = 0.02
    seed: int = 0
    valid: int = 0
    ages: List[int] = field(default_factory=list)
    muted: int = 0
    tick: int = 0
    last_login_tick: int = -10_000
    success_ema: float = 0.5
    fail_ema: float = 0.05
    demand_ema: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.ages = []

    def expire(self) -> None:
        self.ages = [a + 1 for a in self.ages if a + 1 < self.session_ttl]
        self.valid = len(self.ages)

    def traffic(self, pattern: str) -> int:
        t = self.tick
        if pattern == "idle":
            return 1 if self.rng.random() < 0.02 else 0
        if pattern == "busy":
            return self.rng.randint(2, 6)
        if pattern == "burst":
            phase = t % 80
            if phase < 60:
                return 1 if self.rng.random() < 0.05 else 0
            return self.rng.randint(4, 10)
        # diurnal
        hour = (t // 5) % 24
        if 0 <= hour < 7:
            return 1 if self.rng.random() < 0.05 else 0
        return self.rng.randint(1, 4)

    def serve(self, n_req: int) -> Tuple[int, int]:
        ok = fail = 0
        for _ in range(n_req):
            if self.valid > 0:
                self.ages.pop(0)
                self.valid -= 1
                ok += 1
            else:
                fail += 1
        self.demand_ema = 0.9 * self.demand_ema + 0.1 * n_req
        return ok, fail

    def do_login(self, batch: int) -> Tuple[int, int]:
        """补登；返回 (logged, attempted)。池满则不登。"""
        need = max(0, self.pool_target - self.valid)
        if need <= 0:
            return 0, 0
        attempted = min(need, max(1, batch))
        logged = 0
        for _ in range(attempted):
            if self.rng.random() < self.mute_prob:
                self.muted = min(self.pool_target, self.muted + 1)
                self.fail_ema = 0.9 * self.fail_ema + 0.1
                continue
            self.ages.append(0)
            self.valid += 1
            logged += 1
            self.success_ema = 0.9 * self.success_ema + 0.1 * 1.0
            self.fail_ema = 0.9 * self.fail_ema
        self.last_login_tick = self.tick
        return logged, attempted


@dataclass
class PollWorld:
    """Cursor 用量轮询：有请求要新鲜用量；空闲轮询浪费。"""

    threshold: float = 90.0
    usage_drift: float = 1.5
    poll_cost: float = 1.0
    seed: int = 0
    usage: float = 20.0
    last_poll_tick: int = -1000
    last_seen_usage: float = 20.0
    has_token: bool = True
    tick: int = 0
    keys_left: int = 5
    demand_ema: float = 0.0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def traffic(self, pattern: str) -> int:
        t = self.tick
        if pattern == "idle":
            return 1 if self.rng.random() < 0.03 else 0
        if pattern == "busy":
            return self.rng.randint(1, 5)
        if pattern == "burst":
            return self.rng.randint(3, 8) if (t % 70) >= 50 else (1 if self.rng.random() < 0.04 else 0)
        hour = (t // 5) % 24
        return self.rng.randint(1, 3) if hour >= 8 else (1 if self.rng.random() < 0.05 else 0)

    def drift(self, n_req: int) -> None:
        self.usage = min(100.0, self.usage + n_req * self.usage_drift * self.rng.uniform(0.5, 1.2))
        if n_req == 0:
            # 空闲缓慢回落；偶发后台漂移，模拟别的客户端在用
            self.usage = max(0.0, self.usage - 0.25)
            if self.rng.random() < 0.03:
                self.usage = min(100.0, self.usage + self.rng.uniform(2, 8))
        self.demand_ema = 0.9 * self.demand_ema + 0.1 * n_req

    def do_poll(self) -> bool:
        if not self.has_token:
            self.has_token = True
            self.last_poll_tick = self.tick
            self.last_seen_usage = self.usage
            return True
        ok = self.rng.random() > 0.05
        self.last_poll_tick = self.tick
        if ok:
            self.last_seen_usage = self.usage
            if self.usage >= self.threshold:
                self.usage = self.rng.uniform(5, 25)
                self.last_seen_usage = self.usage
                self.keys_left = max(0, self.keys_left - 1)
        return ok

    def request_ok(self) -> bool:
        stale = (self.tick - self.last_poll_tick) > 25
        if not self.has_token:
            return False
        if stale and self.usage >= self.threshold - 8:
            return False
        if self.usage >= self.threshold:
            return False
        return True


# ---------------------------------------------------------------------------
# 特征维度方案（消融）
# ---------------------------------------------------------------------------


def login_features(world: LoginWorld, skips: int, scheme: str, n_req: int) -> List[float]:
    fill = world.valid / max(1, world.pool_target)
    need = max(0, world.pool_target - world.valid) / max(1, world.pool_target)
    zero = 1.0 if world.valid == 0 else 0.0
    low = 1.0 if fill < 0.25 else 0.0
    since = min(1.0, (world.tick - world.last_login_tick) / 60.0)
    hour = ((world.tick // 5) % 24) / 24.0
    demand = _clamp01(world.demand_ema / 6.0)
    req = _clamp01(n_req / 8.0)
    ttl_pressure = 0.0
    if world.ages:
        # 即将过期比例
        ttl_pressure = sum(1 for a in world.ages if a >= world.session_ttl - 5) / len(world.ages)

    if scheme == "minimal":
        # 仅水位：易在 idle 刷满 / busy 反应慢
        return [fill, need, 1.0]
    if scheme == "demand":
        # 水位 + 即时/EMA 需求
        return [fill, need, req, demand, 1.0]
    if scheme == "stability":
        # 长期稳定核心维
        return [fill, need, zero, low, since, req, demand, _clamp01(skips / 8.0), math.sin(2 * math.pi * hour), 1.0]
    if scheme == "full":
        return [
            fill, need, zero, low, since, req, demand, ttl_pressure,
            world.success_ema, world.fail_ema,
            world.muted / max(1, world.pool_target),
            _clamp01(skips / 8.0),
            math.sin(2 * math.pi * hour), math.cos(2 * math.pi * hour),
            1.0,
        ]
    if scheme == "noisy":
        return login_features(world, skips, "full", n_req) + [world.rng.random() for _ in range(6)]
    raise KeyError(scheme)


def poll_features(world: PollWorld, skips: int, scheme: str, n_req: int) -> List[float]:
    usage_n = world.usage / 100.0
    near = 1.0 if world.usage >= world.threshold - 10 else 0.0
    critical = 1.0 if world.usage >= world.threshold - 5 else 0.0
    since = min(1.0, (world.tick - world.last_poll_tick) / 40.0)
    hour = ((world.tick // 5) % 24) / 24.0
    req = _clamp01(n_req / 8.0)
    demand = _clamp01(world.demand_ema / 5.0)
    stale = 1.0 if (world.tick - world.last_poll_tick) > 20 else 0.0

    if scheme == "minimal":
        return [usage_n, near, 1.0]
    if scheme == "demand":
        return [usage_n, near, req, demand, 1.0]
    if scheme == "stability":
        return [
            1.0 if world.has_token else 0.0,
            usage_n, near, critical, since, stale, req, demand,
            _clamp01(skips / 6.0),
            math.sin(2 * math.pi * hour),
            1.0,
        ]
    if scheme == "full":
        return [
            1.0 if world.has_token else 0.0,
            usage_n, near, critical, since, stale, req, demand,
            world.keys_left / 5.0,
            _clamp01(skips / 6.0),
            math.sin(2 * math.pi * hour), math.cos(2 * math.pi * hour),
            1.0,
        ]
    if scheme == "noisy":
        return poll_features(world, skips, "full", n_req) + [world.rng.random() for _ in range(6)]
    raise KeyError(scheme)


# ---------------------------------------------------------------------------
# 奖励与硬约束（面向长期稳定）
# ---------------------------------------------------------------------------


def login_reward(
    *,
    arm: int,
    n_req: int,
    ok: int,
    fail: int,
    logged: int,
    attempted: int,
    fill_before: float,
    fill_after: float,
) -> float:
    """结果导向：服务成功优先，空闲刷池重罚。"""
    served = ok + fail
    avail = ok / served if served else 1.0
    idle = n_req == 0
    if arm == ARM_ACT:
        if attempted == 0:
            # 池满还选登录：浪费
            return 0.05
        useful = logged / max(1, attempted)
        if idle:
            # 空闲只允许「缺口很大」时少量补登
            if fill_before < 0.15:
                return 0.55 * useful + 0.2
            return 0.15 * useful  # 空闲刷池：低奖
        # 有流量：填缺口 + 可用性
        return _clamp01(0.55 * avail + 0.35 * useful + 0.10 * (1.0 - fill_before if fill_before < 0.8 else 0.1))
    # SKIP
    if fail > 0:
        return 0.05
    if idle:
        # 空闲跳过：水位越高越好
        return _clamp01(0.55 + 0.45 * fill_after)
    # 有流量跳过：仅当水位够用
    return _clamp01(0.3 + 0.7 * avail * fill_before)


def poll_reward(
    *,
    arm: int,
    n_req: int,
    ok: int,
    fail: int,
    usage: float,
    threshold: float,
    polled_ok: bool,
) -> float:
    served = ok + fail
    avail = ok / served if served else 1.0
    idle = n_req == 0
    safe = usage < threshold * 0.55
    near = usage >= threshold - 10
    if arm == ARM_ACT:
        if idle and safe:
            return 0.08  # 空闲安全区轮询：浪费
        if near or not polled_ok:
            return 0.9 if polled_ok else 0.2
        if idle:
            return 0.35
        return _clamp01(0.45 * avail + 0.4 + (0.15 if near else 0.0))
    # SKIP
    if fail > 0:
        return 0.05
    if idle and safe:
        return 0.95
    if near:
        return 0.15
    return _clamp01(0.4 + 0.5 * avail)


def login_hard_arm(world: LoginWorld, n_req: int, consec_skip: int) -> int | None:
    fill = world.valid / max(1, world.pool_target)
    # 空池 / 低水位遇流量：必须登
    if world.valid == 0:
        return ARM_ACT
    if n_req > 0 and world.valid < n_req:
        return ARM_ACT
    if n_req > 0 and fill < 0.2:
        return ARM_ACT
    # 空闲且池接近满：禁止刷
    if n_req == 0 and fill >= 0.85 and consec_skip < 4:
        return ARM_SKIP
    # 空闲水位尚可：倾向跳过（硬约束仅前几次，其余交给 bandit）
    if n_req == 0 and fill >= 0.4 and consec_skip < 2:
        return ARM_SKIP
    return None


def poll_hard_arm(world: PollWorld, n_req: int, consec_skip: int) -> int | None:
    if not world.has_token:
        return ARM_ACT
    if world.usage >= world.threshold - 5:
        return ARM_ACT
    if n_req > 0 and (world.tick - world.last_poll_tick) > 22 and world.usage >= world.threshold - 15:
        return ARM_ACT
    if n_req == 0 and world.usage < world.threshold * 0.55 and consec_skip < 3:
        return ARM_SKIP
    return None


def login_batch_size(world: LoginWorld, n_req: int) -> int:
    """按缺口与需求决定一次补登批量，避免 busy 时每次只登 3 个跟不上。"""
    need = max(0, world.pool_target - world.valid)
    demand = max(n_req, int(round(world.demand_ema)))
    if n_req == 0:
        # 空闲最多补一小撮，防「一直 32 清到满」
        return min(need, 2)
    return min(need, max(demand + 2, 4))


# ---------------------------------------------------------------------------
# 仿真主循环
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    algo: str
    scheme: str
    pattern: str
    kind: str
    ticks: int
    actions: int = 0
    skips: int = 0
    load_cost: float = 0.0
    req_ok: int = 0
    req_fail: int = 0
    waste_actions: int = 0
    logged_total: int = 0
    rewards: float = 0.0

    @property
    def availability(self) -> float:
        tot = self.req_ok + self.req_fail
        return self.req_ok / tot if tot else 1.0

    @property
    def action_rate(self) -> float:
        return self.actions / max(1, self.ticks)

    @property
    def waste_rate(self) -> float:
        return self.waste_actions / max(1, self.actions) if self.actions else 0.0

    @property
    def waste_intensity(self) -> float:
        """浪费动作 / tick：空闲高负载的真实度量。"""
        return self.waste_actions / max(1, self.ticks)

    @property
    def load_per_tick(self) -> float:
        return self.load_cost / max(1, self.ticks)

    @property
    def stability_score(self) -> float:
        """可用性优先，惩罚动作率与浪费强度（非条件浪费率）。"""
        return self.availability - 0.40 * self.action_rate - 0.35 * self.waste_intensity - 0.05 * self.load_per_tick


def _make_algo(name: str, dim: int, seed: int):
    if name == "LinUCB":
        return LinUCB(dim, alpha=0.55, seed=seed)
    if name == "LinTS":
        return LinTS(dim, v=0.30, seed=seed)
    if name == "TTPS":
        return TTPS(dim, beta=0.55, v=0.30, seed=seed)
    raise KeyError(name)


def run_login_sim(
    algo_name: str,
    scheme: str,
    pattern: str,
    *,
    ticks: int = 400,
    seed: int = 0,
) -> RunStats:
    world = LoginWorld(seed=seed)
    dim = len(login_features(world, 0, scheme, 0))
    algo = _make_algo(algo_name, dim, seed)
    stats = RunStats(algo_name, scheme, pattern, "login", ticks)
    consec_skip = 0
    for t in range(ticks):
        world.tick = t
        world.expire()
        # 先看潜在需求（用 demand_ema + 抽样预览），再决策维护，再服务
        # 用同一 traffic 抽样：先 peek 再真正 serve
        n_req = world.traffic(pattern)
        fill_before = world.valid / max(1, world.pool_target)
        x = login_features(world, consec_skip, scheme, n_req)
        hard = login_hard_arm(world, n_req, consec_skip)
        arm = hard if hard is not None else algo.select(x)

        logged = attempted = 0
        if arm == ARM_ACT:
            batch = login_batch_size(world, n_req)
            logged, attempted = world.do_login(batch)
            stats.actions += 1
            stats.load_cost += world.login_cost * max(1, attempted)
            stats.logged_total += logged
            # 空闲且水位已够仍登录 → 浪费
            if n_req == 0 and fill_before >= 0.35:
                stats.waste_actions += 1
            consec_skip = 0
        else:
            stats.skips += 1
            consec_skip += 1

        ok, fail = world.serve(n_req)
        stats.req_ok += ok
        stats.req_fail += fail
        fill_after = world.valid / max(1, world.pool_target)
        reward = login_reward(
            arm=arm,
            n_req=n_req,
            ok=ok,
            fail=fail,
            logged=logged,
            attempted=attempted,
            fill_before=fill_before,
            fill_after=fill_after,
        )
        algo.update(arm, x, reward)
        stats.rewards += reward
    return stats


def run_poll_sim(
    algo_name: str,
    scheme: str,
    pattern: str,
    *,
    ticks: int = 400,
    seed: int = 0,
) -> RunStats:
    world = PollWorld(seed=seed)
    dim = len(poll_features(world, 0, scheme, 0))
    algo = _make_algo(algo_name, dim, seed)
    stats = RunStats(algo_name, scheme, pattern, "poll", ticks)
    consec_skip = 0
    for t in range(ticks):
        world.tick = t
        n_req = world.traffic(pattern)
        world.drift(n_req)
        x = poll_features(world, consec_skip, scheme, n_req)
        hard = poll_hard_arm(world, n_req, consec_skip)
        arm = hard if hard is not None else algo.select(x)

        polled_ok = True
        safe_idle = n_req == 0 and world.usage < world.threshold * 0.55
        if arm == ARM_ACT:
            polled_ok = world.do_poll()
            stats.actions += 1
            stats.load_cost += world.poll_cost
            if safe_idle:
                stats.waste_actions += 1
            consec_skip = 0
        else:
            stats.skips += 1
            consec_skip += 1

        ok = fail = 0
        for _ in range(n_req):
            if world.request_ok():
                ok += 1
            else:
                fail += 1
        stats.req_ok += ok
        stats.req_fail += fail
        reward = poll_reward(
            arm=arm,
            n_req=n_req,
            ok=ok,
            fail=fail,
            usage=world.usage,
            threshold=world.threshold,
            polled_ok=polled_ok,
        )
        algo.update(arm, x, reward)
        stats.rewards += reward
    return stats


def _fmt(s: RunStats) -> str:
    return (
        f"{s.algo:5s} {s.scheme:10s} {s.pattern:6s} | "
        f"avail={s.availability:5.1%} act={s.action_rate:5.1%} "
        f"wasteI={s.waste_intensity:5.1%} wasteR={s.waste_rate:5.1%} "
        f"score={s.stability_score:6.3f} load/t={s.load_per_tick:5.2f}"
    )


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

ALGOS = ("LinUCB", "LinTS", "TTPS")
SCHEMES = ("minimal", "demand", "stability", "full", "noisy")
PATTERNS = ("idle", "busy", "burst", "diurnal")


class TestScheduleBanditLimits(unittest.TestCase):
    """极限：空闲压制负载；繁忙保住可用性。"""

    def test_idle_login_should_not_keep_full_pool_hot(self) -> None:
        rows = []
        for algo in ALGOS:
            for scheme in ("minimal", "demand", "stability", "full"):
                st = run_login_sim(algo, scheme, "idle", ticks=300, seed=7)
                rows.append(st)
                self.assertGreaterEqual(st.availability, 0.80, _fmt(st))
                # 空闲绝对浪费强度必须低（核心：别一直清 32 号）
                if scheme in ("stability", "full"):
                    self.assertLess(st.waste_intensity, 0.25, _fmt(st))
                    self.assertLess(st.action_rate, 0.45, _fmt(st))

    def test_busy_login_keeps_availability(self) -> None:
        for algo in ALGOS:
            for scheme in ("demand", "stability", "full"):
                st = run_login_sim(algo, scheme, "busy", ticks=300, seed=11)
                self.assertGreaterEqual(st.availability, 0.90, _fmt(st))

    def test_idle_poll_waste_bounded(self) -> None:
        for algo in ALGOS:
            st = run_poll_sim(algo, "stability", "idle", ticks=300, seed=13)
            self.assertLess(st.action_rate, 0.40, _fmt(st))
            # 用强度而非条件浪费率：空闲时偶发探索的 waste_rate 可偏高，但 /tick 必须低
            self.assertLess(st.waste_intensity, 0.20, _fmt(st))

    def test_burst_poll_availability(self) -> None:
        for algo in ALGOS:
            st = run_poll_sim(algo, "full", "burst", ticks=350, seed=17)
            self.assertGreaterEqual(st.availability, 0.80, _fmt(st))

    def test_idle_vs_busy_load_gap_login(self) -> None:
        """同算法下 idle 负载应显著低于 busy（否则长期空转）。"""
        for algo in ALGOS:
            idle = run_login_sim(algo, "stability", "idle", ticks=280, seed=31)
            busy = run_login_sim(algo, "stability", "busy", ticks=280, seed=31)
            self.assertLess(idle.load_per_tick, busy.load_per_tick * 0.55, f"{algo}\n{_fmt(idle)}\n{_fmt(busy)}")

    def test_idle_vs_busy_load_gap_poll(self) -> None:
        for algo in ALGOS:
            idle = run_poll_sim(algo, "stability", "idle", ticks=280, seed=33)
            busy = run_poll_sim(algo, "stability", "busy", ticks=280, seed=33)
            self.assertLess(idle.action_rate, busy.action_rate + 0.05, f"{algo}\n{_fmt(idle)}\n{_fmt(busy)}")
            self.assertLess(idle.waste_intensity, 0.22, _fmt(idle))


class TestDimensionAblation(unittest.TestCase):
    """维度方案实际对比。"""

    def test_login_dimension_ranking_on_idle(self) -> None:
        scores: Dict[str, List[float]] = {s: [] for s in SCHEMES}
        loads: Dict[str, List[float]] = {s: [] for s in SCHEMES}
        for seed in (1, 2, 3):
            for algo in ALGOS:
                for scheme in SCHEMES:
                    st = run_login_sim(algo, scheme, "idle", ticks=250, seed=seed + (hash(algo) % 50))
                    scores[scheme].append(st.stability_score)
                    loads[scheme].append(st.waste_intensity)
        mean = {k: sum(v) / len(v) for k, v in scores.items()}
        mean_w = {k: sum(v) / len(v) for k, v in loads.items()}
        self.assertGreater(mean["stability"], mean["noisy"] - 0.03)
        self.assertGreater(mean["full"], mean["noisy"] - 0.03)
        # stability 空闲浪费应不差于 minimal
        self.assertLessEqual(mean_w["stability"], mean_w["minimal"] + 0.08)
        print("\n[login idle] mean score / wasteI by scheme:")
        for k, v in sorted(mean.items(), key=lambda kv: -kv[1]):
            print(f"  {k:10s} score={v:7.4f} wasteI={mean_w[k]:6.3f}")

    def test_poll_dimension_ranking_on_idle(self) -> None:
        scores: Dict[str, List[float]] = {s: [] for s in SCHEMES}
        for seed in (4, 5, 6):
            for algo in ALGOS:
                for scheme in SCHEMES:
                    st = run_poll_sim(algo, scheme, "idle", ticks=250, seed=seed)
                    scores[scheme].append(st.stability_score)
        mean = {k: sum(v) / len(v) for k, v in scores.items()}
        self.assertGreater(mean["stability"], mean["noisy"] - 0.03)
        print("\n[poll idle] mean stability_score by scheme:")
        for k, v in sorted(mean.items(), key=lambda kv: -kv[1]):
            print(f"  {k:10s} {v:7.4f}")

    def test_demand_features_help_busy_login(self) -> None:
        """busy 下带需求维应比纯水位 minimal 更稳（可用性）。"""
        min_avails, dem_avails = [], []
        for seed in (8, 9, 10):
            for algo in ALGOS:
                min_avails.append(run_login_sim(algo, "minimal", "busy", ticks=220, seed=seed).availability)
                dem_avails.append(run_login_sim(algo, "demand", "busy", ticks=220, seed=seed).availability)
        self.assertGreaterEqual(sum(dem_avails) / len(dem_avails), sum(min_avails) / len(min_avails) - 0.03)

    def test_algorithm_leaderboard_mixed_traffic(self) -> None:
        board: Dict[str, List[float]] = {a: [] for a in ALGOS}
        for pattern in PATTERNS:
            for seed in (21, 22):
                for algo in ALGOS:
                    login = run_login_sim(algo, "stability", pattern, ticks=200, seed=seed)
                    poll = run_poll_sim(algo, "stability", pattern, ticks=200, seed=seed + 1)
                    board[algo].append(0.55 * login.stability_score + 0.45 * poll.stability_score)
        mean = {k: sum(v) / len(v) for k, v in board.items()}
        print("\n[mixed] algo stability_score:")
        for k, v in sorted(mean.items(), key=lambda kv: -kv[1]):
            print(f"  {k:5s} {v:7.4f}")
        for algo, v in mean.items():
            self.assertGreater(v, 0.35, f"{algo} too unstable: {v}")


class TestOracleGap(unittest.TestCase):
    """与 always-act 上界对比。"""

    def test_login_better_than_always_act_on_idle(self) -> None:
        always_load = 0.0
        world = LoginWorld(seed=99)
        for t in range(250):
            world.tick = t
            world.expire()
            n_req = world.traffic("idle")
            attempted = min(max(0, world.pool_target - world.valid), 3) or 0
            if world.valid < world.pool_target:
                world.do_login(3)
                always_load += max(1, attempted if attempted else 3)
            else:
                always_load += 1  # 仍尝试 act
            world.serve(n_req)
        always_lpt = always_load / 250
        for algo in ALGOS:
            st = run_login_sim(algo, "stability", "idle", ticks=250, seed=99)
            self.assertLess(st.load_per_tick, always_lpt * 0.70, f"{_fmt(st)} always={always_lpt:.2f}")

    def test_poll_better_than_always_act_on_idle(self) -> None:
        for algo in ALGOS:
            st = run_poll_sim(algo, "stability", "idle", ticks=250, seed=101)
            self.assertLess(st.action_rate, 0.55, _fmt(st))
            self.assertLess(st.waste_intensity, 0.25, _fmt(st))


class TestStressMatrix(unittest.TestCase):
    """多维交叉：算法 × 维度 × 流量，输出极限表并设硬底线。"""

    def test_matrix_smoke_floors(self) -> None:
        worst_busy_avail = 1.0
        worst_idle_waste = 0.0
        print("\n=== stress matrix (stability scheme) ===")
        for pattern in PATTERNS:
            for algo in ALGOS:
                login = run_login_sim(algo, "stability", pattern, ticks=240, seed=41)
                poll = run_poll_sim(algo, "stability", pattern, ticks=240, seed=42)
                print(" L", _fmt(login))
                print(" P", _fmt(poll))
                if pattern == "busy":
                    worst_busy_avail = min(worst_busy_avail, login.availability, poll.availability)
                if pattern == "idle":
                    worst_idle_waste = max(worst_idle_waste, login.waste_intensity, poll.waste_intensity)
        self.assertGreaterEqual(worst_busy_avail, 0.85)
        self.assertLess(worst_idle_waste, 0.30)


if __name__ == "__main__":
    print("=== LOGIN limit table ===")
    for pattern in PATTERNS:
        print(f"\npattern={pattern}")
        for scheme in SCHEMES:
            for algo in ALGOS:
                print(" ", _fmt(run_login_sim(algo, scheme, pattern, ticks=300, seed=3)))
    print("\n=== POLL limit table ===")
    for pattern in PATTERNS:
        print(f"\npattern={pattern}")
        for scheme in SCHEMES:
            for algo in ALGOS:
                print(" ", _fmt(run_poll_sim(algo, scheme, pattern, ticks=300, seed=3)))
    unittest.main()
