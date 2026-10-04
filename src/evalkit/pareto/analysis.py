"""Unit-economics math: per-config stats, Pareto frontiers, routers, price what-ifs.

Input rows are eval results ``{tenant, config, task_id, success, cost_usd,
latency_s, confidence?}``. Every config of a tenant is assumed to have run on
the same task set, which is what makes per-task routing simulation possible.
"""

from __future__ import annotations

import fnmatch
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from evalkit.core.llm import stable_hash


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class ConfigStats:
    tenant: str
    config: str
    n: int
    successes: int
    success_rate: float
    ci_lo: float
    ci_hi: float
    mean_cost: float
    cost_per_success: float
    p95_latency: float
    on_frontier: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RouterPoint:
    tenant: str
    cheap: str
    expensive: str
    threshold: float
    escalation_rate: float
    success_rate: float
    mean_cost: float
    cost_per_success: float
    p95_latency: float
    on_frontier: bool = False

    @property
    def config(self) -> str:
        return f"router({self.cheap}→{self.expensive}@{self.threshold:.2f})"

    def to_dict(self) -> dict:
        return {**asdict(self), "config": self.config}


def _stats(tenant: str, config: str, rows: list[dict]) -> ConfigStats:
    n = len(rows)
    succ = sum(bool(r["success"]) for r in rows)
    cost = float(np.mean([r["cost_usd"] for r in rows]))
    lo, hi = wilson(succ, n)
    lat = [r.get("latency_s", 0.0) for r in rows]
    return ConfigStats(
        tenant=tenant,
        config=config,
        n=n,
        successes=succ,
        success_rate=succ / n,
        ci_lo=lo,
        ci_hi=hi,
        mean_cost=cost,
        cost_per_success=cost * n / succ if succ else math.inf,
        p95_latency=float(np.percentile(lat, 95)),
    )


def group(rows: Iterable[dict]) -> dict[str, dict[str, list[dict]]]:
    """``{tenant: {config: [rows]}}`` preserving first-seen order."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["tenant"]][r["config"]].append(r)
    return out


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated ``(cost, success)`` points: lower cost, higher success.

    A point is dominated when another is no more expensive and at least as
    successful, and strictly better on one axis. Returned in ascending cost.
    """
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    frontier, best = [], -math.inf
    for i in order:
        if points[i][1] > best:
            frontier.append(i)
            best = points[i][1]
    return frontier


def summarize(rows: Iterable[dict]) -> dict[str, list[ConfigStats]]:
    """Per-tenant config stats with ``on_frontier`` set."""
    result: dict[str, list[ConfigStats]] = {}
    for tenant, configs in group(rows).items():
        stats = [_stats(tenant, c, rs) for c, rs in configs.items()]
        for i in pareto_frontier([(s.mean_cost, s.success_rate) for s in stats]):
            stats[i].on_frontier = True
        result[tenant] = sorted(stats, key=lambda s: s.mean_cost)
    return result


def simulate_router(
    rows: Iterable[dict],
    tenant: str,
    cheap: str,
    expensive: str,
    thresholds: Iterable[float] | None = None,
    cascade: bool = True,
) -> list[RouterPoint]:
    """Blend a System-One and a System-Two config with a confidence threshold.

    Each task first goes to ``cheap``; if its ``confidence`` is below the
    threshold the task escalates to ``expensive`` and takes that outcome. With
    ``cascade`` the escalated task pays for both calls (the cheap attempt is
    sunk); without it a pre-router is assumed and only the expensive call is
    paid. Rows without ``confidence`` fall back to a deterministic per-task
    hash, i.e. a random router that interpolates linearly.
    """
    by_cfg = group(rows).get(tenant, {})
    if cheap not in by_cfg or expensive not in by_cfg:
        raise KeyError(f"tenant {tenant!r} lacks config {cheap!r} or {expensive!r}")
    a = {r["task_id"]: r for r in by_cfg[cheap]}
    b = {r["task_id"]: r for r in by_cfg[expensive]}
    tasks = sorted(set(a) & set(b))
    if not tasks:
        raise ValueError(f"{cheap} and {expensive} share no task_ids for tenant {tenant!r}")

    def conf(r: dict) -> float:
        if r.get("confidence") is not None:
            return float(r["confidence"])
        return hash_unit(str(r["task_id"]))

    points = []
    for t in thresholds if thresholds is not None else np.linspace(0, 1, 21):
        succ = 0
        costs, lats, escalated = [], [], 0
        for tid in tasks:
            ra, rb = a[tid], b[tid]
            if conf(ra) >= t:
                succ += bool(ra["success"])
                costs.append(ra["cost_usd"])
                lats.append(ra.get("latency_s", 0.0))
            else:
                escalated += 1
                succ += bool(rb["success"])
                costs.append(rb["cost_usd"] + (ra["cost_usd"] if cascade else 0.0))
                lats.append(rb.get("latency_s", 0.0) + (ra.get("latency_s", 0.0) if cascade else 0))
        n = len(tasks)
        mean_cost = float(np.mean(costs))
        points.append(
            RouterPoint(
                tenant=tenant,
                cheap=cheap,
                expensive=expensive,
                threshold=float(t),
                escalation_rate=escalated / n,
                success_rate=succ / n,
                mean_cost=mean_cost,
                cost_per_success=mean_cost * n / succ if succ else math.inf,
                p95_latency=float(np.percentile(lats, 95)),
            )
        )
    return points


def hash_unit(key: str) -> float:
    return stable_hash("router", key) / 2**64


def mark_router_frontier(stats: list[ConfigStats], router: list[RouterPoint]) -> None:
    """Flag router points that sit on the joint (configs + router) frontier."""
    pts = [(s.mean_cost, s.success_rate) for s in stats] + [
        (p.mean_cost, p.success_rate) for p in router
    ]
    front = set(pareto_frontier(pts))
    for j, p in enumerate(router):
        p.on_frontier = len(stats) + j in front


def apply_price_factor(rows: Iterable[dict], pattern: str, factor: float) -> list[dict]:
    """Copy of ``rows`` with ``cost_usd`` scaled by ``factor`` for configs matching ``pattern``."""
    if factor < 0:
        raise ValueError("price factor must be non-negative")
    out = []
    for r in rows:
        r = dict(r)
        if fnmatch.fnmatchcase(r["config"], pattern):
            r["cost_usd"] = r["cost_usd"] * factor
        out.append(r)
    return out


def frontier_diff(
    before: dict[str, list[ConfigStats]], after: dict[str, list[ConfigStats]]
) -> dict[str, dict]:
    """Per tenant: configs that joined/left the frontier and cost-per-success deltas."""
    diff = {}
    for tenant, stats in after.items():
        old = {s.config: s for s in before.get(tenant, [])}
        joined = [s.config for s in stats if s.on_frontier and not old[s.config].on_frontier]
        left = [s.config for s in stats if not s.on_frontier and old[s.config].on_frontier]
        cps = {
            s.config: (old[s.config].cost_per_success, s.cost_per_success)
            for s in stats
            if not math.isclose(old[s.config].mean_cost, s.mean_cost)
        }
        diff[tenant] = {"joined": joined, "left": left, "cost_per_success": cps}
    return diff


def recommend(stats: list[ConfigStats], tolerance: float = 0.02) -> dict[str, str | float]:
    """Best-quality config and the cheapest config within ``tolerance`` of it."""
    best = max(stats, key=lambda s: (s.success_rate, -s.mean_cost))
    near = [s for s in stats if s.success_rate >= best.success_rate - tolerance]
    value = min(near, key=lambda s: s.mean_cost)
    return {
        "best_quality": best.config,
        "best_success": best.success_rate,
        "value_pick": value.config,
        "value_success": value.success_rate,
        "savings_vs_best": 1 - value.mean_cost / best.mean_cost if best.mean_cost else 0.0,
    }
