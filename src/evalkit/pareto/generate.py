"""Synthetic eval results: tenants × configs × shared tasks.

Each tenant has its own difficulty profile and token footprint; each config
has a skill level, per-1M-token prices and a latency profile. A task's latent
difficulty is shared by every config, so outcomes are correlated the way real
evals are, and every result carries the config's (noisily calibrated)
self-reported confidence for router simulation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    skill: float
    price_in: float  # USD per 1M input tokens
    price_out: float  # USD per 1M output tokens
    out_mult: float  # output-token multiplier (reasoning models think out loud)
    latency_s: float
    in_mult: float = 1.0  # input-token multiplier (retrieval stuffs the context)


@dataclass(frozen=True)
class TenantSpec:
    name: str
    difficulty: float
    tokens_in: int
    tokens_out: int


CONFIGS = [
    ConfigSpec("s1-nano", -0.6, 0.05, 0.40, 1.0, 0.6),
    ConfigSpec("s1-mini", 0.2, 0.15, 0.60, 1.0, 0.9),
    ConfigSpec("s1-mini+rag", 0.7, 0.15, 0.60, 1.1, 1.4, in_mult=2.5),
    ConfigSpec("s2-legacy", 0.8, 5.00, 15.0, 1.0, 3.0),
    ConfigSpec("s2-large", 1.2, 2.50, 10.0, 1.0, 2.2),
    ConfigSpec("s2-large+reasoning", 1.7, 2.50, 10.0, 4.0, 7.5),
    ConfigSpec("s2-frontier", 1.8, 15.0, 60.0, 1.5, 6.0),
]
TENANTS = [
    TenantSpec("globex-support", -0.4, 1800, 250),
    TenantSpec("initech-finance", 0.6, 3500, 400),
    TenantSpec("acme-legal", 1.3, 9000, 700),
    TenantSpec("umbrella-health", 1.0, 5000, 500),
]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_rows(
    tasks_per_tenant: int = 120,
    seed: int = 11,
    configs: list[ConfigSpec] | None = None,
    tenants: list[TenantSpec] | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for t in tenants or TENANTS:
        difficulty = rng.normal(t.difficulty, 1.1, tasks_per_tenant)
        size = rng.lognormal(0.0, 0.35, tasks_per_tenant)
        for c in configs or CONFIGS:
            for i in range(tasks_per_tenant):
                p = _sigmoid(1.6 * (c.skill - difficulty[i]) + 0.8)
                success = bool(rng.random() < p)
                tin = t.tokens_in * size[i] * c.in_mult
                tout = t.tokens_out * size[i] * c.out_mult * rng.lognormal(0, 0.2)
                cost = (tin * c.price_in + tout * c.price_out) / 1e6
                conf = _sigmoid(math.log(p / (1 - p)) + rng.normal(0, 0.9))
                rows.append(
                    {
                        "tenant": t.name,
                        "config": c.name,
                        "task_id": f"{t.name}-{i:04d}",
                        "success": success,
                        "cost_usd": round(cost, 7),
                        "latency_s": round(c.latency_s * size[i] * rng.lognormal(0, 0.25), 3),
                        "confidence": round(conf, 4),
                    }
                )
    return rows


def generate(out_path: str | Path, tasks_per_tenant: int = 120, seed: int = 11) -> int:
    rows = generate_rows(tasks_per_tenant, seed)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return len(rows)
