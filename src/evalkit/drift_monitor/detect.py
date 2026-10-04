"""Drift tests against a rolling baseline.

* **z-score** — today's metric vs the baseline mean/std. The std is floored
  at a fraction of the mean and, for rate metrics, at the binomial standard
  error of today's sample, so small samples and near-constant baselines
  cannot explode z (``rate_sd_floor`` is an absolute floor for rates).
* **CUSUM** — one-sided cumulative sum of standardised deviations over the
  last few days; catches slow decay no single day would trip.
* **PSI** — Population Stability Index between baseline and today's
  distributions (answer-length histogram, tool-usage mix).

Only the *bad* direction alerts (a falling refusal rate is not an incident).
Baseline days that themselves raised critical alerts are excluded, so a
sustained regression keeps alerting instead of becoming the new normal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from evalkit.drift_monitor.alerts import Alert
from evalkit.drift_monitor.store import MetricStore

# +1: higher is worse, -1: lower is worse
DIRECTION: dict[str, int] = {
    "refusal_rate": 1,
    "empty_rate": 1,
    "answer_chars_mean": -1,
    "tool_error_rate": 1,
    "latency_p50": 1,
    "latency_p95": 1,
    "cost_usd_mean": 1,
    "judge_score": -1,
}


@dataclass
class DetectorConfig:
    baseline_days: int = 14
    min_baseline: int = 5
    z_warn: float = 3.0
    z_crit: float = 4.0
    std_floor_frac: float = 0.05
    cusum_days: int = 7
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    psi_warn: float = 0.2
    psi_crit: float = 0.3
    rate_sd_floor: float = 0.01
    rate_metrics: frozenset[str] = frozenset({"refusal_rate", "empty_rate", "tool_error_rate"})
    directions: dict[str, int] = field(default_factory=lambda: dict(DIRECTION))


def _sigma(
    baseline: Sequence[float], floor_frac: float, abs_floor: float = 0.0
) -> tuple[float, float]:
    arr = np.asarray(baseline, dtype=float)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mu, max(sd, floor_frac * abs(mu), abs_floor, 1e-9)


def rate_floor(baseline: Sequence[float], n: int | None, min_sd: float) -> float:
    """SD floor for a proportion: binomial SE at sample size ``n``, at least ``min_sd``."""
    p = float(np.mean(baseline))
    se = math.sqrt(max(p * (1 - p), 1e-9) / n) if n else 0.0
    return max(se, min_sd)


def zscore(
    value: float, baseline: Sequence[float], floor_frac: float = 0.05, abs_floor: float = 0.0
) -> float:
    mu, sd = _sigma(baseline, floor_frac, abs_floor)
    return (value - mu) / sd


def cusum(
    values: Sequence[float],
    baseline: Sequence[float],
    direction: int = 1,
    k: float = 0.5,
    floor_frac: float = 0.05,
    abs_floor: float = 0.0,
) -> float:
    """Final one-sided CUSUM statistic of ``values`` relative to ``baseline``."""
    mu, sd = _sigma(baseline, floor_frac, abs_floor)
    s = 0.0
    for v in values:
        s = max(0.0, s + direction * (v - mu) / sd - k)
    return s


def psi(expected: dict[str, int], actual: dict[str, int], alpha: float = 0.5) -> float:
    """Population Stability Index with additive (``alpha``) smoothing of every bin.

    Smoothing matters for daily samples of ~100: a rare bin that happens to be
    empty today would otherwise get a tiny epsilon mass and dominate the index.
    """
    keys = sorted(set(expected) | set(actual))
    te, ta = sum(expected.values()), sum(actual.values())
    if te == 0 or ta == 0:
        return 0.0
    te_s, ta_s = te + alpha * len(keys), ta + alpha * len(keys)
    total = 0.0
    for k in keys:
        e = (expected.get(k, 0) + alpha) / te_s
        a = (actual.get(k, 0) + alpha) / ta_s
        total += (a - e) * math.log(a / e)
    return total


def baseline_dates(day: str, store: MetricStore, cfg: DetectorConfig) -> list[str]:
    """Most recent ``baseline_days`` stored dates before ``day`` without critical alerts."""
    excluded = store.alerted_dates("critical")
    prior = [d for d in store.dates() if d < day and d not in excluded]
    return prior[-cfg.baseline_days :]


def detect(day: str, store: MetricStore, cfg: DetectorConfig | None = None) -> list[Alert]:
    """Run every test for ``day`` (whose metrics must already be stored)."""
    cfg = cfg or DetectorConfig()
    base = baseline_dates(day, store, cfg)
    if len(base) < cfg.min_baseline:
        return []
    recent = [d for d in store.dates() if d <= day][-cfg.cusum_days :]
    info = store.run_info(day)
    n_today = info[1] if info else None
    alerts: list[Alert] = []
    for metric, direction in cfg.directions.items():
        series = store.series(metric)
        if day not in series:
            continue
        hist = [series[d] for d in base if d in series]
        if len(hist) < cfg.min_baseline:
            continue
        value, mu = series[day], float(np.mean(hist))
        floor = rate_floor(hist, n_today, cfg.rate_sd_floor) if metric in cfg.rate_metrics else 0
        z = zscore(value, hist, cfg.std_floor_frac, floor)
        bad_z = z * direction
        if bad_z >= cfg.z_warn:
            sev = "critical" if bad_z >= cfg.z_crit else "warning"
            alerts.append(
                Alert(
                    day,
                    metric,
                    "zscore",
                    sev,
                    value,
                    mu,
                    z,
                    f"{metric} {value:.4g} vs baseline {mu:.4g} (z={z:+.1f})",
                )
            )
        s = cusum(
            [series[d] for d in recent if d in series],
            hist,
            direction,
            cfg.cusum_k,
            cfg.std_floor_frac,
            floor,
        )
        if s >= cfg.cusum_h and bad_z < cfg.z_warn:
            alerts.append(
                Alert(
                    day,
                    metric,
                    "cusum",
                    "warning",
                    value,
                    mu,
                    s,
                    f"{metric} drifting {'up' if direction > 0 else 'down'} for "
                    f"{len(recent)} days (CUSUM={s:.1f})",
                )
            )
    for name in store.dist_names():
        today = store.dist(name, day)
        if not today:
            continue
        pooled: dict[str, int] = {}
        for d in base:
            for k, v in (store.dist(name, d) or {}).items():
                pooled[k] = pooled.get(k, 0) + v
        score = psi(pooled, today)
        if score >= cfg.psi_warn:
            sev = "critical" if score >= cfg.psi_crit else "warning"
            shifted = _biggest_shift(pooled, today)
            alerts.append(
                Alert(
                    day,
                    name,
                    "psi",
                    sev,
                    score,
                    0.0,
                    score,
                    f"{name} distribution shifted (PSI={score:.2f}; {shifted})",
                )
            )
    return alerts


def _biggest_shift(expected: dict[str, int], actual: dict[str, int]) -> str:
    te, ta = sum(expected.values()) or 1, sum(actual.values()) or 1
    deltas = {
        k: actual.get(k, 0) / ta - expected.get(k, 0) / te for k in set(expected) | set(actual)
    }
    k = max(deltas, key=lambda x: abs(deltas[x]))
    return f"'{k}' {expected.get(k, 0) / te:.0%} -> {actual.get(k, 0) / ta:.0%}"
