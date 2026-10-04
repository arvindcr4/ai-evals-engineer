"""Resampling statistics for comparing models on the same eval items.

Everything here is numpy + stdlib: normal quantiles via ``math.erf`` and
Acklam's rational approximation, exact binomial tails via ``math.lgamma``.
The paired, clustered and item-level bootstraps share one engine: a cluster
bootstrap where every item is its own cluster in the unclustered case.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

Correction = Literal["none", "bonferroni", "holm", "bh"]

_CHUNK_CELLS = 4_000_000


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam, |rel err| < 1.2e-9 after one Newton step)."""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p must be in [0, 1], got {p}")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = norm_cdf(x) - p
    return x - e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)


@dataclass
class BootstrapCI:
    estimate: float
    low: float
    high: float
    se: float
    method: str
    confidence: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2


def _cluster_sums(values: np.ndarray, clusters: Sequence | None) -> tuple[np.ndarray, np.ndarray]:
    if clusters is None:
        return values.astype(float), np.ones(len(values))
    _, inv = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    sums = np.bincount(inv, weights=values.astype(float))
    counts = np.bincount(inv).astype(float)
    return sums, counts


def _boot_means(sums: np.ndarray, counts: np.ndarray, n_boot: int,
                rng: np.random.Generator) -> np.ndarray:
    """Ratio-of-sums bootstrap means using multinomial cluster weights, chunked for memory."""
    g = len(sums)
    out = np.empty(n_boot)
    step = max(1, _CHUNK_CELLS // max(g, 1))
    probs = np.full(g, 1.0 / g)
    for start in range(0, n_boot, step):
        stop = min(n_boot, start + step)
        w = rng.multinomial(g, probs, size=stop - start).astype(float)
        out[start:stop] = (w @ sums) / (w @ counts)
    return out


def _bca_bounds(boot: np.ndarray, theta: float, jack: np.ndarray,
                confidence: float) -> tuple[float, float]:
    alpha = 1 - confidence
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / len(boot)
    prop = min(max(prop, 1.0 / (len(boot) + 1)), 1 - 1.0 / (len(boot) + 1))
    z0 = norm_ppf(prop)
    dev = jack.mean() - jack
    denom = 6.0 * (np.sum(dev**2) ** 1.5)
    acc = float(np.sum(dev**3) / denom) if denom > 0 else 0.0
    qs = []
    for tail in (alpha / 2, 1 - alpha / 2):
        z = norm_ppf(tail)
        adj = norm_cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        qs.append(float(np.quantile(boot, min(max(adj, 0.0), 1.0))))
    return qs[0], qs[1]


def bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence | None = None,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    method: Literal["bca", "percentile"] = "bca",
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap CI for a mean; pass per-item paired differences for a paired CI.

    With ``clusters`` the resampling unit is the cluster (conversation, template,
    source document...), which keeps correlated items from shrinking the CI.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        raise ValueError("need at least 2 observations to bootstrap")
    sums, counts = _cluster_sums(x, clusters)
    theta = float(sums.sum() / counts.sum())
    boot = _boot_means(sums, counts, n_boot, np.random.default_rng(seed))
    if method == "percentile" or np.ptp(boot) == 0:
        a = (1 - confidence) / 2
        low, high = float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))
        used = "percentile"
    else:
        tot_s, tot_c = sums.sum(), counts.sum()
        jack = (tot_s - sums) / np.maximum(tot_c - counts, 1e-12)
        low, high = _bca_bounds(boot, theta, jack, confidence)
        used = "bca"
    return BootstrapCI(theta, low, high, float(boot.std(ddof=1)), used, confidence)


def bootstrap_unpaired(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10_000,
    confidence: float = 0.95, seed: int = 0,
) -> BootstrapCI:
    """Percentile CI for mean(b) - mean(a) when the two models saw different items."""
    xa, xb = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ba = _boot_means(xa, np.ones(len(xa)), n_boot, rng)
    bb = _boot_means(xb, np.ones(len(xb)), n_boot, rng)
    diff = bb - ba
    q = (1 - confidence) / 2
    return BootstrapCI(float(xb.mean() - xa.mean()), float(np.quantile(diff, q)),
                       float(np.quantile(diff, 1 - q)), float(diff.std(ddof=1)),
                       "percentile-unpaired", confidence)


def paired_permutation_test(
    diffs: Sequence[float], clusters: Sequence | None = None,
    n_perm: int = 10_000, seed: int = 0,
) -> float:
    """Two-sided sign-flip permutation p-value for H0: mean paired difference = 0.

    Under H0 the A/B labels are exchangeable within each item (or cluster), so
    flipping the sign of each unit's summed difference draws from the null.
    """
    d = np.asarray(diffs, dtype=float)
    sums, counts = _cluster_sums(d, clusters)
    total = counts.sum()
    obs = abs(sums.sum()) / total
    if obs == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    g = len(sums)
    hits = 0
    step = max(1, _CHUNK_CELLS // max(g, 1))
    for start in range(0, n_perm, step):
        k = min(n_perm, start + step) - start
        signs = rng.integers(0, 2, size=(k, g)) * 2 - 1
        stats = np.abs(signs @ sums) / total
        hits += int(np.sum(stats >= obs - 1e-12))
    return (hits + 1) / (n_perm + 1)


def unpaired_permutation_test(
    a: Sequence[float], b: Sequence[float], n_perm: int = 10_000, seed: int = 0,
) -> float:
    """Two-sided label-shuffle permutation p-value for H0: mean(a) = mean(b)."""
    xa, xb = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([xa, xb])
    na = len(xa)
    obs = abs(xb.mean() - xa.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        hits += abs(perm[na:].mean() - perm[:na].mean()) >= obs - 1e-12
    return (int(hits) + 1) / (n_perm + 1)


def _binom_logpmf(k: int, n: int, p: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1 - p))


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> tuple[float, int, int]:
    """Exact McNemar test on paired binary outcomes.

    Returns ``(p_value, n_a_only, n_b_only)``: only discordant items carry
    information, and under H0 each is equally likely to favour either model.
    """
    xa, xb = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    only_a = int(np.sum(xa & ~xb))
    only_b = int(np.sum(~xa & xb))
    n = only_a + only_b
    if n == 0:
        return 1.0, 0, 0
    k = min(only_a, only_b)
    tail = sum(math.exp(_binom_logpmf(i, n, 0.5)) for i in range(k + 1))
    return min(1.0, 2 * tail), only_a, only_b


def adjust_pvalues(pvals: Sequence[float], method: Correction = "holm") -> list[float]:
    """Family-wise (Bonferroni, Holm) or false-discovery-rate (Benjamini-Hochberg) adjustment."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0 or method == "none":
        return p.tolist()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0).tolist()
    order = np.argsort(p)
    ranked = p[order]
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method == "bh":
        adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    else:
        raise ValueError(f"unknown correction {method!r}")
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


# --- power analysis -----------------------------------------------------------


def power_two_proportions(
    n: int, p1: float, p2: float, alpha: float = 0.05,
    paired: bool = False, discordance: float | None = None,
) -> float:
    """Power of a two-sided test of p1 vs p2 with ``n`` items (per arm when unpaired).

    Paired power uses the McNemar normal approximation; ``discordance`` is the
    fraction of items on which the two models disagree. It defaults to the
    independent-errors value p1(1-p2)+p2(1-p1), the worst case for pairing.
    """
    delta = abs(p2 - p1)
    if delta == 0 or n <= 0:
        return alpha
    za = norm_ppf(1 - alpha / 2)
    if paired:
        psi = discordance if discordance is not None else p1 * (1 - p2) + p2 * (1 - p1)
        psi = max(psi, delta + 1e-12)
        z = (delta * math.sqrt(n) - za * math.sqrt(psi)) / math.sqrt(psi - delta**2)
    else:
        pbar = (p1 + p2) / 2
        se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
        se1 = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
        z = (delta - za * se0) / se1
    return norm_cdf(z)


def required_sample_size(
    baseline: float, delta: float, alpha: float = 0.05, power: float = 0.8,
    paired: bool = False, discordance: float | None = None,
) -> int:
    """Smallest n with at least ``power`` to detect ``baseline -> baseline + delta``."""
    p2 = baseline + delta
    if not 0 < p2 < 1:
        raise ValueError("baseline + delta must lie in (0, 1)")
    lo, hi = 1, 2
    while power_two_proportions(hi, baseline, p2, alpha, paired, discordance) < power:
        hi *= 2
        if hi > 10**9:
            raise ValueError("effect too small to detect with any practical n")
    while lo < hi:
        mid = (lo + hi) // 2
        if power_two_proportions(mid, baseline, p2, alpha, paired, discordance) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def minimum_detectable_effect(
    n: int, baseline: float, alpha: float = 0.05, power: float = 0.8,
    paired: bool = False, discordance: float | None = None,
) -> float:
    """Smallest upward delta detectable with ``power`` at sample size ``n``."""
    lo, hi = 1e-6, 1 - baseline - 1e-6
    if power_two_proportions(n, baseline, baseline + hi, alpha, paired, discordance) < power:
        return float("nan")
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_two_proportions(n, baseline, baseline + mid, alpha, paired, discordance) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def mde_continuous(n: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """MDE for a mean of paired differences with standard deviation ``sd``."""
    return (norm_ppf(1 - alpha / 2) + norm_ppf(power)) * sd / math.sqrt(max(n, 1))
