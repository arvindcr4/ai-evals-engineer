"""Model comparisons that end in a sentence a reviewer can act on.

Input rows are ``{item_id, model, score[, cluster]}``. :func:`compare` pairs two
models on their shared items (falling back to an unpaired test when there is
no overlap) and returns a :class:`Comparison` whose :meth:`Comparison.verdict`
reads like "Model B wins by +3.2 pts ± 1.1 (95% CI [2.1, 4.3], p=0.003,
n=500 paired)". :func:`leaderboard` ranks N models with multiplicity-corrected
pairwise tests and groups statistically tied models into tiers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np

from evalkit.significance.stats import (
    BootstrapCI,
    Correction,
    adjust_pvalues,
    bootstrap_mean,
    bootstrap_unpaired,
    mcnemar_exact,
    mde_continuous,
    minimum_detectable_effect,
    norm_ppf,
    paired_permutation_test,
    unpaired_permutation_test,
)

ScoreTable = dict[str, dict[str, tuple[float, str | None]]]


def load_scores(rows: Iterable[dict], metric: str = "score") -> ScoreTable:
    """Index rows as ``{model: {item_id: (score, cluster)}}``.

    ``metric`` names the score field; when rows carry a ``metric`` key (long
    format, several metrics per item) it selects rows with that metric instead.
    Duplicate (model, item) rows are averaged, e.g. repeated samples per item.
    """
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    clusters: dict[tuple[str, str], str | None] = {}
    for r in rows:
        if "metric" in r and metric != "score":
            if r["metric"] != metric:
                continue
            value = r["score"]
        else:
            if metric not in r:
                raise KeyError(f"row missing metric field {metric!r}: {r}")
            value = r[metric]
        model, item = str(r["model"]), str(r["item_id"])
        acc[model][item].append(float(value))
        cl = r.get("cluster")
        clusters[(model, item)] = None if cl is None else str(cl)
    return {
        m: {i: (float(np.mean(v)), clusters[(m, i)]) for i, v in items.items()}
        for m, items in acc.items()
    }


def _is_binary(*arrays: np.ndarray) -> bool:
    return all(np.isin(a, (0.0, 1.0)).all() for a in arrays)


def _is_unit_interval(*arrays: np.ndarray) -> bool:
    return all(len(a) and a.min() >= 0 and a.max() <= 1 for a in arrays)


def simulate_results(
    accuracies: dict[str, float],
    n_items: int = 500,
    n_clusters: int | None = None,
    cluster_sd: float = 0.0,
    seed: int = 0,
    item_sd: float = 1.5,
    model_sd: float = 0.7,
) -> list[dict]:
    """Synthetic 0/1 results with known true accuracies and shared item difficulty.

    Item difficulty (``item_sd``) is common to all models, which is what makes
    pairing powerful; ``model_sd`` is model-specific noise; ``cluster_sd`` adds a per-cluster shift so items within a
    cluster are correlated, as with turns of one conversation.
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(0, item_sd, n_items)
    cl = rng.integers(0, n_clusters, n_items) if n_clusters else None
    if cl is not None and cluster_sd > 0:
        difficulty += rng.normal(0, cluster_sd, n_clusters)[cl]
    rows: list[dict] = []
    for model, acc in accuracies.items():
        # Shift the shared latent so the marginal accuracy is approximately ``acc``.
        sd = float(np.sqrt(item_sd**2 + (cluster_sd**2 if n_clusters else 0.0) + model_sd**2))
        thresh = norm_ppf(acc) * sd
        noise = rng.normal(0, model_sd, n_items)
        correct = (thresh - difficulty + noise) > 0
        for i in range(n_items):
            row = {"item_id": f"q{i:04d}", "model": model, "score": int(correct[i])}
            if cl is not None:
                row["cluster"] = f"c{int(cl[i]):03d}"
            rows.append(row)
    return rows


@dataclass
class Comparison:
    model_a: str
    model_b: str
    mean_a: float
    mean_b: float
    delta: float
    ci: BootstrapCI
    p_value: float
    n: int
    paired: bool
    test: str
    alpha: float = 0.05
    n_clusters: int | None = None
    scale: float = 1.0
    unit: str = ""
    mde: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci.low > 0 or self.ci.high < 0

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha and self.ci_excludes_zero

    @property
    def winner(self) -> str | None:
        if not self.significant:
            return None
        return self.model_b if self.delta > 0 else self.model_a

    def _fmt(self, x: float, sign: bool = False) -> str:
        v = x * self.scale
        return f"{v:+.1f}" if sign else f"{v:.1f}"

    def verdict(self) -> str:
        conf = round(self.ci.confidence * 100)
        p = "p<0.001" if self.p_value < 0.001 else f"p={self.p_value:.3f}"
        design = "paired" if self.paired else "unpaired"
        if self.n_clusters:
            design += f", {self.n_clusters} clusters"
        unit = f" {self.unit}" if self.unit else ""
        if self.significant:
            lo, hi = (self.ci.low, self.ci.high) if self.delta > 0 else (
                -self.ci.high, -self.ci.low)
            stats = (f"({conf}% CI [{self._fmt(lo)}, {self._fmt(hi)}], {p}, "
                     f"n={self.n} {design})")
            lead, trail = (self.model_b, self.model_a) if self.delta > 0 else (
                self.model_a, self.model_b)
            return (f"{lead} beats {trail} by {self._fmt(abs(self.delta), True)}{unit} "
                    f"± {self._fmt(self.ci.half_width)} {stats}")
        stats = (f"({conf}% CI [{self._fmt(self.ci.low)}, {self._fmt(self.ci.high)}], {p}, "
                 f"n={self.n} {design})")
        borderline = (self.p_value < self.alpha) != self.ci_excludes_zero
        tag = " (borderline: test and CI disagree)" if borderline else ""
        msg = (f"No significant difference between {self.model_a} and {self.model_b}{tag}: "
               f"{self.model_b} − {self.model_a} = {self._fmt(self.delta, True)}{unit} {stats}.")
        if self.mde is not None and np.isfinite(self.mde):
            msg += (f" Smallest effect detectable at this n with 80% power ≈ "
                    f"{self._fmt(self.mde)}{unit}.")
        return msg

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(significant=self.significant, winner=self.winner, verdict=self.verdict())
        return d


def compare(
    table: ScoreTable,
    model_a: str,
    model_b: str,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    alpha: float | None = None,
    clustered: bool = True,
    ci_method: str = "bca",
    seed: int = 0,
) -> Comparison:
    """Compare ``model_b`` against ``model_a``; positive delta means B scores higher.

    Paired designs use a (cluster) bootstrap CI of per-item differences, an exact
    McNemar test for 0/1 scores without clusters, and a sign-flip permutation
    test otherwise. Models with no shared items get unpaired tests.
    """
    for m in (model_a, model_b):
        if m not in table:
            raise KeyError(f"model {m!r} not in results (have: {sorted(table)})")
    alpha = 1 - confidence if alpha is None else alpha
    sa, sb = table[model_a], table[model_b]
    shared = sorted(set(sa) & set(sb))
    xa_all = np.array([v[0] for v in sa.values()])
    xb_all = np.array([v[0] for v in sb.values()])
    scale, unit = (100.0, "pts") if _is_unit_interval(xa_all, xb_all) else (1.0, "")
    if len(shared) >= 2:
        xa = np.array([sa[i][0] for i in shared])
        xb = np.array([sb[i][0] for i in shared])
        cl = [sa[i][1] or sb[i][1] for i in shared]
        use_clusters = clustered and any(c is not None for c in cl)
        clusters = [c if c is not None else f"__item_{i}" for c, i in zip(cl, shared)] \
            if use_clusters else None
        diffs = xb - xa
        ci = bootstrap_mean(diffs, clusters, n_boot, confidence, ci_method, seed)  # type: ignore[arg-type]
        extra: dict = {}
        if _is_binary(xa, xb) and not use_clusters:
            p, only_a, only_b = mcnemar_exact(xa, xb)
            test = "mcnemar-exact"
            extra = {"a_only_correct": only_a, "b_only_correct": only_b}
        else:
            p = paired_permutation_test(diffs, clusters, n_boot, seed)
            test = "cluster-permutation" if use_clusters else "paired-permutation"
        n_cl = len(set(clusters)) if clusters else None
        sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
        design_effect = (ci.se * np.sqrt(len(diffs)) / sd) if sd > 0 else 1.0
        mde = mde_continuous(len(diffs), sd * design_effect, alpha) if sd > 0 else None
        return Comparison(model_a, model_b, float(xa.mean()), float(xb.mean()),
                          float(diffs.mean()), ci, p, len(shared), True, test, alpha,
                          n_cl, scale, unit, mde, extra)
    ci = bootstrap_unpaired(xa_all, xb_all, n_boot, confidence, seed)
    p = unpaired_permutation_test(xa_all, xb_all, min(n_boot, 5000), seed)
    mde = None
    if scale == 100.0:
        mde = minimum_detectable_effect(min(len(xa_all), len(xb_all)),
                                        float(xa_all.mean()), alpha)
    return Comparison(model_a, model_b, float(xa_all.mean()), float(xb_all.mean()),
                      ci.estimate, ci, p, len(xa_all) + len(xb_all), False,
                      "unpaired-permutation", alpha, None, scale, unit, mde)


@dataclass
class LeaderboardRow:
    rank: int
    tier: int
    model: str
    mean: float
    ci_low: float
    ci_high: float
    n: int
    delta_vs_top: float
    p_vs_top_adj: float | None


def leaderboard(
    table: ScoreTable,
    n_boot: int = 5_000,
    confidence: float = 0.95,
    correction: Correction = "holm",
    seed: int = 0,
) -> tuple[list[LeaderboardRow], list[Comparison], list[float]]:
    """Rank models; models not significantly worse than a tier's leader share its tier.

    All pairwise comparisons are run and their p-values corrected together
    (Holm controls the family-wise error rate, BH the false discovery rate).
    """
    alpha = 1 - confidence
    models = sorted(table, key=lambda m: -np.mean([v[0] for v in table[m].values()]))
    pairs: list[Comparison] = []
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            pairs.append(compare(table, ma, mb, n_boot, confidence, alpha, seed=seed))
    adj = adjust_pvalues([c.p_value for c in pairs], correction)
    sig = {(c.model_a, c.model_b): (p < alpha and c.ci_excludes_zero) for c, p in zip(pairs, adj)}
    padj = {(c.model_a, c.model_b): p for c, p in zip(pairs, adj)}
    by_pair = {(c.model_a, c.model_b): c for c in pairs}
    rows: list[LeaderboardRow] = []
    tier, leader = 1, models[0] if models else None
    for rank, m in enumerate(models, start=1):
        if leader is not None and m != leader and sig.get((leader, m), False):
            tier += 1
            leader = m
        scores = np.array([v[0] for v in table[m].values()])
        ci = bootstrap_mean(scores, None, n_boot, confidence, "percentile", seed)
        top = models[0]
        comp = by_pair.get((top, m))
        rows.append(LeaderboardRow(
            rank, tier, m, float(scores.mean()), ci.low, ci.high, len(scores),
            comp.delta if comp else 0.0, padj.get((top, m)),
        ))
    return rows, pairs, adj


def leaderboard_markdown(rows: list[LeaderboardRow], scale: float = 100.0) -> str:
    lines = ["| Rank | Tier | Model | Score | 95% CI | Δ vs top | p (adj) | n |",
             "|---:|---:|---|---:|---|---:|---:|---:|"]
    for r in rows:
        p = "—" if r.p_vs_top_adj is None else (
            "<0.001" if r.p_vs_top_adj < 0.001 else f"{r.p_vs_top_adj:.3f}")
        lines.append(
            f"| {r.rank} | {r.tier} | {r.model} | {r.mean * scale:.1f} | "
            f"[{r.ci_low * scale:.1f}, {r.ci_high * scale:.1f}] | "
            f"{r.delta_vs_top * scale:+.1f} | {p} | {r.n} |"
        )
    return "\n".join(lines)
