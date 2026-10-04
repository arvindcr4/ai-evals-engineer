"""All-pairs (strength-2) covering arrays.

Given factors with discrete levels, build a small set of rows such that every
pair of levels from any two factors appears together in at least one row. The
greedy construction is AETG-style: each new row starts from a still-uncovered
pair, then fills the remaining factors one by one with the level that covers
the most new pairs, keeping the best of several seeded candidates.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

Pair = tuple[tuple[int, int], tuple[int, int]]  # ((factor_i, level_i), (factor_j, level_j))


def all_pairs(sizes: Sequence[int]) -> set[Pair]:
    """Every factor-level pair that a strength-2 covering array must hit."""
    out: set[Pair] = set()
    for i, j in itertools.combinations(range(len(sizes)), 2):
        for a in range(sizes[i]):
            for b in range(sizes[j]):
                out.add(((i, a), (j, b)))
    return out


def pairs_of(row: Sequence[int]) -> set[Pair]:
    return {((i, row[i]), (j, row[j])) for i, j in itertools.combinations(range(len(row)), 2)}


def covering_array(sizes: Sequence[int], seed: int = 0, candidates: int = 20) -> list[list[int]]:
    """Greedy pairwise covering array; returns rows of level indices."""
    if any(s < 1 for s in sizes):
        raise ValueError("every factor needs at least one level")
    if len(sizes) < 2:
        return [[a] for a in range(sizes[0])] if sizes else []
    rng = random.Random(seed)
    uncovered = all_pairs(sizes)
    rows: list[list[int]] = []
    while uncovered:
        best_row, best_gain = None, -1
        ordered = sorted(uncovered)
        for _ in range(candidates):
            (fi, li), (fj, lj) = ordered[rng.randrange(len(ordered))]
            row: list[int | None] = [None] * len(sizes)
            row[fi], row[fj] = li, lj
            rest = [k for k in range(len(sizes)) if row[k] is None]
            rng.shuffle(rest)
            for k in rest:
                scores = []
                for lvl in range(sizes[k]):
                    gain = sum(
                        1
                        for m, v in enumerate(row)
                        if v is not None and _key(m, v, k, lvl) in uncovered
                    )
                    scores.append((gain, -rng.random(), lvl))
                row[k] = max(scores)[2]
            full = [int(v) for v in row]  # type: ignore[arg-type]
            gain = len(pairs_of(full) & uncovered)
            if gain > best_gain:
                best_row, best_gain = full, gain
        assert best_row is not None
        rows.append(best_row)
        uncovered -= pairs_of(best_row)
    return rows


def _key(m: int, v: int, k: int, lvl: int) -> Pair:
    return ((m, v), (k, lvl)) if m < k else ((k, lvl), (m, v))


def pairwise_coverage(sizes: Sequence[int], rows: Sequence[Sequence[int]]) -> float:
    """Fraction of all level pairs hit by ``rows`` (1.0 for fewer than two factors)."""
    universe = all_pairs(sizes)
    if not universe:
        return 1.0
    hit: set[Pair] = set()
    for r in rows:
        hit |= pairs_of(r)
    return len(hit & universe) / len(universe)
