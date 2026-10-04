"""MinHash signatures and LSH banding with numpy.

Signatures use ``num_perm`` universal hash functions ``(a*x + b) mod p`` over
31-bit shingle hashes with the Mersenne prime ``p = 2**31 - 1``, so every
product fits in uint64 without overflow. :class:`LSHIndex` splits signatures
into ``bands`` x ``rows`` and buckets eval items by band; any training window
sharing one bucket is a candidate pair, which is then verified on the full
signature.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from evalkit.contamination.text import ngram_hashes

_MERSENNE = np.uint64((1 << 31) - 1)
_EMPTY = np.uint64((1 << 32) - 1)


@dataclass
class MinHasher:
    """Word-shingle MinHash. ``shingle`` is the word n-gram size."""

    num_perm: int = 128
    shingle: int = 3
    seed: int = 1

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._a = rng.integers(1, int(_MERSENNE), size=self.num_perm, dtype=np.uint64)
        self._b = rng.integers(0, int(_MERSENNE), size=self.num_perm, dtype=np.uint64)

    def shingles(self, tokens: Sequence[str]) -> np.ndarray:
        """Unique 31-bit shingle hashes (falls back to the whole text if short)."""
        n = min(self.shingle, len(tokens))
        if n == 0:
            return np.empty(0, dtype=np.uint64)
        h = ngram_hashes(tokens, n) & _MERSENNE
        return np.unique(h)

    def signature_of(self, shingles: np.ndarray) -> np.ndarray:
        if shingles.size == 0:
            return np.full(self.num_perm, _EMPTY, dtype=np.uint64)
        x = shingles[None, :]
        values = (self._a[:, None] * x + self._b[:, None]) % _MERSENNE
        return values.min(axis=1)

    def signature(self, tokens: Sequence[str]) -> np.ndarray:
        return self.signature_of(self.shingles(tokens))


def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimated Jaccard similarity: fraction of agreeing signature slots."""
    if sig_a[0] == _EMPTY or sig_b[0] == _EMPTY:
        return 0.0
    return float(np.mean(sig_a == sig_b))


def containment(j: float, size_a: int, size_b: int) -> float:
    """Estimate ``|A ∩ B| / |A|`` from Jaccard and set sizes.

    ``|A ∩ B| = J (|A| + |B|) / (1 + J)``. Containment of the eval item in a
    training window is what matters when the window is longer than the item.
    """
    if size_a == 0:
        return 0.0
    inter = j * (size_a + size_b) / (1.0 + j)
    return float(min(1.0, inter / size_a))


@dataclass
class LSHIndex:
    """Band-bucket index over eval-item signatures."""

    bands: int = 64
    rows: int = 2
    _buckets: list[dict[bytes, list[int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._buckets = [defaultdict(list) for _ in range(self.bands)]

    def _keys(self, sig: np.ndarray) -> list[bytes]:
        if len(sig) < self.bands * self.rows:
            raise ValueError(
                f"signature length {len(sig)} < bands*rows = {self.bands * self.rows}"
            )
        return [sig[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def add(self, key: int, sig: np.ndarray) -> None:
        if sig[0] == _EMPTY:
            return
        for band, k in enumerate(self._keys(sig)):
            self._buckets[band][k].append(key)

    def query(self, sig: np.ndarray) -> set[int]:
        if sig[0] == _EMPTY:
            return set()
        out: set[int] = set()
        for band, k in enumerate(self._keys(sig)):
            hit = self._buckets[band].get(k)
            if hit:
                out.update(hit)
        return out

    def candidate_probability(self, j: float) -> float:
        """Chance a pair with Jaccard ``j`` collides in at least one band."""
        return 1.0 - (1.0 - j**self.rows) ** self.bands
