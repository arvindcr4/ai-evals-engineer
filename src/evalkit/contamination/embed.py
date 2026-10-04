"""Embeddings for paraphrase-level contamination.

:class:`HashedEmbedder` is dependency-free: content-word unigrams and
character 4-grams (a cheap stemming proxy) are feature-hashed with a sign bit
into ``dim`` buckets, weighted by sublinear TF times an IDF fitted on the eval
set, and L2-normalised. :class:`OpenAIEmbedder` calls any OpenAI-compatible
``/embeddings`` endpoint when real semantic embeddings are wanted.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import numpy as np

from evalkit.contamination.text import token_hash, tokenize

_STOPWORDS_TEXT = """
a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those there here what which who whom whose when where why how do
does did done can could should would will shall may might must not no so than then too
very just also into about over under up down out off again more most such only own same
other some any each both few all i you he she we they me him her us them my your his our
their s t
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


class Embedder(Protocol):
    name: str
    cosine_thresholds: tuple[float, float]

    def fit(self, texts: Sequence[str]) -> None: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class HashedEmbedder:
    """Feature-hashed TF-IDF bag of words and character n-grams.

    Each content word maps to a cached sparse vector (its own feature plus its
    character n-grams, signed and IDF-weighted); a text is the sum of its word
    vectors scaled by sublinear term frequency.

    ``cosine_thresholds`` are the (suspicious, contaminated) defaults: lexical
    embeddings score paraphrases far lower than neural ones (~0.45 vs ~0.9),
    so the bar is set per embedder.
    """

    dim: int = 4096
    char_n: int = 4
    name: str = "hashed"
    cosine_thresholds: tuple[float, float] = (0.4, 0.8)
    cache_size: int = 200_000
    idf: dict[int, float] = field(default_factory=dict)
    default_idf: float = 1.0

    def word_features(self, word: str) -> list[str]:
        """The word itself plus its padded character n-grams.

        Word bigrams are deliberately absent: paraphrases rarely keep them, so
        they add mass that only exact copies share (the n-gram detector's job).
        """
        padded = f"<{word}>"
        grams = [padded[i : i + self.char_n] for i in range(len(padded) - self.char_n + 1)]
        return [f"w:{word}", *(f"c:{g}" for g in grams if len(padded) > self.char_n)]

    def words(self, text: str) -> Counter[str]:
        return Counter(w for w in tokenize(text) if w not in STOPWORDS)

    def fit(self, texts: Sequence[str]) -> None:
        """Fit IDF on the reference (eval) collection; unseen features get max IDF."""
        df: Counter[int] = Counter()
        for text in texts:
            df.update({self._bucket(f)[0] for w in self.words(text) for f in self.word_features(w)})
        n = len(texts)
        self.idf = {b: math.log((1 + n) / (1 + c)) + 1.0 for b, c in df.items()}
        self.default_idf = math.log(1 + n) + 1.0
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = token_hash(feature)
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def _word_vector(self, word: str) -> tuple[np.ndarray, np.ndarray]:
        """Sparse (indices, signed IDF weights) for one word, cached."""
        cache = self.__dict__.setdefault("_cache", {})
        hit = cache.get(word)
        if hit is None:
            pairs = [self._bucket(f) for f in self.word_features(word)]
            idx = np.fromiter((b for b, _ in pairs), dtype=np.int64, count=len(pairs))
            val = np.fromiter(
                (sg * self.idf.get(b, self.default_idf) for b, sg in pairs),
                dtype=np.float32, count=len(pairs),
            )
            if len(cache) >= self.cache_size:
                cache.clear()
            hit = cache[word] = (idx, val)
        return hit

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Sum of per-word vectors scaled by sublinear TF, L2-normalised."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = self.words(text)
            if not counts:
                continue
            parts = [self._word_vector(w) for w in counts]
            tf = np.repeat(
                np.fromiter((1.0 + math.log(c) for c in counts.values()), dtype=np.float32),
                [len(i) for i, _ in parts],
            )
            out[row] = np.bincount(
                np.concatenate([i for i, _ in parts]),
                weights=np.concatenate([v for _, v in parts]) * tf,
                minlength=self.dim,
            )
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)


@dataclass
class OpenAIEmbedder:
    """Any OpenAI-compatible ``/embeddings`` endpoint, batched."""

    model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    batch_size: int = 128
    timeout: float = 60.0
    cosine_thresholds: tuple[float, float] = (0.85, 0.95)

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        key = self.api_key or os.environ.get("EVALKIT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            r = httpx.post(
                f"{self.base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": self.model, "input": list(texts[i : i + self.batch_size])},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            rows.extend(d["embedding"] for d in data)
        arr = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)


def get_embedder(spec: str | None = None) -> Embedder:
    """``hashed`` / ``hashed:<dim>`` → :class:`HashedEmbedder`;
    ``openai:<model>`` or ``<base_url>|<model>`` → :class:`OpenAIEmbedder`."""
    spec = spec or "hashed"
    if spec == "hashed" or spec.startswith("hashed:"):
        return HashedEmbedder(dim=int(spec.split(":", 1)[1]) if ":" in spec else 4096)
    if spec.startswith("openai:"):
        return OpenAIEmbedder(model=spec.split(":", 1)[1])
    if "|" in spec:
        base, model = spec.split("|", 1)
        return OpenAIEmbedder(model=model, base_url=base)
    raise ValueError(f"unknown embedder spec: {spec!r}")
