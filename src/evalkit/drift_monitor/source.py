"""Traffic sources and deterministic sampling.

A source yields one day's request records. ``JsonlDirSource`` reads a log
directory partitioned by date (``<dir>/YYYY-MM-DD.jsonl``). Sampling is
bottom-k hash sampling: a record is kept when ``hash(salt, id)`` falls under
``rate``, and an optional cap keeps the k smallest hashes, so the sample is
a uniform, reproducible reservoir that never depends on file order.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

from evalkit.core.llm import stable_hash

_SPACE = 2**64


class TrafficSource(Protocol):
    def dates(self) -> list[str]: ...

    def read(self, day: str) -> Iterator[dict]: ...


class JsonlDirSource:
    """Logs stored as ``<root>/YYYY-MM-DD.jsonl``, one request per line."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def dates(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("????-??-??.jsonl"))

    def read(self, day: str) -> Iterator[dict]:
        path = self.root / f"{day}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"no traffic log for {day}: {path}")
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def hash_sample(
    records: Iterator[dict] | list[dict],
    rate: float,
    salt: str = "drift-v1",
    cap: int | None = None,
    key: str = "request_id",
) -> tuple[list[dict], int]:
    """Return ``(sample, total_seen)`` using bottom-k hash sampling."""
    threshold = int(rate * _SPACE)
    kept: list[tuple[int, dict]] = []
    total = 0
    for rec in records:
        total += 1
        h = stable_hash(salt, str(rec.get(key, total)))
        if h < threshold:
            kept.append((h, rec))
    kept.sort(key=lambda t: t[0])
    if cap is not None:
        kept = kept[:cap]
    return [r for _, r in kept], total


def date_range(start: str, end: str) -> list[str]:
    """Inclusive list of ISO dates from ``start`` to ``end``."""
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if d1 < d0:
        raise ValueError(f"end {end} before start {start}")
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]
