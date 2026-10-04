"""Text normalization, hashed word n-grams and streaming record readers.

Normalization follows the usual decontamination recipe: Unicode NFKC, case
folding, every non-alphanumeric run collapsed to a single space. N-gram hashes
are computed vectorised with numpy from cached per-token 64-bit hashes, so the
same n-gram always maps to the same integer across processes.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)
_PRIME = np.uint64(1099511628211)

DEFAULT_EVAL_FIELDS = ("prompt", "question", "input", "text", "answer", "output", "reference")
DEFAULT_TRAIN_FIELDS = ("text", "prompt", "completion", "input", "output", "content")


def normalize(text: str) -> str:
    """NFKC + casefold + collapse punctuation and whitespace to single spaces."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return _NON_WORD.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Normalized whitespace tokens."""
    norm = normalize(text)
    return norm.split() if norm else []


@lru_cache(maxsize=1 << 18)
def token_hash(token: str) -> int:
    """Stable 64-bit token hash (bounded LRU cache keeps memory flat)."""
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")


def token_hashes(tokens: Sequence[str]) -> np.ndarray:
    return np.fromiter((token_hash(t) for t in tokens), dtype=np.uint64, count=len(tokens))


def ngram_hashes(tokens: Sequence[str] | np.ndarray, n: int) -> np.ndarray:
    """Polynomial rolling hash of every contiguous word ``n``-gram, in order.

    Accepts tokens or precomputed token hashes. Returns an empty array when
    there are fewer than ``n`` tokens.
    """
    th = tokens if isinstance(tokens, np.ndarray) else token_hashes(tokens)
    count = len(th) - n + 1
    if count <= 0 or n <= 0:
        return np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = np.zeros(count, dtype=np.uint64)
        for k in range(n):
            h = h * _PRIME + th[k : k + count]
    return h


def windows(tokens: Sequence[str], size: int, stride: int) -> list[tuple[int, list[str]]]:
    """Overlapping token windows ``(start, tokens)`` covering the whole sequence."""
    if len(tokens) <= size:
        return [(0, list(tokens))] if tokens else []
    starts = list(range(0, len(tokens) - size + 1, max(1, stride)))
    if starts[-1] + size < len(tokens):
        starts.append(len(tokens) - size)
    return [(s, list(tokens[s : s + size])) for s in starts]


@dataclass
class Record:
    """One document: an id, its raw text, and the original JSON row (if any)."""

    id: str
    text: str
    raw: dict | None = None


def _field_text(value: object) -> str:
    """Flatten strings, chat messages and OpenAI-style content parts to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(t for t in (_field_text(item) for item in value) if t)
    if isinstance(value, dict):
        if "content" in value:
            return _field_text(value["content"])
        text = value.get("text")
        return text if isinstance(text, str) else ""
    return "" if value is None else str(value)


def record_text(row: dict, fields: Sequence[str] | None, defaults: Sequence[str]) -> str:
    """Join the text of ``fields`` (or the defaults present) plus chat ``messages``."""
    keys = list(fields) if fields else [k for k in defaults if k in row]
    if not fields and "messages" in row:
        keys.append("messages")
    if not fields and not keys:
        keys = [k for k, v in row.items() if isinstance(v, str) and k not in ("id", "_id")]
    return "\n".join(t for t in (_field_text(row.get(k)) for k in keys) if t)


def iter_records(
    path: str | Path,
    fields: Sequence[str] | None = None,
    id_field: str = "id",
    defaults: Sequence[str] = DEFAULT_TRAIN_FIELDS,
) -> Iterator[Record]:
    """Stream records from a JSONL file (or one doc per non-empty line of text).

    JSONL rows may be plain ``{"text": ...}`` or chat format
    ``{"messages": [{"role": ..., "content": ...}]}``. Ids default to
    ``<file name>:<line number>`` when the row has no ``id_field``.
    """
    path = Path(path)
    is_json = path.suffix in (".jsonl", ".json", ".ndjson")
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            fallback = f"{path.name}:{lineno}"
            if not is_json:
                yield Record(fallback, line)
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                text = _field_text(row)
                yield Record(fallback, text, {"text": text})
                continue
            rid = row.get(id_field)
            yield Record(
                str(rid) if rid is not None else fallback,
                record_text(row, fields, defaults),
                row,
            )
