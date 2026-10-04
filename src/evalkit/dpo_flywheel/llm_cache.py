"""Metering + persistent completion cache for the flywheel's teacher and judge.

Real-model runs exposed two problems the offline mock hid:

* The nightly job rebuilds the *cumulative* dataset, so without a cache every
  thumbs-down ever seen is re-sent to the teacher and judge every night: spend
  grows linearly with history, and because the teacher samples at T=0.8 the
  dataset (and its ``data_sha256``) changes run to run even with no new data.
* Nothing recorded what a run cost.

``MeteredLLM`` wraps any :class:`~evalkit.core.llm.LLM`. Identical requests
(model + messages + kwargs) are answered from an append-only JSONL cache, and
calls, cache hits, tokens and ``cost_usd`` are tallied. It is thread-safe so the
pair builder can fan out over a worker pool.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from evalkit.core.llm import LLM, Completion, Message


def request_key(model: str, messages: list[Message], kwargs: dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class MeteredLLM:
    inner: LLM
    cache_path: Path | None = None
    usage: Usage = field(default_factory=Usage)
    _cache: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cache_path is not None:
            self.cache_path = Path(self.cache_path)
            if self.cache_path.exists():
                for line in self.cache_path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                        self._cache[row["key"]] = row
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue  # a torn last line from a killed run is not fatal

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "")

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        key = request_key(self.model, messages, kwargs)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self.usage.cache_hits += 1
                return Completion(text=hit["text"], model=self.model, raw={"cached": True})
        try:
            out = self.inner.complete(messages, **kwargs)
        except Exception:
            with self._lock:
                self.usage.errors += 1
            raise
        with self._lock:
            self.usage.calls += 1
            self.usage.tokens_in += out.tokens_in
            self.usage.tokens_out += out.tokens_out
            self.usage.cost_usd += out.cost_usd
            row = {"key": key, "model": self.model, "text": out.text}
            self._cache[key] = row
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
        return out


def usage_of(llm: object) -> dict | None:
    u = getattr(llm, "usage", None)
    return u.to_dict() if isinstance(u, Usage) else None
