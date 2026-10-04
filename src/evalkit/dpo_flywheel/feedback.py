"""Feedback ingestion: thumbs-up/down events, an append-only store and a POST endpoint.

An event is one rated assistant turn::

    {"conversation_id": "c1", "messages": [{"role": "user", "content": "..."}],
     "response": "...", "rating": "down", "correction": "...", "model": "m", "ts": "..."}

``messages`` is the context the model saw; ``response`` is what it said. The
store is a JSONL file that only grows, so a line offset is a stable cursor for
"events since the last nightly run".
"""

from __future__ import annotations

import fcntl
import json
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalkit.core.llm import Message, stable_hash

_UP = {"up", "thumbs_up", "+1", "1", "positive", "good"}
_DOWN = {"down", "thumbs_down", "-1", "0", "negative", "bad"}


def normalize_rating(value: Any) -> str:
    """Map the many spellings of a thumb rating onto ``"up"`` / ``"down"``."""
    if isinstance(value, bool):
        return "up" if value else "down"
    key = str(value).strip().lower()
    if key in _UP:
        return "up"
    if key in _DOWN:
        return "down"
    raise ValueError(f"unrecognised rating: {value!r}")


@dataclass
class FeedbackEvent:
    conversation_id: str
    messages: list[Message]
    response: str
    rating: str
    correction: str | None = None
    model: str = ""
    ts: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def __post_init__(self) -> None:
        self.rating = normalize_rating(self.rating)
        if not self.conversation_id:
            raise ValueError("conversation_id is required")
        if not isinstance(self.messages, list):
            raise TypeError("messages must be a list of {role, content} objects")
        for m in self.messages:
            if not isinstance(m, dict) or not isinstance(m.get("role"), str) or not isinstance(m.get("content"), str):
                raise TypeError("each message needs string 'role' and 'content'")
        if not any(m["role"] == "user" for m in self.messages):
            raise ValueError("messages must contain at least one user turn")
        if not isinstance(self.response, str):
            raise TypeError("response must be a string")
        if self.correction is not None and not isinstance(self.correction, str):
            raise TypeError("correction must be a string")
        if self.correction is not None and not self.correction.strip():
            self.correction = None
        if not self.ts:
            self.ts = datetime.now(UTC).isoformat(timespec="seconds")
        if not self.event_id:
            self.event_id = f"{stable_hash(self.conversation_id, self.prompt_text, self.response, self.rating):016x}"

    @property
    def prompt_text(self) -> str:
        """The conversation rendered as a single prompt string (TRL "standard" format)."""
        return render_prompt(self.messages)

    @property
    def last_user(self) -> str:
        return next(m["content"] for m in reversed(self.messages) if m["role"] == "user")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FeedbackEvent:
        d = dict(d)
        if "messages" not in d and "prompt" in d:
            d["messages"] = [{"role": "user", "content": d.pop("prompt")}]
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        extra = {k: v for k, v in d.items() if k not in cls.__dataclass_fields__}
        if extra:
            known["meta"] = {**known.get("meta", {}), **extra}
        return cls(**known)


def render_prompt(messages: list[Message]) -> str:
    """Single user turn → its text; multi-turn → ``Role: content`` transcript."""
    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


class FeedbackStore:
    """Append-only JSONL store, safe across threads and processes (``flock``)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, events: Iterable[FeedbackEvent], dedupe: bool = True) -> int:
        """Append events, skipping ids already stored. Returns the number written."""
        events = list(events)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                seen: set[str] = set()
                if dedupe:
                    f.seek(0)
                    seen = {json.loads(line)["event_id"] for line in f if line.strip()}
                f.seek(0, 2)
                n = 0
                for ev in events:
                    if dedupe and ev.event_id in seen:
                        continue
                    seen.add(ev.event_id)
                    f.write(json.dumps(ev.to_dict()) + "\n")
                    n += 1
                f.flush()
                return n
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def read(self, start: int = 0) -> list[FeedbackEvent]:
        """Events from line ``start`` onward (a cursor from a previous run)."""
        if not self.path.exists():
            return []
        with open(self.path) as f:
            rows = [line for line in f if line.strip()]
        return [FeedbackEvent.from_dict(json.loads(r)) for r in rows[start:]]

    def __len__(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            return sum(1 for line in f if line.strip())


def load_events(path: str | Path) -> tuple[list[FeedbackEvent], list[tuple[int, str]]]:
    """Parse a JSONL file of raw events. Returns (valid events, [(line_no, error)])."""
    good: list[FeedbackEvent] = []
    bad: list[tuple[int, str]] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                good.append(FeedbackEvent.from_dict(json.loads(line)))
            except (ValueError, TypeError, KeyError, StopIteration) as e:
                bad.append((i, str(e)))
    return good, bad


def create_app(store: FeedbackStore):
    """FastAPI app with ``POST /feedback``, ``GET /stats`` and ``GET /healthz``."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="evalkit dpo-flywheel feedback")

    @app.post("/feedback", status_code=201)
    def post_feedback(payload: dict) -> dict:
        try:
            ev = FeedbackEvent.from_dict(payload)
        except (ValueError, TypeError, StopIteration) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        written = store.append([ev])
        return {"event_id": ev.event_id, "stored": bool(written), "duplicate": not written}

    @app.get("/stats")
    def stats() -> dict:
        events = store.read()
        down = sum(e.rating == "down" for e in events)
        return {
            "events": len(events),
            "down": down,
            "up": len(events) - down,
            "with_correction": sum(e.correction is not None for e in events),
        }

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app
