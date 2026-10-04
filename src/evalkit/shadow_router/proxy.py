"""The shadow proxy: primary on the latency path, candidate in the background.

``ShadowRouter`` owns the routing logic and is framework-free so it can be
tested directly; ``create_app`` wraps it in a FastAPI app exposing
``POST /v1/chat/completions``. Sampling is a deterministic hash of a sticky
key (user id, else request id, else the conversation), so the same user is
always in or out of the shadow cohort and reruns are reproducible.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from evalkit.core.llm import LLM, Completion, Message, stable_hash

try:
    from fastapi import Request
except ImportError:  # pragma: no cover - only the proxy needs fastapi
    Request = Any  # type: ignore[assignment,misc]

log = logging.getLogger("evalkit.shadow_router")

PASSTHROUGH_PARAMS = ("temperature", "top_p", "max_tokens", "stop", "seed")
_BUCKETS = 1_000_000


@dataclass
class ShadowConfig:
    rate: float = 0.05
    salt: str = "shadow-v1"
    log_path: str | Path = "shadow_pairs.jsonl"
    candidate_timeout_s: float = 30.0
    max_workers: int = 4


@dataclass
class Side:
    """One model's half of a shadow pair."""

    model: str
    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None

    @classmethod
    def from_completion(cls, c: Completion) -> Side:
        return cls(c.model, c.text, c.tokens_in, c.tokens_out, c.latency_s, c.cost_usd)


@dataclass
class RouterStats:
    requests: int = 0
    sampled: int = 0
    primary_errors: int = 0
    shadow_errors: int = 0
    shadow_completed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, **deltas: int) -> None:
        with self.lock:
            for k, v in deltas.items():
                setattr(self, k, getattr(self, k) + v)

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "lock"}


def sample_key(body: Mapping[str, Any], headers: Mapping[str, str] | None = None) -> str:
    """Sticky sampling key: user id beats request id beats conversation hash."""
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    for candidate in (headers.get("x-user-id"), body.get("user"), headers.get("x-request-id")):
        if candidate:
            return str(candidate)
    convo = json.dumps(body.get("messages", []), sort_keys=True)
    return f"conv:{stable_hash(convo):016x}"


def should_shadow(key: str, rate: float, salt: str = "shadow-v1") -> bool:
    """Deterministic Bernoulli(rate) on ``key``; stable across processes."""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    return stable_hash(salt, key) % _BUCKETS < int(rate * _BUCKETS)


class PairLogger:
    """Thread-safe append-only JSONL sink for shadow pairs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, default=str) + "\n"
        with self._lock, open(self.path, "a") as f:
            f.write(line)


class ShadowRouter:
    """Serve from ``primary``; mirror sampled requests to ``candidate``.

    The candidate call runs on a worker pool after the primary result is in
    hand, so it never adds latency to the user's response. Candidate
    exceptions and timeouts are caught, logged and recorded in the pair.
    """

    def __init__(self, primary: LLM, candidate: LLM, config: ShadowConfig | None = None):
        self.primary = primary
        self.candidate = candidate
        self.config = config or ShadowConfig()
        self.logger = PairLogger(self.config.log_path)
        self.stats = RouterStats()
        self._orchestrator = ThreadPoolExecutor(self.config.max_workers, "shadow-orch")
        self._calls = ThreadPoolExecutor(self.config.max_workers, "shadow-call")
        self._pending: set[Future] = set()
        self._pending_lock = threading.Lock()

    def handle(
        self, body: Mapping[str, Any], headers: Mapping[str, str] | None = None
    ) -> tuple[Completion, str]:
        """Answer one chat request from the primary; maybe enqueue a shadow call."""
        lower = {k.lower(): v for k, v in (headers or {}).items()}
        request_id = lower.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
        messages: list[Message] = list(body.get("messages", []))
        params = {k: body[k] for k in PASSTHROUGH_PARAMS if k in body}
        key = sample_key(body, headers)
        self.stats.bump(requests=1)
        try:
            primary = self.primary.complete(messages, **params)
        except Exception:
            self.stats.bump(primary_errors=1)
            raise
        if should_shadow(key, self.config.rate, self.config.salt):
            self.stats.bump(sampled=1)
            fut = self._orchestrator.submit(
                self._shadow, request_id, key, messages, params, primary
            )
            with self._pending_lock:
                self._pending.add(fut)
            fut.add_done_callback(self._discard)
        return primary, request_id

    def _discard(self, fut: Future) -> None:
        with self._pending_lock:
            self._pending.discard(fut)

    def _shadow(
        self,
        request_id: str,
        key: str,
        messages: list[Message],
        params: dict,
        primary: Completion,
    ) -> None:
        shadow = Side(model=getattr(self.candidate, "model", "candidate"))
        timeout = self.config.candidate_timeout_s
        try:
            result = self._calls.submit(self.candidate.complete, messages, **params).result(
                timeout=timeout
            )
            shadow = Side.from_completion(result)
            if result.latency_s > timeout:
                shadow.error = f"timeout: {result.latency_s:.2f}s > {timeout:.2f}s"
        except FutureTimeout:
            shadow.error = f"timeout: no response within {timeout:.2f}s"
            shadow.latency_s = timeout
        except Exception as exc:  # noqa: BLE001 - candidate failures must never escape
            shadow.error = f"{type(exc).__name__}: {exc}"
        if shadow.error:
            self.stats.bump(shadow_errors=1)
            log.warning("shadow call failed for %s: %s", request_id, shadow.error)
        record = {
            "ts": time.time(),
            "request_id": request_id,
            "sample_key": key,
            "messages": messages,
            "primary": asdict(Side.from_completion(primary)),
            "shadow": asdict(shadow),
        }
        try:
            self.logger.write(record)
            self.stats.bump(shadow_completed=1)
        except OSError:
            log.exception("could not write shadow pair %s", request_id)

    def flush(self, timeout: float | None = None) -> None:
        """Block until every in-flight shadow call has been logged."""
        with self._pending_lock:
            pending = list(self._pending)
        for fut in pending:
            fut.result(timeout=timeout)

    def close(self) -> None:
        self.flush()
        self._orchestrator.shutdown(wait=True)
        self._calls.shutdown(wait=False, cancel_futures=True)


def completion_response(c: Completion, request_id: str) -> dict:
    """Render a ``Completion`` as an OpenAI ``chat.completion`` object."""
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": c.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": c.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": c.tokens_in,
            "completion_tokens": c.tokens_out,
            "total_tokens": c.tokens_in + c.tokens_out,
        },
    }


def create_app(router: ShadowRouter):
    """FastAPI app: ``/v1/chat/completions``, ``/shadow/stats``, ``/healthz``."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from starlette.concurrency import run_in_threadpool

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        router.close()

    app = FastAPI(title="evalkit shadow router", lifespan=lifespan)
    app.state.router = router

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or not body.get("messages"):
            raise HTTPException(status_code=400, detail="body must include 'messages'")
        try:
            completion, request_id = await run_in_threadpool(
                router.handle, body, dict(request.headers)
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"primary failed: {exc}") from exc
        return completion_response(completion, request_id)

    @app.get("/shadow/stats")
    def stats() -> dict:
        return {"rate": router.config.rate, **router.stats.snapshot()}

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app
