"""Minimal LLM abstraction.

Every system in evalkit talks to models through :class:`LLM`. ``MockLLM`` is
deterministic and offline so tests and demos run without keys;
``OpenAICompatibleLLM`` hits any OpenAI-style ``/chat/completions`` endpoint
(OpenAI, DeepSeek, vLLM, Ollama, OpenRouter...).
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

Message = dict[str, str]


@dataclass
class Completion:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    raw: dict = field(default_factory=dict)


class LLM(Protocol):
    model: str

    def complete(self, messages: list[Message], **kwargs) -> Completion: ...


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def stable_hash(*parts: str) -> int:
    """Deterministic 64-bit hash (Python's hash() is salted per process)."""
    h = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


@dataclass
class MockLLM:
    """Deterministic offline model.

    ``responder`` maps the message list to a reply; by default it echoes a
    stable pseudo-answer derived from the prompt hash. Prices are per 1M tokens.
    """

    model: str = "mock-1"
    responder: Callable[[list[Message]], str] | None = None
    price_in: float = 0.15
    price_out: float = 0.60
    latency_s: float = 0.05

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        prompt = "\n".join(m["content"] for m in messages)
        if self.responder is not None:
            text = self.responder(messages)
        else:
            text = f"[{self.model}] answer-{stable_hash(self.model, prompt) % 1000:03d}"
        tin, tout = _approx_tokens(prompt), _approx_tokens(text)
        return Completion(
            text=text,
            model=self.model,
            tokens_in=tin,
            tokens_out=tout,
            latency_s=self.latency_s,
            cost_usd=(tin * self.price_in + tout * self.price_out) / 1e6,
        )


@dataclass
class OpenAICompatibleLLM:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    price_in: float = 0.0
    price_out: float = 0.0
    timeout: float = 120.0
    extra_body: dict = field(default_factory=dict)
    max_retries: int = 4

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        key = self.api_key or os.environ.get("EVALKIT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        body = {"model": self.model, "messages": messages, **self.extra_body, **kwargs}
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                r = httpx.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                    timeout=self.timeout,
                )
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                retryable = isinstance(e, httpx.TransportError) or (
                    e.response.status_code in (429, 500, 502, 503, 504)
                )
                if not retryable or attempt >= self.max_retries:
                    raise
                time.sleep(min(30.0, 2.0**attempt))
        data = r.json()
        usage = data.get("usage", {})
        tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            model=self.model,
            tokens_in=tin,
            tokens_out=tout,
            latency_s=time.perf_counter() - t0,
            cost_usd=(tin * self.price_in + tout * self.price_out) / 1e6,
            raw=data,
        )


# USD per 1M tokens (cache-miss input, output) at DeepSeek peak rates, Oct 2026.
DEEPSEEK_PRICES = {
    "deepseek-flash": (0.30, 1.20),
    "deepseek-v4-pro": (1.32, 3.96),
}


def get_llm(spec: str | None = None) -> LLM:
    """Build an LLM from a spec string.

    ``mock`` / ``mock:<name>`` → MockLLM; ``openai:<model>`` → OpenAI;
    ``deepseek:<model>`` → DeepSeek; ``<base_url>|<model>`` → any compatible server.
    Falls back to ``$EVALKIT_LLM`` then ``mock``.
    """
    spec = spec or os.environ.get("EVALKIT_LLM", "mock")
    if spec == "mock" or spec.startswith("mock:"):
        return MockLLM(model=spec.split(":", 1)[1] if ":" in spec else "mock-1")
    if spec.startswith("openai:"):
        return OpenAICompatibleLLM(model=spec.split(":", 1)[1])
    if spec.startswith("deepseek:"):
        # ``deepseek:deepseek-flash`` runs with thinking off (fast, cheap);
        # append ``+think`` to enable the reasoning mode.
        model = spec.split(":", 1)[1]
        think = model.endswith("+think")
        model = model.removesuffix("+think")
        price_in, price_out = DEEPSEEK_PRICES.get(model, (0.0, 0.0))
        return OpenAICompatibleLLM(
            model=model,
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            price_in=price_in,
            price_out=price_out,
            extra_body={"thinking": {"type": "enabled" if think else "disabled"}},
        )
    if "|" in spec:
        base, model = spec.split("|", 1)
        return OpenAICompatibleLLM(model=model, base_url=base)
    raise ValueError(f"unknown LLM spec: {spec!r}")
