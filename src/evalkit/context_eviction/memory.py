"""Working-memory strategies under a token budget.

Each strategy ingests turns one at a time and, when asked, renders the context
an agent would actually see (optionally for a specific query). The budget
applies to that rendered context; retrieval's external store does not count.

* ``full``      — keep everything (unbounded baseline, violates the budget).
* ``fifo``      — truncate oldest first, the system prompt included.
* ``window``    — pinned system prompt + the newest turns that fit.
* ``summary``   — pinned system prompt + rolling LLM summary of evicted turns + recent turns.
* ``retrieval`` — pinned system prompt + recent turns + top-k turns by hashed-BoW cosine.
"""

from __future__ import annotations

import itertools
import re
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from evalkit.context_eviction.scenario import Turn
from evalkit.core.llm import LLM, Message, MockLLM, stable_hash

_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "i",
        "in", "is", "it", "its", "me", "my", "of", "on", "or", "s", "so", "that", "the", "this",
        "to", "was", "what", "with", "you", "your", "can", "now",
    }
)
FACT_RE = re.compile(
    r"\b(?:my|the user's) ([a-z][a-z ]{1,30}?) (?:is now|has changed to|changed to|is) "
    r"([^.?!\n]+)[.?!]"
)


def count_tokens(text: str) -> int:
    """Whitespace token count — deterministic and model-agnostic."""
    return len(text.split())


def render(turn: Turn) -> str:
    return f"{turn.role}: {turn.text}"


class Memory(ABC):
    name = "base"

    def __init__(self, system: str, budget: int):
        self.system = f"system: {system}"
        self.budget = budget
        self.llm_calls = 0

    @abstractmethod
    def add(self, turn: Turn) -> None: ...

    @abstractmethod
    def context(self, query: str | None = None) -> list[str]:
        """Lines visible to the model, in chronological order."""

    def tokens(self, query: str | None = None) -> int:
        return sum(count_tokens(x) for x in self.context(query))


def _fit_newest(lines: list[str], budget: int) -> list[str]:
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        t = count_tokens(line)
        if used + t > budget:
            break
        kept.append(line)
        used += t
    return kept[::-1]


class FullMemory(Memory):
    name = "full"

    def __init__(self, system: str, budget: int):
        super().__init__(system, budget)
        self.lines: list[str] = []

    def add(self, turn: Turn) -> None:
        self.lines.append(render(turn))

    def context(self, query: str | None = None) -> list[str]:
        return [self.system, *self.lines]


class FIFOMemory(Memory):
    name = "fifo"

    def __init__(self, system: str, budget: int):
        super().__init__(system, budget)
        self.buffer: list[str] = [self.system]
        self.used = count_tokens(self.system)

    def add(self, turn: Turn) -> None:
        line = render(turn)
        self.buffer.append(line)
        self.used += count_tokens(line)
        while self.used > self.budget and len(self.buffer) > 1:
            self.used -= count_tokens(self.buffer.pop(0))

    def context(self, query: str | None = None) -> list[str]:
        return list(self.buffer)


class WindowMemory(FullMemory):
    name = "window"

    def context(self, query: str | None = None) -> list[str]:
        room = self.budget - count_tokens(self.system)
        return [self.system, *_fit_newest(self.lines, room)]


SUMMARY_PROMPT = """You maintain the long-term memory of a personal assistant.
Merge the previous memory notes and the new conversation turns into updated notes.
Keep every durable fact about the user as a sentence "the user's <attribute> is <value>."
When a fact was updated, keep ONLY the latest value. Drop small talk.
Use at most {limit} words.

PREVIOUS NOTES:
{previous}

NEW TURNS:
{turns}

UPDATED NOTES:"""


def extractive_summarizer(messages: list[Message]) -> str:
    """Offline summarizer: keep fact sentences, latest value per attribute wins."""
    prompt = messages[-1]["content"]
    body = prompt.split("PREVIOUS NOTES:", 1)[-1]
    facts: dict[str, str] = {}
    for attr, value in FACT_RE.findall(body):
        facts.pop(attr, None)
        facts[attr] = value.strip()
    return " ".join(f"the user's {a} is {v}." for a, v in facts.items())


class SummaryMemory(Memory):
    name = "summary"

    def __init__(self, system: str, budget: int, llm: LLM | None = None,
                 summary_frac: float = 0.4):
        super().__init__(system, budget)
        self.llm = llm or MockLLM(model="mock-summarizer", responder=extractive_summarizer)
        self.summary = ""
        self.recent: list[str] = []
        self.summary_cap = int(budget * summary_frac)

    def _summary_line(self) -> list[str]:
        return [f"memory: {self.summary}"] if self.summary else []

    def context(self, query: str | None = None) -> list[str]:
        return [self.system, *self._summary_line(), *self.recent]

    def add(self, turn: Turn) -> None:
        self.recent.append(render(turn))
        while self.tokens() > self.budget and self.recent:
            n = max(1, len(self.recent) // 2)
            chunk, self.recent = self.recent[:n], self.recent[n:]
            prompt = SUMMARY_PROMPT.format(limit=self.summary_cap,
                                           previous=self.summary or "(none)",
                                           turns="\n".join(chunk))
            self.summary = self.llm.complete([{"role": "user", "content": prompt}]).text.strip()
            self.llm_calls += 1
            words = self.summary.split()
            if len(words) > self.summary_cap:  # model ignored the limit: keep the newest notes
                self.summary = " ".join(words[-self.summary_cap:])


class HashingEmbedder:
    """Bag-of-words + bigram feature hashing into a unit vector."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def tokens(self, text: str) -> list[str]:
        words = [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
        return words + [f"{a}_{b}" for a, b in itertools.pairwise(words)]

    def __call__(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for tok in self.tokens(text):
            h = stable_hash(tok)
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v


class RetrievalMemory(Memory):
    name = "retrieval"

    def __init__(self, system: str, budget: int, top_k: int = 6, recent_frac: float = 0.3,
                 embedder: Callable[[str], np.ndarray] | None = None):
        super().__init__(system, budget)
        self.top_k, self.recent_frac = top_k, recent_frac
        self.embed = embedder or HashingEmbedder()
        self.lines: list[str] = []
        self.vecs: list[np.ndarray] = []

    def add(self, turn: Turn) -> None:
        line = render(turn)
        self.lines.append(line)
        self.vecs.append(self.embed(line))

    def context(self, query: str | None = None) -> list[str]:
        room = self.budget - count_tokens(self.system)
        recent_room = room if query is None else int(self.budget * self.recent_frac)
        recent = _fit_newest(self.lines, recent_room)
        start = len(self.lines) - len(recent)
        chosen = set(range(start, len(self.lines)))
        used = sum(count_tokens(x) for x in recent)
        if query is not None and start > 0:
            sims = np.stack(self.vecs[:start]) @ self.embed(query)
            for i in np.argsort(-sims, kind="stable")[: self.top_k]:
                t = count_tokens(self.lines[i])
                if sims[i] <= 0 or used + t > room:
                    continue
                chosen.add(int(i))
                used += t
        return [self.system, *(self.lines[i] for i in sorted(chosen))]


STRATEGIES: dict[str, type[Memory]] = {
    m.name: m for m in (FullMemory, FIFOMemory, WindowMemory, SummaryMemory, RetrievalMemory)
}


def make_memory(name: str, system: str, budget: int, llm: LLM | None = None,
                **kwargs) -> Memory:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    if name == "summary":
        return SummaryMemory(system, budget, llm=llm, **kwargs)
    return STRATEGIES[name](system, budget, **kwargs)
