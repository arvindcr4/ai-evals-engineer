"""Deterministic offline stand-ins for a prod model and a cheaper candidate.

``demo:primary`` and ``demo:candidate`` specs resolve here so the shadow demo
produces realistic, reproducible diffs without network access: the candidate
agrees verbatim most of the time, paraphrases or truncates sometimes, gives a
different answer occasionally, has a fat latency tail and fails ~3% of calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evalkit.core.llm import LLM, Completion, Message, get_llm, stable_hash

_STOP_WORDS = (
    "a an the is are was were be to of in on for with and or how what why when which who "
    "do does did can could should would i my me you your it its this that please about "
    "from at by as into vs fix best practice check first explain briefly handle small team "
    "break under load issues"
)
_STOP = frozenset(_STOP_WORDS.split())

_FACTS = [
    "start with the smallest change that reproduces the behaviour",
    "measure before and after so the effect is attributable",
    "prefer the documented default unless a benchmark says otherwise",
    "keep the rollback path one command away",
    "check the logs for the first error, not the loudest one",
    "pin versions so the result is reproducible",
]
_DETAILS = [
    "This avoids most surprises in production.",
    "Teams that skip this step usually pay for it later.",
    "It also makes the change easy to review.",
    "Document the decision next to the code.",
]


def _topic(messages: list[Message]) -> str:
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", user.lower()) if w not in _STOP]
    return " ".join(words[:4]) or "this"


class CandidateError(RuntimeError):
    pass


@dataclass
class DemoModel:
    """Template-based model whose behaviour is a pure function of the prompt."""

    role: str = "primary"
    model: str = ""
    price_in: float = 2.50
    price_out: float = 10.0

    def __post_init__(self) -> None:
        if not self.model:
            self.model = "demo-large" if self.role == "primary" else "demo-small"
        if self.role != "primary":
            self.price_in, self.price_out = 0.15, 0.60

    def _primary_text(self, messages: list[Message]) -> str:
        topic = _topic(messages)
        h = stable_hash("fact", topic)
        fact = _FACTS[h % len(_FACTS)]
        detail = _DETAILS[(h // 7) % len(_DETAILS)]
        return f"For {topic}, {fact}. {detail}"

    def complete(self, messages: list[Message], **kwargs) -> Completion:
        prompt = "\n".join(m["content"] for m in messages)
        h = stable_hash(self.role, prompt)
        base = self._primary_text(messages)
        if self.role == "primary":
            text = base
            latency = 0.70 + (h % 600) / 1000 + (1.5 if h % 50 == 0 else 0.0)
        else:
            bucket = h % 100
            if bucket < 3:
                raise CandidateError("upstream 503 from candidate endpoint")
            if bucket < 62:
                text = base
            elif bucket < 82:
                text = "In short: " + base.split(". ")[0] + "."
            elif bucket < 92:
                text = base.replace("For ", "When dealing with ", 1).replace(". ", "; ", 1)
            else:
                topic = _topic(messages)
                text = f"I'm not certain about {topic}; it depends on your setup."
            latency = 0.25 + (h % 400) / 1000 + (6.0 if h % 40 == 7 else 0.0)
        tin, tout = max(1, len(prompt) // 4), max(1, len(text) // 4)
        return Completion(
            text=text,
            model=self.model,
            tokens_in=tin,
            tokens_out=tout,
            latency_s=round(latency, 3),
            cost_usd=(tin * self.price_in + tout * self.price_out) / 1e6,
        )


def resolve_llm(spec: str | None) -> LLM:
    """``demo:primary`` / ``demo:candidate`` → :class:`DemoModel`; else ``get_llm``."""
    if spec and spec.startswith("demo:"):
        return DemoModel(role=spec.split(":", 1)[1] or "primary")
    return get_llm(spec)
