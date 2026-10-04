"""Judge prompts, verdict parsing, and a simulated judge with dial-in biases.

:class:`PairwiseJudge` and :class:`PointwiseJudge` work with any :class:`LLM`.
:class:`SimulatedJudge` is a ``MockLLM`` responder that reads the same prompts
a real judge would and answers like a flawed one: it rewards substance, but
also length, the first slot, and its own family's style, by amounts you set.
Because the injected biases are known, tests can check that the audit
recovers them and that calibration removes them.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from evalkit.calibrated_judge.anchors import FACT_SET, FAMILY_STYLE, Anchor
from evalkit.core.llm import LLM, Message, MockLLM, stable_hash

PAIRWISE_SYSTEM = (
    "You are an impartial evaluator. Compare two assistant responses to the same question. "
    "Judge helpfulness, correctness and completeness. Do not let response order, length or "
    "style influence you."
)
PAIRWISE_TEMPLATE = """[Question]
{prompt}

[Response A]
{a}
[End of Response A]

[Response B]
{b}
[End of Response B]

Which response is better? Reply with JSON only:
{{"winner": "A" | "B" | "tie", "confidence": <probability your verdict is right, 0.5-1.0>}}"""

POINTWISE_SYSTEM = "You are a strict evaluator. Rate the response from 1 (useless) to 10 (ideal)."
POINTWISE_TEMPLATE = """[Question]
{prompt}

[Response]
{response}
[End of Response]

Reply with JSON only: {{"score": <integer 1-10>}}"""


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def parse_pairwise(text: str) -> tuple[str, float]:
    """Parse ``(winner in {"A","B","tie"}, confidence)`` from a judge reply."""
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            w = str(d.get("winner", "")).strip().upper()
            conf = float(d.get("confidence", 0.75))
            if w in ("A", "B", "TIE"):
                return ("tie" if w == "TIE" else w), min(max(conf, 0.5), 1.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    m = re.search(r"\[\[(A|B|C|tie)\]\]|winner\W+(A|B|tie)\b", text, re.IGNORECASE)
    if m:
        w = (m.group(1) or m.group(2)).upper()
        return ("tie" if w in ("C", "TIE") else w), 0.75
    return "tie", 0.5


def parse_score(text: str) -> float | None:
    m = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', text) or re.search(r"\b(10|[1-9])\b", text)
    return float(m.group(1)) if m else None


@dataclass
class Verdict:
    """A verdict mapped back to the anchor's own A/B labels."""

    winner: str
    p_a: float
    confidence: float
    swapped: bool
    raw: str = ""


class PairwiseJudge:
    def __init__(self, llm: LLM, system: str = PAIRWISE_SYSTEM,
                 template: str = PAIRWISE_TEMPLATE):
        self.llm, self.system, self.template = llm, system, template

    def judge(self, prompt: str, response_a: str, response_b: str,
              swap: bool = False) -> Verdict:
        """Judge A vs B; with ``swap`` B is shown first and the verdict is mapped back."""
        first, second = (response_b, response_a) if swap else (response_a, response_b)
        messages: list[Message] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.template.format(prompt=prompt, a=first, b=second)},
        ]
        raw = self.llm.complete(messages, temperature=0).text
        slot, conf = parse_pairwise(raw)
        if slot == "tie":
            return Verdict("tie", 0.5, conf, swap, raw)
        picked_first = slot == "A"
        winner = "a" if picked_first != swap else "b"
        p_a = conf if winner == "a" else 1 - conf
        return Verdict(winner, p_a, conf, swap, raw)

    def judge_anchor(self, anchor: Anchor, swap: bool = False) -> Verdict:
        return self.judge(anchor.prompt, anchor.response_a, anchor.response_b, swap)


class PointwiseJudge:
    def __init__(self, llm: LLM, system: str = POINTWISE_SYSTEM,
                 template: str = POINTWISE_TEMPLATE):
        self.llm, self.system, self.template = llm, system, template

    def score(self, prompt: str, response: str) -> float | None:
        messages: list[Message] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.template.format(prompt=prompt, response=response)},
        ]
        return parse_score(self.llm.complete(messages, temperature=0).text)


_BLOCK = re.compile(r"\[Response( [AB])?\]\n(.*?)\n\[End of Response", re.DOTALL)


@dataclass
class SimulatedJudge:
    """Deterministic biased judge, usable as a ``MockLLM`` responder.

    Pairwise logit for "first slot wins" is
    ``skill·Δfacts + verbosity_bias·log(len1/len2) + position_bias
    + self_bias·(own1 − own2) + noise·ε``; ``overconfidence`` > 1 makes the
    stated confidence more extreme than the true accuracy warrants.
    """

    family: str = "nova"
    skill: float = 0.7
    position_bias: float = 1.0
    verbosity_bias: float = 2.5
    self_bias: float = 1.5
    noise: float = 0.8
    overconfidence: float = 2.0
    tie_band: float = 0.25
    seed: int = 0

    def _features(self, text: str) -> tuple[int, int, int]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        facts = sum(s in FACT_SET for s in sentences)
        own = int(text.startswith(FAMILY_STYLE.get(self.family, "\x00")))
        return facts, len(text.split()), own

    def _eps(self, prompt: str) -> float:
        rng = np.random.default_rng(stable_hash(str(self.seed), prompt) % 2**32)
        return float(rng.normal())

    def __call__(self, messages: list[Message]) -> str:
        prompt = messages[-1]["content"]
        blocks = [b[1] for b in _BLOCK.findall(prompt)]
        if len(blocks) == 1:
            facts, words, own = self._features(blocks[0])
            z = (self.skill * (facts - 3) + self.verbosity_bias * math.log(words / 60)
                 + self.self_bias * own + self.noise * self._eps(prompt))
            return json.dumps({"score": int(np.clip(round(1 + 9 * _sigmoid(z)), 1, 10))})
        if len(blocks) != 2:
            return '{"winner": "tie", "confidence": 0.5}'
        (f1, w1, o1), (f2, w2, o2) = self._features(blocks[0]), self._features(blocks[1])
        z = (self.skill * (f1 - f2) + self.verbosity_bias * math.log(w1 / w2)
             + self.position_bias + self.self_bias * (o1 - o2) + self.noise * self._eps(prompt))
        if abs(z) < self.tie_band:
            return json.dumps({"winner": "tie", "confidence": 0.55})
        conf = _sigmoid(self.overconfidence * abs(z))
        return json.dumps({"winner": "A" if z > 0 else "B", "confidence": round(conf, 3)})

    def as_llm(self) -> MockLLM:
        return MockLLM(model=f"{self.family}-judge-sim", responder=self)


def sim_judge_from_args(args: Any) -> SimulatedJudge:
    return SimulatedJudge(
        family=args.judge_family, position_bias=args.sim_position_bias,
        verbosity_bias=args.sim_verbosity_bias, self_bias=args.sim_self_bias,
        noise=args.sim_noise, overconfidence=args.sim_overconfidence, seed=args.seed,
    )
