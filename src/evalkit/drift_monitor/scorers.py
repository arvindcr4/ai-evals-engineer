"""Per-record scorers and the daily aggregation they feed.

A scorer maps one traffic record to named numeric values (``None`` means
"not applicable", e.g. tool-error rate for a record with no tool calls).
``aggregate`` turns a day's scored sample into scalar metrics plus the
distributions used for PSI (answer-length histogram, tool-usage mix).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from evalkit.core.llm import LLM, Message, MockLLM

REFUSAL = re.compile(
    r"\b(i'?m sorry|i can(?:no|')t help|i am unable|i'?m unable|cannot assist|"
    r"not able to (?:help|assist)|as an ai)\b",
    re.IGNORECASE,
)
LENGTH_BINS = [0, 1, 100, 250, 500, 1000, 2000]
LENGTH_LABELS = ["empty", "1-99", "100-249", "250-499", "500-999", "1000-1999", "2000+"]


class Scorer(Protocol):
    name: str

    def score(self, rec: dict) -> dict[str, float | None]: ...


def _answer(rec: dict) -> str:
    return str(rec.get("output") or "")


@dataclass
class RefusalScorer:
    name: str = "refusal"

    def score(self, rec: dict) -> dict[str, float | None]:
        return {"refusal": float(bool(REFUSAL.search(_answer(rec))))}


@dataclass
class LengthScorer:
    name: str = "length"

    def score(self, rec: dict) -> dict[str, float | None]:
        ans = _answer(rec).strip()
        return {"answer_chars": float(len(ans)), "empty": float(not ans)}


@dataclass
class ToolScorer:
    name: str = "tools"

    def score(self, rec: dict) -> dict[str, float | None]:
        calls = rec.get("tool_calls") or []
        if not calls:
            return {"tool_error": None, "tool_calls": 0.0}
        errors = sum(1 for c in calls if not c.get("ok", True))
        return {"tool_error": errors / len(calls), "tool_calls": float(len(calls))}


@dataclass
class LatencyCostScorer:
    name: str = "latency_cost"

    def score(self, rec: dict) -> dict[str, float | None]:
        lat, cost = rec.get("latency_s"), rec.get("cost_usd")
        return {
            "latency_s": None if lat is None else float(lat),
            "cost_usd": None if cost is None else float(cost),
        }


JUDGE_PROMPT = """Rate the assistant's answer to the user on a 1-5 scale
(5 = fully correct and helpful, 1 = useless, refused or empty).
Tool errors during the run: {tool_errors}.
Reply with a single digit.

User: {input}

Assistant: {output}
"""


def heuristic_judge(messages: list[Message]) -> str:
    """Offline judge: penalise refusals, empty/short answers and tool errors."""
    body = messages[-1]["content"]
    answer = body.split("\n\nAssistant: ", 1)[-1]
    errors = int(re.search(r"Tool errors during the run: (\d+)", body).group(1))
    score = 5
    if REFUSAL.search(answer) or not answer.strip():
        score = 1
    elif len(answer) < 120:
        score = 3
    score -= min(errors, 2)
    return str(max(1, score))


@dataclass
class JudgeScorer:
    """LLM-judge quality score in [0, 1]; unparseable replies are skipped."""

    llm: LLM = field(default_factory=lambda: MockLLM(model="mock-judge", responder=heuristic_judge))
    name: str = "judge"

    def score(self, rec: dict) -> dict[str, float | None]:
        errors = sum(1 for c in rec.get("tool_calls") or [] if not c.get("ok", True))
        prompt = JUDGE_PROMPT.format(
            tool_errors=errors, input=rec.get("input", ""), output=_answer(rec)
        )
        reply = self.llm.complete([{"role": "user", "content": prompt}]).text
        m = re.search(r"[1-5]", reply)
        return {"judge_score": (int(m.group()) - 1) / 4 if m else None}


def default_scorers(judge: LLM | None = None) -> list[Scorer]:
    scorers: list[Scorer] = [RefusalScorer(), LengthScorer(), ToolScorer(), LatencyCostScorer()]
    if judge is not None:
        scorers.append(JudgeScorer(judge))
    return scorers


@dataclass
class DayAggregate:
    n: int
    metrics: dict[str, float]
    dists: dict[str, dict[str, int]]


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def aggregate(records: list[dict], scorers: list[Scorer]) -> DayAggregate:
    """Score a day's sample and reduce it to metrics and distributions."""
    rows = []
    for rec in records:
        row: dict[str, float | None] = {}
        for s in scorers:
            row.update(s.score(rec))
        rows.append(row)

    def col(name: str) -> list[float | None]:
        return [r.get(name) for r in rows]

    metrics: dict[str, float | None] = {
        "refusal_rate": _mean(col("refusal")),
        "empty_rate": _mean(col("empty")),
        "answer_chars_mean": _mean(col("answer_chars")),
        "tool_error_rate": _mean(col("tool_error")),
        "tool_calls_mean": _mean(col("tool_calls")),
        "cost_usd_mean": _mean(col("cost_usd")),
        "judge_score": _mean(col("judge_score")),
    }
    lat = [v for v in col("latency_s") if v is not None]
    if lat:
        metrics["latency_p50"] = float(np.percentile(lat, 50))
        metrics["latency_p95"] = float(np.percentile(lat, 95))

    lengths = [len(_answer(r).strip()) for r in records]
    idx = np.digitize(lengths, LENGTH_BINS[1:], right=False) if lengths else []
    length_hist = {label: 0 for label in LENGTH_LABELS}
    for i in idx:
        length_hist[LENGTH_LABELS[int(i)]] += 1
    tools = Counter(c.get("name", "?") for r in records for c in r.get("tool_calls") or [])
    return DayAggregate(
        n=len(records),
        metrics={k: v for k, v in metrics.items() if v is not None},
        dists={"answer_length": length_hist, "tool_mix": dict(sorted(tools.items()))},
    )
