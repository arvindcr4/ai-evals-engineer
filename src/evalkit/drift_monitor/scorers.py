"""Per-record scorers and the daily aggregation they feed.

A scorer maps one traffic record to named numeric values (``None`` means
"not applicable", e.g. tool-error rate for a record with no tool calls).
``aggregate`` turns a day's scored sample into scalar metrics plus the
distributions used for PSI (answer-length histogram, tool-usage mix).
"""

from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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


JUDGE_SYSTEM = (
    "You are a strict quality grader for a business assistant's answers. You never "
    "answer or continue the user's request yourself; you only grade the answer given."
)
JUDGE_PROMPT = """Rate the assistant's answer to the user's request on a 1-5 scale.
5 = directly and specifically answers the request
3 = partially answers it, or is vague
1 = does not address the request (off-topic), refuses, or is empty
You cannot verify company data: assume stated figures are correct and judge
relevance, completeness and helpfulness.
Tool errors during the run: {tool_errors}.

<request>
{input}
</request>

<answer>
{output}
</answer>

Reply with exactly one line: SCORE: <1-5>"""

_SCORE_TAG = re.compile(r"score\W{0,4}([1-5])(?!\d)", re.IGNORECASE)
_OUT_OF_5 = re.compile(r"(?<![\d.])([1-5])\s*(?:/|out of)\s*5\b", re.IGNORECASE)
_LONE = re.compile(r"(?<![\d.\-/])([1-5])(?![\d\-/]|\.\d)")


def parse_rating(reply: str) -> int | None:
    """Extract a 1-5 rating from a judge reply, or ``None`` if it is ambiguous.

    Real models wrap the digit in prose, markdown or fences ("**Score: 4**",
    "I'd rate this 4/5", "```\n4\n```"). The old ``re.search("[1-5]")`` took
    the *first* digit, so "On a 1-5 scale this is a 4" parsed as 1. Preference:
    an explicit ``SCORE: n`` (last one wins), then ``n/5``, then a single
    unambiguous standalone digit; ranges like ``1-5`` never count.
    """
    text = reply.replace("`", " ").replace("*", " ")
    for pat in (_SCORE_TAG, _OUT_OF_5):
        hits = pat.findall(text)
        if hits:
            return int(hits[-1])
    lone = set(_LONE.findall(text))
    return int(lone.pop()) if len(lone) == 1 else None


def _prompt_parts(body: str) -> tuple[str, int]:
    answer = body.split("<answer>\n", 1)[-1].rsplit("\n</answer>", 1)[0]
    errors = int(re.search(r"Tool errors during the run: (\d+)", body).group(1))
    return answer, errors


def heuristic_judge(messages: list[Message]) -> str:
    """Offline judge: penalise refusals, empty/short answers and tool errors."""
    answer, errors = _prompt_parts(messages[-1]["content"])
    score = 5
    if REFUSAL.search(answer) or not answer.strip():
        score = 1
    elif len(answer) < 120:
        score = 3
    score -= min(errors, 2)
    return f"SCORE: {max(1, score)}"


@dataclass
class JudgeScorer:
    """LLM-judge quality score in [0, 1].

    Empty answers score 0 without a call (the rubric fixes them at 1/5 and real
    judges, given nothing to grade, tend to answer the user's request instead).
    Unparseable replies and API errors yield ``None`` and count towards
    ``judge_unscored``, so one bad call never aborts the nightly job. Each
    record also reports the call's ``judge_cost_usd``.
    """

    llm: LLM = field(default_factory=lambda: MockLLM(model="mock-judge", responder=heuristic_judge))
    name: str = "judge"
    temperature: float = 0.0
    max_tokens: int = 16

    def score(self, rec: dict) -> dict[str, float | None]:
        answer = _answer(rec)
        if not answer.strip():
            return {"judge_score": 0.0, "judge_cost_usd": 0.0, "judge_unscored": 0.0}
        errors = sum(1 for c in rec.get("tool_calls") or [] if not c.get("ok", True))
        prompt = JUDGE_PROMPT.format(
            tool_errors=errors,
            input=str(rec.get("input", "")).replace("</request>", "</ request>"),
            output=answer.replace("</answer>", "</ answer>"),
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            c = self.llm.complete(
                messages, temperature=self.temperature, max_tokens=self.max_tokens
            )
        except Exception:  # noqa: BLE001 - a failed judge call must not kill the run
            return {"judge_score": None, "judge_cost_usd": 0.0, "judge_unscored": 1.0}
        rating = parse_rating(c.text)
        return {
            "judge_score": None if rating is None else (rating - 1) / 4,
            "judge_cost_usd": c.cost_usd,
            "judge_unscored": float(rating is None),
        }


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


def _score_record(rec: dict, scorers: list[Scorer]) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for s in scorers:
        row.update(s.score(rec))
    return row


def aggregate(records: list[dict], scorers: list[Scorer], workers: int = 1) -> DayAggregate:
    """Score a day's sample and reduce it to metrics and distributions.

    ``workers > 1`` scores records concurrently (useful when a scorer calls a
    remote LLM judge); row order, and so every metric, is unchanged.
    """
    if workers > 1 and len(records) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda r: _score_record(r, scorers), records))
    else:
        rows = [_score_record(rec, scorers) for rec in records]

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
        "judge_unscored_rate": _mean(col("judge_unscored")),
    }
    judge_cost = [v for v in col("judge_cost_usd") if v is not None]
    if judge_cost:
        metrics["judge_cost_usd"] = float(sum(judge_cost))
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
