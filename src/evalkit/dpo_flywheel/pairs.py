"""Turning thumbs-downs into DPO preference pairs.

Three strategies, tried in order for each thumbs-down event:

* ``correction`` — the user edited the answer: their edit is ``chosen``.
* ``similar_up`` — another conversation with a near-identical prompt got a
  thumbs-up: that response is ``chosen``.
* ``teacher`` — a stronger model drafts N candidates, a judge scores them and
  the best one is ``chosen`` (only if it beats the rejected answer by a margin).

The output rows are TRL ``DPOTrainer``-ready: ``{"prompt", "chosen", "rejected"}``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from evalkit.core.llm import LLM, Message, MockLLM, get_llm, stable_hash
from evalkit.dpo_flywheel.feedback import FeedbackEvent
from evalkit.dpo_flywheel.filters import (
    Decontaminator,
    FilterConfig,
    filter_pairs,
    is_refusal,
    jaccard,
    normalize,
    shingles,
)

STRATEGIES = ("correction", "similar_up", "teacher")
_STOP = frozenset(
    ["a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on", "for", "and", "or", "what", "how", "why", "do", "does", "i", "you", "it", "my", "me"]
)


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    strategy: str
    source_event: str
    margin: float | None = None
    meta: dict = field(default_factory=dict)

    def to_trl(self) -> dict:
        return {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}

    def to_dict(self) -> dict:
        return asdict(self)


class Judge(Protocol):
    def score(self, prompt: str, response: str) -> float: ...


def _content_words(text: str) -> set[str]:
    return {w for w in normalize(text).split() if w not in _STOP and len(w) > 2}


class HeuristicJudge:
    """Offline 0–10 rubric: on-topic coverage, sane length, no refusal, no repetition."""

    def score(self, prompt: str, response: str) -> float:
        words = response.split()
        if not words:
            return 0.0
        s = 4.0
        want = _content_words(prompt)
        if want:
            s += 3.0 * len(want & _content_words(response)) / len(want)
        n = len(words)
        s += 2.0 if 12 <= n <= 300 else (0.5 if 5 <= n < 12 else -1.5)
        uniq = len({w.lower() for w in words}) / n
        if uniq < 0.5:
            s -= 2.0
        if is_refusal(response):
            s -= 4.0
        return round(max(0.0, min(10.0, s)), 3)


_SCORE_RE = re.compile(r"(?:score|rating)\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

JUDGE_PROMPT = """You grade an assistant reply for helpfulness, correctness and safety.
Prompt:
{prompt}

Reply:
{response}

Answer with one line: SCORE: <0-10>"""


@dataclass
class LLMJudge:
    """Rubric judge backed by any :class:`LLM`; falls back to the heuristic on parse failure."""

    llm: LLM
    fallback: HeuristicJudge = field(default_factory=HeuristicJudge)

    def score(self, prompt: str, response: str) -> float:
        out = self.llm.complete(
            [{"role": "user", "content": JUDGE_PROMPT.format(prompt=prompt, response=response)}],
            temperature=0,
        ).text
        m = _SCORE_RE.search(out)
        if not m:
            return self.fallback.score(prompt, response)
        return max(0.0, min(10.0, float(m.group(1))))


def mock_teacher_responder(messages: list[Message]) -> str:
    """Deterministic stand-in for a strong teacher model: an on-topic, structured answer
    whose detail level varies with the candidate seed in the system message."""
    user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    seed = next((m["content"] for m in messages if m["role"] == "system"), "")
    topic = user.strip().rstrip("?.!")
    level = stable_hash(seed, user) % 3
    steps = [
        f"Start by restating the goal: {topic}.",
        "Check the relevant constraints and inputs before acting.",
        "Apply the standard approach step by step and verify the result.",
        "Note edge cases and how to handle them.",
    ]
    body = " ".join(steps[: 2 + level])
    return f"Here is a direct answer to \"{topic}\". {body}"


def resolve_llm(spec: str | None, teacher: bool = False) -> LLM:
    """``get_llm`` but mock specs get a useful deterministic responder."""
    spec = spec or "mock"
    if spec == "mock" or spec.startswith("mock:"):
        name = spec.split(":", 1)[1] if ":" in spec else ("mock-teacher" if teacher else "mock-1")
        return MockLLM(model=name, responder=mock_teacher_responder if teacher else None)
    return get_llm(spec)


@dataclass
class PairBuilder:
    strategies: tuple[str, ...] = STRATEGIES
    teacher: LLM | None = None
    judge: Judge = field(default_factory=HeuristicJudge)
    n_candidates: int = 3
    similar_threshold: float = 0.6

    def build(self, events: list[FeedbackEvent], only: set[str] | None = None) -> tuple[list[PreferencePair], Counter]:
        """Pairs for every thumbs-down (restricted to event ids in ``only`` if given).

        Returns (pairs, unpaired-reason counts). Thumbs-ups are used as
        ``similar_up`` donors across the whole event list.
        """
        ups = [e for e in events if e.rating == "up" and not is_refusal(e.response)]
        up_index = [(e, shingles(e.prompt_text, 1)) for e in ups]
        pairs: list[PreferencePair] = []
        missed: Counter = Counter()
        for ev in events:
            if ev.rating != "down" or (only is not None and ev.event_id not in only):
                continue
            pair = None
            for strat in self.strategies:
                pair = getattr(self, f"_from_{strat}")(ev, up_index)
                if pair is not None:
                    break
            if pair is None:
                missed["no_strategy_applied"] += 1
            else:
                pairs.append(pair)
        return pairs, missed

    def _pair(self, ev: FeedbackEvent, chosen: str, strategy: str, **meta) -> PreferencePair:
        prompt = ev.prompt_text
        margin = self.judge.score(prompt, chosen) - self.judge.score(prompt, ev.response)
        return PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=ev.response,
            strategy=strategy,
            source_event=ev.event_id,
            margin=round(margin, 3),
            meta={"conversation_id": ev.conversation_id, "model": ev.model, **meta},
        )

    def _from_correction(self, ev: FeedbackEvent, _ups) -> PreferencePair | None:
        if ev.correction is None:
            return None
        return self._pair(ev, ev.correction, "correction")

    def _from_similar_up(self, ev: FeedbackEvent, up_index) -> PreferencePair | None:
        target = shingles(ev.prompt_text, 1)
        best, best_sim = None, self.similar_threshold
        for up, sh in up_index:
            if up.conversation_id == ev.conversation_id and up.response == ev.response:
                continue
            sim = jaccard(target, sh)
            if sim >= best_sim and normalize(up.response) != normalize(ev.response):
                best, best_sim = up, sim
        if best is None:
            return None
        return self._pair(ev, best.response, "similar_up", donor=best.event_id, similarity=round(best_sim, 3))

    def _from_teacher(self, ev: FeedbackEvent, _ups) -> PreferencePair | None:
        if self.teacher is None or self.n_candidates < 1:
            return None
        prompt = ev.prompt_text
        scored = []
        for k in range(self.n_candidates):
            msgs = [{"role": "system", "content": f"candidate-{k}: answer helpfully and concisely."}, *ev.messages]
            text = self.teacher.complete(msgs, temperature=0.8).text.strip()
            if text:
                scored.append((self.judge.score(prompt, text), k, text))
        if not scored:
            return None
        score, k, best = max(scored, key=lambda t: (t[0], -t[1]))
        return self._pair(ev, best, "teacher", teacher=getattr(self.teacher, "model", ""), candidates=len(scored), best_score=score)


@dataclass
class BuildResult:
    pairs: list[PreferencePair]
    manifest: dict


def dataset_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True).encode() + b"\n")
    return h.hexdigest()


def build_dataset(
    events: list[FeedbackEvent],
    builder: PairBuilder,
    filter_cfg: FilterConfig | None = None,
    golden: list[str] | None = None,
    only: set[str] | None = None,
) -> BuildResult:
    """Construct, filter and summarise pairs. The manifest records every count."""
    cfg = filter_cfg or FilterConfig()
    raw, missed = builder.build(events, only=only)
    by_strategy_raw = Counter(p.strategy for p in raw)
    decon = Decontaminator(golden) if golden else None
    kept, drops, pii = filter_pairs(raw, cfg, decon)
    rows = [p.to_trl() for p in kept]
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "events": {
            "total": len(events),
            "down": sum(e.rating == "down" for e in events),
            "up": sum(e.rating == "up" for e in events),
        },
        "pairs_raw": len(raw),
        "pairs_kept": len(kept),
        "unpaired": dict(missed),
        "by_strategy_raw": dict(by_strategy_raw),
        "by_strategy_kept": dict(Counter(p.strategy for p in kept)),
        "drops": dict(drops),
        "pii_redactions": dict(pii),
        "data_sha256": dataset_hash(rows),
        "filters": asdict(cfg),
        "golden_items": len(golden or []),
        "judge": type(builder.judge).__name__,
        "teacher": getattr(builder.teacher, "model", None),
    }
    return BuildResult(pairs=kept, manifest=manifest)


def write_dataset(result: BuildResult, out_dir: str | Path) -> Path:
    """Write ``train.jsonl`` (TRL rows), ``pairs.jsonl`` (with provenance) and ``manifest.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        f.writelines(json.dumps(p.to_trl()) + "\n" for p in result.pairs)
    with open(out / "pairs.jsonl", "w") as f:
        f.writelines(json.dumps(p.to_dict()) + "\n" for p in result.pairs)
    (out / "manifest.json").write_text(json.dumps(result.manifest, indent=2) + "\n")
    return out / "train.jsonl"
