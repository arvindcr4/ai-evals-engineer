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
import statistics
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
    leaks_context,
    normalize,
    scrub_pii,
    shingles,
)
from evalkit.dpo_flywheel.llm_cache import usage_of

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


# Tolerates the decorations real judges add: ``**SCORE:** 8``, ``Score = 7/10``,
# ``SCORE: 8.5``. When the reply reasons first and mentions several scores, the
# last one is the verdict.
_SCORE_RE = re.compile(r"(?:score|rating)[*_\s]*[:=][*_\s]*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_score(text: str) -> float | None:
    found = _SCORE_RE.findall(text)
    if not found:
        return None
    return max(0.0, min(10.0, float(found[-1])))

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
    max_tokens: int = 64
    parse_failures: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def score(self, prompt: str, response: str) -> float:
        out = self.llm.complete(
            [{"role": "user", "content": JUDGE_PROMPT.format(prompt=prompt, response=response)}],
            temperature=0,
            max_tokens=self.max_tokens,
        ).text
        s = parse_score(out)
        if s is None:
            # Silent fallback would mix two score scales in one margin; count it.
            with self._lock:
                self.parse_failures += 1
            return self.fallback.score(prompt, response)
        return s


def mock_teacher_responder(messages: list[Message]) -> str:
    """Deterministic stand-in for a strong teacher model: an on-topic, structured answer
    whose detail level varies with the candidate seed in the system message."""
    user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    # Seed on the first system line only, so adding reference answers below it
    # does not change the mock's output.
    seed = next((m["content"] for m in messages if m["role"] == "system"), "").split("\n", 1)[0]
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


TEACHER_SYSTEM = """{seed}: you are this product's support assistant. Answer the user helpfully and concisely.
When the example conversations below show the product's menus or policies, use them. Otherwise give the
general steps a user should check, without inventing product-specific menu names. Write only the reply
the user sees: do not refer to examples, notes, documentation or anything you were shown."""


@dataclass
class PairBuilder:
    strategies: tuple[str, ...] = STRATEGIES
    teacher: LLM | None = None
    judge: Judge = field(default_factory=HeuristicJudge)
    n_candidates: int = 3
    similar_threshold: float = 0.6
    # Trusted answers (thumbs-ups, user corrections) most similar to the prompt are
    # shown to the teacher. Without them a real teacher writes generic
    # "it depends on your platform" answers that ignore the product.
    teacher_refs: int = 3
    teacher_temperature: float = 0.8
    teacher_max_tokens: int = 600
    workers: int = 1
    # Scrub PII from everything sent to the teacher/judge APIs. The dataset filters
    # scrub only *after* pairs are built, so without this the raw emails/cards of
    # thumbs-down users (and of other users, via teacher references) leave the box.
    scrub_llm_inputs: bool = True

    def _clean(self, text: str) -> str:
        return scrub_pii(text)[0] if self.scrub_llm_inputs else text

    def _score(self, prompt: str, response: str) -> float:
        return self.judge.score(self._clean(prompt), self._clean(response))

    def build(self, events: list[FeedbackEvent], only: set[str] | None = None) -> tuple[list[PreferencePair], Counter]:
        """Pairs for every thumbs-down (restricted to event ids in ``only`` if given).

        Returns (pairs, unpaired-reason counts). Thumbs-ups are used as
        ``similar_up`` donors across the whole event list. With ``workers > 1``
        events are processed concurrently; output order stays the event order.
        A teacher/judge API error loses that one event (``<strategy>_error``),
        not the whole nightly run.
        """
        ups = [e for e in events if e.rating == "up" and not is_refusal(e.response)]
        up_index = [(e, shingles(e.prompt_text, 1)) for e in ups]
        self._refs = [
            (e, self._clean(e.correction if e.correction is not None else e.response), shingles(e.prompt_text, 1))
            for e in events
            if (e.rating == "up" and not is_refusal(e.response))
            or (e.correction is not None and not is_refusal(e.correction))
        ]
        todo = [e for e in events if e.rating == "down" and (only is None or e.event_id in only)]

        def one(ev: FeedbackEvent) -> tuple[PreferencePair | None, list[str]]:
            errors = []
            for strat in self.strategies:
                try:
                    pair = getattr(self, f"_from_{strat}")(ev, up_index)
                except Exception:  # noqa: BLE001 - API errors/timeouts after retries
                    errors.append(f"{strat}_error")
                    continue
                if pair is not None:
                    return pair, errors
            return None, errors

        if self.workers > 1 and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                results = list(pool.map(one, todo))
        else:
            results = [one(ev) for ev in todo]
        pairs: list[PreferencePair] = []
        missed: Counter = Counter()
        for pair, errors in results:
            missed.update(errors)
            if pair is None:
                missed["no_strategy_applied"] += 1
            else:
                pairs.append(pair)
        return pairs, missed

    def _references(self, ev: FeedbackEvent) -> list[tuple[str, str]]:
        if self.teacher_refs < 1:
            return []
        target = shingles(ev.prompt_text, 1)
        scored = [
            (jaccard(target, sh), e.prompt_text, answer)
            for e, answer, sh in getattr(self, "_refs", [])
            if e.event_id != ev.event_id
        ]
        scored.sort(key=lambda t: -t[0])
        return [(self._clean(q), a) for _, q, a in scored[: self.teacher_refs]]

    def _pair(self, ev: FeedbackEvent, chosen: str, strategy: str, **meta) -> PreferencePair:
        prompt = ev.prompt_text
        margin = self._score(prompt, chosen) - self._score(prompt, ev.response)
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
        refs = self._references(ev)
        ref_block = "".join(f"\n\nQ: {q}\nA: {a}" for q, a in refs)
        if ref_block:
            ref_block = "\n\nExample conversations with this product's assistant:" + ref_block
        scored, leaked = [], 0
        for k in range(self.n_candidates):
            system = TEACHER_SYSTEM.format(seed=f"candidate-{k}") + ref_block
            msgs = [{"role": "system", "content": system},
                    *({**m, "content": self._clean(m["content"])} for m in ev.messages)]
            text = self.teacher.complete(
                msgs, temperature=self.teacher_temperature, max_tokens=self.teacher_max_tokens
            ).text.strip()
            if not text:
                continue
            if leaks_context(text):
                # Talks about its reference context; never worth a judge call.
                leaked += 1
                continue
            scored.append((self._score(prompt, text), k, text))
        if not scored:
            return None
        score, k, best = max(scored, key=lambda t: (t[0], -t[1]))
        return self._pair(
            ev, best, "teacher", teacher=getattr(self.teacher, "model", ""), candidates=len(scored),
            best_score=score, candidate_scores=[s for s, _, _ in sorted(scored, key=lambda t: t[1])],
            references=len(refs), leaked_candidates=leaked,
        )


@dataclass
class BuildResult:
    pairs: list[PreferencePair]
    manifest: dict


def dataset_hash(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True).encode() + b"\n")
    return h.hexdigest()


def margin_stats(pairs: list[PreferencePair]) -> dict:
    """Per-strategy judge-margin summary (n, min, median, mean, max)."""
    by: dict[str, list[float]] = {}
    for p in pairs:
        if p.margin is not None:
            by.setdefault(p.strategy, []).append(p.margin)
    return {
        s: {"n": len(v), "min": min(v), "median": statistics.median(v), "mean": round(statistics.fmean(v), 3), "max": max(v)}
        for s, v in sorted(by.items())
    }


def llm_usage(builder: PairBuilder) -> dict:
    """Calls/tokens/cost per role when the teacher/judge are :class:`MeteredLLM`."""
    out: dict = {}
    seen: set[int] = set()
    total = 0.0
    for role, llm in (("teacher", builder.teacher), ("judge", getattr(builder.judge, "llm", None))):
        u = usage_of(llm)
        if u is None:
            continue
        out[role] = u
        if id(llm) not in seen:  # one model serving both roles is counted once
            seen.add(id(llm))
            total += u["cost_usd"]
    if out:
        out["total_cost_usd"] = round(total, 6)
    return out


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
        "judge_model": getattr(getattr(builder.judge, "llm", None), "model", None),
        "judge_parse_failures": getattr(builder.judge, "parse_failures", 0),
        "teacher": getattr(builder.teacher, "model", None),
        "margins_raw": margin_stats(raw),
        "margins_kept": margin_stats(kept),
        "llm_usage": llm_usage(builder),
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
