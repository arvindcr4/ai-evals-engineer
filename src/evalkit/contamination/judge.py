"""Optional LLM adjudication of *suspicious* items.

Embedding and partial-overlap evidence is noisy, so a model can be asked
whether the best-matching training passage is a restatement of the eval item.
A confirmed ``YES`` upgrades the item to ``contaminated``; ``NO`` leaves it
suspicious with the verdict recorded. The offline default is a MockLLM whose
responder answers by content-word overlap, so the path is testable without keys.
"""

from __future__ import annotations

import re

from evalkit.contamination.embed import STOPWORDS
from evalkit.contamination.scanner import EvalIndex, ScanReport
from evalkit.contamination.text import tokenize
from evalkit.core.llm import LLM, Message, MockLLM, get_llm

PROMPT = (
    "You audit evaluation datasets for training-data contamination.\n"
    "EVAL ITEM:\n{item}\n\nTRAINING PASSAGE:\n{passage}\n\n"
    "Does the training passage contain this eval item, a paraphrase of it, or its answer "
    "in a way that would let a model memorise it? Reply with YES or NO on the first line, "
    "then one short sentence of justification."
)


def _content(text: str) -> set[str]:
    return {w for w in tokenize(text) if w not in STOPWORDS}


def overlap_responder(messages: list[Message]) -> str:
    """Deterministic stand-in judge: YES when >=55% of the item's content words recur."""
    body = messages[-1]["content"]
    m = re.search(r"EVAL ITEM:\n(.*?)\n\nTRAINING PASSAGE:\n(.*?)\n\nDoes", body, re.DOTALL)
    if not m:
        return "NO\nCould not parse the prompt."
    item, passage = _content(m.group(1)), _content(m.group(2))
    share = len(item & passage) / max(1, len(item))
    verdict = "YES" if share >= 0.55 else "NO"
    return f"{verdict}\n{share:.0%} of the item's content words appear in the passage."


def judge_llm(spec: str | None) -> LLM:
    llm = get_llm(spec)
    if isinstance(llm, MockLLM) and llm.responder is None:
        llm.responder = overlap_responder
    return llm


def adjudicate(report: ScanReport, index: EvalIndex, llm: LLM) -> int:
    """Ask ``llm`` about every suspicious item with evidence; returns upgrades."""
    texts = dict(zip(index.ids, index.texts))
    upgraded = 0
    for r in report.items:
        if r.status != "suspicious" or not r.evidence:
            continue
        prompt = PROMPT.format(item=texts[r.id], passage=r.evidence)
        reply = llm.complete([{"role": "user", "content": prompt}], temperature=0).text.strip()
        first = reply.splitlines()[0].strip().upper() if reply else ""
        r.judge = " ".join(reply.split())[:240]
        if first.startswith("YES"):
            r.status = "contaminated"
            r.reasons.append(f"[contaminated] LLM judge ({llm.model}) confirmed the leak")
            if "judge" not in r.methods:
                r.methods.append("judge")
            upgraded += 1
    return upgraded
