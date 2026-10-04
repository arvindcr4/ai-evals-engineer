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


_VERDICT = re.compile(
    r"^(?:(?:final\s+)?(?:verdict|answer|decision|judgement|judgment|label)\s*[:=\-]?\s*)?"
    r"(yes|no)\b",
    re.IGNORECASE,
)


def parse_verdict(reply: str) -> str | None:
    """``"YES"`` / ``"NO"`` from a judge reply, or ``None`` when it is unreadable.

    Real models do not always answer with a bare first-line ``YES``: they bold it
    (``**YES**``), label it (``Verdict: NO``), put it in a code fence, or open
    with a sentence and give the verdict later. The first non-empty line is
    read after stripping markdown; failing that, an upper-case ``YES``/``NO``
    that occurs alone in the reply is accepted.
    """
    lines = [ln for ln in (reply or "").splitlines() if ln.strip().strip("`")]
    for line in lines[:1]:
        cleaned = re.sub(r"[*_#>`\"'\[\]()]", "", line).strip()
        m = _VERDICT.match(cleaned)
        if m:
            return m.group(1).upper()
    found = set(re.findall(r"\b(YES|NO)\b", reply or ""))
    return found.pop() if len(found) == 1 else None


def adjudicate(report: ScanReport, index: EvalIndex, llm: LLM) -> int:
    """Ask ``llm`` about every suspicious item with evidence; returns upgrades.

    Usage (calls, tokens, USD cost, verdict counts) is recorded on
    ``report.judge_usage``. A failed call or an unreadable reply leaves the
    item suspicious with the problem noted, rather than aborting the scan.
    """
    texts = dict(zip(index.ids, index.texts))
    usage = {
        "model": llm.model, "calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
        "yes": 0, "no": 0, "unparsed": 0, "errors": 0,
    }
    for r in report.items:
        if r.status != "suspicious" or not r.evidence:
            continue
        prompt = PROMPT.format(item=texts[r.id], passage=r.evidence)
        usage["calls"] += 1
        try:
            comp = llm.complete([{"role": "user", "content": prompt}], temperature=0)
        except Exception as e:  # noqa: BLE001 - one bad call must not sink the scan
            usage["errors"] += 1
            r.judge = f"error: {type(e).__name__}: {e}"[:240]
            r.reasons.append(f"[note] LLM judge ({llm.model}) failed; left suspicious")
            continue
        usage["tokens_in"] += comp.tokens_in
        usage["tokens_out"] += comp.tokens_out
        usage["cost_usd"] += comp.cost_usd
        reply = (comp.text or "").strip()
        verdict = parse_verdict(reply)
        r.judge = " ".join(reply.split())[:240] or "(empty reply)"
        if verdict == "YES":
            usage["yes"] += 1
            r.status = "contaminated"
            r.reasons.append(f"[contaminated] LLM judge ({llm.model}) confirmed the leak")
            if "judge" not in r.methods:
                r.methods.append("judge")
        elif verdict == "NO":
            usage["no"] += 1
        else:
            usage["unparsed"] += 1
            r.reasons.append(f"[note] LLM judge ({llm.model}) reply had no YES/NO verdict")
    usage["cost_usd"] = round(usage["cost_usd"], 6)
    report.judge_usage = usage
    return usage["yes"]
