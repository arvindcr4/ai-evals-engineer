"""Recall test with model-written leaks: which detector catches which kind of leak?

The hand-planted demo corpus has only a couple of paraphrases. ``leaktest``
asks a real model to write, for every eval item, several *kinds* of leak and a
same-topic non-leak, scans one corpus per kind, and reports per-detector catch
rates. Optionally a judge model then adjudicates every (item, document) pair
directly, giving the judge's own recall on leaks and false-positive rate on
the non-leaks.

Kinds:

- ``verbatim``: the item copied into a short web-page context (built locally,
  no model call; a positive control).
- ``light_edit``: a copy with roughly one word in eight swapped for a synonym.
- ``paraphrase``: the same question and answer restated in new wording.
- ``heavy_paraphrase``: restated as flowing prose with fresh vocabulary and a
  different structure (no question form).
- ``answer_only``: a passage that states the answer's key facts but not the
  question.
- ``negative``: a passage on the same topic that neither contains the
  question nor reveals its answer (false-positive probe).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from evalkit.contamination.judge import PROMPT as JUDGE_PROMPT
from evalkit.contamination.judge import adjudicate, parse_verdict
from evalkit.contamination.scanner import ContaminationScanner, EvalIndex, Thresholds
from evalkit.contamination.text import Record
from evalkit.core.llm import LLM, Message

LLM_KINDS = ("light_edit", "paraphrase", "heavy_paraphrase", "answer_only", "negative")
KINDS = ("verbatim", *LLM_KINDS)
POSITIVE_KINDS = tuple(k for k in KINDS if k != "negative")

GEN_PROMPT = """You help test a training-data contamination detector.
Here is an evaluation item (question and reference answer):

<<<
{item}
>>>

Write five short training-corpus documents (each 40-120 words) and return them as a
single JSON object with exactly these string keys:

"light_edit": the item copied almost verbatim inside a short study-notes or forum context,
  but with roughly one word in every eight replaced by a synonym or small rewording.
"paraphrase": a forum or Q&A page that asks the same question and gives the same answer
  in clearly different wording (keep the technical terms that have no synonym).
"heavy_paraphrase": an explanatory paragraph (not in question form) that conveys the same
  question's content and its answer using as much fresh vocabulary and different sentence
  structure as possible.
"answer_only": a passage (e.g. a textbook or bulletin excerpt) that states the key facts of
  the reference answer without asking or restating the question.
"negative": a passage on the SAME broad topic that does NOT contain the question and does
  NOT reveal its answer (discuss a neighbouring fact instead).

Return only the JSON object."""

SYNTH_PROMPT = """Write {n} diverse evaluation items for a question-answering benchmark.
Cover varied domains (science, maths word problems, programming, history, geography,
medicine, law, economics, everyday reasoning). Each item has a self-contained question of
15-40 words and a 1-2 sentence reference answer. Return a JSON array of objects with keys
"question" and "answer". Return only the JSON array."""

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


def extract_json(text: str) -> object:
    """Parse the first JSON object/array in a model reply.

    Real replies wrap JSON in code fences, prefix it with prose ("Here are the
    documents:"), or append notes after it. Tries fenced blocks first, then
    raw-decodes from every ``{`` / ``[`` until one parses. Raises ``ValueError``
    when nothing does.
    """
    candidates = [m.group(1) for m in _FENCE.finditer(text or "")] + [text or ""]
    dec = json.JSONDecoder()
    for cand in candidates:
        for m in re.finditer(r"[\[{]", cand):
            try:
                obj, _ = dec.raw_decode(cand, m.start())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, (dict, list)) and obj:
                return obj
    raise ValueError(f"no JSON object in reply: {(text or '')[:120]!r}")


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):  # e.g. {"text": "..."} or {"title":..., "body":...}
        return " ".join(_as_text(v) for v in value.values() if v).strip()
    if isinstance(value, list):
        return " ".join(_as_text(v) for v in value).strip()
    return ""


def parse_leaks(reply: str) -> dict[str, str]:
    """Variant texts by kind; tolerant to key case/spacing and nested objects."""
    obj = extract_json(reply)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        raise ValueError("expected a JSON object of leak variants")  # noqa: TRY004 - parse error, retried
    norm = {re.sub(r"[^a-z]+", "_", str(k).lower()).strip("_"): v for k, v in obj.items()}
    out = {k: _as_text(norm[k]) for k in LLM_KINDS if k in norm and _as_text(norm[k])}
    if not out:
        raise ValueError(f"no leak kinds in reply keys {sorted(norm)}")
    return out


def verbatim_doc(item: str) -> str:
    return f"Practice question from a study site. {item} Try the next one on your own."


@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    failures: list[str] = field(default_factory=list)

    def add(self, comp) -> None:
        self.calls += 1
        self.tokens_in += comp.tokens_in
        self.tokens_out += comp.tokens_out
        self.cost_usd += comp.cost_usd

    def to_dict(self) -> dict:
        return {
            "calls": self.calls, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6), "failures": self.failures,
        }


def _ask_json(llm: LLM, prompt: str, usage: Usage, parse, retries: int = 1):
    messages: list[Message] = [{"role": "user", "content": prompt}]
    last = ""
    for _ in range(retries + 1):
        comp = llm.complete(messages, temperature=0)
        usage.add(comp)
        try:
            return parse(comp.text)
        except ValueError as e:
            last = str(e)
            messages = [
                *messages[:1],
                {"role": "assistant", "content": comp.text or ""},
                {"role": "user", "content": "That was not valid JSON. Return only the JSON."},
            ]
    raise ValueError(last)


def synth_items(llm: LLM, n: int, usage: Usage, prefix: str = "s") -> list[dict]:
    """Ask ``llm`` for ``n`` extra eval items (``{"id", "question", "answer"}``)."""

    def parse(text: str) -> list[dict]:
        obj = extract_json(text)
        if isinstance(obj, dict):  # {"items": [...]}
            obj = next((v for v in obj.values() if isinstance(v, list)), [])
        rows = [
            r for r in obj
            if isinstance(r, dict) and _as_text(r.get("question")) and _as_text(r.get("answer"))
        ]
        if not rows:
            raise ValueError("no question/answer rows")
        return rows

    rows = _ask_json(llm, SYNTH_PROMPT.format(n=n), usage, parse)
    return [
        {"id": f"{prefix}{k + 1:02d}", "question": _as_text(r["question"]),
         "answer": _as_text(r["answer"])}
        for k, r in enumerate(rows[:n])
    ]


def generate_leaks(items: Sequence[tuple[str, str]], llm: LLM, usage: Usage) -> list[dict]:
    """One row per (item, kind): ``{"id", "eval_id", "kind", "text"}``."""
    rows: list[dict] = []
    for k, (eid, text) in enumerate(items):
        rows.append({"id": f"verbatim-{k:03d}", "eval_id": eid, "kind": "verbatim",
                     "text": verbatim_doc(text)})
        try:
            variants = _ask_json(llm, GEN_PROMPT.format(item=text), usage, parse_leaks)
        except Exception as e:  # noqa: BLE001 - record and continue with other items
            usage.failures.append(f"{eid}: {type(e).__name__}: {e}"[:200])
            continue
        for kind in LLM_KINDS:
            if kind in variants:
                rows.append({"id": f"{kind}-{k:03d}", "eval_id": eid, "kind": kind,
                             "text": variants[kind]})
            else:
                usage.failures.append(f"{eid}: missing {kind}")
    return rows


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def detector_recall(
    index: EvalIndex,
    leaks: Sequence[dict],
    thresholds: Thresholds | None = None,
    judge: LLM | None = None,
    usage: Usage | None = None,
) -> dict[str, dict]:
    """Scan one corpus per kind; per-kind rates of flagging the item via its own doc.

    An item counts as caught by a method only when that method flagged it and
    the implicated document is the item's own leak document (cross-item
    matches are reported separately as ``cross_flags``). With ``judge``, the
    scan's suspicious items are adjudicated exactly as ``scan --llm`` does and
    ``pipeline_rate`` is the share that ends up contaminated (the end-to-end
    result: detector flag, then judge upgrade).
    """
    known = set(index.ids)
    out: dict[str, dict] = {}
    for kind in KINDS:
        docs = [r for r in leaks if r["kind"] == kind and r["eval_id"] in known]
        if not docs:
            continue
        own = {r["eval_id"]: r["id"] for r in docs}
        rep = ContaminationScanner(index, thresholds).scan(
            Record(r["id"], r["text"], r) for r in docs
        )
        flagged_before = {r.id: r.status for r in rep.items}
        if judge is not None:
            adjudicate(rep, index, judge)
            if usage is not None:
                ju = rep.judge_usage
                usage.calls += ju["calls"] - ju["errors"]
                usage.tokens_in += ju["tokens_in"]
                usage.tokens_out += ju["tokens_out"]
                usage.cost_usd += ju["cost_usd"]
        stats = {"items": len(own), "flagged": 0, "contaminated": 0, "cross_flags": 0,
                 "pipeline": 0, "by_method": {"ngram": 0, "minhash": 0, "embedding": 0}}
        per_item = {}
        for r in rep.items:
            if r.id not in own:
                if r.status != "clean":
                    stats["cross_flags"] += 1
                continue
            mine = own[r.id]
            method_docs = {
                "ngram": set(r.ngram_docs), "minhash": {r.containment_doc},
                "embedding": {r.cosine_doc},
            }
            caught = [m for m in r.methods if m in method_docs and mine in method_docs[m]]
            for m in caught:
                stats["by_method"][m] += 1
            if caught:
                stats["flagged"] += 1
                stats["contaminated"] += flagged_before[r.id] == "contaminated"
                stats["pipeline"] += r.status == "contaminated"
            elif flagged_before[r.id] != "clean":
                stats["cross_flags"] += 1
            per_item[r.id] = {
                "status": flagged_before[r.id] if caught else "clean", "methods": caught,
                "final": r.status if caught else "clean", "judge": r.judge,
                "cosine": r.max_cosine, "containment": r.max_containment,
                "overlap": r.overlap_ratio,
            }
        n = stats["items"]
        stats["flag_rate"] = _rate(stats["flagged"], n)
        stats["contaminated_rate"] = _rate(stats["contaminated"], n)
        if judge is not None:
            stats["pipeline_rate"] = _rate(stats["pipeline"], n)
        stats["method_rate"] = {m: _rate(v, n) for m, v in stats["by_method"].items()}
        # Whole-document cosine of each item with its own doc, unfloored (the
        # scanner drops window cosines below 0.75 x the suspicious bar to 0).
        pos = {eid: k for k, eid in enumerate(index.ids)}
        if index.embeddings is not None:
            vecs = index.embedder.embed([r["text"] for r in docs])
            cos = sorted(
                round(float(vecs[k] @ index.embeddings[pos[r["eval_id"]]]), 4)
                for k, r in enumerate(docs)
            )
        else:
            cos = []
        stats["doc_cosine"] = (
            {"min": cos[0], "median": cos[len(cos) // 2], "max": cos[-1]} if cos else {}
        )
        stats["per_item"] = per_item
        out[kind] = stats
    return out


def judge_pairs(
    index: EvalIndex, leaks: Sequence[dict], llm: LLM, usage: Usage, max_chars: int = 1200
) -> dict[str, dict]:
    """Ask the judge about every (item, own doc) pair; YES rate per kind."""
    texts = dict(zip(index.ids, index.texts))
    out: dict[str, dict] = {}
    for row in leaks:
        if row["eval_id"] not in texts:
            continue
        s = out.setdefault(row["kind"], {"pairs": 0, "yes": 0, "no": 0, "unparsed": 0,
                                         "errors": 0, "minority_ids": []})
        s["pairs"] += 1
        prompt = JUDGE_PROMPT.format(item=texts[row["eval_id"]], passage=row["text"][:max_chars])
        try:
            comp = llm.complete([{"role": "user", "content": prompt}], temperature=0)
        except Exception as e:  # noqa: BLE001
            s["errors"] += 1
            usage.failures.append(f"judge {row['id']}: {type(e).__name__}")
            continue
        usage.add(comp)
        v = parse_verdict(comp.text)
        s[{"YES": "yes", "NO": "no"}.get(v or "", "unparsed")] += 1
        # keep the surprising verdicts (YES on a non-leak, NO/unparsed on a leak)
        if (v == "YES") == (row["kind"] == "negative"):
            s["minority_ids"].append(f"{row['eval_id']}: {' '.join(comp.text.split())[:160]}")
    for s in out.values():
        s["yes_rate"] = _rate(s["yes"], s["pairs"])
    return out


def _cos(d: dict | None) -> str:
    return f"{d['min']:.2f} / {d['median']:.2f} / {d['max']:.2f}" if d else "-"


def to_markdown(result: dict) -> str:
    rec, jud = result["detectors"], result.get("judge") or {}
    lines = [
        "# Leak recall test",
        "",
        f"{result['items']} eval items; leaks written by `{result['generator']}`"
        + (f", judged by `{result['judge_model']}`" if jud else "") + ".",
        "",
        (
            "Share of eval items flagged via their own leak document (for `negative` "
            "every flag is a false positive):"
        ),
        "",
        "| kind | items | any detector | contaminated | n-gram | MinHash | embedding "
        "| own-doc cosine min / median / max |"
        + (" scan + judge -> contaminated | judge YES on every pair |" if jud else ""),
        "|---|---|---|---|---|---|---|---|" + ("---|---|" if jud else ""),
    ]
    for kind in KINDS:
        if kind not in rec:
            continue
        s = rec[kind]
        mr = s["method_rate"]
        row = (
            f"| {kind} | {s['items']} | {s['flag_rate']:.0%} | {s['contaminated_rate']:.0%} "
            f"| {mr['ngram']:.0%} | {mr['minhash']:.0%} | {mr['embedding']:.0%} "
            f"| {_cos(s.get('doc_cosine'))} |"
        )
        if jud:
            j = jud.get(kind)
            pr = s.get("pipeline_rate")
            row += f" {pr:.0%} |" if pr is not None else " - |"
            row += f" {j['yes_rate']:.0%} ({j['yes']}/{j['pairs']}) |" if j else " - |"
        lines.append(row)
    odd = [(k, i) for k in KINDS for i in (jud.get(k) or {}).get("minority_ids", [])]
    if odd:
        lines += [
            "",
            "Judge verdicts against the generator's label (YES on a non-leak, NO on a leak):",
            "",
        ]
        lines += [f"- `{k}` {i}" for k, i in odd]
    u = result["usage"]
    lines += [
        "",
        (
            f"API usage: {u['calls']} calls, {u['tokens_in']:,} in + {u['tokens_out']:,} out "
            f"tokens, ${u['cost_usd']:.4f}."
        ),
    ]
    if u["failures"]:
        lines += ["", "Failures:", *(f"- {f}" for f in u["failures"])]
    return "\n".join(lines) + "\n"
