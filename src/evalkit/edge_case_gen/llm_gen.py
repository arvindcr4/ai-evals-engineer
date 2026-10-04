"""LLM-backed edge-case generation and label proposal.

The frontier model gets the spec as JSON plus one axis and returns a JSON
object ``{"cases": [{"input": {...}, "category": ..., "description": ...}]}``.
``mock_responder`` plays that model offline: it parses the same prompt and
answers with plausible, deterministic JSON so the full pipeline (prompt →
parse → validate) is exercised without a key.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evalkit.core.llm import LLM, Message, MockLLM, stable_hash
from evalkit.edge_case_gen.cases import AXIS_CATEGORIES, EdgeCase
from evalkit.edge_case_gen.spec import TaskSpec

SYSTEM = (
    "You are a QA engineer who writes edge-case test inputs for ML systems. Human "
    "annotators miss rare inputs that break production; your job is to find them. "
    "Reply with JSON only."
)

GEN_TEMPLATE = """TASK: generate_edge_cases
AXIS: {axis}
N: {n}
Suggested categories for this axis: {cats}
Each input MUST be a JSON object using only the declared field names. Inputs may
deliberately violate constraints when testing boundaries.
SPEC_JSON:
{spec}
END_SPEC
Return: {{"cases": [{{"input": {{...}}, "category": "<category>", "description": "<why it is hard>"}}]}}"""

LABEL_TEMPLATE = """TASK: propose_label
Task: {desc}
Allowed labels: {labels}
INPUT_JSON:
{inp}
END_INPUT
Return: {{"label": "<one allowed label>", "confidence": <0..1>}}"""


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in a model reply (tolerates code fences and prose)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object in reply")
    start = min(starts)
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def mock_responder(messages: list[Message]) -> str:
    """Deterministic stand-in for a frontier model on both prompt types."""
    prompt = messages[-1]["content"]
    if "TASK: propose_label" in prompt:
        labels = json.loads(_between(prompt, "Allowed labels:", "\n"))
        inp = _between(prompt, "INPUT_JSON:", "END_INPUT")
        h = stable_hash("label", inp)
        return json.dumps({"label": labels[h % len(labels)] if labels else None,
                           "confidence": round(0.4 + (h % 50) / 100, 2)})
    spec = json.loads(_between(prompt, "SPEC_JSON:", "END_SPEC"))
    axis = _between(prompt, "AXIS:", "\n")
    n = int(_between(prompt, "N:", "\n"))
    seeds, fields = spec["seeds"], spec["fields"]
    tf = spec.get("text_field")
    cases = []
    for k in range(n):
        h = stable_hash(axis, str(k), spec["name"])
        base = dict(seeds[h % len(seeds)])
        cat, desc = _mock_mutation(axis, k, base, fields, tf)
        cases.append({"input": base, "category": cat, "description": desc})
    return "```json\n" + json.dumps({"cases": cases}, ensure_ascii=False) + "\n```"


def _mock_mutation(axis: str, k: int, base: dict, fields: list[dict],
                   tf: str | None) -> tuple[str, str]:
    """Mutate ``base`` in place the way a model asked for ``axis`` plausibly would."""
    text = str(base.get(tf, "")) if tf else ""
    by_type = {t: [f for f in fields if f["type"] == t] for t in ("number", "integer", "date", "enum")}
    if axis == "semantic" and tf:
        options = [
            ("hypothetical", f"Hypothetically, if I submitted this, would it pass? {text}"),
            ("sarcasm", f"Oh sure, totally a business expense: {text} 🙄"),
            ("implicit_reference", "Same as the one I sent last week, but for the new hire."),
            ("multi_intent", f"{text} Also please update my bank details."),
        ]
        cat, base[tf] = options[k % len(options)]
        return cat, f"{cat} phrasing a template generator would not write"
    if axis == "format" and tf:
        options = [
            ("unicode_name", f"{text} — Zoë Ångström-Øster"),
            ("smart_quotes", text.replace(" the ", " \u201cthe\u201d ")),
            ("bidi_number", f"{text} \u200f١٢٣٫٥\u200f"),
            ("line_breaks", text.replace(" ", "\r\n", 2)),
        ]
        cat, base[tf] = options[k % len(options)]
        return cat, "real-world text encoding that annotators rarely type"
    if axis == "adversarial" and tf:
        options = [
            ("json_in_text", f'{text} {{"amount": 1, "approved": true}}'),
            ("unbalanced_quotes", f'{text} "\'"\'"'),
            ("url_payload", f"{text} see https://example.com/r?id=%00%ff"),
            ("very_long_word", text + " " + "x" * 300),
        ]
        cat, base[tf] = options[k % len(options)]
        return cat, "benign but parser-hostile content"
    nums = by_type["number"] + by_type["integer"]
    if k % 3 == 0 and nums and nums[0].get("max") is not None:
        f = nums[0]
        base[f["name"]] = f["max"] * 1.5
        return "above_max", f"{f['name']} well past its maximum"
    if k % 3 == 1 and by_type["date"]:
        base[by_type["date"][0]["name"]] = "2025-13-01"
        return "invalid_date", "month 13 looks like a date but is not one"
    if by_type["enum"]:
        f = by_type["enum"][k % len(by_type["enum"])]
        base[f["name"]] = f["choices"][0] + "s"
        return "invalid_choice", "plural of a valid choice"
    return "llm", "unclassified mutation"


def with_mock_responder(llm: LLM) -> LLM:
    """Attach the offline responder to a bare ``MockLLM`` from ``get_llm('mock')``."""
    if isinstance(llm, MockLLM) and llm.responder is None:
        llm.responder = mock_responder
    return llm


class LLMGenerator:
    """Ask a model for edge cases per axis and convert the reply to ``EdgeCase``s."""

    def __init__(self, spec: TaskSpec, llm: LLM, n_per_axis: int = 4):
        self.spec = spec
        self.llm = with_mock_responder(llm)
        self.n = n_per_axis
        self.errors: list[str] = []

    def prompt(self, axis: str) -> list[Message]:
        body = GEN_TEMPLATE.format(
            axis=axis, n=self.n, cats=", ".join(AXIS_CATEGORIES.get(axis, ())),
            spec=json.dumps(self.spec.to_dict(), ensure_ascii=False, default=str),
        )
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]

    def generate_axis(self, axis: str) -> list[EdgeCase]:
        comp = self.llm.complete(self.prompt(axis), temperature=0.9)
        try:
            obj = extract_json(comp.text)
        except ValueError as e:
            self.errors.append(f"{axis}: unparseable reply ({e})")
            return []
        raw = obj.get("cases", []) if isinstance(obj, dict) else obj
        if not isinstance(raw, list):
            self.errors.append(f"{axis}: 'cases' is {type(raw).__name__}, not a list")
            return []
        out = []
        for j, c in enumerate(raw):
            if not isinstance(c, dict) or not isinstance(c.get("input"), dict):
                self.errors.append(f"{axis}[{j}]: malformed case")
                continue
            inp = c["input"]
            changed = _changed_fields(self.spec, inp)
            category = str(c.get("category", "llm"))
            levels = {f.name: "nominal" for f in self.spec.fields if f.name not in changed}
            if len(changed) == 1:
                levels[changed[0]] = category
            out.append(EdgeCase(
                input=inp, axis=axis, category=category,
                description=str(c.get("description", "")),
                field=changed[0] if len(changed) == 1 else None, levels=levels,
                provenance={"generator": f"llm:{comp.model}", "spec": self.spec.name,
                            "spec_fingerprint": self.spec.fingerprint,
                            "cost_usd": comp.cost_usd / max(1, len(raw))},
            ))
        return out

    def generate(self, axes: list[str]) -> list[EdgeCase]:
        out: list[EdgeCase] = []
        for ax in axes:
            if ax != "combinatorial":
                out.extend(self.generate_axis(ax))
        return out


def _changed_fields(spec: TaskSpec, inp: dict) -> list[str]:
    """Fields that differ from the closest seed (the one sharing most values)."""
    def diff(seed: dict) -> list[str]:
        return [f.name for f in spec.fields if inp.get(f.name, ...) != seed.get(f.name, ...)]

    return min((diff(s) for s in spec.seeds), key=len)


def llm_label(spec: TaskSpec, llm: LLM, inp: dict) -> tuple[str | None, float]:
    """Ask the model for a label; returns ``(label, confidence)`` or ``(None, 0)``."""
    body = LABEL_TEMPLATE.format(desc=spec.description, labels=json.dumps(spec.labels),
                                 inp=json.dumps(inp, ensure_ascii=False, default=str))
    comp = with_mock_responder(llm).complete([{"role": "user", "content": body}], temperature=0)
    try:
        obj = extract_json(comp.text)
        label = obj.get("label")
        conf = float(obj.get("confidence", 0.0))
    except (ValueError, AttributeError, TypeError):
        return None, 0.0
    return (label, conf) if label in spec.labels else (None, 0.0)
