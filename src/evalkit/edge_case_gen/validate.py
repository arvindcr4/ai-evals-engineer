"""Validation: schema check, dedupe, novelty vs seeds, and label proposal.

Raw generated cases are noisy. This stage keeps only cases worth a golden-set
slot: structurally usable inputs (unknown fields or non-object inputs are
rejected; constraint violations are kept, they are the point), not duplicates
of each other, and not copies of a seed. Labels come from a reference oracle
and/or an LLM; anything an oracle cannot vouch for is flagged for human review.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalkit.core.llm import LLM
from evalkit.edge_case_gen.cases import EdgeCase, canonical
from evalkit.edge_case_gen.claims import claim_holds
from evalkit.edge_case_gen.llm_gen import llm_label
from evalkit.edge_case_gen.spec import TaskSpec, validate_input

Oracle = Callable[[dict[str, Any]], Any]
REVIEW_AXES = ("semantic",)


def shingles(text: str, k: int = 4) -> set[str]:
    """Character k-shingles of NFC-normalised text.

    Whitespace is deliberately not collapsed: tabs, CRLF, NBSP and ideographic
    spaces are what format-axis cases test, so two inputs that differ only in
    *which* odd whitespace they carry are distinct edge cases, not duplicates.
    """
    text = unicodedata.normalize("NFC", text)
    if len(text) <= k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def load_callable(ref: str, base_dir: str | Path | None = None) -> Callable:
    """Resolve ``path/to/file.py:fn`` or ``package.module:fn`` to a callable."""
    target, _, attr = ref.rpartition(":")
    if not target or not attr:
        raise ValueError(f"expected <file.py|module>:<function>, got {ref!r}")
    if target.endswith(".py"):
        p = Path(target)
        if not p.is_absolute() and base_dir is not None and not p.exists():
            p = Path(base_dir) / p
        name = f"_evalkit_ext_{p.stem}"
        mod_spec = importlib.util.spec_from_file_location(name, p)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"cannot load {p}")
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = mod
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return getattr(mod, attr)


@dataclass
class ValidationReport:
    kept: list[EdgeCase] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        reasons: dict[str, int] = {}
        for r in self.rejected:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        return {
            "kept": len(self.kept),
            "rejected": len(self.rejected),
            "rejected_by_reason": reasons,
            "schema_invalid_kept": sum(1 for c in self.kept if not c.schema_valid),
            "needs_human_review": sum(1 for c in self.kept if c.needs_human_review),
        }


class Validator:
    """Filter and annotate generated cases.

    ``near_dup`` is the shingle-Jaccard threshold above which two cases with
    identical structured fields count as duplicates; ``min_novelty`` drops cases
    whose whole input is that close to a seed (novelty = 1 − max similarity).
    ``check_claims`` rejects cases whose input lacks the edge their category
    names (see :mod:`claims`), e.g. a ``zero_width`` case with no zero-width
    character, which real models return often.
    """

    def __init__(self, spec: TaskSpec, near_dup: float = 0.95, min_novelty: float = 0.0,
                 check_claims: bool = True):
        self.spec = spec
        self.check_claims = check_claims
        self.near_dup = near_dup
        self.min_novelty = min_novelty
        self._seed_sh = [shingles(canonical(s)) for s in spec.seeds]
        self._seed_keys = {canonical(s) for s in spec.seeds}

    def _bucket(self, inp: dict, errors: list[str]) -> str:
        """Near-dup bucket: same structured fields, schema-error kinds and text-length scale."""
        kinds = sorted(re.sub(r"[\d.]+", "#", e) for e in errors)
        rest = {k: v for k, v in inp.items() if k != self.spec.text_field}
        return canonical([rest, kinds, len(self._text(inp)).bit_length()])

    def _text(self, inp: dict) -> str:
        return str(inp.get(self.spec.text_field, "")) if self.spec.text_field else ""

    def run(self, cases: list[EdgeCase]) -> ValidationReport:
        rep = ValidationReport()
        known = {f.name for f in self.spec.fields}
        seen: set[str] = set()
        by_struct: dict[str, list[tuple[set[str], str]]] = {}
        for c in cases:
            if not isinstance(c.input, dict):
                rep.rejected.append({"id": c.id, "reason": "malformed", "detail": "not an object"})
                continue
            extra = [k for k in c.input if k not in known]
            if extra:
                rep.rejected.append({"id": c.id, "reason": "unknown_fields", "detail": extra})
                continue
            key = canonical(c.input)
            if key in self._seed_keys:
                rep.rejected.append({"id": c.id, "reason": "copy_of_seed", "detail": c.category})
                continue
            if key in seen:
                rep.rejected.append({"id": c.id, "reason": "duplicate", "detail": c.category})
                continue
            if self.check_claims and claim_holds(self.spec, c.category, c.input,
                                                 c.field) is False:
                rep.rejected.append({"id": c.id, "reason": "claim_not_in_input",
                                     "detail": f"{c.axis}/{c.category}: {c.description[:120]}"})
                continue
            errors = validate_input(self.spec, c.input)
            sh = shingles(self._text(c.input))
            bucket = by_struct.setdefault(self._bucket(c.input, errors), [])
            twin = next((cid for s, cid in bucket if jaccard(sh, s) >= self.near_dup), None)
            if twin is not None:
                rep.rejected.append({"id": c.id, "reason": "near_duplicate", "detail": twin})
                continue
            novelty = 1.0 - max(jaccard(shingles(key), s) for s in self._seed_sh)
            if novelty < self.min_novelty:
                rep.rejected.append({"id": c.id, "reason": "low_novelty", "detail": novelty})
                continue
            seen.add(key)
            bucket.append((sh, c.id))
            c.schema_errors = errors
            c.schema_valid = not c.schema_errors
            c.novelty = round(novelty, 4)
            rep.kept.append(c)
        return rep


def propose_labels(spec: TaskSpec, cases: list[EdgeCase], oracle: Oracle | None = None,
                   llm: LLM | None = None, min_confidence: float = 0.7) -> dict:
    """Fill ``label`` in place and set ``needs_human_review`` with reasons.

    Returns the LLM usage (calls, tokens, cost) spent on label proposals.

    Schema-invalid inputs get ``spec.invalid_label`` when the spec defines one.
    Otherwise the oracle labels; the LLM labels when there is no oracle and is
    cross-checked when there is. Semantic cases always go to review, since the
    oracle reads structured fields and cannot judge what the text means.
    """
    usage: dict = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    for c in cases:
        reasons: list[str] = []
        c.label, c.label_source = None, None
        if c.schema_valid is False and spec.invalid_label:
            c.label, c.label_source = spec.invalid_label, "schema"
        else:
            if oracle is not None:
                try:
                    c.label, c.label_source = str(oracle(dict(c.input))), "oracle"
                except Exception as e:  # noqa: BLE001 - an oracle crash is a review signal
                    reasons.append(f"oracle_error: {type(e).__name__}: {e}")
            if llm is not None:
                lab, conf = llm_label(spec, llm, c.input, usage=usage)
                if c.label is None and lab is not None:
                    c.label, c.label_source = lab, "llm"
                    if conf < min_confidence:
                        reasons.append(f"low_llm_confidence: {conf:.2f}")
                elif lab is not None and lab != c.label:
                    reasons.append(f"oracle_llm_disagree: llm={lab}")
            if c.label is None:
                reasons.append("no_label")
        if c.axis in REVIEW_AXES:
            reasons.append(f"{c.axis}_axis")
        c.review_reasons = reasons
        c.needs_human_review = bool(reasons)
    return usage
