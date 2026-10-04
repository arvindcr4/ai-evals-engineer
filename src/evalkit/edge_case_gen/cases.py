"""The edge-case record written to the golden-set JSONL."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from dataclasses import field as dc_field
from typing import Any

from evalkit.core.llm import stable_hash

AXES = ("boundary", "format", "semantic", "adversarial", "combinatorial")

AXIS_CATEGORIES: dict[str, tuple[str, ...]] = {
    "boundary": (
        "min", "max", "below_min", "above_max", "zero", "negative", "overflow",
        "empty", "at_max_length", "over_max_length", "whitespace_only", "missing",
        "null", "wrong_type", "invalid_choice", "invalid_date", "leap_day",
    ),
    "format": (
        "homoglyph", "rtl", "emoji", "whitespace", "casing", "typos",
        "mixed_language", "zero_width", "combining_marks", "fullwidth", "date_format",
    ),
    "semantic": ("ambiguous", "contradictory", "multi_intent", "out_of_scope", "negated"),
    "adversarial": (
        "long_input", "nested_quoting", "markup", "instruction_like", "delimiter_collision",
    ),
    "combinatorial": ("pairwise",),
}


def canonical(inp: Any) -> str:
    return json.dumps(inp, sort_keys=True, ensure_ascii=False, default=str)


@dataclass
class EdgeCase:
    """One generated input plus where it came from and what we know about it."""

    input: dict[str, Any]
    axis: str
    category: str
    description: str = ""
    field: str | None = None
    levels: dict[str, str] = dc_field(default_factory=dict)
    provenance: dict[str, Any] = dc_field(default_factory=dict)
    id: str = ""
    schema_valid: bool | None = None
    schema_errors: list[str] = dc_field(default_factory=list)
    novelty: float | None = None
    label: str | None = None
    label_source: str | None = None
    needs_human_review: bool = False
    review_reasons: list[str] = dc_field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            h = stable_hash(canonical(self.input), self.axis, self.category)
            self.id = f"ec-{h % 16**10:010x}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> EdgeCase:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})
