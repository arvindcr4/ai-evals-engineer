"""Task specs: the input schema edge cases are generated against.

A spec is a small YAML file describing the task, its input fields (type plus
constraints), the label set, and a handful of seed examples drawn from the
golden dataset. Everything downstream (generation, validation, coverage) is
driven from it, so the schema check here is the single source of truth for
whether a generated input is well-formed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

FIELD_TYPES = ("string", "integer", "number", "date", "enum", "boolean")


@dataclass
class FieldSpec:
    name: str
    type: str
    required: bool = True
    min: Any = None
    max: Any = None
    min_length: int | None = None
    max_length: int | None = None
    choices: list[str] = field(default_factory=list)
    pattern: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.type not in FIELD_TYPES:
            raise ValueError(f"field {self.name!r}: unknown type {self.type!r}")
        if self.type == "enum" and not self.choices:
            raise ValueError(f"field {self.name!r}: enum needs choices")
        if self.type == "date":
            self.min = str(self.min) if self.min is not None else None
            self.max = str(self.max) if self.max is not None else None


@dataclass
class TaskSpec:
    name: str
    description: str
    fields: list[FieldSpec]
    seeds: list[dict[str, Any]]
    labels: list[str] = field(default_factory=list)
    text_field: str | None = None
    invalid_label: str | None = None
    out_of_scope: list[str] = field(default_factory=list)
    path: str | None = None

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("spec needs at least one seed example")
        names = {f.name for f in self.fields}
        if self.text_field is None:
            self.text_field = next((f.name for f in self.fields if f.type == "string"), None)
        elif self.text_field not in names:
            raise ValueError(f"text_field {self.text_field!r} is not a declared field")
        for i, s in enumerate(self.seeds):
            errs = validate_input(self, s)
            if errs:
                raise ValueError(f"seed {i} violates the schema: {'; '.join(errs)}")

    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def example_value(self, name: str, prefer: int = 0) -> Any:
        """A representative value for ``name``: the preferred seed's, else any seed's.

        Optional fields may be absent from some or all seeds; generators still
        need a nominal value to mutate, so fall back to a type default.
        """
        order = [prefer % len(self.seeds)] + list(range(len(self.seeds)))
        for i in order:
            if self.seeds[i].get(name) is not None:
                return self.seeds[i][name]
        f = self.field(name)
        if f.type in ("integer", "number"):
            return next((v for v in (f.min, f.max) if v is not None), 1)
        if f.type == "enum":
            return f.choices[0]
        if f.type == "date":
            return f.min or f.max or "2025-01-01"
        if f.type == "boolean":
            return True
        return "x" * max(1, f.min_length or 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("path", None)
        return d

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def spec_from_dict(d: dict, path: str | None = None) -> TaskSpec:
    fields = [FieldSpec(**f) for f in d.get("fields", [])]
    if not fields:
        raise ValueError("spec needs at least one field")
    return TaskSpec(
        name=d.get("name", "task"),
        description=d.get("description", ""),
        fields=fields,
        seeds=[_normalize_seed(s) for s in d.get("seeds", [])],
        labels=list(d.get("labels", [])),
        text_field=d.get("text_field"),
        invalid_label=d.get("invalid_label"),
        out_of_scope=list(d.get("out_of_scope", [])),
        path=path,
    )


def load_spec(path: str | Path) -> TaskSpec:
    with open(path) as f:
        return spec_from_dict(yaml.safe_load(f), path=str(path))


def _normalize_seed(seed: dict) -> dict:
    # YAML parses bare 2025-03-04 into a date object; the schema works on ISO strings.
    return {k: v.isoformat() if isinstance(v, dt.date) else v for k, v in seed.items()}


def parse_iso_date(value: Any) -> dt.date | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def validate_value(f: FieldSpec, value: Any) -> list[str]:
    """Return schema violations for one field value (empty list = valid)."""
    n = f.name
    if value is None:
        return [f"{n}: null"]
    if f.type in ("integer", "number"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{n}: expected {f.type}, got {type(value).__name__}"]
        if f.type == "integer" and not isinstance(value, int):
            return [f"{n}: expected integer, got float"]
        if isinstance(value, float) and not math.isfinite(value):
            return [f"{n}: non-finite"]
        errs = []
        if f.min is not None and value < f.min:
            errs.append(f"{n}: {value} < min {f.min}")
        if f.max is not None and value > f.max:
            errs.append(f"{n}: {value} > max {f.max}")
        return errs
    if f.type == "boolean":
        return [] if isinstance(value, bool) else [f"{n}: expected boolean"]
    if not isinstance(value, str):
        return [f"{n}: expected string, got {type(value).__name__}"]
    if f.type == "enum":
        return [] if value in f.choices else [f"{n}: {value!r} not in {f.choices}"]
    if f.type == "date":
        d = parse_iso_date(value)
        if d is None:
            return [f"{n}: {value!r} is not an ISO date"]
        errs = []
        if f.min and d < dt.date.fromisoformat(f.min):
            errs.append(f"{n}: {value} before {f.min}")
        if f.max and d > dt.date.fromisoformat(f.max):
            errs.append(f"{n}: {value} after {f.max}")
        return errs
    errs = []
    if f.min_length is not None and len(value) < f.min_length:
        errs.append(f"{n}: length {len(value)} < {f.min_length}")
    if f.max_length is not None and len(value) > f.max_length:
        errs.append(f"{n}: length {len(value)} > {f.max_length}")
    if f.pattern and not re.fullmatch(f.pattern, value):
        errs.append(f"{n}: does not match {f.pattern!r}")
    return errs


def validate_input(spec: TaskSpec, inp: Any) -> list[str]:
    """Schema-check a whole input dict against the spec."""
    if not isinstance(inp, dict):
        return [f"input is {type(inp).__name__}, not an object"]
    known = {f.name for f in spec.fields}
    errs = [f"unknown field {k!r}" for k in inp if k not in known]
    for f in spec.fields:
        if f.name not in inp:
            if f.required:
                errs.append(f"{f.name}: missing")
            continue
        errs.extend(validate_value(f, inp[f.name]))
    return errs
