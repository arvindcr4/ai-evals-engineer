"""Eval-suite config: parameterised cases, targets and scorers.

A suite is a YAML file::

    name: ticket-triage
    target: {callable: sut.py:triage}          # or {llm: mock, prompt: "...{text}..."}
    dataset: dataset.jsonl                     # rows {id, input: {...}, expected: {...}}
    params: {channel: [email, chat]}           # matrix, crossed with every item
    scorers:
      - {type: exact, field: category}
      - {type: numeric, field: amount, tolerance: 0.01}
    gate: {max_success_drop: 0.02, max_p95_latency_increase: 0.25}

Every (item × param combination) is one case; a case succeeds when all of its
scorers pass.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evalkit.core.trajectory import read_jsonl

Target = Callable[..., Any]


@dataclass
class Case:
    case_id: str
    item_id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatePolicy:
    max_success_drop: float = 0.02
    alpha: float = 0.05
    require_significance: bool = True
    max_p95_latency_increase: float = 0.25
    min_latency_increase_ms: float = 5.0
    n_boot: int = 5000

    @classmethod
    def from_dict(cls, d: dict | None) -> GatePolicy:
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown gate keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class Scorer:
    """One check on a case output.

    ``field`` is a dotted path into the output (dict, or JSON string for
    ``json_field``); ``None`` scores the whole output. The expected value comes
    from ``value`` if given, else ``expected[expected_field or field]``.
    """

    type: str
    field: str | None = None
    expected_field: str | None = None
    value: Any = None
    pattern: str | None = None
    tolerance: float = 0.0
    relative: bool = False
    case_sensitive: bool = False
    name: str = ""

    TYPES = ("exact", "contains", "regex", "json_field", "numeric")

    def __post_init__(self) -> None:
        if self.type not in self.TYPES:
            raise ValueError(f"unknown scorer type {self.type!r}; expected one of {self.TYPES}")
        if self.type == "regex" and not self.pattern:
            raise ValueError("regex scorer needs 'pattern'")
        self.name = self.name or f"{self.type}:{self.field or 'output'}"

    def expected_value(self, case: Case) -> Any:
        if self.value is not None:
            return self.value
        key = self.expected_field or self.field
        if key is None:
            return case.expected.get("output", case.expected)
        return _get_path(case.expected, key)

    def score(self, output: Any, case: Case) -> tuple[bool, Any, Any]:
        """Return ``(passed, got, expected)``."""
        if self.type == "json_field" and isinstance(output, str):
            try:
                output = json.loads(_strip_fences(output))
            except json.JSONDecodeError:
                return False, output, self.expected_value(case)
        got = _get_path(output, self.field) if self.field else output
        exp = self.pattern if self.type == "regex" else self.expected_value(case)
        if got is _MISSING:
            return False, None, exp
        if self.type == "regex":
            return bool(re.search(self.pattern or "", str(got))), got, exp
        if self.type == "numeric":
            if exp is None or got is None:
                return exp is None and got is None, got, exp
            try:
                g, e = float(got), float(exp)
            except (TypeError, ValueError):
                return False, got, exp
            tol = self.tolerance * abs(e) if self.relative else self.tolerance
            return abs(g - e) <= tol + 1e-12, got, exp
        if self.type == "contains":
            needles = exp if isinstance(exp, list) else [exp]
            hay = str(got) if self.case_sensitive else str(got).lower()
            ok = all((str(n) if self.case_sensitive else str(n).lower()) in hay for n in needles)
            return ok, got, exp
        if isinstance(got, str) and isinstance(exp, str):
            if self.case_sensitive:
                return got.strip() == exp.strip(), got, exp
            return got.strip().lower() == exp.strip().lower(), got, exp
        return got == exp, got, exp


_MISSING = object()


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def _get_path(obj: Any, path: str | None) -> Any:
    if path is None:
        return obj
    for part in path.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
            obj = obj[int(part)]
        else:
            return _MISSING
    return obj


@dataclass
class Suite:
    name: str
    cases: list[Case]
    scorers: list[Scorer]
    target: dict[str, Any]
    gate: GatePolicy
    root: Path = Path(".")

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        path = Path(path)
        cfg = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(cfg, path.parent)

    @classmethod
    def from_dict(cls, cfg: dict, root: Path | str = ".") -> Suite:
        root = Path(root)
        items = list(cfg.get("cases") or [])
        if cfg.get("dataset"):
            items += read_jsonl(root / cfg["dataset"])
        if not items:
            raise ValueError("suite has no cases (set 'dataset' and/or 'cases')")
        scorers = [Scorer(**s) for s in cfg.get("scorers") or [{"type": "exact"}]]
        return cls(
            name=cfg.get("name", "suite"),
            cases=expand_cases(items, cfg.get("params") or {}),
            scorers=scorers,
            target=cfg.get("target") or {},
            gate=GatePolicy.from_dict(cfg.get("gate")),
            root=root,
        )


def expand_cases(items: list[dict], params: dict[str, list]) -> list[Case]:
    """Cross every item with the parameter matrix."""
    keys = sorted(params)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(params[k] for k in keys))]
    cases: list[Case] = []
    seen: set[str] = set()
    for idx, it in enumerate(items):
        item_id = str(it.get("id", idx))
        inp = it.get("input", {k: v for k, v in it.items() if k not in ("id", "expected")})
        if not isinstance(inp, dict):
            inp = {"input": inp}
        exp = it.get("expected", {})
        if not isinstance(exp, dict):
            exp = {"output": exp}
        for combo in combos or [{}]:
            suffix = ",".join(f"{k}={combo[k]}" for k in keys)
            cid = f"{item_id}[{suffix}]" if suffix else item_id
            if cid in seen:
                raise ValueError(f"duplicate case id {cid!r}")
            seen.add(cid)
            cases.append(Case(cid, item_id, dict(inp), dict(exp), dict(combo)))
    return cases


def load_callable(spec: str, root: Path | str = ".") -> Target:
    """Resolve ``pkg.module:function`` or ``path/to/file.py:function`` (relative to ``root``)."""
    if ":" not in spec:
        raise ValueError(f"callable spec must be 'module:function' or 'file.py:function': {spec!r}")
    mod_part, fn_name = spec.rsplit(":", 1)
    if mod_part.endswith(".py"):
        file = Path(mod_part)
        if not file.is_absolute():
            file = Path(root) / file
        mod_name = f"_evalkit_target_{file.stem}"
        loader_spec = importlib.util.spec_from_file_location(mod_name, file)
        if loader_spec is None or loader_spec.loader is None:
            raise ImportError(f"cannot load {file}")
        module = importlib.util.module_from_spec(loader_spec)
        sys.modules[mod_name] = module
        loader_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)
    fn = getattr(module, fn_name, None)
    if not callable(fn):
        raise TypeError(f"{spec!r} is not callable")
    return fn
