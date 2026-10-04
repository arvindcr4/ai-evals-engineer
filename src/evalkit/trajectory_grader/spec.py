"""Grading spec: the deterministic contract an agent's tool calls must satisfy.

A spec declares every tool the agent may call, the schema of each tool's
parameters, the dependency DAG between tools (``requires``), which tools are
safety checks, which tools must appear in every run, and which are forbidden.

Example (YAML)::

    name: refund-agent
    max_steps: 8
    required: [lookup_order, issue_refund]
    forbidden: [delete_account]
    tools:
      verify_identity:
        safety: true
        params:
          customer_id: {type: string, required: true}
          method: {type: string, enum: [otp, kba], required: true}
      issue_refund:
        requires: [lookup_order, verify_identity]
        params:
          amount: {type: number, required: true, min: 0, max: 500}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PARAM_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


class SpecError(ValueError):
    """Raised when a spec is malformed (unknown types, dangling dependencies, cycles)."""


@dataclass
class ParamSpec:
    name: str
    type: str = "string"
    required: bool = False
    enum: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    pattern: str | None = None

    def type_ok(self, value: Any) -> bool:
        if self.type in ("integer", "number") and isinstance(value, bool):
            return False
        return isinstance(value, PARAM_TYPES[self.type])


@dataclass
class ToolSpec:
    name: str
    params: dict[str, ParamSpec] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)
    safety: bool = False
    allow_extra_params: bool = False
    max_calls: int | None = None

    @property
    def required_params(self) -> list[str]:
        return [p.name for p in self.params.values() if p.required]


@dataclass
class Weights:
    """Score penalties; hard failures also flip the run verdict to FAIL."""

    hard: float = 0.5
    soft: float = 0.1


@dataclass
class GradingSpec:
    name: str
    tools: dict[str, ToolSpec]
    required: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    max_steps: int | None = None
    pass_threshold: float = 0.0
    weights: Weights = field(default_factory=Weights)

    @property
    def safety_tools(self) -> list[str]:
        return [t.name for t in self.tools.values() if t.safety]

    @property
    def mandatory(self) -> list[str]:
        """Tools that must succeed at least once: declared ``required`` plus every safety check."""
        out = list(self.required)
        out += [t for t in self.safety_tools if t not in out]
        return out

    def ancestors(self, tool: str) -> set[str]:
        """Transitive closure of ``requires`` for ``tool``."""
        seen: set[str] = set()
        stack = list(self.tools[tool].requires) if tool in self.tools else []
        while stack:
            t = stack.pop()
            if t not in seen:
                seen.add(t)
                stack.extend(self.tools[t].requires)
        return seen

    def topological_order(self) -> list[str]:
        """A valid execution order of the DAG; raises :class:`SpecError` on cycles."""
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(n: str, path: list[str]) -> None:
            if state.get(n) == 2:
                return
            if state.get(n) == 1:
                raise SpecError(f"dependency cycle: {' -> '.join(path + [n])}")
            state[n] = 1
            for d in self.tools[n].requires:
                visit(d, path + [n])
            state[n] = 2
            order.append(n)

        for name in self.tools:
            visit(name, [])
        return order


def _parse_param(name: str, raw: Any) -> ParamSpec:
    if isinstance(raw, str):
        raw = {"type": raw}
    raw = dict(raw or {})
    p = ParamSpec(
        name=name,
        type=raw.get("type", "string"),
        required=bool(raw.get("required", False)),
        enum=raw.get("enum"),
        min=raw.get("min"),
        max=raw.get("max"),
        pattern=raw.get("pattern"),
    )
    if p.type not in PARAM_TYPES:
        raise SpecError(f"param {name!r}: unknown type {p.type!r}")
    return p


def parse_spec(data: dict[str, Any]) -> GradingSpec:
    """Build and validate a :class:`GradingSpec` from a plain dict."""
    tools: dict[str, ToolSpec] = {}
    for tname, traw in (data.get("tools") or {}).items():
        traw = dict(traw or {})
        params = {pn: _parse_param(pn, pr) for pn, pr in (traw.get("params") or {}).items()}
        tools[tname] = ToolSpec(
            name=tname,
            params=params,
            requires=list(traw.get("requires") or []),
            safety=bool(traw.get("safety", False)),
            allow_extra_params=bool(traw.get("allow_extra_params", False)),
            max_calls=traw.get("max_calls"),
        )
    w = data.get("weights") or {}
    spec = GradingSpec(
        name=data.get("name", "spec"),
        tools=tools,
        required=list(data.get("required") or []),
        forbidden=list(data.get("forbidden") or []),
        max_steps=data.get("max_steps"),
        pass_threshold=float(data.get("pass_threshold", 0.0)),
        weights=Weights(hard=float(w.get("hard", 0.5)), soft=float(w.get("soft", 0.1))),
    )
    for t in spec.tools.values():
        for d in t.requires:
            if d not in spec.tools:
                raise SpecError(f"tool {t.name!r} requires undeclared tool {d!r}")
    for r in spec.required:
        if r not in spec.tools:
            raise SpecError(f"required tool {r!r} is not declared")
    overlap = set(spec.forbidden) & set(spec.tools)
    if overlap:
        raise SpecError(f"tools both declared and forbidden: {sorted(overlap)}")
    spec.topological_order()
    return spec


def load_spec(path: str | Path) -> GradingSpec:
    with open(path) as f:
        return parse_spec(yaml.safe_load(f) or {})
