"""Execute a suite against a target and record success + latency per case."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evalkit.core.llm import LLM, get_llm
from evalkit.regression_gate.suite import Case, Suite, Target, load_callable


@dataclass
class CaseResult:
    case_id: str
    item_id: str
    params: dict[str, Any]
    success: bool
    checks: dict[str, bool]
    latency_s: float
    output: Any = None
    got: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RunResult:
    suite: str
    cases: list[CaseResult]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return float(np.mean([c.success for c in self.cases])) if self.cases else 0.0

    def latency_quantile(self, q: float) -> float:
        return float(np.quantile([c.latency_s for c in self.cases], q)) if self.cases else 0.0

    def summary(self) -> dict[str, Any]:
        checks: dict[str, list[bool]] = {}
        for c in self.cases:
            for k, v in c.checks.items():
                checks.setdefault(k, []).append(v)
        return {
            "n_cases": len(self.cases),
            "success_rate": self.success_rate,
            "errors": sum(c.error is not None for c in self.cases),
            "latency_p50_s": self.latency_quantile(0.5),
            "latency_p95_s": self.latency_quantile(0.95),
            "scorer_pass_rates": {k: float(np.mean(v)) for k, v in checks.items()},
        }

    def to_dict(self) -> dict:
        return {"suite": self.suite, "meta": self.meta, "summary": self.summary(),
                "cases": [asdict(c) for c in self.cases]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=str) + "\n")

    @classmethod
    def from_dict(cls, d: dict) -> RunResult:
        return cls(d["suite"], [CaseResult(**c) for c in d["cases"]], d.get("meta", {}))

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _llm_target(cfg: dict[str, Any], llm: LLM | None) -> Target:
    model = llm or get_llm(cfg.get("llm"))
    template: str = cfg.get("prompt", "{input}")
    system: str | None = cfg.get("system")

    def run(inp: dict[str, Any], **params: Any) -> tuple[str, float]:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": template.format(**inp, **params)}]
        comp = model.complete(messages)
        return comp.text, comp.latency_s

    run.reports_latency = True  # type: ignore[attr-defined]
    return run


def resolve_target(suite: Suite, llm: LLM | None = None) -> Target:
    cfg = suite.target
    if "callable" in cfg:
        return load_callable(cfg["callable"], suite.root)
    if "llm" in cfg or "prompt" in cfg or llm is not None:
        return _llm_target(cfg, llm)
    raise ValueError("suite target needs 'callable' or 'llm'/'prompt'")


def run_case(target: Target, case: Case, suite: Suite) -> CaseResult:
    t0 = time.perf_counter()
    try:
        out = target(case.input, **case.params)
        latency = time.perf_counter() - t0
        if getattr(target, "reports_latency", False):
            out, latency = out
        error = None
    except Exception as exc:  # noqa: BLE001 — a crashing case fails, the gate keeps going
        out, latency, error = None, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"
    checks: dict[str, bool] = {}
    got: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for sc in suite.scorers:
        ok, g, exp = sc.score(out, case) if error is None else (False, None, None)
        checks[sc.name], got[sc.name], expected[sc.name] = bool(ok), g, exp
    return CaseResult(case.case_id, case.item_id, case.params, error is None and all(
        checks.values()), checks, float(latency), out, got, expected, error)


def run_suite(suite: Suite, target: Target | None = None, llm: LLM | None = None,
              meta: dict[str, Any] | None = None) -> RunResult:
    """Run every case; ``target`` overrides the suite's configured target."""
    fn = target or resolve_target(suite, llm)
    results = [run_case(fn, c, suite) for c in suite.cases]
    info = {"git_sha": os.environ.get("GITHUB_SHA"), "ref": os.environ.get("GITHUB_REF")}
    info = {k: v for k, v in info.items() if v}
    info.update(meta or {})
    return RunResult(suite.name, results, info)
