"""Execute a suite against a target and record success + latency per case."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


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
            "tokens_in": sum(c.tokens_in for c in self.cases),
            "tokens_out": sum(c.tokens_out for c in self.cases),
            "cost_usd": sum(c.cost_usd for c in self.cases),
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


class TruncatedOutputError(RuntimeError):
    """The model stopped at ``max_tokens`` (``finish_reason == "length"``)."""

    def __init__(self, message: str, usage: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.usage = usage or {}  # the tokens were still billed


# Keys an llm target may set. Anything else is a typo that would otherwise be
# silently ignored (e.g. ``temprature: 0`` leaving sampling on).
LLM_TARGET_KEYS = {"llm", "prompt", "system", "workers"}
LLM_GEN_KEYS = {"temperature", "max_tokens", "top_p", "seed", "stop", "response_format"}

_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_template(template: str, values: dict[str, Any]) -> str:
    """Fill ``{name}`` placeholders, leaving every other brace alone.

    ``str.format`` raises on the literal JSON a prompt usually contains
    (``Answer like {"category": ...}``); here only ``{identifier}`` is a
    placeholder, ``{{``/``}}`` still mean literal braces, and an unknown
    ``{identifier}`` is an error so a misspelt field can't ship silently.
    """
    def sub(m: re.Match) -> str:
        if m.group(0) == "{{":
            return "{"
        if m.group(0) == "}}":
            return "}"
        name = m.group(1)
        if name not in values:
            raise KeyError(f"prompt placeholder {{{name}}} has no input field or param")
        return str(values[name])

    return _PLACEHOLDER.sub(sub, template)


def _llm_target(cfg: dict[str, Any], llm: LLM | None) -> Target:
    unknown = set(cfg) - LLM_TARGET_KEYS - LLM_GEN_KEYS
    if unknown:
        raise ValueError(f"unknown llm target keys: {sorted(unknown)}; allowed: "
                         f"{sorted(LLM_TARGET_KEYS | LLM_GEN_KEYS)}")
    model = llm or get_llm(cfg.get("llm"))
    template: str = cfg.get("prompt", "{input}")
    system: str | None = cfg.get("system")
    gen = {k: cfg[k] for k in sorted(LLM_GEN_KEYS) if k in cfg}

    def run(inp: dict[str, Any], **params: Any) -> tuple[str, float, dict[str, Any]]:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": render_template(template, {**inp, **params})}]
        comp = model.complete(messages, **gen)
        usage = {"tokens_in": comp.tokens_in, "tokens_out": comp.tokens_out,
                 "cost_usd": comp.cost_usd}
        choices = comp.raw.get("choices") or [{}]
        if choices[0].get("finish_reason") == "length":
            # A reasoning model can spend the whole max_tokens budget thinking and
            # return "" or half a JSON object; that is a config/infra failure,
            # not a wrong answer, so surface it as an error rather than a miss.
            raise TruncatedOutputError(
                f"output hit max_tokens={gen.get('max_tokens', 'default')} "
                f"({comp.tokens_out} tokens out); partial text: {comp.text[:60]!r}", usage)
        return comp.text, comp.latency_s, usage

    thinking = (getattr(model, "extra_body", None) or {}).get("thinking", {})
    label = model.model + ("+think" if thinking.get("type") == "enabled" else "")
    fingerprint = json.dumps({"system": system, "prompt": template, **gen}, sort_keys=True)
    run.reports_latency = True  # type: ignore[attr-defined]
    run.workers = int(cfg.get("workers", 1))  # type: ignore[attr-defined]
    run.meta = {"model": label,  # type: ignore[attr-defined]
                "prompt_sha": hashlib.sha256(fingerprint.encode()).hexdigest()[:12], **gen}
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
        usage: dict[str, Any] = {}
        if getattr(target, "reports_latency", False):
            out, latency, *rest = out
            usage = rest[0] if rest else {}
        error = None
    except Exception as exc:  # noqa: BLE001 — a crashing case fails, the gate keeps going
        out, latency, error = None, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"
        usage = getattr(exc, "usage", {})
    checks: dict[str, bool] = {}
    got: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for sc in suite.scorers:
        ok, g, exp = sc.score(out, case) if error is None else (False, None, None)
        checks[sc.name], got[sc.name], expected[sc.name] = bool(ok), g, exp
    return CaseResult(case.case_id, case.item_id, case.params, error is None and all(
        checks.values()), checks, float(latency), out, got, expected, error,
        int(usage.get("tokens_in", 0)), int(usage.get("tokens_out", 0)),
        float(usage.get("cost_usd", 0.0)))


def run_suite(suite: Suite, target: Target | None = None, llm: LLM | None = None,
              meta: dict[str, Any] | None = None, workers: int | None = None) -> RunResult:
    """Run every case; ``target`` overrides the suite's configured target.

    ``workers`` > 1 runs cases concurrently (useful for network-bound LLM
    targets; default: the target's ``workers`` setting, else 1). Result order
    always follows the suite.
    """
    fn = target or resolve_target(suite, llm)
    n = max(1, workers or getattr(fn, "workers", 1))
    if n == 1:
        results = [run_case(fn, c, suite) for c in suite.cases]
    else:
        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(lambda c: run_case(fn, c, suite), suite.cases))
    info = {"git_sha": os.environ.get("GITHUB_SHA"), "ref": os.environ.get("GITHUB_REF")}
    info = {k: v for k, v in info.items() if v}
    info.update(getattr(fn, "meta", {}))
    info.update(meta or {})
    return RunResult(suite.name, results, info)
