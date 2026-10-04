"""Step-level trajectory grading against a :class:`GradingSpec`.

The grader walks every ``tool_call`` step of a trajectory in order and checks
it against the spec: is the tool known and allowed, are its arguments real
(no hallucinated parameters), complete, well-typed and in range, and have all
of its DAG prerequisites already *succeeded*? Safety-check prerequisites get
their own verdict because skipping one is the failure that matters most.

Hard issues fail the run and mark the first failing step; soft issues only
lower the score. A run-level pass also requires every mandatory tool (declared
``required`` + all safety checks) to have succeeded at least once.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from evalkit.core.trajectory import Step, Trajectory
from evalkit.trajectory_grader.spec import GradingSpec, ParamSpec

HARD = {
    "unknown_tool",
    "forbidden_tool",
    "hallucinated_param",
    "missing_required_param",
    "type_error",
    "invalid_value",
    "order_violation",
    "dependency_failed",
    "skipped_safety_check",
    "missing_required_step",
}
SOFT = {"redundant", "max_calls_exceeded", "max_steps_exceeded"}
SEVERITY_ORDER = [
    "forbidden_tool",
    "unknown_tool",
    "skipped_safety_check",
    "order_violation",
    "hallucinated_param",
    "missing_required_param",
    "type_error",
    "invalid_value",
    "dependency_failed",
    "max_calls_exceeded",
    "redundant",
    "max_steps_exceeded",
]


@dataclass
class Issue:
    code: str
    detail: str

    @property
    def hard(self) -> bool:
        return self.code in HARD


@dataclass
class StepVerdict:
    step_index: int
    call_index: int
    tool: str
    args: dict[str, Any]
    issues: list[Issue] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not self.issues:
            return "ok"
        codes = [i.code for i in self.issues]
        return min(codes, key=lambda c: SEVERITY_ORDER.index(c) if c in SEVERITY_ORDER else 99)

    @property
    def hard(self) -> bool:
        return any(i.hard for i in self.issues)


@dataclass
class RunReport:
    task_id: str
    passed: bool
    score: float
    first_failure: int | None
    steps: list[StepVerdict]
    run_issues: list[Issue] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.passed:
            return "pass"
        if self.first_failure is not None:
            return next(s.verdict for s in self.steps if s.step_index == self.first_failure)
        return self.run_issues[0].code if self.run_issues else "below_threshold"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        for s, sd in zip(self.steps, d["steps"]):
            sd["verdict"] = s.verdict
        return d


def _check_value(p: ParamSpec, value: Any) -> Issue | None:
    if not p.type_ok(value):
        return Issue("type_error", f"{p.name}: expected {p.type}, got {type(value).__name__}")
    if p.enum is not None and value not in p.enum:
        return Issue("invalid_value", f"{p.name}={value!r} not in {p.enum}")
    if p.min is not None and value < p.min:
        return Issue("invalid_value", f"{p.name}={value!r} < min {p.min}")
    if p.max is not None and value > p.max:
        return Issue("invalid_value", f"{p.name}={value!r} > max {p.max}")
    if p.pattern is not None and not re.fullmatch(p.pattern, str(value)):
        return Issue("invalid_value", f"{p.name}={value!r} does not match /{p.pattern}/")
    return None


def _args_key(name: str, args: dict[str, Any]) -> str:
    return name + json.dumps(args, sort_keys=True, default=str)


class TrajectoryGrader:
    """Grades trajectories step by step against a spec."""

    def __init__(self, spec: GradingSpec):
        self.spec = spec

    def check_args(self, tool: str, args: dict[str, Any]) -> list[Issue]:
        """Schema checks for one call's arguments (no ordering)."""
        t = self.spec.tools[tool]
        issues: list[Issue] = []
        if not t.allow_extra_params:
            for a in args:
                if a not in t.params:
                    issues.append(
                        Issue("hallucinated_param", f"{tool} has no parameter {a!r}")
                    )
        for p in t.params.values():
            if p.name not in args:
                if p.required:
                    issues.append(Issue("missing_required_param", f"{tool} needs {p.name!r}"))
                continue
            bad = _check_value(p, args[p.name])
            if bad:
                issues.append(bad)
        return issues

    def grade(self, traj: Trajectory) -> RunReport:
        spec = self.spec
        succeeded: set[str] = set()
        failed_at: dict[str, int] = {}
        seen_calls: set[str] = set()
        call_counts: dict[str, int] = {}
        verdicts: list[StepVerdict] = []
        calls = [(i, s) for i, s in enumerate(traj.steps) if s.kind == "tool_call"]

        for call_idx, (step_idx, step) in enumerate(calls):
            v = self._grade_call(
                step_idx, call_idx, step, succeeded, failed_at, seen_calls, call_counts
            )
            if not v.hard and step.name in spec.tools:
                succeeded.add(step.name)
            elif v.hard:
                failed_at.setdefault(step.name, step_idx)
            verdicts.append(v)

        run_issues: list[Issue] = []
        for tool in spec.mandatory:
            if tool not in succeeded:
                code = "skipped_safety_check" if spec.tools[tool].safety else "missing_required_step"
                run_issues.append(Issue(code, f"{tool} never succeeded in this run"))

        n_hard = sum(1 for v in verdicts for i in v.issues if i.hard) + len(run_issues)
        n_soft = sum(1 for v in verdicts for i in v.issues if not i.hard)
        score = max(0.0, 1.0 - spec.weights.hard * n_hard - spec.weights.soft * n_soft)
        first = next((v.step_index for v in verdicts if v.hard), None)
        passed = n_hard == 0 and score >= spec.pass_threshold
        return RunReport(traj.task_id, passed, round(score, 4), first, verdicts, run_issues)

    def _grade_call(
        self,
        step_idx: int,
        call_idx: int,
        step: Step,
        succeeded: set[str],
        failed_at: dict[str, int],
        seen_calls: set[str],
        call_counts: dict[str, int],
    ) -> StepVerdict:
        spec = self.spec
        name, args = step.name, dict(step.args or {})
        v = StepVerdict(step_idx, call_idx, name, args)
        if name in spec.forbidden:
            v.issues.append(Issue("forbidden_tool", f"{name} is forbidden"))
            return v
        if name not in spec.tools:
            v.issues.append(Issue("unknown_tool", f"{name} is not a declared tool"))
            return v
        if spec.max_steps is not None and call_idx >= spec.max_steps:
            v.issues.append(Issue("max_steps_exceeded", f"call #{call_idx + 1} > {spec.max_steps}"))

        tool = spec.tools[name]
        for dep in tool.requires:
            if dep in succeeded:
                continue
            if dep in failed_at:
                v.issues.append(Issue(
                    "dependency_failed",
                    f"{name} ran although prerequisite {dep} failed at step {failed_at[dep]}",
                ))
            elif spec.tools[dep].safety:
                v.issues.append(
                    Issue("skipped_safety_check", f"{name} ran before safety check {dep}")
                )
            else:
                v.issues.append(Issue("order_violation", f"{name} ran before {dep}"))
        v.issues.extend(self.check_args(name, args))

        call_counts[name] = call_counts.get(name, 0) + 1
        if tool.max_calls is not None and call_counts[name] > tool.max_calls:
            v.issues.append(
                Issue("max_calls_exceeded", f"{name} called {call_counts[name]}x > {tool.max_calls}")
            )
        key = _args_key(name, args)
        if key in seen_calls:
            v.issues.append(Issue("redundant", f"identical repeat of {name}"))
        seen_calls.add(key)
        return v


@dataclass
class Summary:
    n: int
    passed: int
    mean_score: float
    verdict_counts: dict[str, int]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0


def summarize(reports: list[RunReport]) -> Summary:
    counts: dict[str, int] = {}
    for r in reports:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    n = len(reports)
    mean = sum(r.score for r in reports) / n if n else 0.0
    return Summary(n, sum(r.passed for r in reports), round(mean, 4), counts)


def format_report(reports: list[RunReport], verbose: bool = True) -> str:
    """Human-readable table of run verdicts plus the failing steps."""
    lines = [f"{'task':<24} {'verdict':<22} {'score':>6}  first_fail"]
    for r in reports:
        ff = "-" if r.first_failure is None else f"step {r.first_failure}"
        lines.append(f"{r.task_id:<24} {r.verdict:<22} {r.score:>6.2f}  {ff}")
        if verbose:
            for s in r.steps:
                for i in s.issues:
                    tag = "HARD" if i.hard else "soft"
                    lines.append(f"    [{tag}] step {s.step_index} {s.tool}: {i.code} - {i.detail}")
            for i in r.run_issues:
                lines.append(f"    [HARD] run: {i.code} - {i.detail}")
    s = summarize(reports)
    lines.append(
        f"\n{s.passed}/{s.n} runs passed ({s.pass_rate:.0%}), mean score {s.mean_score:.2f}"
    )
    lines.append("verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(s.verdict_counts.items())))
    return "\n".join(lines)
