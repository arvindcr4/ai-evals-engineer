"""Baseline-vs-candidate gate decision and the Markdown PR report.

A drop in task success blocks only when it is both larger than the policy
threshold *and* statistically significant (paired, clustered by dataset item
so parameter variants of one item are not counted as independent evidence).
Latency blocks when p95 rises by more than the relative threshold and by more
than an absolute floor, so sub-millisecond jitter cannot fail a build.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from evalkit.regression_gate.runner import CaseResult, RunResult
from evalkit.regression_gate.suite import GatePolicy
from evalkit.significance import Comparison, compare

REPORT_MARKER = "<!-- evalkit-regression-gate -->"


@dataclass
class GateReport:
    passed: bool
    reasons: list[str]
    baseline: RunResult
    candidate: RunResult
    policy: GatePolicy
    comparison: Comparison | None
    p95_base: float
    p95_cand: float
    newly_failing: list[tuple[CaseResult, CaseResult]] = field(default_factory=list)
    newly_passing: list[str] = field(default_factory=list)
    missing_cases: list[str] = field(default_factory=list)
    added_cases: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    @property
    def p95_change(self) -> float:
        return (self.p95_cand - self.p95_base) / self.p95_base if self.p95_base > 0 else 0.0

    def to_markdown(self, max_rows: int = 10) -> str:
        status = "PASS" if self.passed else "FAIL — merge blocked"
        b, c = self.baseline.summary(), self.candidate.summary()
        lines = [REPORT_MARKER, f"## Eval regression gate: **{status}**", "",
                 f"Suite `{self.candidate.suite}` · {c['n_cases']} cases"]
        if self.comparison is not None:
            lines += ["", f"> {self.comparison.verdict()}"]
        lines += ["", "| Metric | Baseline | Candidate | Δ | Limit |", "|---|---:|---:|---:|---|",
                  f"| Task success | {b['success_rate']:.1%} | {c['success_rate']:.1%} | "
                  f"{(c['success_rate'] - b['success_rate']) * 100:+.1f} pts | "
                  f"drop ≤ {self.policy.max_success_drop * 100:.1f} pts"
                  + (" or not significant" if self.policy.require_significance else "") + " |",
                  (f"| Latency p50 | {b['latency_p50_s'] * 1000:.1f} ms | "
                  f"{c['latency_p50_s'] * 1000:.1f} ms | "
                  f"{(c['latency_p50_s'] - b['latency_p50_s']) * 1000:+.1f} ms | — |"),
                  (f"| Latency p95 | {self.p95_base * 1000:.1f} ms | {self.p95_cand * 1000:.1f} ms"
                  f" | {self.p95_change:+.0%} | rise ≤ {self.policy.max_p95_latency_increase:.0%}"
                  f" (or < {self.policy.min_latency_increase_ms:g} ms) |"),
                  f"| Errors | {b['errors']} | {c['errors']} | {c['errors'] - b['errors']:+d} | — |"]
        for name in sorted(set(b["scorer_pass_rates"]) | set(c["scorer_pass_rates"])):
            pb, pc = b["scorer_pass_rates"].get(name), c["scorer_pass_rates"].get(name)
            if pb is None or pc is None:
                continue
            lines.append(f"| `{name}` | {pb:.1%} | {pc:.1%} | {(pc - pb) * 100:+.1f} pts | — |")
        if self.reasons:
            lines += ["", "**Blocking:**"] + [f"- {r}" for r in self.reasons]
        if self.newly_failing:
            lines += ["", (f"<details><summary>{len(self.newly_failing)} newly failing cases"
                          f"</summary>"), "", "| Case | Failed check: got → expected |",
                      "|---|---|"]
            for _, cand in self.newly_failing[:max_rows]:
                if cand.error:
                    detail = f"error: {cand.error}"
                else:
                    detail = "; ".join(
                        f"`{k}`: {cand.got.get(k)!r} → {cand.expected.get(k)!r}"
                        for k, ok in cand.checks.items() if not ok)
                detail = detail.replace("|", "\\|")[:200]
                lines.append(f"| `{cand.case_id}` | {detail} |")
            if len(self.newly_failing) > max_rows:
                lines.append(f"| … {len(self.newly_failing) - max_rows} more | |")
            lines += ["", "</details>"]
        extras = []
        if self.newly_passing:
            extras.append(f"{len(self.newly_passing)} newly passing")
        if self.missing_cases:
            extras.append(f"{len(self.missing_cases)} baseline cases missing from candidate")
        if self.added_cases:
            extras.append(f"{len(self.added_cases)} new cases (not gated)")
        if extras:
            lines += ["", "_" + "; ".join(extras) + "._"]
        return "\n".join(lines) + "\n"


def evaluate_gate(baseline: RunResult, candidate: RunResult,
                  policy: GatePolicy | None = None) -> GateReport:
    """Decide pass/fail for ``candidate`` against ``baseline`` on their shared cases."""
    policy = policy or GatePolicy()
    base = {c.case_id: c for c in baseline.cases}
    cand = {c.case_id: c for c in candidate.cases}
    shared = [k for k in base if k in cand]
    reasons: list[str] = []
    comparison = None
    if len(shared) >= 2:
        table = {
            "baseline": {k: (float(base[k].success), base[k].item_id) for k in shared},
            "candidate": {k: (float(cand[k].success), cand[k].item_id) for k in shared},
        }
        comparison = compare(table, "baseline", "candidate", n_boot=policy.n_boot,
                             confidence=1 - policy.alpha, alpha=policy.alpha)
        drop = -comparison.delta
        if drop > policy.max_success_drop:
            if not policy.require_significance:
                reasons.append(f"task success dropped {drop * 100:.1f} pts "
                               f"(limit {policy.max_success_drop * 100:.1f})")
            elif comparison.significant:
                reasons.append(f"task success dropped {drop * 100:.1f} pts "
                               f"(limit {policy.max_success_drop * 100:.1f}), significant at "
                               f"p={comparison.p_value:.3g}")
    elif not shared:
        reasons.append("no cases in common with the baseline — regenerate the baseline")
    p95_b = RunResult(baseline.suite, [base[k] for k in shared]).latency_quantile(0.95)
    p95_c = RunResult(candidate.suite, [cand[k] for k in shared]).latency_quantile(0.95)
    rise = p95_c - p95_b
    if p95_b > 0 and rise / p95_b > policy.max_p95_latency_increase and \
            rise * 1000 >= policy.min_latency_increase_ms:
        reasons.append(f"p95 latency rose {rise / p95_b:+.0%} ({p95_b * 1000:.1f} → "
                       f"{p95_c * 1000:.1f} ms, limit +{policy.max_p95_latency_increase:.0%})")
    return GateReport(
        passed=not reasons, reasons=reasons, baseline=baseline, candidate=candidate,
        policy=policy, comparison=comparison, p95_base=p95_b, p95_cand=p95_c,
        newly_failing=[(base[k], cand[k]) for k in shared if base[k].success and not
                       cand[k].success],
        newly_passing=[k for k in shared if cand[k].success and not base[k].success],
        missing_cases=[k for k in base if k not in cand],
        added_cases=[k for k in cand if k not in base],
    )


def write_step_summary(markdown: str) -> Path | None:
    """Append to ``$GITHUB_STEP_SUMMARY`` when running inside GitHub Actions."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return None
    with open(target, "a") as f:
        f.write(markdown + "\n")
    return Path(target)
