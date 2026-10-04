# 01 — Trajectory Grading Engine

> Final-answer evals hide the exact step where agents break.

## What and why

A step-level evaluator that grades every tool call an agent makes against a
deterministic contract: a YAML spec of tools, parameter schemas and a dependency
DAG. A run fails the moment the agent hallucinates an API parameter, sends a
string where a number belongs, calls a tool before its prerequisites, or skips a
safety check, and the report gives the index of the **first failing step**.

A refund agent that skips `verify_identity` and still produces "Refund issued!"
passes any final-answer eval. Graded step by step, it fails at step 4 with
`skipped_safety_check`.

## Design

```
spec.yaml ──parse/validate──▶ GradingSpec ─┐
runs.jsonl (core Trajectory) ──────────────┴─▶ TrajectoryGrader.grade() ─▶ RunReport per run
```

**Spec** (`trajectory_grader/spec.py`):

| key | meaning |
|---|---|
| `tools.<name>.params` | `{type, required, enum, min, max, pattern}`; types `string integer number boolean array object` (bools are not numbers) |
| `tools.<name>.requires` | DAG edges: tools that must have **succeeded** earlier |
| `tools.<name>.safety` | safety check: mandatory in every run, and its own verdict when skipped |
| `tools.<name>.max_calls` / `allow_extra_params` | call cap; opt out of the hallucinated-param check |
| `required` / `forbidden` | tools that must succeed at least once / must never be called |
| `max_steps`, `pass_threshold`, `weights.{hard,soft}` | budget, minimum score, penalties |

At load time the spec is validated: dangling `requires`, undeclared `required`
tools, tools that are both declared and forbidden, unknown types and dependency
cycles all raise `SpecError`. `check-spec` prints a topological order.

**Grading** (`grader.py`). The grader walks `tool_call` steps in order and keeps
the set of tools that have *succeeded* so far. A call that has a hard issue does
not satisfy later dependencies, so passing a malformed `verify_identity` does
not unlock `issue_refund`. Per-step verdicts:

| verdict | severity | trigger |
|---|---|---|
| `forbidden_tool` | hard | tool is in `forbidden` |
| `unknown_tool` | hard | tool not declared |
| `skipped_safety_check` | hard | a `safety: true` prerequisite has not succeeded (also raised at run level if a safety tool never succeeds) |
| `order_violation` | hard | a non-safety prerequisite has not run yet |
| `hallucinated_param` | hard | argument not in the tool's schema |
| `missing_required_param` / `type_error` / `invalid_value` | hard | schema, enum, range or pattern failures |
| `dependency_failed` | hard | prerequisite was attempted but failed: a cascade, kept separate from the root cause |
| `max_calls_exceeded`, `redundant`, `max_steps_exceeded` | soft | call caps, identical repeated calls, step budget |
| `missing_required_step` | hard (run) | a `required` tool never succeeded |

When a step has several issues, its verdict is the most severe one. Score is
`1 − hard·n_hard − soft·n_soft`, floored at 0. A run passes when it has no hard
issues and its score is at least `pass_threshold`. The run verdict is the verdict
of the first failing step, or the first run-level issue.

The grader is deliberately LLM-free: the contract is deterministic, so the
verdict is reproducible and cheap enough to run on every CI trajectory. It reads
core `Trajectory` JSONL, which means cassettes recorded by the replay debugger
(#11) can be graded directly.

## How to run

```bash
uv run --no-sync evalkit trajectory-grader check-spec --spec examples/01-trajectory-grader/spec.yaml
uv run --no-sync evalkit trajectory-grader grade --spec examples/01-trajectory-grader/spec.yaml \
    --trajectories examples/01-trajectory-grader/runs.jsonl [--json] [--quiet] [--strict]
examples/01-trajectory-grader/run.sh
```

`--strict` exits 1 if any run fails, which makes it usable as a CI gate.

## Sample output (from `run.sh`)

The example is a customer-support refund agent: 5 tools, `verify_identity` as the
safety check, and 8 runs.

```
spec 'refund-agent': 5 tools, DAG order lookup_order -> verify_identity -> check_refund_policy -> issue_refund -> send_email
mandatory: ['lookup_order', 'issue_refund', 'verify_identity']  safety: ['verify_identity']  forbidden: ['delete_account', 'override_fraud_flag']

task                     verdict                 score  first_fail
good-path                pass                     1.00  -
hallucinated-param       hallucinated_param       0.00  step 6
    [HARD] step 6 issue_refund: hallucinated_param - issue_refund has no parameter 'expedite'
    [HARD] step 8 send_email: dependency_failed - send_email ran although prerequisite issue_refund failed at step 6
    [HARD] run: missing_required_step - issue_refund never succeeded in this run
skipped-safety-check     skipped_safety_check     0.00  step 4
    [HARD] step 4 issue_refund: skipped_safety_check - issue_refund ran before safety check verify_identity
    [HARD] step 6 send_email: dependency_failed - send_email ran although prerequisite issue_refund failed at step 4
    [HARD] run: missing_required_step - issue_refund never succeeded in this run
    [HARD] run: skipped_safety_check - verify_identity never succeeded in this run
out-of-order             order_violation          0.00  step 4
    [HARD] step 4 issue_refund: order_violation - issue_refund ran before check_refund_policy
    ...
type-error-amount        type_error               0.00  step 6
    [HARD] step 6 issue_refund: type_error - amount: expected number, got str
    ...
forbidden-override       forbidden_tool           0.50  step 2
    [HARD] step 2 override_fraud_flag: forbidden_tool - override_fraud_flag is forbidden
redundant-lookups        pass                     0.90  -
    [soft] step 2 lookup_order: redundant - identical repeat of lookup_order
bad-enum-and-unknown     invalid_value            0.00  step 2
    [HARD] step 2 verify_identity: invalid_value - method='sms' not in ['otp', 'kba']
    [HARD] step 4 lookup_customer: unknown_tool - lookup_customer is not a declared tool
    ...

2/8 runs passed (25%), mean score 0.30
verdicts: forbidden_tool=1, hallucinated_param=1, invalid_value=1, order_violation=1, pass=2, skipped_safety_check=1, type_error=1
strict: exit 1 (failing runs present) — as expected for this dataset
```

## Limitations

- Dependencies are all-of edges between tool *names*. The grader cannot express
  "either OTP or KBA", argument data-flow (the refunded `order_id` must equal the
  looked-up one), or conditions on tool *results*, such as "refund only if the
  policy check returned eligible".
- Tool results are not inspected, so a call that errored at runtime still counts
  as succeeded if its arguments were valid.
- Safety checks are global per run. If a run handles two customers, one
  verification satisfies both.
- Scoring is a simple linear penalty. Calibrate `weights` and `pass_threshold`
  per product.
