import json
from pathlib import Path

import pytest

from evalkit.cli import main
from evalkit.core.trajectory import Step, Trajectory, write_jsonl
from evalkit.trajectory_grader import SpecError, TrajectoryGrader, parse_spec

SPEC = {
    "name": "refund",
    "max_steps": 6,
    "required": ["lookup_order", "issue_refund"],
    "forbidden": ["delete_account"],
    "tools": {
        "lookup_order": {"params": {"order_id": {"type": "string", "required": True}}},
        "verify_identity": {
            "safety": True,
            "params": {
                "customer_id": {"type": "string", "required": True},
                "method": {"type": "string", "enum": ["otp", "kba"], "required": True},
            },
        },
        "issue_refund": {
            "requires": ["lookup_order", "verify_identity"],
            "max_calls": 1,
            "params": {
                "order_id": {"type": "string", "required": True},
                "amount": {"type": "number", "required": True, "min": 0, "max": 500},
            },
        },
        "send_email": {"requires": ["issue_refund"], "params": {"to": "string"}},
    },
}

LO = ("lookup_order", {"order_id": "O1"})
VI = ("verify_identity", {"customer_id": "C1", "method": "otp"})
IR = ("issue_refund", {"order_id": "O1", "amount": 10.0})


def traj(*calls, task_id="t"):
    steps = []
    for name, args in calls:
        steps.append(Step(kind="llm", name="m", output="thinking"))
        steps.append(Step(kind="tool_call", name=name, args=args))
        steps.append(Step(kind="tool_result", name=name, output="ok"))
    return Trajectory(task_id=task_id, input="refund please", steps=steps)


@pytest.fixture
def grader():
    return TrajectoryGrader(parse_spec(SPEC))


def test_good_run_passes(grader):
    r = grader.grade(traj(LO, VI, IR, ("send_email", {"to": "a@b"})))
    assert r.passed and r.score == 1.0 and r.first_failure is None
    assert [s.verdict for s in r.steps] == ["ok"] * 4


def test_hallucinated_param_fails_at_exact_step(grader):
    r = grader.grade(traj(LO, VI, ("issue_refund", {"order_id": "O1", "amount": 5, "rush": 1})))
    assert not r.passed
    assert r.verdict == "hallucinated_param"
    assert r.first_failure == 7  # steps: llm,call,result x2 then llm -> index 7
    assert r.steps[2].verdict == "hallucinated_param"


def test_skipped_safety_check(grader):
    r = grader.grade(traj(LO, IR))
    assert r.verdict == "skipped_safety_check"
    assert any(i.code == "skipped_safety_check" for i in r.run_issues)


def test_safety_check_after_action_is_still_a_violation(grader):
    r = grader.grade(traj(LO, IR, VI))
    assert not r.passed
    assert r.steps[1].verdict == "skipped_safety_check"
    assert r.steps[2].verdict == "ok"


def test_order_violation_and_dependency_cascade(grader):
    r = grader.grade(traj(VI, IR, LO, ("send_email", {"to": "x"})))
    assert r.steps[1].verdict == "order_violation"
    assert r.steps[3].verdict == "dependency_failed"
    assert r.first_failure == r.steps[1].step_index


def test_failed_safety_call_does_not_satisfy_dependency(grader):
    bad_vi = ("verify_identity", {"customer_id": "C1", "method": "sms"})
    r = grader.grade(traj(LO, bad_vi, IR))
    assert r.steps[1].verdict == "invalid_value"
    assert r.steps[2].verdict == "dependency_failed"
    assert any(i.code == "skipped_safety_check" for i in r.run_issues)


@pytest.mark.parametrize(
    "args,code",
    [
        ({"order_id": "O1"}, "missing_required_param"),
        ({"order_id": "O1", "amount": "10"}, "type_error"),
        ({"order_id": "O1", "amount": True}, "type_error"),
        ({"order_id": "O1", "amount": 9999}, "invalid_value"),
        ({"order_id": "O1", "amount": -1}, "invalid_value"),
    ],
)
def test_param_schema_errors(grader, args, code):
    r = grader.grade(traj(LO, VI, ("issue_refund", args)))
    assert r.steps[2].verdict == code and not r.passed


def test_unknown_and_forbidden_tools(grader):
    r = grader.grade(traj(LO, ("delete_account", {}), VI, ("get_weather", {}), IR))
    assert r.steps[1].verdict == "forbidden_tool"
    assert r.steps[3].verdict == "unknown_tool"
    assert r.verdict == "forbidden_tool"


def test_soft_issues_lower_score_but_pass(grader):
    r = grader.grade(traj(LO, LO, VI, IR))
    assert r.passed
    assert r.steps[1].verdict == "redundant"
    assert r.score == pytest.approx(0.9)


def test_max_calls_and_max_steps_are_soft(grader):
    ir2 = ("issue_refund", {"order_id": "O1", "amount": 11.0})
    r = grader.grade(traj(LO, VI, IR, ir2, LO, VI, ("lookup_order", {"order_id": "O2"})))
    codes = {i.code for s in r.steps for i in s.issues}
    assert {"max_calls_exceeded", "max_steps_exceeded"} <= codes
    assert r.passed and r.score < 1.0


def test_missing_required_step(grader):
    r = grader.grade(traj(LO, VI))
    assert not r.passed and r.first_failure is None
    assert r.verdict == "missing_required_step"


def test_spec_validation():
    bad = {"tools": {"a": {"requires": ["b"]}}}
    with pytest.raises(SpecError):
        parse_spec(bad)
    cyc = {"tools": {"a": {"requires": ["b"]}, "b": {"requires": ["a"]}}}
    with pytest.raises(SpecError, match="cycle"):
        parse_spec(cyc)
    with pytest.raises(SpecError):
        parse_spec({"tools": {"a": {"params": {"x": "complex"}}}})
    spec = parse_spec(SPEC)
    order = spec.topological_order()
    assert order.index("verify_identity") < order.index("issue_refund") < order.index("send_email")
    assert spec.ancestors("send_email") == {"issue_refund", "lookup_order", "verify_identity"}


def test_cli_grade_json(tmp_path: Path, capsys):
    import yaml

    spec_p = tmp_path / "spec.yaml"
    spec_p.write_text(yaml.safe_dump(SPEC))
    runs_p = tmp_path / "runs.jsonl"
    write_jsonl(runs_p, [traj(LO, VI, IR, task_id="good"), traj(LO, IR, task_id="bad")])
    assert main(["trajectory-grader", "grade", "--spec", str(spec_p),
                 "--trajectories", str(runs_p), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["passed"] == 1
    assert [r["verdict"] for r in out["runs"]] == ["pass", "skipped_safety_check"]
    assert main(["trajectory-grader", "grade", "--spec", str(spec_p),
                 "--trajectories", str(runs_p), "--strict", "--quiet"]) == 1


def test_example_dataset_verdicts():
    root = Path(__file__).resolve().parents[1] / "examples" / "01-trajectory-grader"
    from evalkit.core.trajectory import read_jsonl
    from evalkit.trajectory_grader import load_spec

    g = TrajectoryGrader(load_spec(root / "spec.yaml"))
    verdicts = {d["task_id"]: g.grade(Trajectory.from_dict(d)).verdict
                for d in read_jsonl(root / "runs.jsonl")}
    assert verdicts["good-path"] == "pass"
    assert verdicts["hallucinated-param"] == "hallucinated_param"
    assert verdicts["skipped-safety-check"] == "skipped_safety_check"
    assert verdicts["out-of-order"] == "order_violation"
