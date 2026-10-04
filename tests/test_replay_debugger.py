import json

import pytest

from evalkit.cli import main
from evalkit.core.llm import MockLLM
from evalkit.core.trajectory import read_jsonl
from evalkit.replay_debugger import (
    Cassette,
    Override,
    ReplayDivergence,
    Replayer,
    bisect,
    load_cassettes,
    numeric_checker,
    parse_action,
    record,
    resolve_llm,
    save_cassettes,
    toy_tools,
)
from evalkit.replay_debugger.agent import expected_answer, safe_eval

TASK = "What do 3 widgets cost in EUR?"
EXPECTED = 11.73


def test_parse_action():
    assert parse_action('CALL calc {"expr": "1+1"}') == ("call", "calc", {"expr": "1+1"})
    assert parse_action("FINAL 12.50") == ("final", "", "12.50")
    assert parse_action("thinking...\nFINAL 3") == ("final", "", "3")
    assert parse_action("The FINAL answer is 3")[0] == "error"
    assert parse_action("CALL calc {not json}")[0] == "error"


def test_safe_eval_rejects_code():
    assert safe_eval("3 * 4.25 * 0.92") == pytest.approx(11.73)
    with pytest.raises(ValueError):
        safe_eval("__import__('os').getcwd()")


def test_record_careful_passes_and_roundtrips(tmp_path):
    c = record(TASK, resolve_llm("toy"), toy_tools(), task_id="w", meta={"expected": EXPECTED})
    assert [n.kind for n in c.nodes] == ["llm", "tool", "llm", "tool", "llm", "tool", "llm"]
    assert numeric_checker(EXPECTED)(c.final_answer)
    assert expected_answer(TASK) == EXPECTED
    p = tmp_path / "c.jsonl"
    save_cassettes(p, [c])
    row = read_jsonl(p)[0]
    assert [s["kind"] for s in row["steps"]][-1] == "final"
    back = load_cassettes(p)[0]
    assert [n.key for n in back.nodes] == [n.key for n in c.nodes]
    assert back.expected == EXPECTED


def test_replay_without_change_reproduces_from_cassette():
    c = record(TASK, resolve_llm("toy:sloppy"), toy_tools())
    res = Replayer(resolve_llm("toy:sloppy"), toy_tools()).replay(
        c, Override(1, output=c.nodes[1].output))
    assert res.replayed.final_answer == c.final_answer
    assert res.live_calls == 0  # everything after the swap is served from the cassette
    assert res.first_divergence is None


def test_replay_override_llm_node_fixes_hallucination():
    sloppy = resolve_llm("toy:sloppy")
    c = record(TASK, sloppy, toy_tools())
    assert not numeric_checker(EXPECTED)(c.final_answer)
    res = Replayer(sloppy, toy_tools()).replay(c, Override(2, llm=resolve_llm("toy")))
    assert numeric_checker(EXPECTED)(res.replayed.final_answer)
    assert [n.source for n in res.replayed.nodes[:3]] == ["cassette", "cassette", "override"]
    assert res.replayed.nodes[3].name == "fx_rate"
    assert res.first_divergence == 2


def test_replay_cached_vs_live_after_mode():
    calls = {"n": 0}

    def counting_calc(expr):
        calls["n"] += 1
        return round(safe_eval(expr), 4)

    tools = {**toy_tools(), "calc": counting_calc}
    c = record(TASK, resolve_llm("toy"), tools)
    calls["n"] = 0
    Replayer(resolve_llm("toy"), tools, after="cached").replay(c, Override(0, llm=resolve_llm("toy")))
    assert calls["n"] == 0
    Replayer(resolve_llm("toy"), tools, after="live").replay(c, Override(0, llm=resolve_llm("toy")))
    assert calls["n"] == 1


def test_bisect_finds_hallucinating_llm_hop():
    c = record(TASK, resolve_llm("toy:sloppy"), toy_tools())
    r = bisect(c, numeric_checker(EXPECTED), llm=resolve_llm("toy:sloppy"), tools=toy_tools(),
               ref_llm=resolve_llm("toy"), oracle_tools=toy_tools())
    assert not r.baseline_passed
    assert r.root_cause == 2 and r.culprits == [2]
    statuses = {t.index: t.status for t in r.trials}
    assert statuses[0] == statuses[1] == "identical"


def test_bisect_finds_faulty_tool_output():
    c = record(TASK, resolve_llm("toy"), toy_tools("stale_fx"))
    r = bisect(c, numeric_checker(EXPECTED), llm=resolve_llm("toy"), tools=toy_tools("stale_fx"),
               ref_llm=resolve_llm("toy"), oracle_tools=toy_tools())
    assert r.root_cause == 3
    assert c.nodes[3].name == "fx_rate"
    assert all(t.status == "identical" for t in r.trials if t.kind == "llm")


def test_bisect_reports_no_single_cause_for_two_faults():
    # sloppy model AND stale tool: fixing the hop exposes the stale tool, fixing nothing alone works
    c = record(TASK, resolve_llm("toy:sloppy"), toy_tools("stale_fx"))
    r = bisect(c, numeric_checker(EXPECTED), llm=resolve_llm("toy:sloppy"),
               tools=toy_tools("stale_fx"), ref_llm=resolve_llm("toy"), oracle_tools=toy_tools())
    assert r.root_cause is None


def test_bisect_skips_passing_runs():
    c = record(TASK, resolve_llm("toy"), toy_tools())
    r = bisect(c, numeric_checker(EXPECTED), llm=resolve_llm("toy"), tools=toy_tools())
    assert r.baseline_passed and r.trials == []


def test_divergence_detected_when_agent_changes():
    c = record(TASK, resolve_llm("toy"), toy_tools())
    c.nodes[0].input["messages"][0]["content"] = "an older system prompt"
    with pytest.raises(ReplayDivergence):
        Replayer(resolve_llm("toy"), toy_tools()).replay(c, Override(3, output=0.92))


def test_tool_errors_become_observations():
    llm = MockLLM(responder=lambda m: 'CALL lookup_price {"item": "unobtainium"}'
                  if len(m) == 2 else "FINAL 0")
    c = record("What do 1 unobtainium cost in EUR?", llm, toy_tools())
    assert str(c.nodes[1].output).startswith("ERROR: KeyError")
    assert c.final_answer == "0"


def test_cli_record_and_bisect(tmp_path, capsys):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"task_id": "w", "input": TASK, "expected": EXPECTED}) + "\n")
    cas = tmp_path / "c.jsonl"
    assert main(["replay-debugger", "record", "--tasks", str(tasks), "--out", str(cas),
                 "--llm", "toy:sloppy"]) == 0
    assert isinstance(load_cassettes(cas)[0], Cassette)
    capsys.readouterr()
    main(["replay-debugger", "bisect", "--cassettes", str(cas), "--llm", "toy:sloppy",
          "--ref-llm", "toy", "--oracle-tools", "toy"])
    out = capsys.readouterr().out
    assert "ROOT CAUSE: node 2" in out
    main(["replay-debugger", "replay", "--cassettes", str(cas), "--at", "2",
          "--llm", "toy:sloppy", "--output", 'CALL fx_rate {"base": "USD", "quote": "EUR"}'])
    assert "(PASS" in capsys.readouterr().out
