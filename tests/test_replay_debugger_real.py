"""Regression tests for integration bugs seen running the replay debugger on DeepSeek.

Each MockLLM responder mimics a real reply shape observed in the Oct 2026 run.
"""

import pytest

from evalkit.cli import main
from evalkit.core.llm import MockLLM
from evalkit.replay_debugger import (
    Override,
    Replayer,
    bisect,
    numeric_checker,
    parse_action,
    record,
    toy_policy,
    toy_tools,
)
from evalkit.replay_debugger.agent import _ACTION

TASK = "What do 4 gizmos cost in INR?"

# deepseek-flash, gizmos-inr hop 4 on the stale tool: one sentence of reasoning, a blank
# line, then the action. The old first-line parser rejected it and burned an extra hop.
PROSE_THEN_CALL = ("I need the INR-per-USD rate, but the quote returned 0.012 (which appears "
                   'to be USD per INR). I\'ll compute using that.\n\nCALL calc {"expr": '
                   '"4 * 7.5 / 0.012"}')


@pytest.mark.parametrize("text, expected", [
    (PROSE_THEN_CALL, ("call", "calc", {"expr": "4 * 7.5 / 0.012"})),
    ('```\nCALL fx_rate {"base": "USD",\n "quote": "EUR"}\n```',
     ("call", "fx_rate", {"base": "USD", "quote": "EUR"})),
    ('```text\nCALL lookup_price {"item": "gizmo"}\n```',
     ("call", "lookup_price", {"item": "gizmo"})),
    ("**FINAL** 11.73", ("final", "", "11.73")),
    ("FINAL: 11.73", ("final", "", "11.73")),
    ("`FINAL 11.73`", ("final", "", "11.73")),
    ("FINAL 13.86\nbecause 3 * 4.25 * 1.087 = 13.86", ("final", "", "13.86")),
    # one action per hop: a CALL followed by an anticipated FINAL executes only the CALL
    ('CALL calc {"expr": "2 * 12.0"}\nFINAL 24.00', ("call", "calc", {"expr": "2 * 12.0"})),
])
def test_parse_action_real_model_shapes(text, expected):
    assert parse_action(text) == expected


def test_parse_action_still_rejects_non_protocol_prose():
    assert parse_action("The FINAL answer is 3")[0] == "error"
    assert parse_action("Let me look up the price first.")[0] == "error"
    assert parse_action('CALL calc {"expr": oops}')[0] == "error"
    assert _ACTION.match("FINALLY done") is None


def _prose_policy(messages):
    """Careful policy that prefixes every action with a reasoning sentence."""
    return "Thinking about the next step.\n\n" + toy_policy(False)(messages)


def test_prose_prefixed_agent_still_completes_in_minimal_hops():
    c = record(TASK, MockLLM(responder=_prose_policy), toy_tools(), max_hops=8)
    assert numeric_checker(2493.0)(c.final_answer)
    assert len(c.nodes) == 7  # no ERROR retry hops


class _Recorder:
    """MockLLM wrapper capturing the kwargs each completion was called with."""

    def __init__(self, responder):
        self.inner = MockLLM(model="rec", responder=responder)
        self.model = self.inner.model
        self.kwargs: list[dict] = []

    def complete(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        return self.inner.complete(messages, **kwargs)


def test_live_hops_are_sampled_at_temperature_zero_by_default():
    llm = _Recorder(toy_policy(False))
    record(TASK, llm, toy_tools())
    assert llm.kwargs and all(k == {"temperature": 0.0} for k in llm.kwargs)
    llm.kwargs.clear()
    record(TASK, llm, toy_tools(), temperature=None)  # provider default
    assert all(k == {} for k in llm.kwargs)


def test_bisect_ref_and_replay_use_temperature_zero():
    agent = _Recorder(toy_policy(False))
    c = record(TASK, agent, toy_tools("stale_fx"))
    ref = _Recorder(toy_policy(False))
    agent.kwargs.clear()
    bisect(c, numeric_checker(2493.0), llm=agent, tools=toy_tools("stale_fx"), ref_llm=ref,
           oracle_tools=toy_tools(), after="live")
    assert ref.kwargs and all(k == {"temperature": 0.0} for k in ref.kwargs)
    assert agent.kwargs and all(k == {"temperature": 0.0} for k in agent.kwargs)


def _flaky_same_model():
    """Same model name as the agent; a re-sample of the calc hop happens to divide by the
    stale rate (what deepseek-flash did on widgets-eur) and so 'repairs' the run."""
    base = toy_policy(False)

    def respond(messages):
        out = base(messages)
        if out.startswith("CALL calc") and "/" not in out:
            return out.replace("* 1.087", "/ 1.087")
        return out

    return MockLLM(model="flash", responder=respond)


def test_same_model_resample_is_unstable_not_root_cause():
    agent = MockLLM(model="flash", responder=toy_policy(False))
    tools = toy_tools("stale_fx")
    c = record("What do 3 widgets cost in EUR?", agent, tools)
    r = bisect(c, numeric_checker(11.73), llm=agent, tools=tools,
               ref_llm=_flaky_same_model(), oracle_tools=toy_tools())
    assert r.unstable == [4]
    assert r.culprits == [3] and r.root_cause == 3  # only the stale tool is blamed
    unstable = next(t for t in r.trials if t.index == 4)
    assert unstable.replay_answer == "11.73" and "nondeterministic" in unstable.detail


def test_different_ref_model_flip_still_counts():
    agent = MockLLM(model="flash", responder=toy_policy(False))
    tools = toy_tools("stale_fx")
    c = record("What do 3 widgets cost in EUR?", agent, tools)
    ref = _flaky_same_model()
    ref.model = "pro"
    r = bisect(c, numeric_checker(11.73), llm=agent, tools=tools, ref_llm=ref,
               oracle_tools=toy_tools())
    assert r.culprits == [3, 4] and r.unstable == []


def test_cost_is_tracked_for_override_and_bisect():
    agent = MockLLM(model="flash", responder=toy_policy(False), price_in=1e6, price_out=0.0)
    tools = toy_tools("stale_fx")
    c = record("What do 3 widgets cost in EUR?", agent, tools)
    assert all(n.cost_usd > 0 for n in c.nodes if n.kind == "llm")
    swap = MockLLM(model="pro", responder=toy_policy(False), price_in=1e6, price_out=0.0)
    res = Replayer(agent, tools, after="live").replay(c, Override(4, llm=swap))
    node = res.replayed.nodes[4]
    assert node.source == "override" and node.name == "pro" and node.cost_usd > 0
    assert res.cost_usd == pytest.approx(
        sum(n.cost_usd for n in res.replayed.nodes if n.source != "cassette"))
    r = bisect(c, numeric_checker(11.73), llm=agent, tools=tools, ref_llm=swap,
               oracle_tools=toy_tools())
    assert r.cost_usd > 0


def test_cli_prints_cost_and_temperature_flag(tmp_path, capsys):
    tasks = tmp_path / "t.jsonl"
    tasks.write_text('{"task_id": "w", "input": "What do 3 widgets cost in EUR?", '
                     '"expected": 11.73}\n')
    out = tmp_path / "c.jsonl"
    assert main(["replay-debugger", "record", "--tasks", str(tasks), "--llm", "toy",
                 "--tools", "toy:stale_fx", "--temperature", "0", "--out", str(out)]) == 0
    assert "0/1 pass, api cost $" in capsys.readouterr().out
    assert main(["replay-debugger", "bisect", "--cassettes", str(out), "--llm", "toy",
                 "--tools", "toy:stale_fx", "--ref-llm", "toy", "--oracle-tools", "toy"]) == 0
    assert "root causes isolated: 1  (api cost $" in capsys.readouterr().out
