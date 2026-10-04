"""Regression tests for the real-model (DeepSeek) integration of 04 regression_gate.

Each responder below mimics an output actually returned by deepseek-flash during
the Oct 2026 real run (examples/04-regression-gate/real_run.sh).
"""

import json
import threading
import time
from dataclasses import dataclass, field

import pytest

from evalkit.core.llm import Completion, MockLLM
from evalkit.regression_gate import RunResult, Scorer, Suite, evaluate_gate, run_suite
from evalkit.regression_gate.runner import render_template
from evalkit.regression_gate.suite import Case, extract_json

# Seen with the "simplified" prompt: one sentence of reasoning, then BARE JSON (no fences).
PROSE_THEN_BARE = (
    "The customer reports a duplicate subscription charge, which is a billing issue that "
    "should be handled with high priority since it involves an incorrect payment.  \n"
    '{"category":"billing","priority":"high","order_id":null,"amount":null}')
# Seen in the same run: reasoning, then a ```json fenced, pretty-printed object.
PROSE_THEN_FENCE = (
    "This is a straightforward refund request tied to a specific order, so it falls under "
    'billing with normal priority.\n\n```json\n{\n  "category": "billing",\n  "priority": '
    '"normal",\n  "order_id": "ORD-10231",\n  "amount": 49.99\n}\n```')
EXP_T00 = {"category": "billing", "priority": "high", "order_id": None, "amount": None}
EXP_T01 = {"category": "billing", "priority": "normal", "order_id": "ORD-10231",
           "amount": 49.99}


def _case(expected):
    return Case("c", "c", {}, expected)


@pytest.mark.parametrize("text,expected", [(PROSE_THEN_BARE, EXP_T00),
                                           (PROSE_THEN_FENCE, EXP_T01)])
def test_json_field_parses_json_after_prose(text, expected):
    # Before the fix every check on PROSE_THEN_BARE failed although the answer was right.
    for f in expected:
        assert Scorer("json_field", f).score(text, _case(expected))[0], f


def test_extract_json_prefers_last_object_and_reports_missing():
    assert extract_json('I considered {"category": "shipping"} but {"category": "billing"}') \
        == {"category": "billing"}
    assert extract_json('[1, 2]') == [1, 2]
    from evalkit.regression_gate.suite import _MISSING
    assert extract_json("no json here {oops") is _MISSING
    assert not Scorer("json_field", "category").score("Category: billing", _case(
        {"category": "billing"}))[0]


@pytest.mark.parametrize("scorer,text,expected,ok", [
    (Scorer("numeric", "amount", tolerance=0.005), '{"amount": "$1,234.50"}',
     {"amount": 1234.5}, True),
    (Scorer("numeric", "amount", tolerance=0.005), '{"amount": 120.00}', {"amount": 120}, True),
    (Scorer("numeric", "amount"), '{"amount": true}', {"amount": 1}, False),
    (Scorer("exact", "order_id"), PROSE_THEN_FENCE, {"order_id": "ORD-10231"}, True),
    (Scorer("regex", "order_id", pattern=r"^ORD-\d+$"), PROSE_THEN_FENCE, {}, True),
])
def test_field_scorers_parse_llm_text(scorer, text, expected, ok):
    assert scorer.score(text, _case(expected))[0] is ok


def test_render_template_keeps_literal_json_braces():
    tpl = 'Answer like {"category": "billing"}.\nChannel: {channel}\nTicket: {text} {{x}}'
    out = render_template(tpl, {"channel": "chat", "text": "refund pls"})
    assert out == 'Answer like {"category": "billing"}.\nChannel: chat\nTicket: refund pls {x}'
    with pytest.raises(KeyError, match="chanel"):
        render_template("{chanel}", {"channel": "chat"})


@dataclass
class RecordingLLM:
    """Records kwargs; replies like DeepSeek incl. usage, cost and finish_reason."""

    model: str = "deepseek-flash"
    text: str = '{"category": "billing"}'
    finish_reason: str = "stop"
    tokens_out: int = 12
    extra_body: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def complete(self, messages, **kwargs):
        self.calls.append(kwargs)
        return Completion(self.text, self.model, tokens_in=200, tokens_out=self.tokens_out,
                          latency_s=0.6, cost_usd=0.0001,
                          raw={"choices": [{"finish_reason": self.finish_reason}]})


def _llm_suite(n=3, **target):
    items = [{"id": f"t{i}", "input": {"text": f"ticket {i}"},
              "expected": {"category": "billing"}} for i in range(n)]
    return Suite.from_dict({"name": "llm", "cases": items,
                            "target": {"prompt": 'Reply {"category": ...} for: {text}', **target},
                            "scorers": [{"type": "json_field", "field": "category"}]})


def test_generation_settings_are_forwarded_and_usage_recorded():
    # Before the fix `temperature: 0` in the suite was silently dropped (sampling stayed on).
    llm = RecordingLLM()
    res = run_suite(_llm_suite(temperature=0, max_tokens=200), llm=llm)
    assert llm.calls == [{"max_tokens": 200, "temperature": 0}] * 3
    s = res.summary()
    assert res.success_rate == 1.0 and s["tokens_in"] == 600 and s["tokens_out"] == 36
    assert s["cost_usd"] == pytest.approx(0.0003)
    assert res.meta["model"] == "deepseek-flash" and res.meta["temperature"] == 0
    assert len(res.meta["prompt_sha"]) == 12


def test_unknown_target_key_is_rejected():
    with pytest.raises(ValueError, match="temprature"):
        run_suite(_llm_suite(temprature=0), llm=RecordingLLM())


def test_truncated_reasoning_output_is_an_error_not_a_wrong_answer():
    # deepseek-flash+think with max_tokens=200 spent the budget reasoning and returned ""
    # or '{"category": "account", ... "amount":' with finish_reason="length".
    llm = RecordingLLM(text='{"category": "account", "priority": "high", "amount":',
                       finish_reason="length", tokens_out=200,
                       extra_body={"thinking": {"type": "enabled"}})
    res = run_suite(_llm_suite(max_tokens=200), llm=llm)
    assert res.summary()["errors"] == 3 and res.success_rate == 0
    assert "TruncatedOutputError: output hit max_tokens=200" in res.cases[0].error
    assert res.meta["model"] == "deepseek-flash+think"


def test_workers_run_concurrently_and_keep_suite_order():
    active, peak, lock = [0], [0], threading.Lock()

    def responder(messages):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        return json.dumps({"category": "billing", "echo": messages[-1]["content"]})

    res = run_suite(_llm_suite(n=12, workers=4), llm=MockLLM(responder=responder))
    assert [c.case_id for c in res.cases] == [f"t{i}" for i in range(12)]
    assert all(f"ticket {i}" in c.output for i, c in enumerate(res.cases))
    assert peak[0] > 1 and res.success_rate == 1.0


def test_report_shows_build_and_cost_and_flags_new_errors(tmp_path):
    base = run_suite(_llm_suite(), llm=RecordingLLM())
    cand = run_suite(_llm_suite(max_tokens=50), llm=RecordingLLM(finish_reason="length"))
    md = evaluate_gate(base, cand).to_markdown()
    assert "| baseline | `deepseek-flash` |" in md and "| Cost | $0.0003 | $0.0003 |" in md
    assert "3 more errored cases than the baseline" in md
    # results written before cost tracking existed still load
    old = base.to_dict()
    for c in old["cases"]:
        for k in ("tokens_in", "tokens_out", "cost_usd"):
            c.pop(k)
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    assert RunResult.load(path).summary()["cost_usd"] == 0
