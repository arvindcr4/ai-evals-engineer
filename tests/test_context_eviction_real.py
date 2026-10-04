"""Regressions from running context-eviction against real DeepSeek output (Oct 2026)."""

from evalkit.cli import main
from evalkit.context_eviction import Probe, Turn, classify, count_tokens, make_memory, sweep
from evalkit.context_eviction.harness import (
    CostMeter,
    clean_answer,
    default_reader,
    extractive_reader,
)
from evalkit.context_eviction.memory import cap_summary, clean_summary, extractive_summarizer
from evalkit.core.llm import Completion, MockLLM

# Shape of a real deepseek-flash summary: capitalised fact sentences, a long
# small-talk "interests" list and a colleague fact, well over the word cap.
REAL_NOTES = (
    "The user's favorite airline is Qantas. The user's monthly budget is 3100 dollars. "
    "The user's car is a blue Corolla. The user's manager is Grace Okafor. "
    "The user's interests include jazz history, film photography, solar panels, chess "
    "openings, sourdough starters, tide pools, Roman aqueducts, marathon training, origami "
    "cranes, bird migration, mechanical keyboards, medieval maps. "
    "The user's colleague's car is a white Golf."
)


def test_classify_ignores_articles():
    # real reader answered 'blue Corolla' for 'a blue Corolla' and was scored a miss
    p = Probe("car", "q", current="a blue Corolla", stale=["a red Swift"])
    assert classify("blue Corolla", p) == "correct"
    assert classify("The blue Corolla.", p) == "correct"
    assert classify("red Swift", p) == "stale"
    assert classify("unknown", p) == "miss"


def test_cap_summary_drops_filler_before_facts():
    out = cap_summary(REAL_NOTES, 40)
    assert count_tokens(out) <= 40
    for fact in ("Qantas", "3100 dollars", "blue Corolla", "Grace Okafor"):
        assert fact in out, fact
    assert "interests" not in out  # oldest filler goes first; the colleague line fits
    tight = cap_summary(REAL_NOTES, 30)
    assert "colleague" not in tight and "Grace Okafor" in tight


def test_cap_summary_drops_oldest_facts_whole_when_facts_alone_overflow():
    notes = " ".join(f"the user's slot{i} is value{i}." for i in range(10))
    out = cap_summary(notes, 20)
    assert count_tokens(out) <= 20
    assert out.endswith("the user's slot9 is value9.")
    assert out.startswith("the user's")  # whole sentences, never a cut-off tail
    assert cap_summary("short notes.", 20) == "short notes."
    assert count_tokens(cap_summary("word " * 100, 10)) == 10


def test_old_word_truncation_regression_keeps_facts_in_summary_memory():
    # pre-fix: words[-cap:] cut the OLDEST facts and kept the filler list
    llm = MockLLM(responder=lambda m: REAL_NOTES)
    mem = make_memory("summary", "be helpful", 100, llm=llm)
    for i in range(30):
        mem.add(Turn("user", f"noise turn number {i} " * 3))
    assert count_tokens(mem.summary) <= mem.summary_cap
    assert "Qantas" in mem.summary and "3100 dollars" in mem.summary
    assert mem.tokens() <= 100


def test_clean_summary_strips_fences_labels_and_bullets():
    raw = "```\nUPDATED NOTES:\n- the user's car is a red Swift\n* The user's dentist is Dr Osei.\n```"
    assert clean_summary(raw) == "the user's car is a red Swift. The user's dentist is Dr Osei."
    assert clean_summary("**Updated notes:** the user's name is Maya.") == \
        "the user's name is Maya."


def test_capitalised_facts_are_read_by_extractive_components():
    prompt = ("MEMORY:\nmemory: The user's manager is Grace Okafor. The user's car is a red "
              "Swift.\nQUESTION: What is the user's manager?\nANSWER:")
    assert extractive_reader([{"role": "user", "content": prompt}]) == "Grace Okafor"
    merged = extractive_summarizer([{"role": "user", "content":
                                     "PREVIOUS NOTES:\nThe user's Car is a red Swift.\n"
                                     "NEW TURNS:\nuser: Correction, my car is now a white Golf."}])
    assert merged == "the user's car is a white Golf."


def test_clean_answer():
    assert clean_answer("ANSWER: 14 Elm Street") == "14 Elm Street"
    assert clean_answer('"Dr Osei"') == "Dr Osei"
    assert clean_answer("```\nIST\n```") == "IST"
    assert clean_answer("unknown") == "unknown"


class Recorder:
    model = "rec"

    def __init__(self, text: str):
        self.text, self.kwargs = text, []

    def complete(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        return Completion(self.text, self.model, tokens_in=10, tokens_out=2, cost_usd=0.001)


def test_reader_and_summarizer_use_temperature_zero_and_cost_is_metered():
    from evalkit.context_eviction import generate, run_scenario

    sc = generate(40, seed=1)
    summ, reader = Recorder("the user's name is Maya."), Recorder("unknown")
    meter = CostMeter(reader)
    mem = make_memory("summary", sc.system, 120, llm=summ)
    run_scenario(sc, mem, meter)
    assert summ.kwargs and all(k.get("temperature") == 0 for k in summ.kwargs)
    assert all(k.get("temperature") == 0 for k in reader.kwargs)
    assert meter.calls == len(sc.probes) and meter.tokens_in == 10 * len(sc.probes)
    assert abs(meter.cost_usd - 0.001 * len(sc.probes)) < 1e-12
    assert meter.to_dict()["model"] == "rec"


def test_parallel_sweep_matches_sequential():
    kw = {"budget": 200, "trials": 2, "seed": 3}
    seq = sweep(["window", "summary", "retrieval"], [0, 60], **kw)
    par = sweep(["window", "summary", "retrieval"], [0, 60], workers=4,
                reader=CostMeter(default_reader()), **kw)
    assert seq == par


def test_cli_role_overrides_print_usage(capsys):
    assert main(["context-eviction", "run", "--strategies", "summary", "--noise", "60",
                 "--trials", "1", "--budget", "150", "--reader", "mock",
                 "--summarizer", "mock"]) == 0
    out = capsys.readouterr().out
    assert "llm usage: reader mock-extractive-reader" in out and "summarizer mock-summarizer" in out
