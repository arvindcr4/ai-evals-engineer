import json

import numpy as np
import pytest

from evalkit.cli import main
from evalkit.context_eviction import (
    HashingEmbedder,
    Probe,
    Turn,
    classify,
    count_tokens,
    extractive_reader,
    generate,
    make_memory,
    run_scenario,
    sweep,
)
from evalkit.context_eviction.harness import default_reader
from evalkit.context_eviction.memory import extractive_summarizer
from evalkit.core.llm import MockLLM

BUDGET = 400


def run(strategy, noise, seed=0, budget=BUDGET, **kw):
    sc = generate(noise, seed=seed, **kw)
    mem = make_memory(strategy, sc.system, budget)
    return run_scenario(sc, mem, default_reader())


def by_slot(answers):
    return {a["slot"]: a for a in answers}


def test_scenario_is_deterministic_and_plants_needles():
    a, b = generate(100, seed=3), generate(100, seed=3)
    assert [t.text for t in a.turns] == [t.text for t in b.turns]
    assert [t.text for t in generate(100, seed=4).turns] != [t.text for t in a.turns]
    kinds = [t.kind for t in a.turns]
    assert kinds.count("fact") == 6 and kinds.count("update") == 4  # 3 slots + 1 pinned
    assert len(a.turns) == 110
    for p in a.probes:
        if p.updated and not p.pinned:
            fact = next(t for t in a.turns if t.slot == p.slot and t.kind == "fact")
            upd = next(t for t in a.turns if t.slot == p.slot and t.kind == "update")
            assert fact.index < upd.index
            assert p.stale[0] in fact.text and p.current in upd.text
    pinned_upd = [p for p in a.probes if p.pinned and p.updated]
    assert len(pinned_upd) == 1 and pinned_upd[0].stale[0] in a.system


def test_zero_noise_everyone_perfect():
    for s in ("full", "fifo", "window", "summary", "retrieval"):
        res, _ = run(s, 0)
        assert res.recall == 1.0, s


def test_reader_takes_latest_and_ignores_colleague_distractor():
    memory = ("user: Please remember that my manager is Priya Nair.\n"
              "user: Unrelated, but my colleague's manager is Luis Romero.\n"
              "user: Update: my manager has changed to Aiko Tanaka.\n")
    prompt = f"MEMORY:\n{memory}\nQUESTION: What is the user's manager?\nANSWER:"
    assert extractive_reader([{"role": "user", "content": prompt}]) == "Aiko Tanaka"
    only_colleague = "MEMORY:\nuser: my colleague's manager is Luis.\nQUESTION: What is the user's manager?"
    assert extractive_reader([{"role": "user", "content": only_colleague}]) == "unknown"


def test_classify():
    p = Probe("car", "q", current="a red Swift", stale=["a blue Corolla"])
    assert classify("It's a red Swift.", p) == "correct"
    assert classify("a blue Corolla", p) == "stale"
    assert classify("unknown", p) == "miss"


def test_fifo_evicts_pinned_system_prompt_but_respects_budget():
    res, answers = run("fifo", 800)
    assert res.compliance == 1.0 and res.peak_tokens <= BUDGET
    assert res.pinned_correct == 0
    assert by_slot(answers)["name"]["verdict"] == "miss"


def test_window_keeps_pinned_but_goes_stale_on_pinned_update():
    res, answers = run("window", 800)
    assert res.compliance == 1.0
    rows = by_slot(answers)
    assert rows["name"]["verdict"] == "correct"
    assert rows["timezone"]["verdict"] == "stale"  # pinned old value, update evicted


def test_full_memory_recalls_everything_but_blows_budget():
    res, _ = run("full", 200)
    assert res.recall == 1.0
    assert res.compliance < 0.2 and res.peak_tokens > 10 * BUDGET


def test_summary_preserves_updates_within_budget():
    res, _ = run("summary", 800)
    assert res.compliance == 1.0
    assert res.llm_calls > 10
    assert res.recall == 1.0 and res.stale == 0


def test_summary_truncates_an_overlong_summary():
    verbose = MockLLM(responder=lambda m: "word " * 1000)
    mem = make_memory("summary", "be helpful", 100, llm=verbose)
    for i in range(30):
        mem.add(Turn("user", f"noise turn number {i} " * 3))
    assert count_tokens(mem.summary) <= mem.summary_cap
    assert mem.tokens() <= 100


def test_extractive_summarizer_keeps_latest_value():
    prompt = ("instructions\nPREVIOUS NOTES:\nthe user's car is a red Swift.\nNEW TURNS:\n"
              "user: Correction, my car is now a white Golf.\nassistant: chess openings are fun.")
    out = extractive_summarizer([{"role": "user", "content": prompt}])
    assert out == "the user's car is a white Golf."


def test_retrieval_finds_deep_needles_window_cannot():
    r_res, _ = run("retrieval", 200)
    w_res, _ = run("window", 200)
    assert r_res.compliance == 1.0
    assert r_res.recall > w_res.recall + 0.4


def test_hashing_embedder_similarity():
    e = HashingEmbedder()
    q = e("What is the user's home address?")
    hit = e("user: Please remember that my home address is 14 Elm Street.")
    miss = e("assistant: Sourdough starters are surprisingly subtle.")
    assert float(q @ hit) > float(q @ miss)
    assert np.isclose(np.linalg.norm(hit), 1.0)


def test_sweep_is_paired_and_degrades_with_noise():
    cells = sweep(["fifo", "summary"], [0, 400], trials=2, seed=1)
    get = {(c.strategy, c.noise): c for c in cells}
    assert get[("fifo", 0)].recall == get[("summary", 0)].recall == 1.0
    assert get[("fifo", 400)].recall < 0.5
    assert get[("summary", 400)].recall > get[("fifo", 400)].recall
    assert sweep(["fifo"], [400], trials=2, seed=1)[0] == get[("fifo", 400)]


def test_unknown_strategy():
    with pytest.raises(ValueError):
        make_memory("lru", "sys", 100)


def test_cli_run_json(capsys):
    assert main(["context-eviction", "run", "--strategies", "window,retrieval",
                 "--noise", "0,100", "--trials", "1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {(c["strategy"], c["noise"]) for c in data["cells"]} == {
        ("window", 0), ("window", 100), ("retrieval", 0), ("retrieval", 100)}
    assert all(c["compliance"] == 1.0 for c in data["cells"])


def test_scenario_roundtrip_and_inspect_cli(tmp_path, capsys):
    from evalkit.context_eviction import Scenario

    sc = generate(60, seed=5)
    p = tmp_path / "s.json"
    sc.save(p)
    back = Scenario.load(p)
    assert [t.text for t in back.turns] == [t.text for t in sc.turns]
    assert back.probes == sc.probes
    assert main(["context-eviction", "inspect", "--scenario", str(p), "--strategy", "summary",
                 "--budget", "150"]) == 0
    out = capsys.readouterr().out
    assert "summary: recall" in out and "budget compliance 100%" in out
