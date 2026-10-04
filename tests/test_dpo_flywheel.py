import argparse
import json
from datetime import date

import pytest

from evalkit.core.llm import MockLLM
from evalkit.dpo_flywheel.cli import register
from evalkit.dpo_flywheel.feedback import FeedbackEvent, FeedbackStore, create_app, load_events
from evalkit.dpo_flywheel.filters import (
    Decontaminator,
    FilterConfig,
    filter_pairs,
    is_refusal,
    jaccard,
    scrub_pii,
    shingles,
)
from evalkit.dpo_flywheel.nightly import (
    DryRunTrainer,
    MockEvaluator,
    NightlyConfig,
    TrainConfig,
    TRLTrainer,
    run_nightly,
    validate_dataset,
)
from evalkit.dpo_flywheel.pairs import (
    HeuristicJudge,
    LLMJudge,
    PairBuilder,
    PreferencePair,
    build_dataset,
    resolve_llm,
    write_dataset,
)

GOOD = "Open Settings, choose Security, click Reset password and follow the emailed link within 30 minutes."


def ev(cid, prompt, response, rating="down", correction=None):
    return FeedbackEvent(cid, [{"role": "user", "content": prompt}], response, rating, correction)


def pair(prompt="How do I reset my password?", chosen=GOOD, rejected="Contact support.", strategy="teacher", margin=3.0):
    return PreferencePair(prompt, chosen, rejected, strategy, "e1", margin)


# --- feedback -----------------------------------------------------------------


def test_event_validation_and_rating_normalisation():
    assert ev("c", "q", "r", rating="thumbs_up").rating == "up"
    assert ev("c", "q", "r", rating=-1).rating == "down"
    with pytest.raises(ValueError):
        ev("c", "q", "r", rating="meh")
    with pytest.raises(ValueError):
        FeedbackEvent("c", [{"role": "assistant", "content": "hi"}], "r", "down")
    assert ev("c", "q", "r", correction="   ").correction is None


def test_event_id_stable_and_multiturn_prompt():
    a, b = ev("c", "q", "r"), ev("c", "q", "r")
    assert a.event_id == b.event_id and a.event_id != ev("c", "q", "other").event_id
    multi = FeedbackEvent(
        "c",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "help"}],
        "r",
        "down",
    )
    assert multi.prompt_text == "User: hi\nAssistant: hello\nUser: help"
    assert multi.last_user == "help"


def test_store_dedupes_and_cursor(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    assert store.append([ev("c1", "q1", "r"), ev("c2", "q2", "r")]) == 2
    assert store.append([ev("c1", "q1", "r"), ev("c3", "q3", "r")]) == 1
    assert len(store) == 3
    assert [e.conversation_id for e in store.read(start=2)] == ["c3"]


def test_load_events_reports_bad_lines(tmp_path):
    p = tmp_path / "in.jsonl"
    p.write_text(
        json.dumps({"conversation_id": "c", "prompt": "q", "response": "r", "rating": "down", "user_tier": "pro"})
        + "\n{not json\n"
        + json.dumps({"conversation_id": "c", "messages": [], "response": "r", "rating": "down"})
        + "\n"
    )
    good, bad = load_events(p)
    assert len(good) == 1 and good[0].meta == {"user_tier": "pro"}
    assert [line for line, _ in bad] == [2, 3]


def test_fastapi_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    store = FeedbackStore(tmp_path / "fb.jsonl")
    client = TestClient(create_app(store))
    body = {"conversation_id": "c1", "messages": [{"role": "user", "content": "q"}], "response": "r", "rating": "down"}
    r = client.post("/feedback", json=body)
    assert r.status_code == 201 and r.json()["stored"] is True
    assert client.post("/feedback", json=body).json()["duplicate"] is True
    assert client.post("/feedback", json={**body, "rating": "meh"}).status_code == 422
    assert client.get("/stats").json() == {"events": 1, "down": 1, "up": 0, "with_correction": 0}


# --- filters ------------------------------------------------------------------


def test_scrub_pii_luhn_and_phone():
    text, c = scrub_pii("mail a.b@x.io, card 4111 1111 1111 1111, call +1 415 555 0134, order 1234567890123456")
    assert "[EMAIL]" in text and "[CARD]" in text and "[PHONE]" in text
    assert "1234567890123456" in text  # fails Luhn and too long for a phone: kept
    assert c == {"email": 1, "card": 1, "phone": 1}
    assert scrub_pii("version 2.4.1 shipped in 2026")[1] == {}


def test_shingles_and_refusals():
    assert jaccard(shingles("the quick brown fox"), shingles("The quick, brown fox!")) == 1.0
    assert jaccard(set(), set()) == 1.0
    assert is_refusal("I’m unable to do that") and not is_refusal("Sure, here's how")


def test_decontaminator_exact_and_ngram():
    d = Decontaminator(["What is the refund window for annual plans on the pro tier?"], n=5, threshold=0.5)
    assert d.overlap("what is the REFUND window for annual plans on the pro tier") == 1.0
    assert d.is_contaminated("Tell me: what is the refund window for annual plans on the pro tier")
    assert not d.is_contaminated("How do I reset my password today please?")


def test_filter_drop_reasons():
    pairs = [
        pair(),
        pair(chosen="contact support", rejected="Contact support!"),  # identical
        pair(chosen="Ok."),  # length
        pair(chosen="I'm sorry, but I can't help with resetting passwords here at all."),  # refusal
        pair(margin=0.2),  # low margin
        pair(margin=0.2, strategy="correction"),  # trusted correction; then duplicate of #1
        pair(prompt="What is the refund window for annual plans?"),  # contaminated
        pair(chosen=GOOD + " Thanks"),  # near duplicate
        pair(chosen="Email me at a@b.co", prompt="Which address?", rejected="no"),
    ]
    decon = Decontaminator(["What is the refund window for annual plans?"])
    kept, drops, pii = filter_pairs(pairs, FilterConfig(), decon)
    assert drops == {
        "identical": 1, "length": 1, "refusal": 1, "low_margin": 1,
        "duplicate": 1, "contaminated": 1, "near_duplicate": 1,
    }
    assert len(kept) == 2 and kept[1].chosen == "Email me at [EMAIL]" and pii["email"] == 1


# --- pair construction ----------------------------------------------------------


def test_strategies_in_priority_order():
    events = [
        ev("c1", "How do I reset my password?", "Contact support.", correction=GOOD),
        ev("c2", "How do I export invoices as CSV?", "Contact support."),
        ev("u2", "How can I export invoices as CSV?", "Billing > Invoices > Export CSV, then pick a date range.", "up"),
        ev("c3", "How do I enable two-factor authentication?", "Not possible."),
    ]
    builder = PairBuilder(teacher=resolve_llm("mock", teacher=True))
    pairs, missed = builder.build(events)
    assert [p.strategy for p in pairs] == ["correction", "similar_up", "teacher"]
    assert pairs[1].meta["donor"] == events[2].event_id
    assert "two-factor" in pairs[2].chosen and pairs[2].margin > 1
    assert not missed

    no_teacher, missed = PairBuilder(teacher=None).build(events)
    assert len(no_teacher) == 2 and missed == {"no_strategy_applied": 1}


def test_only_restricts_to_given_events():
    events = [ev("c1", "q one here", "bad", correction=GOOD), ev("c2", "q two here", "bad", correction=GOOD)]
    pairs, _ = PairBuilder(teacher=None).build(events, only={events[1].event_id})
    assert [p.meta["conversation_id"] for p in pairs] == ["c2"]


def test_heuristic_and_llm_judge():
    h = HeuristicJudge()
    q = "How do I reset my password?"
    assert h.score(q, GOOD) > h.score(q, "I'm sorry, but I can't help with that.") + 3
    assert h.score(q, "") == 0.0
    judge = LLMJudge(MockLLM(responder=lambda m: "SCORE: 8.5"))
    assert judge.score(q, GOOD) == 8.5
    garbled = LLMJudge(MockLLM(responder=lambda m: "looks fine"))
    assert garbled.score(q, GOOD) == h.score(q, GOOD)


def test_build_and_write_dataset(tmp_path):
    events = [
        ev("c1", "How do I reset my password? mail me at x@y.com", "Contact support.", correction=GOOD),
        ev("c2", "How do I reset my password? mail me at x@y.com", "Contact support.", correction=GOOD),
        ev("u", "unrelated", "fine", "up"),
    ]
    res = build_dataset(events, PairBuilder(teacher=None))
    assert res.manifest["pairs_raw"] == 2 and res.manifest["pairs_kept"] == 1
    assert res.manifest["drops"] == {"duplicate": 1} and res.manifest["pii_redactions"] == {"email": 2}
    path = write_dataset(res, tmp_path / "ds")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert set(rows[0]) == {"prompt", "chosen", "rejected"} and "[EMAIL]" in rows[0]["prompt"]
    again = build_dataset(events, PairBuilder(teacher=None))
    assert again.manifest["data_sha256"] == res.manifest["data_sha256"]


# --- nightly ------------------------------------------------------------------


def test_validate_dataset_catches_problems(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"prompt":"q","chosen":"a","rejected":"a"}\n{"prompt":"q","chosen":""}\nnope\n')
    rows, problems = validate_dataset(p)
    assert len(rows) == 2 and len(problems) == 3
    res = DryRunTrainer().train(p, TrainConfig(), tmp_path / "ad")
    assert res.status == "failed" and (tmp_path / "ad" / "plan.json").exists()


def test_trl_backend_errors_cleanly_without_extra(tmp_path):
    try:
        import trl  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="train"):
            TRLTrainer().train(tmp_path / "x.jsonl", TrainConfig(), tmp_path / "out")


def _seed_store(path, n, prefix="c"):
    topics = ["reset password", "export invoices", "enable sso", "rotate webhook secret", "merge contacts",
              "cancel subscription", "change language", "view audit logs", "restore project", "add teammate"]
    FeedbackStore(path).append(
        ev(f"{prefix}{i}", f"How do I {topics[i % 10]} for account {prefix}{i}?", "Not possible.",
           correction=f"Open the admin console and {topics[i % 10]} under settings for workspace {prefix}{i}.")
        for i in range(n)
    )


def test_nightly_gate_version_and_promotion(tmp_path):
    store = tmp_path / "fb.jsonl"
    _seed_store(store, 3)
    cfg = NightlyConfig(root=tmp_path / "fw", store=store, min_new_pairs=5, run_date=date(2026, 10, 3))
    builder, ev_ = PairBuilder(teacher=None), MockEvaluator()

    rec = run_nightly(cfg, builder, DryRunTrainer(), ev_)
    assert rec["status"] == "skipped"
    state = json.loads((tmp_path / "fw" / "state.json").read_text())
    assert state["cursor"] == 0  # feedback keeps accumulating

    _seed_store(store, 12, prefix="d")
    rec = run_nightly(cfg, builder, DryRunTrainer(), ev_)
    assert rec["status"] == "promoted" and rec["pairs_new"] == 15
    ds = tmp_path / "fw" / "datasets" / "2026-10-03"
    assert {p.name for p in ds.iterdir()} == {"train.jsonl", "pairs.jsonl", "manifest.json", "train_config.yaml"}
    assert "lora_r: 16" in (ds / "train_config.yaml").read_text()

    _seed_store(store, 6, prefix="e")
    rec2 = run_nightly(cfg, builder, DryRunTrainer(), MockEvaluator(gain_per_pair=0.0, noise=0.0))
    assert rec2["dataset"].endswith("2026-10-03-2")  # same-day rerun gets a new version
    assert rec2["status"] == "rejected" and rec2["gate"]["incumbent"].endswith("adapters/2026-10-03")
    state = json.loads((tmp_path / "fw" / "state.json").read_text())
    assert state["promoted_adapter"].endswith("adapters/2026-10-03") and state["cursor"] == 21


def test_nightly_train_failure_does_not_promote(tmp_path):
    class Broken(DryRunTrainer):
        def train(self, dataset_path, config, output_dir):
            res = super().train(dataset_path, config, output_dir)
            res.status = "failed"
            return res

    store = tmp_path / "fb.jsonl"
    _seed_store(store, 5)
    cfg = NightlyConfig(root=tmp_path / "fw", store=store, min_new_pairs=1)
    rec = run_nightly(cfg, PairBuilder(teacher=None), Broken(), MockEvaluator())
    assert rec["status"] == "train_failed"
    state = json.loads((tmp_path / "fw" / "state.json").read_text())
    assert state["promoted_adapter"] is None and state["cursor"] == 0


def main(argv):
    parser = argparse.ArgumentParser(prog="evalkit")
    register(parser.add_subparsers(dest="system", required=True))
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def test_cli_end_to_end(tmp_path, capsys):
    src = tmp_path / "in.jsonl"
    prompts = ["How do I configure billing alerts?", "How do I configure billing alerts?",
               "Where are audit logs stored?", "Can I rename a workspace?"]
    src.write_text("".join(
        json.dumps({"conversation_id": f"c{i}", "prompt": q, "response": "Not possible.", "rating": "down"}) + "\n"
        for i, q in enumerate(prompts)
    ))
    store = str(tmp_path / "fb.jsonl")
    assert main(["dpo-flywheel", "ingest", str(src), "--store", store]) == 0
    assert main(["dpo-flywheel", "build-pairs", "--store", store, "--out", str(tmp_path / "ds")]) == 0
    capsys.readouterr()
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    assert manifest["by_strategy_kept"] == {"teacher": 3} and manifest["drops"] == {"duplicate": 1}
    rc = main(["dpo-flywheel", "nightly", "--store", store, "--root", str(tmp_path / "fw"),
               "--min-new-pairs", "2", "--date", "2026-10-03", "--dry-run", "--teacher", "none"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["status"] == "skipped" and out["pairs_new"] == 0
