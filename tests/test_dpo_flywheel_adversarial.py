"""Adversarial regression tests for dpo_flywheel: malformed input, decontamination
blind spots, PII over-redaction, crashing trainers and seeded determinism."""

import json
from datetime import date

import pytest

from evalkit.dpo_flywheel.feedback import FeedbackEvent, FeedbackStore, create_app, load_events
from evalkit.dpo_flywheel.filters import Decontaminator, FilterConfig, filter_pairs, scrub_pii
from evalkit.dpo_flywheel.nightly import DryRunTrainer, MockEvaluator, NightlyConfig, run_nightly
from evalkit.dpo_flywheel.pairs import PairBuilder, PreferencePair, build_dataset, resolve_llm

MALFORMED = [
    {"conversation_id": "c", "messages": ["hi"], "response": "x", "rating": "down"},
    {"conversation_id": "c", "messages": "hello", "response": "x", "rating": "down"},
    {"conversation_id": "c", "messages": [{"role": "user", "content": None}], "response": "x", "rating": "down"},
    {"conversation_id": "c", "messages": [{"role": "user", "content": "q"}], "response": "x", "rating": "down",
     "correction": 42},
]


@pytest.mark.parametrize("payload", MALFORMED)
def test_malformed_messages_raise_value_or_type_error(payload):
    with pytest.raises((ValueError, TypeError)):
        FeedbackEvent.from_dict(payload)


def test_ingest_and_endpoint_survive_malformed_rows(tmp_path):
    src = tmp_path / "in.jsonl"
    good = {"conversation_id": "ok", "messages": [{"role": "user", "content": "q?"}], "response": "a", "rating": "up"}
    src.write_text("\n".join(json.dumps(r) for r in [*MALFORMED, good]) + "\n")
    events, bad = load_events(src)
    assert len(events) == 1 and len(bad) == len(MALFORMED)

    from fastapi.testclient import TestClient

    client = TestClient(create_app(FeedbackStore(tmp_path / "s.jsonl")))
    for payload in MALFORMED:
        assert client.post("/feedback", json=payload).status_code == 422


def test_decontamination_catches_eval_item_embedded_in_long_prompt():
    golden = "What is the capital city of France and why was it chosen"
    d = Decontaminator([golden])
    long_prompt = ("Long preamble about my day at work, my cat, my commute and a dozen other unrelated "
                   "things that happened this week. ") + golden
    assert d.is_contaminated(long_prompt)
    assert not d.is_contaminated("Tell me about the history of Lyon and its silk industry in detail please")


def test_decontamination_short_golden_items():
    d = Decontaminator(["Define overfitting", "Hello"])
    assert d.is_contaminated("Quick one: define overfitting?")
    assert not d.is_contaminated("Hello, how do I export invoices to CSV?")  # one-word item: exact only
    assert d.is_contaminated("hello!")


def test_scrub_keeps_ip_addresses_but_removes_phone():
    text, counts = scrub_pii("Ping 192.168.100.200 or call +1 (415) 555-0134 now")
    assert "192.168.100.200" in text and "[PHONE]" in text and counts == {"phone": 1}


def test_filter_never_keeps_identical_after_pii_scrub():
    # chosen and rejected differ only in the email address -> identical once scrubbed
    p = PreferencePair("How do I reach support?", "Write to alice@example.com and we reply within one day.",
                       "Write to bob@example.org and we reply within one day.", "teacher", "e", 5.0)
    kept, drops, pii = filter_pairs([p], FilterConfig())
    assert kept == [] and drops == {"identical": 1} and pii == {"email": 2}


def _seed(store, n, prefix):
    FeedbackStore(store).append(
        FeedbackEvent(f"{prefix}{i}", [{"role": "user", "content": f"How do I archive project {prefix}{i}?"}],
                      "Not possible.", "down",
                      f"Open project {prefix}{i}, choose Settings, then Archive and confirm the dialog.")
        for i in range(n)
    )


def test_crashing_trainer_is_recorded_not_raised(tmp_path):
    class Exploding(DryRunTrainer):
        name = "boom"

        def train(self, dataset_path, config, output_dir):
            raise RuntimeError("CUDA out of memory")

    store = tmp_path / "fb.jsonl"
    _seed(store, 4, "c")
    cfg = NightlyConfig(root=tmp_path / "fw", store=store, min_new_pairs=1, run_date=date(2026, 10, 3))
    rec = run_nightly(cfg, PairBuilder(teacher=None), Exploding(), MockEvaluator())
    assert rec["status"] == "train_failed" and "out of memory" in rec["train"]["metrics"]["error"]
    state = json.loads((tmp_path / "fw" / "state.json").read_text())
    assert state["cursor"] == 0 and state["promoted_adapter"] is None and len(state["runs"]) == 1


def test_zero_threshold_with_no_pairs_does_not_promote(tmp_path):
    store = tmp_path / "fb.jsonl"
    FeedbackStore(store).append([FeedbackEvent("u", [{"role": "user", "content": "hi"}], "hello there", "up")])
    cfg = NightlyConfig(root=tmp_path / "fw", store=store, min_new_pairs=0, run_date=date(2026, 10, 3))
    rec = run_nightly(cfg, PairBuilder(teacher=None), DryRunTrainer(), MockEvaluator())
    assert rec["status"] == "train_failed" and rec["pairs_total"] == 0


def test_build_is_deterministic_with_mock_teacher(tmp_path):
    store = tmp_path / "fb.jsonl"
    FeedbackStore(store).append(
        FeedbackEvent(f"t{i}", [{"role": "user", "content": f"How do I rotate the API key for service {i}?"}],
                      "Idk.", "down")
        for i in range(6)
    )
    events = FeedbackStore(store).read()
    hashes = {
        build_dataset(events, PairBuilder(teacher=resolve_llm("mock", teacher=True))).manifest["data_sha256"]
        for _ in range(3)
    }
    assert len(hashes) == 1
