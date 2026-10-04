"""Nightly orchestrator: gate → version dataset → train config → train → promote.

Layout under ``root``::

    state.json                      cursor into the feedback store, promoted adapter, run history
    datasets/<YYYY-MM-DD>[-N]/      train.jsonl, pairs.jsonl, manifest.json, train_config.yaml
    adapters/<YYYY-MM-DD>[-N]/      trainer output (plan.json for the dry-run backend)

The dataset is cumulative (every thumbs-down ever seen, re-filtered), but a run
only trains when at least ``min_new_pairs`` of the kept pairs come from events
that arrived since the last *successful* run; otherwise the cursor stays put so
new feedback keeps accumulating toward the threshold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import yaml

from evalkit.core.llm import LLM, stable_hash
from evalkit.dpo_flywheel.feedback import FeedbackStore
from evalkit.dpo_flywheel.filters import FilterConfig
from evalkit.dpo_flywheel.pairs import HeuristicJudge, PairBuilder, build_dataset, write_dataset

REQUIRED_KEYS = ("prompt", "chosen", "rejected")


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    beta: float = 0.1
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    seed: int = 0


@dataclass
class TrainResult:
    backend: str
    status: str  # "trained" | "planned" | "failed"
    adapter_path: str
    metrics: dict = field(default_factory=dict)


class Trainer(Protocol):
    name: str

    def train(self, dataset_path: Path, config: TrainConfig, output_dir: Path) -> TrainResult: ...


def validate_dataset(path: Path) -> tuple[list[dict], list[str]]:
    """Rows plus a list of problems (missing keys, empty fields, chosen == rejected)."""
    rows, problems = [], []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {i}: invalid json ({e.msg})")
                continue
            missing = [k for k in REQUIRED_KEYS if not isinstance(row.get(k), str) or not row[k].strip()]
            if missing:
                problems.append(f"line {i}: missing/empty {missing}")
            elif row["chosen"] == row["rejected"]:
                problems.append(f"line {i}: chosen == rejected")
            rows.append(row)
    return rows, problems


class DryRunTrainer:
    """Validates the data and writes the plan a real run would execute. No GPU, no deps."""

    name = "dry"

    def train(self, dataset_path: Path, config: TrainConfig, output_dir: Path) -> TrainResult:
        rows, problems = validate_dataset(dataset_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        approx_tokens = sum(len(" ".join(r.get(k, "") for k in REQUIRED_KEYS)) // 4 for r in rows)
        eff_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
        steps = max(1, -(-len(rows) // eff_batch)) * config.num_train_epochs
        plan = {
            "backend": self.name,
            "dataset": str(dataset_path),
            "rows": len(rows),
            "problems": problems,
            "approx_tokens": approx_tokens,
            "effective_batch": eff_batch,
            "optimizer_steps": steps,
            "config": asdict(config),
            "command": f"evalkit dpo-flywheel nightly --backend trl  # would train {config.base_model}",
        }
        (output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        status = "failed" if problems or not rows else "planned"
        return TrainResult(self.name, status, str(output_dir), {"rows": len(rows), "steps": steps, "problems": len(problems)})


class TRLTrainer:
    """TRL ``DPOTrainer`` + PEFT LoRA. Heavy imports happen only inside :meth:`train`."""

    name = "trl"

    def train(self, dataset_path: Path, config: TrainConfig, output_dir: Path) -> TrainResult:
        try:
            from datasets import load_dataset
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import DPOConfig, DPOTrainer
        except ImportError as e:  # pragma: no cover - exercised only with the train extra
            raise RuntimeError(
                "TRL backend needs the 'train' extra: uv sync --extra train"
            ) from e

        rows, problems = validate_dataset(dataset_path)
        if problems or not rows:
            return TrainResult(self.name, "failed", str(output_dir), {"problems": problems[:20]})
        ds = load_dataset("json", data_files=str(dataset_path), split="train")
        tok = AutoTokenizer.from_pretrained(config.base_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.base_model)
        peft_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            task_type="CAUSAL_LM",
        )
        args = DPOConfig(
            output_dir=str(output_dir),
            beta=config.beta,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=config.max_length,
            seed=config.seed,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
        )
        trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
        out = trainer.train()
        trainer.save_model(str(output_dir))
        return TrainResult(self.name, "trained", str(output_dir), dict(out.metrics))


TRAINERS: dict[str, type] = {"dry": DryRunTrainer, "trl": TRLTrainer}


class Evaluator(Protocol):
    def evaluate(self, adapter_path: str | None, golden: list[str]) -> float: ...


@dataclass
class MockEvaluator:
    """Deterministic stand-in for "score the model on the golden set".

    The base scores ``base_score``; an adapter gains ``gain_per_pair`` per training
    row (capped) plus seeded noise, so small datasets can lose and get rejected.
    """

    base_score: float = 0.70
    gain_per_pair: float = 0.004
    max_gain: float = 0.08
    noise: float = 0.02
    seed: int = 0

    def evaluate(self, adapter_path: str | None, golden: list[str]) -> float:
        if adapter_path is None:
            return self.base_score
        rows = 0
        plan = Path(adapter_path) / "plan.json"
        if plan.exists():
            rows = json.loads(plan.read_text()).get("rows", 0)
        jitter = (stable_hash(str(self.seed), Path(adapter_path).name) % 2001 / 1000 - 1) * self.noise
        return round(self.base_score + min(self.max_gain, rows * self.gain_per_pair) + jitter, 4)


@dataclass
class WinRateEvaluator:
    """Head-to-head on golden prompts: candidate (adapter served behind ``candidate``) vs
    ``base``, scored by a judge. Returns the candidate win rate; base scores 0.5."""

    base: LLM
    candidate: LLM
    judge: HeuristicJudge = field(default_factory=HeuristicJudge)

    def evaluate(self, adapter_path: str | None, golden: list[str]) -> float:
        if adapter_path is None or not golden:
            return 0.5
        wins = 0.0
        for q in golden:
            msgs = [{"role": "user", "content": q}]
            a = self.judge.score(q, self.candidate.complete(msgs).text)
            b = self.judge.score(q, self.base.complete(msgs).text)
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
        return wins / len(golden)


@dataclass
class NightlyConfig:
    root: Path
    store: Path
    min_new_pairs: int = 20
    min_win: float = 0.01
    run_date: date | None = None
    golden: list[str] = field(default_factory=list)
    train: TrainConfig = field(default_factory=TrainConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)


def _load_state(root: Path) -> dict:
    p = root / "state.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"cursor": 0, "promoted_adapter": None, "promoted_score": None, "runs": []}


def _save_state(root: Path, state: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def _version_dir(base: Path, day: date) -> Path:
    stem = day.isoformat()
    cand, n = base / stem, 1
    while cand.exists():
        n += 1
        cand = base / f"{stem}-{n}"
    return cand


def run_nightly(
    cfg: NightlyConfig,
    builder: PairBuilder,
    trainer: Trainer,
    evaluator: Evaluator,
) -> dict:
    """One flywheel turn. Returns the run record (also appended to ``state.json``)."""
    root = Path(cfg.root)
    state = _load_state(root)
    store = FeedbackStore(cfg.store)
    events = store.read()
    cursor = state["cursor"]
    new_ids = {e.event_id for e in events[cursor:]}
    record: dict = {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": trainer.name,
        "events_total": len(events),
        "events_new": len(new_ids),
    }

    result = build_dataset(events, builder, cfg.filters, cfg.golden)
    new_pairs = sum(p.source_event in new_ids for p in result.pairs)
    record.update(pairs_total=len(result.pairs), pairs_new=new_pairs, drops=result.manifest["drops"])
    if result.manifest.get("llm_usage"):
        record["llm_usage"] = result.manifest["llm_usage"]
    if new_pairs < cfg.min_new_pairs:
        record["status"] = "skipped"
        record["reason"] = f"only {new_pairs} new pairs (< {cfg.min_new_pairs})"
        state["runs"].append(record)
        _save_state(root, state)
        return record

    day = cfg.run_date or datetime.now(UTC).date()
    ds_dir = _version_dir(root / "datasets", day)
    train_path = write_dataset(result, ds_dir)
    (ds_dir / "train_config.yaml").write_text(yaml.safe_dump(asdict(cfg.train), sort_keys=False))
    adapter_dir = root / "adapters" / ds_dir.name
    try:
        tr = trainer.train(train_path, cfg.train, adapter_dir)
    except Exception as e:  # noqa: BLE001 - a crashed trainer must still leave a run record
        tr = TrainResult(trainer.name, "failed", str(adapter_dir), {"error": f"{type(e).__name__}: {e}"})
    record.update(dataset=str(ds_dir), data_sha256=result.manifest["data_sha256"], train=asdict(tr))
    if tr.status == "failed":
        record["status"] = "train_failed"
        state["runs"].append(record)
        _save_state(root, state)
        return record

    base_score = (
        state["promoted_score"]
        if state["promoted_adapter"] is not None
        else evaluator.evaluate(None, cfg.golden)
    )
    cand_score = evaluator.evaluate(tr.adapter_path, cfg.golden)
    promote = cand_score - base_score >= cfg.min_win
    record["gate"] = {
        "incumbent": state["promoted_adapter"] or "base",
        "incumbent_score": base_score,
        "candidate_score": cand_score,
        "min_win": cfg.min_win,
        "promoted": promote,
    }
    if promote:
        state["promoted_adapter"] = tr.adapter_path
        state["promoted_score"] = cand_score
    record["status"] = "promoted" if promote else "rejected"
    state["cursor"] = len(events)
    state["runs"].append(record)
    _save_state(root, state)
    return record
