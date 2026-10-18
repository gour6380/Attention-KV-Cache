"""Small offline CPU integration checks, intentionally unexecuted at delivery."""

import pytest
import torch

import src.train as training
from src.checkpoints import load_model, load_portable_model, read_checkpoint
from src.config import BenchmarkConfig, DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.io import atomic_json, read_json, sha256_file
from src.runs import start_run
from src.tracking import FitTracker


class OfflineTokens:
    def __init__(self, directory):
        self.directory = directory
        self.manifest = {"context_length": 4, "fixture": True}
        directory.mkdir(exist_ok=True)
        atomic_json(directory / "manifest.json", self.manifest)

    def verify(self):
        assert (self.directory / "manifest.json").is_file()

    def dataset(self, split):
        return [
            (torch.tensor([1, 4 + i, 5 + i, 6 + i]), torch.tensor([4 + i, 5 + i, 6 + i, 2]))
            for i in range(5 if split == "train" else 2)
        ]


def setup(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "config.py").write_text("# source fixture\n")
    (root / "requirements.txt").write_text("# deps fixture\n")
    cfg = ExperimentConfig(
        data=DataConfig(train_stories=5, vocab_size=300, context_length=4),
        model=ModelConfig(
            vocab_size=300, num_layers=1, d_model=16, num_query_heads=4, ffn_dim=32, max_seq_len=32
        ),
        train=TrainConfig(
            epochs=2, microbatch_size=2, accumulation_steps=2, checkpoint_interval=1, log_interval=1
        ),
        benchmark=BenchmarkConfig(
            prompt_lengths=(4,), batch_sizes=(1,), decode_steps=2, warmup_trials=0, measured_trials=1
        ),
    )
    tokens = OfflineTokens(root / "data")
    monkeypatch.setattr(training, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training, "assert_correctness", lambda *args: {"passed": True})
    return root, cfg, tokens


def bind(run, tokens):
    run.bind_data(
        "tokens",
        {
            "directory": str(tokens.directory),
            "manifest_sha256": sha256_file(tokens.directory / "manifest.json"),
            "manifest": tokens.manifest,
        },
    )


def test_partial_accumulation_exact_budget_and_reload(tmp_path, monkeypatch):
    root, cfg, tokens = setup(tmp_path, monkeypatch)
    run = start_run(root, cfg)
    bind(run, tokens)
    result = training.train_variant(run, tokens, "mha")
    assert result["tokens_seen"] == 2 * 5 * 4
    assert result["optimizer_updates"] == 4
    first = load_model(run, "mha", device=torch.device("cpu"))
    second = load_model(run, "mha", device=torch.device("cpu"))
    with torch.inference_mode():
        ids = torch.tensor([[1, 4, 5]])
        torch.testing.assert_close(first(ids), second(ids), atol=0, rtol=0)
    monkeypatch.setattr(
        training, "Decoder", lambda *_: pytest.fail("Completed fit must not instantiate a model")
    )
    assert training.train_variant(run, tokens, "mha") == result


def test_cpu_resume_matches_uninterrupted_parameters(tmp_path, monkeypatch):
    root, cfg, tokens = setup(tmp_path, monkeypatch)
    whole = start_run(root, cfg)
    bind(whole, tokens)
    training.train_variant(whole, tokens, "mha")
    interrupted = start_run(root, cfg)
    bind(interrupted, tokens)
    original = FitTracker.log

    def interrupt(self, step, kind, values):
        if step == 2 and kind == "train":
            raise KeyboardInterrupt()
        return original(self, step, kind, values)

    monkeypatch.setattr(FitTracker, "log", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.train_variant(interrupted, tokens, "mha")
    assert read_checkpoint(interrupted, "mha", "latest")["position"]["step"] == 1
    monkeypatch.setattr(FitTracker, "log", original)
    resumed = start_run(root, cfg, "resume", interrupted.path.name)
    training.train_variant(resumed, tokens, "mha")
    a = read_checkpoint(whole, "mha")["model_state"]
    b = read_checkpoint(resumed, "mha")["model_state"]
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_schedule_final_and_partial_groups():
    cfg = TrainConfig()
    assert training.learning_rate_for_update(cfg, 99, 100) == pytest.approx(cfg.min_learning_rate)
    assert [list(g) for g in training.accumulation_groups(5, 0, 4)] == [[0, 1, 2, 3], [4]]


def test_resume_discards_uncommitted_validation_at_same_optimizer_step(tmp_path, monkeypatch):
    import json

    root, cfg, tokens = setup(tmp_path, monkeypatch)
    run = start_run(root, cfg)
    bind(run, tokens)
    original = FitTracker.log

    def interrupt_after_record(self, step, kind, values):
        original(self, step, kind, values)
        if step == 2 and kind == "validation":
            raise KeyboardInterrupt()

    monkeypatch.setattr(FitTracker, "log", interrupt_after_record)
    with pytest.raises(KeyboardInterrupt):
        training.train_variant(run, tokens, "mha")
    latest = read_checkpoint(run, "mha", "latest")
    assert latest["position"]["step"] == 2
    assert latest["position"]["epoch"] == 0
    monkeypatch.setattr(FitTracker, "log", original)
    resumed = start_run(root, cfg, "resume", run.path.name)
    training.train_variant(resumed, tokens, "mha")
    records = [json.loads(line) for line in (run.model_dir("mha") / "metrics.jsonl").read_text().splitlines()]
    validations = [row for row in records if row["kind"] == "validation"]
    assert [(row["epoch"], row["step"]) for row in validations] == [(1, 2), (2, 4)]


def test_latest_checksum_interruption_falls_back_to_verified_previous(tmp_path, monkeypatch):
    root, cfg, tokens = setup(tmp_path, monkeypatch)
    run = start_run(root, cfg)
    bind(run, tokens)
    training.train_variant(run, tokens, "mha")
    directory = run.model_dir("mha")
    previous = torch.load(directory / "latest.previous.pt", map_location="cpu", weights_only=True)
    receipt = read_json(directory / "latest.json")
    atomic_json(directory / "latest.json", {**receipt, "sha256": "interrupted-sidecar"})
    recovered = read_checkpoint(run, "mha", "latest")
    assert recovered["position"] == previous["position"]
    # Final-checkpoint corruption must fail closed rather than selecting a backup.
    final_receipt = read_json(directory / "final.json")
    atomic_json(directory / "final.json", {**final_receipt, "bytes": -1})
    with pytest.raises(RuntimeError, match="checksum"):
        read_checkpoint(run, "mha", "final")


def test_resume_counter_validation_rejects_partial_update_or_token_drift():
    cfg = TrainConfig(epochs=2, microbatch_size=2, accumulation_steps=2)
    training._validate_resume_position(
        {"epoch": 0, "next_batch": 2, "step": 1, "tokens_seen": 16},
        cfg,
        windows=5,
        context=4,
        batches=3,
        updates_per_epoch=2,
    )
    with pytest.raises(RuntimeError, match="boundary"):
        training._validate_resume_position(
            {"epoch": 0, "next_batch": 1, "step": 1, "tokens_seen": 8},
            cfg,
            windows=5,
            context=4,
            batches=3,
            updates_per_epoch=2,
        )
    with pytest.raises(RuntimeError, match="counters"):
        training._validate_resume_position(
            {"epoch": 0, "next_batch": 2, "step": 1, "tokens_seen": 12},
            cfg,
            windows=5,
            context=4,
            batches=3,
            updates_per_epoch=2,
        )


def test_portable_checkpoint_load_after_moving_artifact_pair(tmp_path, monkeypatch):
    import shutil

    root, cfg, tokens = setup(tmp_path, monkeypatch)
    run = start_run(root, cfg)
    bind(run, tokens)
    training.train_variant(run, tokens, "mha")
    model = load_model(run, "mha", device="cpu")
    portable = tmp_path / "portable"
    portable.mkdir()
    for suffix in ("pt", "json"):
        shutil.copy2(run.model_dir("mha") / f"final.{suffix}", portable / f"final.{suffix}")
    (root / "src" / "config.py").write_text("# changed original source\n")
    with pytest.raises(RuntimeError, match="identity"):
        load_model(run, "mha", device="cpu")
    recovered = load_portable_model(portable / "final.pt", device="cpu")
    ids = torch.tensor([[1, 4, 5]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids), recovered(ids), atol=0, rtol=0)
