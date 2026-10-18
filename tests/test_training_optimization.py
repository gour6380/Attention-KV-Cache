"""Behavioral coverage for larger microbatches and group-level synchronization.

Only tiny, deterministic CPU fixtures are used. These tests never consume the
real TinyStories cache or create artifacts in the project's runs directory.
"""

import json
import math

import pytest
import torch

import src.train as training
from src.checkpoints import read_checkpoint
from src.config import BenchmarkConfig, DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from src.io import atomic_json, sha256_file
from src.runs import close_run, start_run


class WindowTokens:
    def __init__(self, directory, windows):
        self.directory = directory
        self.windows = windows
        self.manifest = {"context_length": 4, "fixture": "optimization", "windows": windows}
        directory.mkdir(parents=True)
        atomic_json(directory / "manifest.json", self.manifest)

    def verify(self):
        assert self.directory.is_dir()

    def dataset(self, split):
        count = self.windows if split == "train" else 2
        return [
            (
                torch.tensor([1, 4 + index, 5 + index, 6 + index]),
                torch.tensor([4 + index, 5 + index, 6 + index, 2]),
            )
            for index in range(count)
        ]


def fixture_run(
    tmp_path, monkeypatch, *, windows=9, microbatch=4, accumulation=2, log_interval=1, checkpoint_interval=100
):
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "config.py").write_text("# tiny optimization-test source\n")
    (root / "requirements.txt").write_text("# no remote data or runtime dependencies installed by tests\n")
    tokens = WindowTokens(root / "data", windows)
    config = ExperimentConfig(
        data=DataConfig(train_stories=windows, vocab_size=300, context_length=4),
        model=ModelConfig(
            vocab_size=300,
            num_layers=1,
            d_model=16,
            num_query_heads=4,
            ffn_dim=32,
            max_seq_len=32,
        ),
        train=TrainConfig(
            epochs=1,
            microbatch_size=microbatch,
            accumulation_steps=accumulation,
            checkpoint_interval=checkpoint_interval,
            log_interval=log_interval,
        ),
        benchmark=BenchmarkConfig(
            prompt_lengths=(4,), batch_sizes=(1,), decode_steps=2, warmup_trials=0, measured_trials=1
        ),
    )
    monkeypatch.setattr(training, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training, "assert_correctness", lambda *args: {"passed": True})
    run = start_run(root, config)
    run.bind_data(
        "tokens",
        {
            "directory": str(tokens.directory),
            "manifest_sha256": sha256_file(tokens.directory / "manifest.json"),
            "manifest": tokens.manifest,
        },
    )
    return run, tokens


def fit_and_read(run, tokens):
    try:
        result = training.train_variant(run, tokens, "mha")
        checkpoint = read_checkpoint(run, "mha", "final")
        records = [
            json.loads(line) for line in (run.model_dir("mha") / "metrics.jsonl").read_text().splitlines()
        ]
        return result, checkpoint, [record for record in records if record["kind"] == "train"]
    finally:
        close_run(run)


def test_default_microbatch_change_preserves_full_update_budget():
    config = ExperimentConfig()
    assert (config.train.microbatch_size, config.train.accumulation_steps) == (4, 2)
    assert config.train.microbatch_size * config.train.accumulation_steps * config.data.context_length == 8192


@pytest.mark.parametrize("windows", [8, 9, 13])
def test_larger_microbatch_preserves_ordered_optimizer_groups(tmp_path, monkeypatch, windows):
    old_run, old_tokens = fixture_run(
        tmp_path / "old", monkeypatch, windows=windows, microbatch=2, accumulation=4
    )
    new_run, new_tokens = fixture_run(
        tmp_path / "new", monkeypatch, windows=windows, microbatch=4, accumulation=2
    )
    old_result, old_checkpoint, old_metrics = fit_and_read(old_run, old_tokens)
    new_result, new_checkpoint, new_metrics = fit_and_read(new_run, new_tokens)

    expected_updates = math.ceil(windows / 8)
    assert old_result["optimizer_updates"] == new_result["optimizer_updates"] == expected_updates
    assert old_result["tokens_seen"] == new_result["tokens_seen"] == windows * 4
    assert [row["tokens_seen"] for row in old_metrics] == [row["tokens_seen"] for row in new_metrics]
    assert [row["tokens_seen"] for row in new_metrics] == [
        min(8 * step, windows) * 4 for step in range(1, expected_updates + 1)
    ]
    assert [row["nll"] for row in old_metrics] == pytest.approx(
        [row["nll"] for row in new_metrics], abs=2e-6, rel=2e-6
    )
    assert old_checkpoint["position"]["tokens_seen"] == new_checkpoint["position"]["tokens_seen"]
    for name in old_checkpoint["model_state"]:
        torch.testing.assert_close(
            old_checkpoint["model_state"][name], new_checkpoint["model_state"][name], atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("failure", ["infinite_loss_finite_gradients", "finite_loss_nan_gradients"])
def test_nonfinite_update_never_reaches_optimizer_step(tmp_path, monkeypatch, failure):
    run, tokens = fixture_run(tmp_path, monkeypatch)
    cross_entropy = training.F.cross_entropy
    steps = []
    original_step = torch.optim.AdamW.step

    class FiniteLossBadBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, grad_output):
            return torch.full_like(grad_output, float("nan"))

    def failing_loss(*args, **kwargs):
        value = cross_entropy(*args, **kwargs)
        if failure == "infinite_loss_finite_gradients":
            # The derivative of this detached constant is zero: a gradient-only
            # guard would wrongly allow AdamW to update finite parameters.
            return value + value.detach().new_tensor(float("inf"))
        return FiniteLossBadBackward.apply(value)

    def count_step(self, *args, **kwargs):
        steps.append(True)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(training.F, "cross_entropy", failing_loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", count_step)
    try:
        with pytest.raises((FloatingPointError, RuntimeError), match="(?i)(non.?finite|finite)"):
            training.train_variant(run, tokens, "mha")
        assert steps == []
        assert not (run.model_dir("mha") / "final.pt").exists()
        latest = read_checkpoint(run, "mha", "latest")
        assert latest["position"]["step"] == 0
        assert latest["position"]["tokens_seen"] == 0
    finally:
        close_run(run)


@pytest.mark.parametrize("windows,log_interval", [(13, 2), (25, 3)])
def test_logging_interval_is_token_weighted_and_resets_at_flush(tmp_path, monkeypatch, windows, log_interval):
    reference_run, reference_tokens = fixture_run(
        tmp_path / "reference", monkeypatch, windows=windows, log_interval=1
    )
    grouped_run, grouped_tokens = fixture_run(
        tmp_path / "grouped", monkeypatch, windows=windows, log_interval=log_interval
    )
    _, _, reference = fit_and_read(reference_run, reference_tokens)
    result, _, grouped = fit_and_read(grouped_run, grouped_tokens)
    previous_tokens = 0
    source_cursor = 0
    for record in grouped:
        interval = reference[source_cursor : source_cursor + log_interval]
        sizes = []
        for source_record in interval:
            sizes.append(source_record["tokens_seen"] - previous_tokens)
            previous_tokens = source_record["tokens_seen"]
        total_tokens = sum(sizes)
        expected_nll = (
            sum(source_record["nll"] * size for source_record, size in zip(interval, sizes)) / total_tokens
        )
        assert record["nll"] == pytest.approx(expected_nll, rel=2e-6, abs=2e-6)
        assert record["perplexity"] == pytest.approx(math.exp(expected_nll), rel=3e-6)
        assert record["tokens_in_interval"] == total_tokens
        assert record["updates_in_interval"] == len(interval)
        assert record["tokens_seen"] == previous_tokens
        assert record["training_interval_seconds"] > 0
        assert record["tokens_per_second"] == pytest.approx(
            total_tokens / record["training_interval_seconds"]
        )
        assert isinstance(record["throughput_scope"], str) and record["throughput_scope"]
        source_cursor += len(interval)
    assert source_cursor == result["optimizer_updates"]
    assert sum(record["tokens_in_interval"] for record in grouped) == windows * 4


@pytest.mark.parametrize("microbatch_size", [1, 2, 4])
def test_gradient_group_reads_host_once_regardless_of_microbatch_count(monkeypatch, microbatch_size):
    """Catch accidental reintroduction of .item()/float(loss) inside the loop."""
    model = torch.nn.Sequential(torch.nn.Embedding(16, 4), torch.nn.Linear(4, 16, bias=False))
    inputs = torch.arange(32, dtype=torch.long).reshape(8, 4) % 16
    targets = (inputs + 1) % 16
    microbatches = [
        (inputs[start : start + microbatch_size], targets[start : start + microbatch_size])
        for start in range(0, len(inputs), microbatch_size)
    ]
    reads = {"cpu": 0, "tolist": 0, "item": 0, "float": 0}

    def observe(name, original):
        def wrapper(self, *args, **kwargs):
            reads[name] += 1
            return original(self, *args, **kwargs)

        return wrapper

    for name, method in (("cpu", "cpu"), ("tolist", "tolist"), ("item", "item"), ("float", "__float__")):
        monkeypatch.setattr(torch.Tensor, method, observe(name, getattr(torch.Tensor, method)))
    mean_nll, gradient_norm = training._accumulate_gradients(
        model, microbatches, group_tokens=targets.numel(), device=torch.device("cpu"), gradient_clip=1.0
    )
    assert isinstance(mean_nll, float) and math.isfinite(mean_nll)
    assert isinstance(gradient_norm, float) and math.isfinite(gradient_norm)
    assert reads == {"cpu": 1, "tolist": 1, "item": 0, "float": 0}


def test_resume_replays_checkpoint_flushed_interval_without_duplicate_loss(tmp_path, monkeypatch):
    whole, whole_tokens = fixture_run(
        tmp_path / "whole", monkeypatch, windows=33, log_interval=3, checkpoint_interval=2
    )
    interrupted, interrupted_tokens = fixture_run(
        tmp_path / "interrupted", monkeypatch, windows=33, log_interval=3, checkpoint_interval=2
    )
    _, reference_checkpoint, reference_metrics = fit_and_read(whole, whole_tokens)
    original_log = training.FitTracker.log

    def interrupt_after_uncommitted_row(self, step, kind, values):
        original_log(self, step, kind, values)
        if kind == "train" and step == 3:
            raise KeyboardInterrupt()

    monkeypatch.setattr(training.FitTracker, "log", interrupt_after_uncommitted_row)
    try:
        with pytest.raises(KeyboardInterrupt):
            training.train_variant(interrupted, interrupted_tokens, "mha")
        committed = read_checkpoint(interrupted, "mha", "latest")
        assert committed["position"]["step"] == 2
        assert committed["position"]["tokens_seen"] == 64
        assert committed["tracker_state"]["records"] == 1
    finally:
        close_run(interrupted)
    monkeypatch.setattr(training.FitTracker, "log", original_log)
    resumed = start_run(
        interrupted.project_root, interrupted.config, mode="resume", run_id=interrupted.path.name
    )
    _, recovered_checkpoint, recovered_metrics = fit_and_read(resumed, interrupted_tokens)
    assert [record["step"] for record in recovered_metrics] == [2, 3, 4, 5]
    assert [record["updates_in_interval"] for record in recovered_metrics] == [2, 1, 1, 1]
    assert [record["tokens_in_interval"] for record in recovered_metrics] == [64, 32, 32, 4]
    assert sum(record["tokens_in_interval"] for record in recovered_metrics) == 33 * 4
    for reference, recovered in zip(reference_metrics, recovered_metrics):
        for field in (
            "step",
            "nll",
            "perplexity",
            "gradient_norm",
            "tokens_seen",
            "updates_in_interval",
            "tokens_in_interval",
        ):
            assert recovered[field] == reference[field]
    for name in reference_checkpoint["model_state"]:
        torch.testing.assert_close(
            recovered_checkpoint["model_state"][name],
            reference_checkpoint["model_state"][name],
            atol=0,
            rtol=0,
        )
