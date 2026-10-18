"""Benchmark protocol tests use recording fakes, not timed training models."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import benchmark
from src.config import BenchmarkConfig, ExperimentConfig
from src.io import read_json


class RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, ids, cache=None, last_only=False):
        self.calls.append((ids.clone(), cache is not None, last_only))


def test_forced_append_count_never_uses_eos_stopping():
    full = torch.full((2, 8), 2, dtype=torch.long)  # Every value could be EOS.
    uncached = RecordingModel()
    benchmark._forward_appends(uncached, full, prompt_length=3, steps=5)
    assert [ids.shape[1] for ids, _, _ in uncached.calls] == [4, 5, 6, 7, 8]
    assert all(last_only for _, _, last_only in uncached.calls)
    cached = RecordingModel()
    benchmark._forward_appends(cached, full, prompt_length=3, steps=5, cache=object())
    assert len(cached.calls) == 5
    assert all(ids.shape == (2, 1) and uses_cache for ids, uses_cache, _ in cached.calls)
    for index, (ids, _, _) in enumerate(cached.calls):
        torch.testing.assert_close(ids, uncached.calls[index][0][:, -1:])


def test_timer_synchronizes_before_clock_and_after_work(monkeypatch):
    events = []

    def sync(device):
        events.append("sync")

    def timer():
        events.append("clock")
        return {2: 1.0, 5: 1.25}[len(events)]

    monkeypatch.setattr(benchmark, "synchronize", sync)
    monkeypatch.setattr(benchmark.time, "perf_counter", timer)
    assert benchmark._elapsed_ms(lambda: events.append("work"), "cpu") == 250
    assert events == ["sync", "clock", "work", "sync", "clock"]


def test_fixtures_are_same_exact_token_workloads_for_all_models(tmp_path, monkeypatch):
    config = ExperimentConfig(
        benchmark=BenchmarkConfig(
            prompt_lengths=(2, 4), batch_sizes=(1, 2), decode_steps=3, warmup_trials=0, measured_trials=2
        )
    )
    run = SimpleNamespace(path=tmp_path, config=config, stage=lambda name: nullcontext())
    tokens = SimpleNamespace(array=lambda split: np.arange(30, dtype=np.uint32))
    monkeypatch.setattr(benchmark, "_verify_tokens", lambda *_: "token-identity")
    path = benchmark.prepare_benchmark_fixtures(run, tokens)
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["p4_b2_prompt"], [[0, 1, 2, 3], [7, 8, 9, 10]])
        np.testing.assert_array_equal(arrays["p4_b2_append"], [[4, 5, 6], [11, 12, 13]])
        np.testing.assert_array_equal(arrays["p2_b1_prompt"], [[0, 1]])
        np.testing.assert_array_equal(arrays["p2_b1_append"], [[2, 3, 4]])
    assert benchmark.prepare_benchmark_fixtures(run, tokens) == path
    assert not (tmp_path / "data_manifest").exists()
    assert read_json(path.with_suffix(".json"))["row_start_offsets"] == [0, 7]


def test_changed_fixture_bytes_rejected_on_reuse(tmp_path, monkeypatch):
    config = ExperimentConfig(
        benchmark=BenchmarkConfig(prompt_lengths=(2,), batch_sizes=(1,), decode_steps=2)
    )
    run = SimpleNamespace(path=tmp_path, config=config, stage=lambda name: nullcontext())
    monkeypatch.setattr(benchmark, "_verify_tokens", lambda *_: "tokens")
    path = benchmark.prepare_benchmark_fixtures(run, SimpleNamespace(array=lambda _: np.arange(10)))
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="do not match"):
        benchmark.prepare_benchmark_fixtures(run, SimpleNamespace())


def test_trial_statistics_are_not_individual_token_percentiles():
    common = dict(
        variant="gqa",
        prompt_length=128,
        batch_size=1,
        decode_steps=128,
        capacity=256,
        logical_kv_bytes=100,
        allocated_kv_bytes=100,
        theoretical_kv_bytes=100,
        theoretical_prefill_kv_bytes=50,
        kv_ratio_to_mha=0.25,
        systems_only=False,
    )
    rows = [{**common, "cached_decode_ms": ms, "cached_ms_per_step": ms / 128} for ms in (10, 20, 30)]
    summary = benchmark._summarize_case(rows)
    assert summary["cached_decode_ms_median"] == 20
    assert summary["cached_decode_ms_p95"] == 29
    assert summary["cached_ms_per_step_median"] == 20 / 128
    assert "not token tail" in summary["quantile_unit"]
    assert summary["trials"] == 3


def test_too_short_stream_is_rejected_not_repeated(tmp_path, monkeypatch):
    config = ExperimentConfig(
        benchmark=BenchmarkConfig(prompt_lengths=(4,), batch_sizes=(2,), decode_steps=3)
    )
    run = SimpleNamespace(path=tmp_path, config=config)
    monkeypatch.setattr(benchmark, "_verify_tokens", lambda *_: "tokens")
    with pytest.raises(ValueError, match="at least 14"):
        benchmark.prepare_benchmark_fixtures(run, SimpleNamespace(array=lambda _: np.arange(13)))
