"""Controlled fixed-token inference trials, with synchronization outside timing."""

from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tqdm.auto import tqdm

from .checkpoints import assert_correctness, load_model, require_primary_selection
from .diagnostics import parity_check
from .evaluate import _atomic_npz, _verify_tokens
from .io import atomic_json, canonical_hash, read_json, sha256_file
from .runtime import memory_snapshot, release_memory, select_device, synchronize

if TYPE_CHECKING:
    from .data import TokenData
    from .runs import RunContext


def fixture_key(prompt_length: int, batch_size: int, kind: str) -> str:
    return f"p{prompt_length}_b{batch_size}_{kind}"


def prepare_benchmark_fixtures(run: RunContext, tokens: TokenData) -> Path:
    """Save identical contiguous held-out token workloads for all three models.

    Each batch row starts at a fixed disjoint position in the test stream. Its
    shorter prompts are prefixes of the longest prompt. Padding is never used.
    These fixtures do not become new training/checkpoint data bindings.
    """
    token_identity = _verify_tokens(run, tokens)
    config = run.config.benchmark
    identity = {"token_manifest_sha256": token_identity, "benchmark_config": asdict(config)}
    path = run.path / "benchmarks" / "fixtures.npz"
    receipt = path.with_suffix(".json")
    if path.exists() or receipt.exists():
        if not path.exists() or not receipt.exists():
            raise RuntimeError("Incomplete fixture artifacts; recover explicitly before retrying")
        metadata = read_json(receipt)
        if metadata["identity_sha256"] != canonical_hash(identity) or metadata["sha256"] != sha256_file(path):
            raise RuntimeError("Existing benchmark fixtures do not match this experiment")
        return path
    width = max(config.prompt_lengths) + config.decode_steps
    rows = max(config.batch_sizes)
    stream = tokens.array("test")
    if stream.shape[0] < rows * width:
        raise ValueError(f"Need at least {rows * width} held-out tokens for the fixed benchmark")
    arrays = {}
    # Host arrays are small; disk-backed training data remains unchanged.
    fixed = np.asarray(stream[: rows * width], dtype=np.int64).reshape(rows, width)
    for prompt_length in config.prompt_lengths:
        for batch_size in config.batch_sizes:
            arrays[fixture_key(prompt_length, batch_size, "prompt")] = fixed[
                :batch_size, :prompt_length
            ].copy()
            arrays[fixture_key(prompt_length, batch_size, "append")] = fixed[
                :batch_size, prompt_length : prompt_length + config.decode_steps
            ].copy()
    with run.stage("prepare_benchmark_fixtures"):
        _atomic_npz(path, **arrays)
        atomic_json(
            receipt,
            {
                **identity,
                "identity_sha256": canonical_hash(identity),
                "sha256": sha256_file(path),
                "split": "test",
                "row_start_offsets": [i * width for i in range(rows)],
                "protocol": "Exactly decode_steps appended-token forwards; no EOS-dependent stopping",
            },
        )
    return path


def _elapsed_ms(call, device) -> float:
    synchronize(device)
    started = time.perf_counter()
    call()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def _forward_appends(model, full_tokens: torch.Tensor, prompt_length: int, steps: int, cache=None) -> None:
    """Exact work count: EOS values never change the number of forward calls."""
    for index in range(steps):
        position = prompt_length + index
        current = (
            full_tokens[:, position : position + 1] if cache is not None else full_tokens[:, : position + 1]
        )
        model(current, cache=cache, last_only=True)


def _cached_trial(model, full_tokens, prompt_length, steps, device) -> dict:
    batch_size, capacity = full_tokens.shape
    cache = model.make_cache(batch_size=batch_size, capacity=capacity)
    prefill_ms = _elapsed_ms(
        lambda: model(full_tokens[:, :prompt_length], cache=cache, last_only=True), device
    )
    decode_ms = _elapsed_ms(lambda: _forward_appends(model, full_tokens, prompt_length, steps, cache), device)
    if cache.length != capacity:
        raise RuntimeError("Benchmark cache did not process every forced append token")
    result = {
        "cached_prefill_ms": prefill_ms,
        "cached_decode_ms": decode_ms,
        "cached_total_ms": prefill_ms + decode_ms,
        "cached_ms_per_step": decode_ms / steps,
        "cached_tokens_per_second": batch_size * steps * 1000 / decode_ms,
        "logical_kv_bytes": cache.logical_bytes,
        "allocated_kv_bytes": cache.allocated_bytes,
        "cached_memory_snapshot": memory_snapshot(device),
    }
    cache = None
    return result


def _uncached_trial(model, full_tokens, prompt_length, steps, device) -> dict:
    prefill_ms = _elapsed_ms(lambda: model(full_tokens[:, :prompt_length], last_only=True), device)
    decode_ms = _elapsed_ms(lambda: _forward_appends(model, full_tokens, prompt_length, steps), device)
    return {
        "uncached_prefill_ms": prefill_ms,
        "uncached_decode_ms": decode_ms,
        "uncached_total_ms": prefill_ms + decode_ms,
        "uncached_ms_per_step": decode_ms / steps,
        "uncached_tokens_per_second": full_tokens.shape[0] * steps * 1000 / decode_ms,
        "uncached_memory_snapshot": memory_snapshot(device),
    }


def _summarize_case(rows: list[dict]) -> dict:
    first = rows[0]
    summary = {
        key: first[key]
        for key in (
            "variant",
            "prompt_length",
            "batch_size",
            "decode_steps",
            "capacity",
            "logical_kv_bytes",
            "allocated_kv_bytes",
            "theoretical_kv_bytes",
            "theoretical_prefill_kv_bytes",
            "kv_ratio_to_mha",
            "systems_only",
        )
    }
    timing_keys = [key for key in first if key.endswith(("_ms", "_ms_per_step", "_tokens_per_second"))]
    for key in timing_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_median"] = float(np.median(values))
        summary[f"{key}_p95"] = float(np.percentile(values, 95))
    summary["trials"] = len(rows)
    summary["quantile_unit"] = "trial; ms_per_step is each trial's mean, not token tail latency"
    return summary


def _workload_parity(model, full_tokens, prompt_length: int, steps: int, atol: float, rtol: float) -> dict:
    """Check the endpoint at the actual workload length outside measured trials."""
    expected = model(full_tokens, last_only=True)
    cache = model.make_cache(batch_size=full_tokens.shape[0], capacity=full_tokens.shape[1])
    actual = model(full_tokens[:, :prompt_length], cache=cache, last_only=True)
    for index in range(steps):
        actual = model(
            full_tokens[:, prompt_length + index : prompt_length + index + 1], cache=cache, last_only=True
        )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    return {
        "passed": True,
        "batch_size": full_tokens.shape[0],
        "tokens": full_tokens.shape[1],
        "prompt_tokens": prompt_length,
        "forced_append_steps": steps,
        "atol": atol,
        "rtol": rtol,
        "max_absolute_logit_error": float((actual - expected).abs().max().item()),
        "comparison": "last logits after every forced append versus full-prefix endpoint",
    }


def benchmark_variant(run: RunContext, fixtures: Path, variant: str) -> dict:
    """Benchmark a frozen final checkpoint; return paths and CPU scalars only.

    Cache allocation and input copies occur outside the measured regions. Timing
    includes Python launch overhead, model work, and device synchronization at
    the end of the region. Mode order alternates to reduce a fixed order bias.
    """
    run.require_identity()
    selected = require_primary_selection(run, variant)
    fixtures = Path(fixtures)
    receipt = read_json(fixtures.with_suffix(".json"))
    if receipt["sha256"] != sha256_file(fixtures):
        raise RuntimeError("Benchmark fixture bytes changed")
    token_binding = read_json(run.path / "data_manifest" / "tokens.json")
    expected_identity = {
        "token_manifest_sha256": canonical_hash(token_binding["manifest"]),
        "benchmark_config": asdict(run.config.benchmark),
    }
    if receipt["identity_sha256"] != canonical_hash(expected_identity):
        raise RuntimeError("Benchmark fixtures belong to different data or configuration")
    config = run.config.benchmark
    model_config = run.config.model.for_variant(variant)
    device = select_device()
    assert_correctness(run, device)
    model = full_tokens = None
    attempts = run.path / "benchmarks" / variant / uuid.uuid4().hex[:12]
    attempts.mkdir(parents=True, exist_ok=False)
    raw_path = attempts / "trials.jsonl"
    summaries = []
    metadata = {
        "variant": variant,
        "device": str(device),
        "precision": "float32",
        "backend": "compact_grouped_bmm_explicit_scores",
        "checkpoint_kind": "final",
        "checkpoint_sha256": selected["sha256"],
        "fixtures_sha256": receipt["sha256"],
        "benchmark_config": asdict(config),
        "quantile_unit": "trial, not individual-token tail latency",
        "memory_measurement": "Snapshots only; RSS, tensor allocation, and driver allocation are distinct, not exact peaks",
        "lengths_beyond_training": "systems-only; no long-context quality claim",
        "cache_allocation_timed": False,
        "input_transfer_timed": False,
        "append_tokens": "fixed held-out values; EOS does not stop or shorten workloads",
        "mode_order": "cached first on even trials, uncached first on odd trials",
    }
    with run.stage(f"benchmark_{variant}"):
        try:
            release_memory(device)
            metadata["before_model_memory_snapshot"] = memory_snapshot(device)
            model = load_model(run, variant, kind="final", device=device)
            model.eval()
            synchronize(device)
            metadata["loaded_model_memory_snapshot"] = memory_snapshot(device)
            atomic_json(attempts / "metadata.json", metadata)
            with (
                torch.inference_mode(),
                np.load(fixtures, allow_pickle=False) as arrays,
                raw_path.open("w", encoding="utf-8") as raw,
            ):
                cases = [(length, batch) for length in config.prompt_lengths for batch in config.batch_sizes]
                for prompt_length, batch_size in tqdm(cases, desc=f"Benchmark {variant.upper()}"):
                    prompt = arrays[fixture_key(prompt_length, batch_size, "prompt")]
                    append = arrays[fixture_key(prompt_length, batch_size, "append")]
                    if prompt.shape != (batch_size, prompt_length) or append.shape != (
                        batch_size,
                        config.decode_steps,
                    ):
                        raise ValueError("Unexpected fixture shape")
                    host_tokens = np.concatenate([prompt, append], axis=1)
                    if host_tokens.min() < 0 or host_tokens.max() >= model_config.vocab_size:
                        raise ValueError("Fixture contains out-of-vocabulary token IDs")
                    full_tokens = torch.from_numpy(host_tokens).to(device=device, dtype=torch.long)
                    synchronize(device)
                    parity = {
                        "representative_full_token_chunk_paths": parity_check(
                            model,
                            full_tokens[:, : min(32, full_tokens.shape[1])],
                            atol=run.config.atol,
                            rtol=run.config.rtol,
                        ),
                        "actual_workload_endpoint": _workload_parity(
                            model,
                            full_tokens,
                            prompt_length,
                            config.decode_steps,
                            run.config.atol,
                            run.config.rtol,
                        ),
                    }
                    atomic_json(attempts / f"parity_p{prompt_length}_b{batch_size}.json", parity)
                    measured = []
                    capacity = prompt_length + config.decode_steps
                    per_token_kv_bytes = (
                        2
                        * model_config.num_layers
                        * batch_size
                        * model_config.num_kv_heads
                        * model_config.head_dim
                        * 4
                    )
                    for trial in range(config.warmup_trials + config.measured_trials):
                        # Reclaim prior tensor references between modes; allocator
                        # reuse is part of warm inference, so do not flush each step.
                        modes = (
                            (_cached_trial, _uncached_trial)
                            if trial % 2 == 0
                            else (_uncached_trial, _cached_trial)
                        )
                        row = {
                            "variant": variant,
                            "prompt_length": prompt_length,
                            "batch_size": batch_size,
                            "decode_steps": config.decode_steps,
                            "capacity": capacity,
                            "trial": trial - config.warmup_trials,
                            "warmup": trial < config.warmup_trials,
                            "first_mode": "cached" if trial % 2 == 0 else "uncached",
                            "theoretical_kv_bytes": per_token_kv_bytes * capacity,
                            "theoretical_prefill_kv_bytes": per_token_kv_bytes * prompt_length,
                            "kv_ratio_to_mha": model_config.num_kv_heads / model_config.num_query_heads,
                            "systems_only": capacity > run.config.data.context_length,
                        }
                        for mode in modes:
                            row.update(mode(model, full_tokens, prompt_length, config.decode_steps, device))
                        if row["logical_kv_bytes"] != row["theoretical_kv_bytes"]:
                            raise RuntimeError(
                                "Actual KV storage differs from the declared theoretical shape"
                            )
                        # All serialization and progress are outside timed regions.
                        raw.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        raw.flush()
                        if not row["warmup"]:
                            measured.append(row)
                    summaries.append({**_summarize_case(measured), "checkpoint_parity_passed": True})
                    full_tokens = None
            csv_path = attempts / "summary.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
                writer.writeheader()
                writer.writerows(summaries)
            result = {
                **metadata,
                "cases": summaries,
                "raw_trials_path": str(raw_path),
                "raw_trials_sha256": sha256_file(raw_path),
                "csv_path": str(csv_path),
            }
            atomic_json(attempts / "summary.json", result)
            path = run.path / "benchmarks" / f"{variant}_summary.json"
            atomic_json(path, result)
            return {**result, "path": str(path)}
        finally:
            del model, full_tokens
            release_memory(device)
