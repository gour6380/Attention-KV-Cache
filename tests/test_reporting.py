"""Reporting consumes saved evidence and never starts missing computation."""

import ast
from contextlib import nullcontext
from pathlib import Path

import pytest

from src.config import VARIANTS, ExperimentConfig
from src.io import atomic_json, read_json, sha256_file
from src.reporting import assemble_report, comparison_table, quality_cost_table


class ReportRun:
    def __init__(self, path):
        self.path = path
        self.config = ExperimentConfig()
        for name in (
            "evaluation",
            "benchmarks",
            "models",
            "config",
            "report",
            "figures",
            "data_manifest",
            "generation",
        ):
            (path / name).mkdir(parents=True)
        atomic_json(
            path / "config" / "selection.json",
            {"variants": {name: {"sha256": f"hash-{name}"} for name in VARIANTS}},
        )

    def require_identity(self):
        pass

    def stage(self, name):
        return nullcontext()

    def model_dir(self, variant):
        return self.path / "models" / variant


def write_quality(run, variant, *, denominator=100, token_identity="same", checkpoint=None):
    result = {
        "variant": variant,
        "split": "test",
        "nll": 2.0,
        "perplexity": 7.389056,
        "predicted_tokens": denominator,
        "token_manifest_sha256": token_identity,
        "checkpoint_kind": "final",
        "checkpoint_sha256": checkpoint or f"hash-{variant}",
        "parameter_count": 12000000,
    }
    atomic_json(run.path / "evaluation" / f"{variant}_test.json", result)


def write_benchmark(run, variant):
    raw = run.path / "benchmarks" / f"{variant}_trials.jsonl"
    raw.write_text('{"trial":0}\n')
    case = {
        "variant": variant,
        "prompt_length": 128,
        "batch_size": 1,
        "decode_steps": 128,
        "capacity": 256,
        "systems_only": False,
        "logical_kv_bytes": 1024,
        "allocated_kv_bytes": 1024,
        "cached_prefill_ms_median": 1.0,
        "cached_ms_per_step_median": 0.5,
        "cached_ms_per_step_p95": 0.6,
        "uncached_ms_per_step_median": 2.5,
        "cached_tokens_per_second_median": 2000.0,
    }
    atomic_json(
        run.path / "benchmarks" / f"{variant}_summary.json",
        {
            "variant": variant,
            "checkpoint_sha256": f"hash-{variant}",
            "checkpoint_kind": "final",
            "fixtures_sha256": "same-fixtures",
            "device": "cpu",
            "cases": [case],
            "raw_trials_path": str(raw),
            "raw_trials_sha256": sha256_file(raw),
        },
    )


def test_partial_report_identifies_missing_evidence(tmp_path):
    run = ReportRun(tmp_path / "run_001")
    report = assemble_report(run)
    assert report.is_file()
    manifest = read_json(report.parent / "manifest.json")
    assert manifest["status"] == "partial"
    assert "mha test evaluation" in manifest["missing"]
    assert "passed attention/cache correctness record" in manifest["missing"]
    assert "pending or failed" in report.read_text()
    assert not list((run.path / "models").rglob("*.pt"))
    assert not list((run.path / "benchmarks").glob("*_summary.json"))


def test_quality_comparison_never_mixes_validation_and_test(tmp_path):
    run = ReportRun(tmp_path / "run_001")
    write_quality(run, "mha")
    atomic_json(run.path / "evaluation" / "unigram_validation.json", {"nll": 9.0, "split": "validation"})
    table = comparison_table(run)
    assert table.loc[table.variant == "mha", "status"].item() == "measured"
    assert table.loc[table.variant == "unigram", "status"].item() == "pending"
    assert set(table.split) == {"test"}


def test_comparison_rejects_different_denominators(tmp_path):
    run = ReportRun(tmp_path / "run_001")
    write_quality(run, "mha", denominator=100)
    write_quality(run, "gqa", denominator=99)
    with pytest.raises(RuntimeError, match="different held-out"):
        comparison_table(run)


def test_comparison_rejects_nonprimary_checkpoint(tmp_path):
    run = ReportRun(tmp_path / "run_001")
    write_quality(run, "mha", checkpoint="other-checkpoint")
    with pytest.raises(RuntimeError, match="primary checkpoint"):
        comparison_table(run)


def test_quality_cost_join_keeps_workload_identifiers(tmp_path):
    run = ReportRun(tmp_path / "run_001")
    write_quality(run, "mha")
    write_benchmark(run, "mha")
    table = quality_cost_table(run)
    assert table.shape[0] == 1
    assert table.iloc[0]["batch_size"] == 1
    assert table.iloc[0]["prompt_length"] == 128
    assert table.iloc[0]["nll"] == 2.0


def test_report_source_has_no_computation_entrypoint_calls():
    source = Path(__file__).parents[1] / "src" / "reporting.py"
    tree = ast.parse(source.read_text())
    forbidden = {
        "train_variant",
        "evaluate_variant",
        "evaluate_unigram",
        "benchmark_variant",
        "load_model",
        "download_source",
        "prepare_tokens",
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & forbidden
