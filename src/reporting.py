"""Notebook figures and evidence reports built only from existing artifacts.

This module never downloads data, creates a model, trains, or benchmarks. A
partial report is labeled partial and lists missing evidence explicitly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import VARIANTS
from .data import cache_binding, iter_stories, require_data_binding, verify_cache
from .io import atomic_json, read_json, sha256_file

if TYPE_CHECKING:
    from .runs import RunContext

COLORS = {"mha": "#2563eb", "gqa": "#d97706", "mqa": "#059669", "unigram": "#6b7280"}


def _save_figure(run, figure, name: str):
    directory = run.path / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.png", dpi=170, bbox_inches="tight")
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight")
    return figure


def _training_metrics(run) -> pd.DataFrame:
    records = []
    for variant in VARIANTS:
        path = run.model_dir(variant) / "metrics.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                records.extend({"variant": variant, **json.loads(line)} for line in stream if line.strip())
    if not records:
        raise FileNotFoundError(
            "No existing training metrics. Execute a training cell before plotting its history."
        )
    return pd.DataFrame(records)


def plot_story_lengths(run: RunContext, splits: Path):
    """Character lengths before tokenization; this is not a token-length plot."""
    run.require_identity()
    manifest = verify_cache(splits)
    require_data_binding(run, "splits", cache_binding(Path(splits), manifest))
    lengths = np.fromiter(
        (len(row["text"]) for row in iter_stories(Path(splits) / manifest["splits"]["train"]["file"])),
        dtype=np.int64,
        count=manifest["splits"]["train"]["stories"],
    )
    if not len(lengths):
        raise ValueError("Training split has no stories")
    with run.stage("plot_story_lengths"):
        figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
        counts, edges, _ = axis.hist(lengths, bins=50, color=COLORS["mha"], alpha=0.85)
        axis.set(
            xlabel="Normalized story length (characters)",
            ylabel="Training stories",
            title=f"Selected training-story lengths (n={len(lengths):,})",
        )
        axis.grid(axis="y", alpha=0.2)
        atomic_json(
            run.path / "figures" / "story_length_distribution.json",
            {
                "unit": "characters",
                "stories": len(lengths),
                "median": float(np.median(lengths)),
                "p95": float(np.percentile(lengths, 95)),
                "bin_edges": edges.tolist(),
                "bin_counts": counts.astype(int).tolist(),
            },
        )
        return _save_figure(run, figure, "story_lengths")


def plot_training_curves(run: RunContext):
    run.require_identity()
    records = _training_metrics(run)
    with run.stage("plot_training_curves"):
        figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for variant in VARIANTS:
            for kind, style in (("train", "-"), ("validation", "--")):
                rows = records[(records.variant == variant) & (records.kind == kind)].sort_values("step")
                if not rows.empty:
                    axis.plot(
                        rows.step,
                        rows.nll,
                        style,
                        marker="o" if kind == "validation" else None,
                        color=COLORS[variant],
                        alpha=0.9,
                        label=f"{variant.upper()} {kind}",
                    )
        axis.set(
            xlabel="Completed optimizer updates",
            ylabel="Mean negative log-likelihood (nats/token)",
            title="Observed training groups and full-validation loss",
        )
        axis.grid(alpha=0.2)
        axis.legend(ncols=2, fontsize=8)
        return _save_figure(run, figure, "training_loss")


def plot_learning_rate(run: RunContext):
    run.require_identity()
    records = _training_metrics(run)
    with run.stage("plot_learning_rate"):
        figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
        for variant in VARIANTS:
            rows = records[(records.variant == variant) & (records.kind == "train")].sort_values("step")
            if not rows.empty:
                axis.plot(
                    rows.step, rows.learning_rate, label=variant.upper(), color=COLORS[variant], alpha=0.85
                )
        axis.set(
            xlabel="Completed optimizer updates",
            ylabel="Learning rate",
            title="Recorded optimizer learning rates",
        )
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.grid(alpha=0.2)
        axis.legend()
        return _save_figure(run, figure, "learning_rate")


def _check_selected_artifact(run, result: dict, variant: str) -> None:
    selection = run.path / "config" / "selection.json"
    if not selection.is_file():
        raise RuntimeError("Measured model results require an existing frozen checkpoint selection")
    selected = read_json(selection)["variants"][variant]
    if result["checkpoint_sha256"] != selected["sha256"] or result.get("checkpoint_kind") != "final":
        raise RuntimeError(f"{variant} result does not match the frozen primary checkpoint")


def comparison_table(run: RunContext) -> pd.DataFrame:
    """Test-only quality comparison. Missing evaluations remain explicitly pending."""
    run.require_identity()
    records = []
    identities, denominators = set(), set()
    for variant in (*VARIANTS, "unigram"):
        path = run.path / "evaluation" / f"{variant}_test.json"
        if not path.exists():
            records.append({"variant": variant, "status": "pending", "split": "test"})
            continue
        result = read_json(path)
        if result.get("split") != "test" or result["variant"] != variant:
            raise RuntimeError(f"Mislabelled evaluation artifact: {path}")
        if variant != "unigram":
            _check_selected_artifact(run, result, variant)
        identities.add(result["token_manifest_sha256"])
        denominators.add(result["predicted_tokens"])
        records.append(
            {
                "variant": variant,
                "status": "measured",
                "split": "test",
                "nll": result["nll"],
                "perplexity": result["perplexity"],
                "predicted_tokens": result["predicted_tokens"],
                "parameter_count": result.get("parameter_count"),
                "checkpoint_sha256": result.get("checkpoint_sha256"),
            }
        )
    if len(identities) > 1 or len(denominators) > 1:
        raise RuntimeError("Quality results use different held-out tokens or denominators")
    return pd.DataFrame(records)


def _benchmark_cases(run: RunContext) -> pd.DataFrame:
    records = []
    fixture_ids, devices = set(), set()
    for variant in VARIANTS:
        path = run.path / "benchmarks" / f"{variant}_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)
        _check_selected_artifact(run, summary, variant)
        raw_path = Path(summary["raw_trials_path"])
        if not raw_path.is_file() or sha256_file(raw_path) != summary["raw_trials_sha256"]:
            raise RuntimeError(f"Missing or modified raw benchmark measurements for {variant}")
        fixture_ids.add(summary["fixtures_sha256"])
        devices.add(summary["device"])
        records.extend({**case, "device": summary["device"]} for case in summary["cases"])
    if len(fixture_ids) > 1 or len(devices) > 1:
        raise RuntimeError("Cannot compare benchmarks with different fixed fixtures or devices")
    if not records:
        raise FileNotFoundError(
            "No existing benchmark results. Execute benchmark cells before requesting benchmark figures."
        )
    return pd.DataFrame(records)


def _plot_benchmark(
    run,
    columns: tuple[tuple[str, str, str], ...],
    *,
    title: str,
    ylabel: str,
    name: str,
    divisor: float = 1.0,
    xcolumn: str = "prompt_length",
):
    run.require_identity()
    records = _benchmark_cases(run)
    batches = sorted(records.batch_size.unique())
    with run.stage(f"plot_{name}"):
        figure, axes = plt.subplots(
            1, len(batches), figsize=(5.2 * len(batches), 4.6), squeeze=False, layout="constrained"
        )
        for axis, batch in zip(axes[0], batches):
            for variant in VARIANTS:
                rows = records[(records.variant == variant) & (records.batch_size == batch)].sort_values(
                    xcolumn
                )
                if rows.empty:
                    continue
                for column, suffix, linestyle in columns:
                    axis.plot(
                        rows[xcolumn],
                        rows[column] / divisor,
                        color=COLORS[variant],
                        linestyle=linestyle,
                        marker="o",
                        markersize=4,
                        label=f"{variant.upper()} {suffix}".strip(),
                    )
            axis.set(
                title=f"Batch size {int(batch)}",
                xlabel="Tokens in cache after append" if xcolumn == "capacity" else "Prompt length (tokens)",
                ylabel=ylabel,
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        figure.suptitle(title)
        figure.text(
            0.5,
            -0.015,
            "Workloads exceeding the trained context are systems-only; see the workload table.",
            ha="center",
            fontsize=8,
        )
        return _save_figure(run, figure, name)


def plot_cache_growth(run: RunContext):
    return _plot_benchmark(
        run,
        (("logical_kv_bytes", "logical", "-"), ("allocated_kv_bytes", "allocated", "--")),
        title="KV cache storage after all fixed append steps",
        ylabel="KV storage (MiB)",
        name="cache_growth",
        divisor=1024**2,
        xcolumn="capacity",
    )


def plot_prefill_latency(run: RunContext):
    return _plot_benchmark(
        run,
        (("cached_prefill_ms_median", "median", "-"), ("cached_prefill_ms_p95", "p95 trial", "--")),
        title="Cached prefill latency; warm trials, FP32",
        ylabel="Prefill duration (ms)",
        name="prefill_latency",
    )


def plot_decode_latency(run: RunContext):
    return _plot_benchmark(
        run,
        (("cached_ms_per_step_median", "cached", "-"), ("uncached_ms_per_step_median", "uncached", "--")),
        title="Median of trial-mean decode-step latency",
        ylabel="Mean decode-step duration (ms)",
        name="decode_latency",
    )


def quality_cost_table(run: RunContext) -> pd.DataFrame:
    """Join test quality onto each workload; never mix batch sizes/contexts in a rank."""
    quality = comparison_table(run)
    costs = _benchmark_cases(run)
    metrics = [column for column in ("variant", "nll", "perplexity", "parameter_count") if column in quality]
    joined = costs.merge(quality[metrics], on="variant", how="left", validate="many_to_one")
    columns = [
        "variant",
        "device",
        "prompt_length",
        "batch_size",
        "decode_steps",
        "systems_only",
        "nll",
        "perplexity",
        "parameter_count",
        "logical_kv_bytes",
        "allocated_kv_bytes",
        "cached_prefill_ms_median",
        "cached_ms_per_step_median",
        "cached_ms_per_step_p95",
        "uncached_ms_per_step_median",
        "cached_tokens_per_second_median",
    ]
    return joined[[column for column in columns if column in joined]].sort_values(
        ["batch_size", "prompt_length", "variant"]
    )


def _markdown_table(table: pd.DataFrame) -> str:
    """Markdown without an optional tabulate dependency."""

    def render(value):
        if value is None or isinstance(value, float) and math.isnan(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(table.columns) + " |", "| " + " | ".join("---" for _ in table.columns) + " |"]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def assemble_report(run: RunContext) -> Path:
    """Write a report/claim ledger from saved evidence; missing stages stay missing."""
    run.require_identity()
    quality = comparison_table(run)
    missing = [f"{row.variant} test evaluation" for row in quality.itertuples() if row.status != "measured"]
    existing_summaries = [run.path / "benchmarks" / f"{variant}_summary.json" for variant in VARIANTS]
    missing.extend(
        f"{variant} benchmark" for variant, path in zip(VARIANTS, existing_summaries) if not path.exists()
    )
    costs = quality_cost_table(run) if any(path.exists() for path in existing_summaries) else pd.DataFrame()
    correctness_path = run.path / "evaluation" / "correctness.json"
    correctness = read_json(correctness_path) if correctness_path.exists() else {}
    correctness_status = "passed" if correctness.get("passed") else "pending or failed"
    if not correctness.get("passed"):
        missing.append("passed attention/cache correctness record")
    training = []
    for variant in VARIANTS:
        result_path = run.model_dir(variant) / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            training.append(
                {
                    key: result.get(key)
                    for key in ("variant", "status", "epochs", "tokens_seen", "optimizer_updates", "device")
                }
            )
            if result.get("status") != "complete":
                missing.append(f"{variant} completed training")
        else:
            training.append({"variant": variant, "status": "pending"})
            missing.append(f"{variant} completed training")
    evidence = {}
    for directory in (
        "config",
        "data_manifest",
        "evaluation",
        "benchmarks",
        "generation",
        "figures",
        "models",
    ):
        for path in sorted((run.path / directory).rglob("*")):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".png", ".svg", ".txt"}:
                evidence[str(path.relative_to(run.path))] = sha256_file(path)
    status = "complete" if not missing else "partial"
    query_heads = run.config.model.num_query_heads
    kv_heads = {variant: run.config.model.for_variant(variant).num_kv_heads for variant in VARIANTS}
    kv_ratios = {variant: kv_heads[variant] / query_heads for variant in ("gqa", "mqa")}
    intro = [
        f"# Attention and KV cache — {run.path.name}",
        "",
        f"Evidence status: **{status}**. This document reads existing artifacts; assembling it executes no experiment.",
        "",
        "## Question and protocol",
        "",
        "With identical data, model width/depth, and training-token budget, how does reducing KV heads affect language-model quality, cache storage, and decoding speed?",
        "",
        f"Compare independently trained MHA ({kv_heads['mha']} KV heads), GQA ({kv_heads['gqa']}), and MQA ({kv_heads['mqa']}), with {query_heads} query heads. The primary checkpoint is the final checkpoint after equal fixed epochs; validation-best checkpoints are secondary artifacts. Parameter counts differ and must be reported.",
        "",
        f"Seed: {run.config.train.seed}; planned epochs: {run.config.train.epochs}; training context: {run.config.data.context_length} tokens; precision: FP32. Device preference: MPS → CUDA → CPU, with no silent runtime fallback.",
        "",
        "## Completeness and correctness",
        "",
        f"Correctness record status: **{correctness_status}**; configured tolerances are atol={run.config.atol}, rtol={run.config.rtol}. Tests and this record are different artifacts: the notebook correctness cell checks the active source/config/device.",
        "",
        "Missing evidence: "
        + (", ".join(missing) if missing else "none from the required quality/cost protocol")
        + ".",
        "",
        _markdown_table(pd.DataFrame(training)),
        "",
        "## Test quality",
        "",
        "The test set is derived from the official TinyStories validation split. Loss is total negative log-likelihood divided by the recorded number of predicted tokens; perplexity is exp(aggregate loss). Exact deduplication does not rule out near duplicates.",
        "",
        _markdown_table(quality.drop(columns=["checkpoint_sha256"], errors="ignore")),
        "",
        "## System measurements",
        "",
        "Timing uses fixed held-out tokens and fixed append-step counts. EOS never shortens a workload. Cache allocation, tokenization, transfers, progress bars, and logging are outside the timed regions. Device synchronization bounds timing. Quantiles summarize trials; p95 of each trial's mean step latency is not individual-token tail latency.",
        "",
        "Logical KV bytes and allocated cache capacity are storage measures. RSS, MPS tensor allocations, and Metal driver allocations are separate snapshots in the raw trials; none is an exact peak-memory measurement.",
        "",
        "Any benchmark whose processed context exceeds the training context is systems-only. It cannot establish long-context language quality.",
        "",
        _markdown_table(costs) if not costs.empty else "No benchmark measurements exist yet.",
        "",
        "## Interpretation and limitations",
        "",
        f"The strongest structural conclusion is that compact K/V storage scales with the KV-head count: GQA uses {kv_ratios['gqa']:.3%} and MQA {kv_ratios['mqa']:.3%} of the MHA KV storage under this configuration. This predicts cache bytes, not total process memory or a speedup. Timing and quality conclusions require the recorded measurements above.",
        "",
        "One seed, synthetic TinyStories text, different parameter counts, and the common explicit grouped-matmul attention implementation limit generalization. These results are neither a paper reproduction nor a universal comparison of optimized attention kernels. Generation samples illustrate behavior and are not a quality ranking.",
        "",
        "Next action: "
        + (
            "complete the missing evidence before making a measured comparative claim."
            if missing
            else "inspect the measured quality/storage/latency tradeoff at each workload; replicate selected comparisons with additional seeds before broader claims."
        ),
        "",
        "## Reproduction and evidence",
        "",
        "Open notebooks/attention_kv_cache.ipynb with the registered project kernel. Set an explicit fresh or resume mode, then execute one cell at a time. Run All intentionally includes training. Configuration, source snapshots, environment, split/tokenizer identities, final checkpoint identities, and artifact hashes are saved with this run.",
        "",
        "Evidence hashes: [manifest.json](manifest.json). Claim ledger: [claims.json](claims.json). Machine-readable quality/cost tables accompany this report.",
        "",
        "## Research basis",
        "",
        "- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)",
        "- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)",
        "- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/)",
        "- [TinyStories](https://arxiv.org/abs/2305.07759)",
        "",
    ]
    figures = [
        name
        for name in (
            "story_lengths",
            "training_loss",
            "learning_rate",
            "cache_growth",
            "prefill_latency",
            "decode_latency",
        )
        if (run.path / "figures" / f"{name}.png").is_file()
    ]
    if figures:
        intro.extend(["## Existing figures", ""])
        for name in figures:
            intro.extend([f"![{name.replace('_', ' ')}](../figures/{name}.png)", ""])
    claims = [
        {
            "kind": "structural",
            "claim": "For fixed head_dim/layers/batch/length/dtype, KV bytes scale linearly with KV heads.",
            "ratios_to_mha": kv_ratios,
            "scope": "KV tensors only; no measured latency claim",
        },
        {
            "kind": "observation",
            "status": "available" if not quality[quality.status == "measured"].empty else "pending",
            "claim": "Test NLL/perplexity and token denominators are recorded in quality.csv where status is measured.",
            "evidence": [
                key for key in evidence if key.startswith("evaluation/") and key.endswith("_test.json")
            ],
            "limitation": "One seed; synthetic text; parameter counts differ.",
        },
        {
            "kind": "observation",
            "status": "available" if not costs.empty else "pending",
            "claim": "Workload-specific timings and cache bytes are recorded in quality_cost.csv when benchmarks exist.",
            "evidence": [
                key for key in evidence if key.startswith("benchmarks/") and key.endswith("_summary.json")
            ],
            "limitation": "Common explicit grouped attention; warm trials; snapshot memory; no universal speed claim.",
        },
    ]
    with run.stage("assemble_report"):
        directory = run.path / "report"
        directory.mkdir(exist_ok=True)
        quality.to_csv(directory / "quality.csv", index=False)
        if not costs.empty:
            costs.to_csv(directory / "quality_cost.csv", index=False)
        atomic_json(
            directory / "manifest.json", {"status": status, "missing": missing, "evidence_sha256": evidence}
        )
        atomic_json(
            directory / "claims.json", {"claims": claims, "public_action": "none; this is a local report"}
        )
        destination = directory / "report.md"
        destination.write_text("\n".join(intro), encoding="utf-8")
        return destination
