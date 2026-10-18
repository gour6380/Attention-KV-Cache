"""Token-weighted evaluation; these functions never start a training fit."""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoints import assert_correctness, load_model, require_primary_selection
from .io import atomic_json, canonical_hash, read_json, sha256_file
from .runtime import release_memory, select_device, synchronize

if TYPE_CHECKING:
    from .data import TokenData
    from .runs import RunContext


def loss_summary(total_nll: float, predicted_tokens: int) -> dict:
    """Aggregate loss before exponentiation, including its explicit denominator."""
    if predicted_tokens <= 0:
        raise ValueError("Evaluation contains no predicted tokens")
    if not math.isfinite(total_nll):
        raise FloatingPointError("Non-finite evaluation loss")
    nll = total_nll / predicted_tokens
    # JSON has no portable representation for infinity.
    perplexity = math.exp(nll) if nll < math.log(np.finfo(float).max) else None
    return {
        "total_nll": total_nll,
        "predicted_tokens": predicted_tokens,
        "nll": nll,
        "perplexity": perplexity,
        "perplexity_overflow": perplexity is None,
    }


def evaluate_loss(model, dataset, batch_size: int, device, progress: bool = True) -> dict:
    """Evaluate every target once; ignore -100 padding and restore model mode.

    Safe for epoch validation: no checkpoint selection or test-set access occurs
    here. ``dataset`` yields pairs of one-dimensional IDs and target tensors.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    was_training = model.training
    total_nll, predicted_tokens, windows, padding = 0.0, 0, 0, 0
    model.eval()
    synchronize(device)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for ids, labels in tqdm(loader, desc="Evaluate", disable=not progress, leave=False):
                ids, labels = ids.to(device), labels.to(device)
                valid = int(labels.ne(-100).sum().item())
                windows += ids.shape[0]
                padding += labels.numel() - valid
                if valid == 0:
                    continue
                logits = model(ids)
                summed = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                    ignore_index=-100,
                )
                total_nll += float(summed.item())
                predicted_tokens += valid
        synchronize(device)
    finally:
        model.train(was_training)
    result = loss_summary(total_nll, predicted_tokens)
    result.update(windows=windows, padding_targets=padding, elapsed_seconds=time.perf_counter() - started)
    return result


def _verify_tokens(run: RunContext, tokens: TokenData) -> str:
    """Verify existing immutable token metadata without changing run bindings."""
    run.require_identity()
    tokens.verify()
    expected = {
        "directory": str(tokens.directory),
        "manifest_sha256": sha256_file(tokens.directory / "manifest.json"),
        "manifest": tokens.manifest,
    }
    if read_json(run.path / "data_manifest" / "tokens.json") != expected:
        raise RuntimeError("Token data is not the immutable data bound to this run")
    return canonical_hash(tokens.manifest)


def evaluate_variant(run: RunContext, tokens: TokenData, variant: str) -> dict:
    """Evaluate the frozen final checkpoint on the derived test set."""
    token_identity = _verify_tokens(run, tokens)
    selected = require_primary_selection(run, variant)
    device = select_device()
    assert_correctness(run, device)
    model = None
    with run.stage(f"evaluate_{variant}"):
        try:
            model = load_model(run, variant, kind="final", device=device)
            result = evaluate_loss(model, tokens.dataset("test"), run.config.train.microbatch_size, device)
            result.update(
                variant=variant,
                split="test",
                split_origin="derived from the official validation split",
                checkpoint_kind="final",
                checkpoint_sha256=selected["sha256"],
                token_manifest_sha256=token_identity,
                device=str(device),
                precision="float32",
                parameter_count=sum(p.numel() for p in model.parameters()),
            )
            path = run.path / "evaluation" / f"{variant}_test.json"
            atomic_json(path, result)
            return {**result, "path": str(path)}
        finally:
            del model
            release_memory(device)


def _atomic_npz(path: Path, **arrays) -> None:
    """Write arrays atomically without numpy adding an unexpected extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".arrays-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fit_unigram(run: RunContext, tokens: TokenData) -> Path:
    """Fit an add-one-smoothed baseline using training targets only."""
    token_identity = _verify_tokens(run, tokens)
    path = run.path / "evaluation" / "unigram.npz"
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        if not path.exists() or not metadata_path.exists():
            raise RuntimeError("Incomplete unigram artifacts; recover them explicitly before retrying")
        saved = read_json(metadata_path)
        if saved["token_manifest_sha256"] != token_identity or saved["sha256"] != sha256_file(path):
            raise RuntimeError("Existing unigram identity does not match this run")
        return path
    with run.stage("fit_unigram"):
        vocab_size = run.config.model.vocab_size
        counts = np.zeros(vocab_size, dtype=np.int64)
        # Dataset targets match the training objective, including any discarded tail.
        loader = DataLoader(tokens.dataset("train"), batch_size=128, shuffle=False, num_workers=0)
        for _, labels in tqdm(loader, desc="Fit unigram", leave=False):
            valid = labels[labels.ne(-100)].numpy()
            counts += np.bincount(valid, minlength=vocab_size)
        token_count = int(counts.sum())
        if token_count == 0:
            raise ValueError("Cannot fit the baseline without training targets")
        log_probs = np.log((counts.astype(np.float64) + 1.0) / (token_count + vocab_size))
        _atomic_npz(path, counts=counts, log_probs=log_probs)
        atomic_json(
            metadata_path,
            {
                "token_manifest_sha256": token_identity,
                "sha256": sha256_file(path),
                "training_predicted_tokens": token_count,
                "vocab_size": vocab_size,
                "smoothing": "add_one",
                "fit_split": "train",
            },
        )
    return path


def evaluate_unigram(run: RunContext, tokens: TokenData, baseline_path: Path, split: str = "test") -> dict:
    """Evaluate an existing baseline. Test evaluation requires frozen models."""
    if split not in {"validation", "test"}:
        raise ValueError("Baseline evaluation split must be validation or test")
    token_identity = _verify_tokens(run, tokens)
    if split == "test":
        for variant in ("mha", "gqa", "mqa"):
            require_primary_selection(run, variant)
    baseline_path = Path(baseline_path)
    metadata = read_json(baseline_path.with_suffix(".json"))
    if metadata["token_manifest_sha256"] != token_identity or metadata["sha256"] != sha256_file(
        baseline_path
    ):
        raise RuntimeError("Baseline or token metadata changed")
    with np.load(baseline_path, allow_pickle=False) as saved:
        log_probs = saved["log_probs"].copy()
    total_nll, count, windows = 0.0, 0, 0
    with run.stage(f"evaluate_unigram_{split}"):
        loader = DataLoader(tokens.dataset(split), batch_size=128, shuffle=False, num_workers=0)
        for _, labels in tqdm(loader, desc=f"Unigram {split}", leave=False):
            valid = labels[labels.ne(-100)].numpy()
            total_nll += float(-log_probs[valid].sum(dtype=np.float64))
            count += valid.size
            windows += labels.shape[0]
        result = loss_summary(total_nll, count)
        result.update(
            variant="unigram",
            split=split,
            windows=windows,
            split_origin="derived from the official validation split",
            baseline_sha256=metadata["sha256"],
            token_manifest_sha256=token_identity,
        )
        path = run.path / "evaluation" / f"unigram_{split}.json"
        atomic_json(path, result)
        return {**result, "path": str(path)}
