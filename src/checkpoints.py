"""Portable state-dict checkpoints and immutable primary evaluation selection."""

import os
import random
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import VARIANTS, ModelConfig
from src.io import atomic_json, read_json, sha256_file
from src.runtime import select_device


def capture_rng() -> dict:
    np_state = np.random.get_state()
    value = {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "cpu": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available():
        value["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng(value: dict) -> None:
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(value["cpu"])
    if "mps" in value and torch.backends.mps.is_available():
        torch.mps.set_rng_state(value["mps"])
    if "cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda"])


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    return value


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(_cpu_tree(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Updating a model file and a separate checksum cannot be one atomic
        # rename. Preserve one verified latest checkpoint before replacing it.
        # An interrupt between the two new renames can then replay safely.
        if path.name == "latest.pt" and _verified_artifact(path):
            previous = path.with_name("latest.previous.pt")
            backup_fd, backup_temp = tempfile.mkstemp(prefix=".previous-", dir=path.parent)
            try:
                with os.fdopen(backup_fd, "wb") as destination, path.open("rb") as source:
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                os.replace(backup_temp, previous)
                atomic_json(previous.with_suffix(".json"), read_json(path.with_suffix(".json")))
            finally:
                if os.path.exists(backup_temp):
                    os.unlink(backup_temp)
        os.replace(temp, path)
        atomic_json(path.with_suffix(".json"), {"sha256": sha256_file(path), "bytes": path.stat().st_size})
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _verified_artifact(path: Path) -> bool:
    try:
        receipt = read_json(path.with_suffix(".json"))
        return path.stat().st_size == receipt["bytes"] and sha256_file(path) == receipt["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def read_checkpoint(run, variant: str, kind: str = "final") -> dict:
    run.require_identity()
    if kind not in {"final", "best", "latest"}:
        raise ValueError("Checkpoint kind must be final, best, or latest")
    path = run.model_dir(variant) / f"{kind}.pt"
    if not _verified_artifact(path):
        previous = path.with_name("latest.previous.pt")
        if kind != "latest" or not _verified_artifact(previous):
            raise RuntimeError(f"Checkpoint/checksum pair missing or inconsistent: {path}")
        run.logger.warning("Latest checkpoint update was incomplete; replaying verified %s", previous.name)
        path = previous
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        value.get("format_version") != 1
        or value.get("variant") != variant
        or value.get("precision") != "float32"
    ):
        raise RuntimeError("Checkpoint format, variant or precision mismatch")
    if value["identity"] != run.identity or value["data_identity"] != run.data_identity:
        raise RuntimeError("Checkpoint provenance differs from this run")
    if value["model_config"] != asdict(run.config.model.for_variant(variant)):
        raise RuntimeError("Checkpoint model architecture mismatch")
    return value


def load_model(run, variant: str, kind: str = "final", device=None):
    from src.models.decoder import Decoder

    value = read_checkpoint(run, variant, kind)
    model = Decoder(ModelConfig(**value["model_config"]))
    model.load_state_dict(value["model_state"], strict=True)
    model.to(device if device is not None else select_device())
    model.eval()
    del value
    return model


def load_portable_model(checkpoint_path: Path | str, device="cpu"):
    """Recover a moved checkpoint for inference, outside original-run claims.

    Keep the adjacent .json checksum receipt with the .pt file. This checks
    artifact integrity and reconstructs architecture/weights, but intentionally
    does not certify the original source, environment, data, or benchmark run.
    """
    from src.models.decoder import Decoder

    path = Path(checkpoint_path)
    if not _verified_artifact(path):
        raise RuntimeError("Portable checkpoint requires an intact .pt/.json checksum pair")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format_version") != 1 or value.get("precision") != "float32":
        raise RuntimeError("Unsupported portable checkpoint format or precision")
    model = Decoder(ModelConfig(**value["model_config"])).float()
    model.load_state_dict(value["model_state"], strict=True)
    model.to(torch.device(device))
    model.eval()
    del value
    return model


def assert_correctness(run, device=None) -> dict:
    run.require_identity()
    result = read_json(run.path / "evaluation" / "correctness.json")
    selected_device = str(device if device is not None else select_device())
    if (
        not result.get("passed")
        or result.get("run_identity") != run.identity
        or result.get("device") != selected_device
    ):
        raise RuntimeError(
            "Run the notebook correctness cell for this source/config/device before training or measuring"
        )
    if result.get("atol") != run.config.atol or result.get("rtol") != run.config.rtol:
        raise RuntimeError("Parity tolerances changed")
    if (
        result.get("model_config") != asdict(run.config.model)
        or not result.get("tests")
        or not all(test.get("passed") for test in result["tests"])
    ):
        raise RuntimeError("Correctness receipt lacks the required model configuration/tests")
    return result


def freeze_primary_checkpoints(run) -> dict:
    run.require_identity()
    assert_correctness(run)
    variants = {}
    counts = set()
    for variant in VARIANTS:
        value = read_checkpoint(run, variant, "final")
        if value["position"]["epoch"] != run.config.train.epochs:
            raise RuntimeError(f"Incomplete training: {variant}")
        if run.read_status()["stages"].get(f"train_{variant}", {}).get("status") != "complete":
            raise RuntimeError(f"Training stage not complete: {variant}")
        counts.add(value["position"]["tokens_seen"])
        path = run.model_dir(variant) / "final.pt"
        variants[variant] = {
            "path": str(path.relative_to(run.path)),
            "sha256": sha256_file(path),
            "tokens_seen": value["position"]["tokens_seen"],
        }
        del value
    if len(counts) != 1:
        raise RuntimeError("Primary checkpoints received unequal token budgets")
    payload = {
        "identity": run.identity,
        "data_identity": run.data_identity,
        "policy": "final_after_equal_fixed_epochs",
        "variants": variants,
    }
    target = run.path / "config" / "selection.json"
    if target.exists() and read_json(target) != payload:
        raise RuntimeError("Primary selection was already frozen with different artifacts")
    atomic_json(target, payload)
    return payload


def require_primary_selection(run, variant: str) -> dict:
    run.require_identity()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    payload = read_json(run.path / "config" / "selection.json")
    if payload["identity"] != run.identity or payload["data_identity"] != run.data_identity:
        raise RuntimeError("Primary selection identity mismatch")
    selected = payload["variants"][variant]
    if selected["path"] != str((run.model_dir(variant) / "final.pt").relative_to(run.path)):
        raise RuntimeError("Frozen selection points outside the variant's final checkpoint")
    if sha256_file(run.path / selected["path"]) != selected["sha256"]:
        raise RuntimeError("Frozen primary checkpoint changed")
    return selected
