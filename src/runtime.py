"""Explicit device selection and honestly labeled memory observations."""

import gc
import os
import random
from typing import Any

import numpy as np
import psutil
import torch


def select_device() -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError(
            "Disable PYTORCH_ENABLE_MPS_FALLBACK and restart the kernel; silent operation fallback invalidates comparisons"
        )
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def release_memory(device: torch.device | str) -> None:
    gc.collect()
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def memory_snapshot(device: torch.device | str) -> dict[str, Any]:
    device = torch.device(device)
    values = {
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "system_available_bytes": psutil.virtual_memory().available,
        "system_swap_used_bytes": psutil.swap_memory().used,
        "measurement": "instantaneous snapshot, not an exact peak; unified-memory categories overlap",
    }
    if device.type == "mps":
        values.update(
            mps_tensor_bytes=torch.mps.current_allocated_memory(),
            mps_driver_bytes=torch.mps.driver_allocated_memory(),
        )
    elif device.type == "cuda":
        values.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
        )
    return values


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inspect_environment(run) -> dict[str, Any]:
    from src.io import atomic_json

    run.require_identity()
    device = select_device()
    result = {
        "device": str(device),
        "torch": torch.__version__,
        "precision": "float32",
        "fallback": False,
        "memory": memory_snapshot(device),
        "bitwise_reproducibility_claim": False,
    }
    atomic_json(run.path / "environment" / "device.json", result)
    return result
