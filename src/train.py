"""Train exactly one variant; notebook cells own sequencing, never a CLI."""

import math
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.checkpoints import assert_correctness, atomic_checkpoint, capture_rng, read_checkpoint, restore_rng
from src.evaluate import _verify_tokens, evaluate_loss
from src.io import atomic_json, read_json, sha256_file
from src.models.decoder import Decoder
from src.runtime import release_memory, seed_everything, select_device, synchronize
from src.tracking import FitTracker


def learning_rate_for_update(config, update: int, total: int) -> float:
    """Zero-based schedule; a one-update fixture uses the configured peak LR."""
    if total < 1 or not 0 <= update < total:
        raise ValueError("Learning-rate update must be inside the registered schedule")
    if total == 1:
        return config.learning_rate
    if update == total - 1:
        return config.min_learning_rate
    warmup = (
        min(total - 1, max(1, math.ceil(total * config.warmup_fraction))) if config.warmup_fraction else 0
    )
    if update < warmup:
        return config.learning_rate * (update + 1) / warmup
    progress = (update - warmup) / max(1, total - warmup - 1)
    return config.min_learning_rate + 0.5 * (config.learning_rate - config.min_learning_rate) * (
        1 + math.cos(math.pi * min(1.0, progress))
    )


def epoch_permutation(run, size: int, epoch: int) -> np.ndarray:
    directory = run.path / "config" / "permutations"
    directory.mkdir(exist_ok=True)
    path = directory / f"epoch_{epoch + 1:03d}.npy"
    receipt = path.with_suffix(".json")
    if path.exists():
        expected = read_json(receipt)
        if expected != {"sha256": sha256_file(path), "size": size, "seed": run.config.train.seed + epoch}:
            raise RuntimeError("Saved epoch permutation changed")
        return np.load(path, allow_pickle=False)
    values = np.random.default_rng(run.config.train.seed + epoch).permutation(size)
    np.save(path, values, allow_pickle=False)
    atomic_json(receipt, {"sha256": sha256_file(path), "size": size, "seed": run.config.train.seed + epoch})
    return values


def accumulation_groups(number_of_batches: int, start: int, accumulation: int):
    """Yield exact optimizer groups; a short final group is retained."""
    for first in range(start, number_of_batches, accumulation):
        yield range(first, min(first + accumulation, number_of_batches))


def _accumulate_gradients(
    model, microbatches, group_tokens: int, device: torch.device, gradient_clip: float
) -> tuple[float, float]:
    """Accumulate exact token-normalized gradients with one host read per update.

    Detached losses stay on device through all microbatches. Inspect loss and
    the pre-clipping gradient norm together before the caller may step AdamW.
    Non-finite work can reach backward/clipping, but never an optimizer update.
    """
    if group_tokens <= 0:
        raise ValueError("An optimizer group must contain predicted tokens")
    detached_losses = []
    for inputs, labels in microbatches:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        (loss_sum / group_tokens).backward()
        detached_losses.append(loss_sum.detach())
        # Release the vocabulary logits before allocating the next microbatch.
        del inputs, labels, logits, loss_sum
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=False)
    mean_loss = torch.stack(detached_losses).sum() / group_tokens
    mean_nll, gradient_norm = torch.stack((mean_loss, norm.detach())).cpu().tolist()
    if not math.isfinite(mean_nll):
        raise FloatingPointError("Non-finite training loss; optimizer update was not applied")
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("Non-finite gradient norm; optimizer update was not applied")
    return mean_nll, gradient_norm


def _validate_resume_position(
    position: dict, cfg, windows: int, context: int, batches: int, updates_per_epoch: int
) -> None:
    if set(position) != {"epoch", "next_batch", "step", "tokens_seen"}:
        raise RuntimeError("Checkpoint training position fields are invalid")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in position.values()):
        raise RuntimeError("Checkpoint training counters must be nonnegative integers")
    epoch, next_batch = position["epoch"], position["next_batch"]
    if epoch > cfg.epochs or next_batch > batches or (epoch == cfg.epochs and next_batch):
        raise RuntimeError("Checkpoint position exceeds the registered training budget")
    if next_batch != batches and next_batch % cfg.accumulation_steps:
        raise RuntimeError("Checkpoint was not saved at an optimizer-update boundary")
    expected_step = epoch * updates_per_epoch + math.ceil(next_batch / cfg.accumulation_steps)
    expected_tokens = (epoch * windows + min(next_batch * cfg.microbatch_size, windows)) * context
    if position["step"] != expected_step or position["tokens_seen"] != expected_tokens:
        raise RuntimeError("Checkpoint optimizer/token counters disagree with its sampler position")


def train_variant(run, tokens, variant: str) -> dict:
    run.require_identity()
    _verify_tokens(run, tokens)
    device = select_device()
    assert_correctness(run, device)
    cfg = run.config.train
    directory = run.model_dir(variant)
    stage_name = f"train_{variant}"
    state = run.read_status()["stages"].get(stage_name, {}).get("status")
    if state == "complete":
        read_checkpoint(run, variant, "final")
        return read_json(directory / "result.json")
    if (run.path / "config" / "selection.json").exists():
        raise RuntimeError("Primary evaluation is frozen; create a fresh run to train")
    if state in {"interrupted", "failed", "running"} and run.mode != "resume":
        raise RuntimeError("Select mode='resume' and this run ID before continuing interrupted work")
    dataset = tokens.dataset("train")
    validation = tokens.dataset("validation")
    if len(dataset) == 0:
        raise ValueError("Training data has no complete next-token windows")
    batches = math.ceil(len(dataset) / cfg.microbatch_size)
    updates_per_epoch = math.ceil(batches / cfg.accumulation_steps)
    total_updates = cfg.epochs * updates_per_epoch
    model = optimizer = tracker = saved = None
    parameter = param_group = microbatches = None
    decay, no_decay = [], []
    position = {"epoch": 0, "next_batch": 0, "step": 0, "tokens_seen": 0}
    best_nll = math.inf
    validation_result = None
    latest = directory / "latest.pt"
    resumed = run.mode == "resume" and (latest.exists() or (directory / "latest.previous.pt").exists())

    def save(kind: str, include_optimizer: bool = True) -> None:
        payload = {
            "format_version": 1,
            "variant": variant,
            "identity": run.identity,
            "data_identity": run.data_identity,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "precision": "float32",
            "device": str(device),
            "position": dict(position),
            "best_validation_nll": best_nll,
            "validation": validation_result,
            "scheduler_state": {
                "total_updates": total_updates,
                "completed_updates": position["step"],
                "warmup_fraction": cfg.warmup_fraction,
            },
            "sampler": {
                "seed": cfg.seed,
                "epoch": position["epoch"],
                "next_batch": position["next_batch"],
                "windows": len(dataset),
            },
            "tracker_state": tracker.checkpoint_state(),
            "rng": capture_rng(),
        }
        if include_optimizer:
            payload["optimizer_state"] = optimizer.state_dict()
        atomic_checkpoint(directory / f"{kind}.pt", payload)
        run.logger.info("Checkpoint %s/%s step=%s", variant, kind, position["step"])

    try:
        with run.stage(stage_name):
            seed_everything(cfg.seed)
            model = Decoder(run.config.model.for_variant(variant)).float().to(device)
            for _, parameter in model.named_parameters():
                (decay if parameter.ndim >= 2 else no_decay).append(parameter)
            optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": cfg.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=cfg.learning_rate,
                betas=cfg.betas,
                foreach=False,
            )
            if resumed:
                saved = read_checkpoint(run, variant, "latest")
                if saved["device"] != str(device):
                    raise RuntimeError(
                        "Resume on the original device; changing backends requires a fresh run"
                    )
                position = dict(saved["position"])
                _validate_resume_position(
                    position, cfg, len(dataset), run.config.data.context_length, batches, updates_per_epoch
                )
                if (
                    saved["scheduler_state"]["total_updates"] != total_updates
                    or saved["scheduler_state"]["completed_updates"] != position["step"]
                    or saved["sampler"]
                    != {
                        "seed": cfg.seed,
                        "epoch": position["epoch"],
                        "next_batch": position["next_batch"],
                        "windows": len(dataset),
                    }
                ):
                    raise RuntimeError("Optimizer schedule or sampler changed on resume")
                model.load_state_dict(saved["model_state"], strict=True)
                optimizer.load_state_dict(saved["optimizer_state"])
                best_nll = saved["best_validation_nll"]
                validation_result = saved.get("validation")
            tracker = FitTracker(
                run,
                variant,
                position["step"] if resumed else None,
                resume_state=saved["tracker_state"] if resumed else None,
            )
            if resumed:
                restore_rng(saved["rng"])
                saved = None
            # Keep the detailed inspection summary; save the plain config separately.
            atomic_json(directory / "model_config.json", asdict(model.config))
            (directory / "architecture.txt").write_text(str(model) + "\n", encoding="utf-8")
            atomic_json(
                directory / "parameters.json",
                {
                    "total": sum(p.numel() for p in model.parameters()),
                    "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                },
            )
            if not resumed:
                save("latest")
            for epoch in range(position["epoch"], cfg.epochs):
                permutation = epoch_permutation(run, len(dataset), epoch)
                start_batch = position["next_batch"] if position["epoch"] == epoch else 0
                groups = list(accumulation_groups(batches, start_batch, cfg.accumulation_steps))
                progress = tqdm(
                    groups, desc=f"{variant.upper()} epoch {epoch + 1}/{cfg.epochs}", unit="update"
                )
                synchronize(device)
                window_started = None
                window_tokens = window_updates = 0
                window_nll_sum = 0.0
                for group in progress:
                    if window_started is None:
                        window_started = time.perf_counter()
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    # CPU tensors only: retain this small group to count exact targets.
                    microbatches = []
                    for batch_index in group:
                        indices = permutation[
                            batch_index * cfg.microbatch_size : (batch_index + 1) * cfg.microbatch_size
                        ]
                        samples = [dataset[int(i)] for i in indices]
                        microbatches.append(
                            (torch.stack([s[0] for s in samples]), torch.stack([s[1] for s in samples]))
                        )
                    group_tokens = sum(int((labels != -100).sum()) for _, labels in microbatches)
                    if group_tokens != sum(labels.numel() for _, labels in microbatches):
                        raise ValueError("Training windows must contain only complete, unpadded targets")
                    lr = learning_rate_for_update(cfg, position["step"], total_updates)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    mean_nll, gradient_norm = _accumulate_gradients(
                        model, microbatches, group_tokens, device, cfg.gradient_clip
                    )
                    optimizer.step()
                    position.update(
                        epoch=epoch,
                        next_batch=group.stop,
                        step=position["step"] + 1,
                        tokens_seen=position["tokens_seen"] + group_tokens,
                    )
                    window_tokens += group_tokens
                    window_updates += 1
                    window_nll_sum += mean_nll * group_tokens
                    checkpoint_due = position["step"] % cfg.checkpoint_interval == 0
                    if position["step"] % cfg.log_interval == 0 or checkpoint_due or group.stop == batches:
                        # Fence a whole reporting interval, never every microbatch.
                        # Include batch preparation; exclude metric/checkpoint I/O.
                        synchronize(device)
                        elapsed = time.perf_counter() - window_started
                        window_nll = window_nll_sum / window_tokens
                        tracker.log(
                            position["step"],
                            "train",
                            {
                                "epoch": epoch + 1,
                                "nll": window_nll,
                                "perplexity": math.exp(window_nll) if window_nll < 700 else None,
                                "learning_rate": lr,
                                "gradient_norm": gradient_norm,
                                "tokens_seen": position["tokens_seen"],
                                "tokens_per_second": window_tokens / elapsed,
                                "tokens_in_interval": window_tokens,
                                "updates_in_interval": window_updates,
                                "training_interval_seconds": elapsed,
                                "throughput_scope": "synchronized training interval including batch preparation; excluding logging and checkpoints",
                            },
                        )
                        progress.set_postfix(nll=f"{window_nll:.3f}", lr=f"{lr:.2g}")
                        window_started = None
                        window_tokens = window_updates = 0
                        window_nll_sum = 0.0
                    microbatches = None
                    if checkpoint_due:
                        save("latest")
                optimizer.zero_grad(set_to_none=True)
                validation_result = evaluate_loss(model, validation, cfg.microbatch_size, device)
                tracker.log(position["step"], "validation", {"epoch": epoch + 1, **validation_result})
                position.update(epoch=epoch + 1, next_batch=0)
                if validation_result["nll"] < best_nll:
                    best_nll = validation_result["nll"]
                    save("best", include_optimizer=False)
                save("latest")
            expected_tokens = cfg.epochs * len(dataset) * run.config.data.context_length
            if position["tokens_seen"] != expected_tokens or position["step"] != total_updates:
                raise RuntimeError(
                    "Final training token/update budget does not match the registered protocol"
                )
            save("final", include_optimizer=False)
            result = {
                "variant": variant,
                "status": "complete",
                "epochs": cfg.epochs,
                "tokens_seen": position["tokens_seen"],
                "optimizer_updates": position["step"],
                "final_checkpoint": str((directory / "final.pt").relative_to(run.path)),
                "final_sha256": sha256_file(directory / "final.pt"),
                "best_validation_nll": best_nll,
                "validation_at_final": validation_result,
                "device": str(device),
                "single_seed": cfg.seed,
            }
            atomic_json(directory / "result.json", result)
            tracker.close()
            tracker = None
        return result
    finally:
        if tracker is not None:
            try:
                tracker.close()
            except Exception:
                run.logger.exception("Could not finish metric exports while unwinding training")
        # Optimizer groups and loop aliases also hold parameter tensors. Empty
        # every local owner before releasing the accelerator allocation cache.
        decay.clear()
        no_decay.clear()
        parameter = param_group = microbatches = None
        saved = tracker = optimizer = model = None
        release_memory(device)
