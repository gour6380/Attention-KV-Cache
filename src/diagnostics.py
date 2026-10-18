"""Explicit, user-invoked architecture inspection and fail-closed parity checks."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .config import VARIANTS
from .io import atomic_json
from .models import Decoder, causal_mask, grouped_attention
from .runtime import release_memory, select_device, synchronize


def inspect_model(run, variant: str) -> dict[str, Any]:
    """Save architecture/config without retaining a model in notebook state."""
    run.require_identity()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    with run.stage(f"inspect_{variant}"):
        config = run.config.model.for_variant(variant)
        model = Decoder(config).float()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        summary = {
            "variant": variant,
            "config": asdict(config),
            "attention_backend": "compact_grouped_bmm",
            "parameter_count": parameter_count,
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "weight_bytes_fp32": parameter_count * 4,
            "kv_bytes_per_token_per_sequence_fp32": 2
            * config.num_layers
            * config.num_kv_heads
            * config.head_dim
            * 4,
            "tied_embedding_and_output": model.embedding.weight is model.lm_head.weight,
            "initialization": "SHA256(seed:parameter_name); normal std=0.02; residual projections scaled by sqrt(2*layers); RMSNorm=1",
            "dropout": 0.0,
            "parameter_shapes": {name: list(value.shape) for name, value in model.named_parameters()},
        }
        destination = run.model_dir(variant)
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "architecture.json", summary)
        (destination / "architecture.txt").write_text(str(model) + "\n", encoding="utf-8")
        del model
    return summary


def _comparison(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return {"max_absolute_error": float((actual.detach() - expected.detach()).abs().max().cpu())}


def _attention_reference_checks(atol: float, rtol: float) -> list[dict]:
    """Expanded K/V are confined to this tiny CPU test oracle, never production."""
    tests = []
    for kv_heads in (8, 2, 1):
        for offset, length in ((0, 5), (3, 2), (6, 1)):
            generator = torch.Generator().manual_seed(11 + kv_heads + offset)
            q = torch.randn(2, 8, length, 8, generator=generator, requires_grad=True)
            k = torch.randn(2, kv_heads, offset + length, 8, generator=generator, requires_grad=True)
            v = torch.randn(2, kv_heads, offset + length, 8, generator=generator, requires_grad=True)
            reference_inputs = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
            rq, rk, rv = reference_inputs
            actual = grouped_attention(q, k, v, position_offset=offset)
            allowed = causal_mask(length, offset + length, offset, "cpu")
            expected = F.scaled_dot_product_attention(
                rq,
                rk.repeat_interleave(8 // kv_heads, dim=1),
                rv.repeat_interleave(8 // kv_heads, dim=1),
                attn_mask=allowed,
                dropout_p=0.0,
            )
            errors = _comparison(actual, expected, atol, rtol)
            probe = torch.randn(actual.shape, generator=generator)
            actual_grads = torch.autograd.grad((actual * probe).sum(), (q, k, v))
            expected_grads = torch.autograd.grad((expected * probe).sum(), reference_inputs)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
                _comparison(actual_grad, expected_grad, atol, rtol)
            tests.append(
                {
                    "name": "grouped_bmm_output_and_gradients",
                    "device": "cpu",
                    "kv_heads": kv_heads,
                    "offset": offset,
                    "query_length": length,
                    "passed": True,
                    **errors,
                }
            )
    return tests


def _decoder_checks(config, device: torch.device, atol: float, rtol: float) -> list[dict]:
    tests = []
    small = replace(
        config, vocab_size=97, num_layers=2, d_model=config.num_query_heads * 8, ffn_dim=96, max_seq_len=64
    )
    ids = torch.randint(0, small.vocab_size, (2, 13), generator=torch.Generator().manual_seed(101)).to(device)
    for variant in VARIANTS:
        model = Decoder(small.for_variant(variant)).float().to(device).eval()
        with torch.inference_mode():
            full, weights = model(ids, return_attention=True)
            cache = model.make_cache(batch_size=2, capacity=32)
            allocated = cache.allocated_bytes
            token_outputs = []
            for position in range(ids.shape[1]):
                token_outputs.append(model(ids[:, position : position + 1], cache=cache))
                if cache.length != position + 1 or cache.allocated_bytes != allocated:
                    raise AssertionError("Cache length/storage changed incorrectly")
            token_result = torch.cat(token_outputs, dim=1)
            errors = _comparison(token_result, full, atol, rtol)
            del token_outputs, token_result
            cache.reset()
            chunk_outputs = []
            offset = 0
            for length in (5, 2, 1, 5):
                chunk_outputs.append(model(ids[:, offset : offset + length], cache=cache))
                offset += length
            _comparison(torch.cat(chunk_outputs, dim=1), full, atol, rtol)
            expected_logical = 2 * small.num_layers * 2 * model.config.num_kv_heads * 13 * small.head_dim * 4
            expected_capacity = expected_logical * 32 // 13
            if cache.logical_bytes != expected_logical or allocated != expected_capacity:
                raise AssertionError("Cache storage does not match the compact KV formula")
            changed = ids.clone()
            changed[:, 7:] = (changed[:, 7:] + 1) % small.vocab_size
            _comparison(model(changed)[:, :7], full[:, :7], atol, rtol)
            for attention in weights:
                if torch.count_nonzero(torch.triu(attention, diagonal=1)).item() != 0:
                    raise AssertionError("Attention leaked across the causal mask")
                _comparison(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol, rtol)
            _comparison(model(ids, last_only=True), full[:, -1:], atol, rtol)
            cache.reset()
            _comparison(model(ids, cache=cache), full, atol, rtol)
            if model.lm_head.weight is not model.embedding.weight:
                raise AssertionError("Output and token embedding weights are not tied")
            tests.append(
                {
                    "name": "decoder_cache_parity_causality_storage_reset",
                    "variant": variant,
                    "device": str(device),
                    "passed": True,
                    "tokens": 13,
                    "batch_size": 2,
                    "allocated_cache_bytes": allocated,
                    **errors,
                }
            )
        synchronize(device)
        del model, cache, full, weights, attention, chunk_outputs, changed
        release_memory(device)
    return tests


def run_correctness_checks(run) -> dict[str, Any]:
    """Run only when the notebook user invokes this cell; persist failures too.

    Small CPU reference/gradient checks plus CPU and selected-device decoder
    checks certify the implementation, not the quality of an untrained model.
    """
    run.require_identity()
    selected_device = select_device()
    result: dict[str, Any] = {
        "passed": False,
        "atol": run.config.atol,
        "rtol": run.config.rtol,
        "device": str(selected_device),
        "run_identity": run.identity,
        "model_config": asdict(run.config.model),
        "tests": [],
        "backend": "compact_grouped_bmm",
        "precision": "float32",
        "scope": "small implementation fixtures; no training or dataset required",
    }
    receipt = run.path / "evaluation" / "correctness.json"
    with run.stage("correctness"):
        try:
            result["tests"].extend(_attention_reference_checks(run.config.atol, run.config.rtol))
            devices = [torch.device("cpu")]
            if selected_device.type != "cpu":
                devices.append(selected_device)
            for device in devices:
                result["tests"].extend(
                    _decoder_checks(run.config.model, device, run.config.atol, run.config.rtol)
                )
            result["passed"] = True
        except BaseException as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            atomic_json(receipt, result)
            raise
        atomic_json(receipt, result)
    return result


def parity_check(
    model: Decoder, input_ids: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-4
) -> dict[str, Any]:
    """Verify a specific checkpoint on full, token, chunk and last-only paths.

    The caller owns model lifetime. This function retains no model or tensor in
    its result and leaves training/eval mode unchanged. Benchmarks should pass
    a bounded fixture so full-vocabulary logits do not dominate memory.
    """
    if model.training:
        raise ValueError("Set the model to eval() before checkpoint parity checks")
    if atol <= 0 or rtol <= 0:
        raise ValueError("Parity tolerances must be positive")
    if input_ids.ndim != 2 or min(input_ids.shape) <= 0:
        raise ValueError("Expected nonempty [batch, tokens] input_ids")
    batch, length = input_ids.shape
    with torch.inference_mode():
        full = model(input_ids)
        cache = model.make_cache(batch, length)
        token_outputs = [model(input_ids[:, pos : pos + 1], cache=cache) for pos in range(length)]
        token_errors = _comparison(torch.cat(token_outputs, dim=1), full, atol, rtol)
        del token_outputs
        cache.reset()
        # Include non-square multi-query chunks after a prefix whenever possible.
        split = max(1, length // 3)
        chunk_outputs = []
        chunk_lengths = []
        for start in range(0, length, split):
            current = input_ids[:, start : start + split]
            chunk_outputs.append(model(current, cache=cache))
            chunk_lengths.append(current.shape[1])
        chunk_errors = _comparison(torch.cat(chunk_outputs, dim=1), full, atol, rtol)
        del chunk_outputs
        last_errors = _comparison(model(input_ids, last_only=True), full[:, -1:], atol, rtol)
        cache.reset()
        prefill_size = max(1, length // 2)
        prefill = model(input_ids[:, :prefill_size], cache=cache, last_only=True)
        _comparison(prefill, full[:, prefill_size - 1 : prefill_size], atol, rtol)
        for pos in range(prefill_size, length):
            _comparison(
                model(input_ids[:, pos : pos + 1], cache=cache, last_only=True),
                full[:, pos : pos + 1],
                atol,
                rtol,
            )
        if cache.length != length:
            raise AssertionError("Checkpoint parity left an incorrect cache length")
        synchronize(input_ids.device)
        result = {
            "passed": True,
            "device": str(input_ids.device),
            "batch_size": batch,
            "tokens": length,
            "atol": atol,
            "rtol": rtol,
            "full_vs_token": token_errors,
            "full_vs_chunk": chunk_errors,
            "full_vs_last_only": last_errors,
            "chunk_lengths": chunk_lengths,
            "prefill_tokens": prefill_size,
            "kv_heads": model.config.num_kv_heads,
            "cache_allocated_bytes": cache.allocated_bytes,
        }
    return result


def _walkthrough_tokens(run, model: Decoder, text: str, max_tokens: int) -> tuple[torch.Tensor, list[str]]:
    from .tokenizer import load_tokenizer

    run.require_identity()
    if model.training:
        raise ValueError("Set the walkthrough model to eval()")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide nonempty text for the walkthrough")
    tokenizer = load_tokenizer(run.path / "tokenizer" / "tokenizer.json")
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not encoded.ids:
        raise ValueError("The tokenizer produced no tokens")
    bos_id = tokenizer.token_to_id("<|bos|>")
    if bos_id is None:
        raise ValueError("Tokenizer is missing the training BOS marker")
    token_ids, labels = [bos_id, *encoded.ids], ["<|bos|>", *encoded.tokens]
    if len(token_ids) > min(max_tokens, model.config.max_seq_len):
        raise ValueError(
            f"Walkthrough text has {len(token_ids)} tokens including BOS; use at most {min(max_tokens, model.config.max_seq_len)}"
        )
    tensor = torch.tensor([token_ids], dtype=torch.long, device=model.embedding.weight.device)
    return tensor, labels


def _variant_name(model: Decoder) -> str:
    if model.config.num_kv_heads == model.config.num_query_heads:
        return "mha"
    return "mqa" if model.config.num_kv_heads == 1 else "gqa"


def plot_attention(run, model: Decoder, text: str, layer: int = 0, head: int = 0):
    """Plot one head from one layer and save the same figure to the current run."""
    import matplotlib.pyplot as plt

    ids, labels = _walkthrough_tokens(run, model, text, max_tokens=96)
    if not 0 <= layer < model.config.num_layers or not 0 <= head < model.config.num_query_heads:
        raise ValueError("Attention layer or head is out of range")
    with torch.inference_mode():
        _, attentions = model(ids, return_attention=True)
        matrix = attentions[layer][0, head].detach().cpu().numpy()
    figure, ax = plt.subplots(
        figsize=(min(16, max(7, len(labels) * 0.25)), min(14, max(6, len(labels) * 0.23)))
    )
    im = ax.imshow(matrix, origin="upper", aspect="auto", interpolation="nearest", cmap="viridis")
    ticks = range(0, len(labels), max(1, (len(labels) + 31) // 32))
    ax.set_xticks(list(ticks), [labels[index] for index in ticks], rotation=90)
    ax.set_yticks(list(ticks), [labels[index] for index in ticks])
    ax.set(
        xlabel="Key position / token",
        ylabel="Query position / token",
        title=f"{_variant_name(model).upper()} · layer {layer}, query head {head} · causal attention",
    )
    figure.colorbar(im, ax=ax, label="Attention probability")
    figure.tight_layout()
    path: Path = run.path / "figures" / f"attention_{_variant_name(model)}_layer{layer}_head{head}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    del attentions
    return figure


def inspect_cache_growth(run, model: Decoder, text: str) -> list[dict[str, Any]]:
    """Show exact compact logical storage growth with fixed allocated capacity."""
    ids, _ = _walkthrough_tokens(run, model, text, max_tokens=256)
    records = []
    with torch.inference_mode():
        cache = model.make_cache(1, ids.shape[1])
        for index in range(ids.shape[1]):
            model(ids[:, index : index + 1], cache=cache, last_only=True)
            records.append(
                {
                    "tokens": cache.length,
                    "capacity": cache.capacity,
                    "kv_heads": model.config.num_kv_heads,
                    "logical_bytes": cache.logical_bytes,
                    "allocated_bytes": cache.allocated_bytes,
                }
            )
    atomic_json(run.path / "evaluation" / f"cache_growth_{_variant_name(model)}.json", records)
    return records


def compare_cached_outputs(run, model: Decoder, text: str) -> dict[str, Any]:
    """Check trained-model parity on walkthrough text, raising on any mismatch."""
    ids, _ = _walkthrough_tokens(run, model, text, max_tokens=128)
    result = parity_check(model, ids, atol=run.config.atol, rtol=run.config.rtol)
    result["variant"] = _variant_name(model)
    atomic_json(run.path / "evaluation" / f"walkthrough_parity_{_variant_name(model)}.json", result)
    return result
