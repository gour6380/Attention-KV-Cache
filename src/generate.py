"""Illustrative generation from a saved final checkpoint, never a quality score."""

from __future__ import annotations

import math
import uuid
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from .checkpoints import assert_correctness, load_model, require_primary_selection
from .diagnostics import parity_check
from .io import atomic_json, read_json, sha256_file
from .runtime import release_memory, select_device
from .tokenizer import load_tokenizer

if TYPE_CHECKING:
    from .runs import RunContext


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator) -> int:
    """Sample on CPU with an explicit generator (also supported with MPS models)."""
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p in (0, 1]")
    scores = logits.detach().to(device="cpu", dtype=torch.float32) / temperature
    if (
        scores.ndim != 1
        or torch.isnan(scores).any()
        or torch.isposinf(scores).any()
        or not torch.isfinite(scores).any()
    ):
        raise FloatingPointError("Sampling requires a finite one-dimensional logits vector")
    sorted_logits, order = torch.sort(scores, descending=True)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    # Keep the first token that crosses p, including the leading token at small p.
    remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.softmax(sorted_logits, dim=-1)
    selected = torch.multinomial(filtered, num_samples=1, generator=generator)
    return int(order[selected].item())


def generate_variant(
    run: RunContext,
    variant: str,
    prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    seed: int = 2027,
) -> dict:
    """Generate one saved example within the trained context, stopping on EOS.

    PAD/BOS are suppressed after the initial BOS. The sampling RNG lives on CPU
    so device selection never silently substitutes a different model backend.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be text")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p in (0, 1]")
    run.require_identity()
    selected = require_primary_selection(run, variant)
    tokenizer_path = run.path / "tokenizer" / "tokenizer.json"
    tokenizer_identity = read_json(run.path / "data_manifest" / "tokenizer.json")["tokenizer_sha256"]
    if sha256_file(tokenizer_path) != tokenizer_identity:
        raise RuntimeError("Portable tokenizer differs from the tokenizer bound to this run")
    tokenizer = load_tokenizer(tokenizer_path)
    special = {name: tokenizer.token_to_id(f"<|{name}|>") for name in ("pad", "bos", "eos")}
    if any(value is None for value in special.values()):
        raise ValueError("Saved tokenizer is missing PAD, BOS, or EOS")
    ids = [special["bos"], *tokenizer.encode(prompt, add_special_tokens=False).ids]
    if len(ids) + max_new_tokens > run.config.data.context_length:
        raise ValueError(
            "Prompt plus requested generation exceeds the trained context; shorten one explicitly"
        )
    device = select_device()
    assert_correctness(run, device)
    model = cache = logits = scores = None
    generated: list[int] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with run.stage(f"generate_{variant}"):
        try:
            model = load_model(run, variant, kind="final", device=device)
            model.eval()
            with torch.inference_mode():
                parity = parity_check(
                    model,
                    torch.tensor([ids[:32]], device=device, dtype=torch.long),
                    atol=run.config.atol,
                    rtol=run.config.rtol,
                )
                cache = model.make_cache(batch_size=1, capacity=len(ids) + max_new_tokens)
                logits = model(
                    torch.tensor([ids], device=device, dtype=torch.long), cache=cache, last_only=True
                )
                for index in tqdm(range(max_new_tokens), desc=f"Generate {variant.upper()}", leave=False):
                    scores = logits[0, -1].clone()
                    scores[[special["pad"], special["bos"]]] = float("-inf")
                    token_id = sample_top_p(scores, temperature, top_p, generator)
                    generated.append(token_id)
                    if token_id == special["eos"]:
                        break
                    if index + 1 < max_new_tokens:
                        logits = model(
                            torch.tensor([[token_id]], device=device, dtype=torch.long),
                            cache=cache,
                            last_only=True,
                        )
                cache = logits = scores = None
            record = {
                "variant": variant,
                "prompt": prompt,
                "prompt_token_ids": ids,
                "generated_token_ids": generated,
                "completion": tokenizer.decode(generated, skip_special_tokens=True),
                "text": tokenizer.decode(ids + generated, skip_special_tokens=True),
                "generated_tokens": len(generated),
                "stop_reason": "eos" if generated[-1] == special["eos"] else "max_new_tokens",
                "checkpoint_kind": "final",
                "checkpoint_sha256": selected["sha256"],
                "checkpoint_parity": parity,
                "tokenizer_sha256": sha256_file(tokenizer_path),
                "device": str(device),
                "sampling_device": "cpu",
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "seed": seed,
                "suppressed_token_ids": [special["pad"], special["bos"]],
                "interpretation": "Illustrative sample; not an objective quality ranking",
            }
            path = run.path / "generation" / f"{variant}_{uuid.uuid4().hex[:12]}.json"
            atomic_json(path, record)
            return {**record, "path": str(path)}
        finally:
            # Exceptions retain their traceback; only release references owned here.
            del model, cache, logits, scores
            release_memory(device)
