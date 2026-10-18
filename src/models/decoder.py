"""Small decoder-only language model; attention differs only in KV head count."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..cache import KVCache
from ..config import ModelConfig
from .attention import GroupedSelfAttention


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        stable = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = stable * torch.rsqrt(stable.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_width: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden_width, bias=False)
        self.up_proj = nn.Linear(width, hidden_width, bias=False)
        self.down_proj = nn.Linear(hidden_width, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GroupedSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.ffn_dim)

    def forward(
        self,
        x: Tensor,
        *,
        position_offset: int,
        cache: KVCache | None,
        layer_index: int,
        return_attention: bool,
        causal_blocked_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        attention_output, weights = self.attention(
            self.attention_norm(x),
            position_offset=position_offset,
            cache=cache,
            layer_index=layer_index,
            return_attention=return_attention,
            causal_blocked_mask=causal_blocked_mask,
        )
        x = x + attention_output
        return x + self.mlp(self.ffn_norm(x)), weights


def parameter_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class Decoder(nn.Module):
    """All logits retain [batch, time, vocabulary], including last_only=True."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self._cache_owner = object()
        # A single model-local mask serves every attention layer. It grows
        # lazily, never exceeds max_seq_len squared bytes, and is reconstructed
        # rather than serialized with checkpoints. Keep even this empty seed a
        # normal tensor if the model is being inspected under inference_mode.
        with torch.inference_mode(False):
            self.register_buffer(
                "_causal_blocked_mask", torch.empty((0, 0), dtype=torch.bool), persistent=False
            )
        # Module constructors initialize randomly; restore the caller's CPU RNG
        # afterward and replace every parameter with a stable name-seeded value.
        with torch.random.fork_rng(devices=[]):
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
            self.final_norm = RMSNorm(config.d_model, config.norm_eps)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.embedding.weight
            self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if parameter.ndim == 1:
                    parameter.fill_(1.0)
                    continue
                generator = torch.Generator(device="cpu").manual_seed(parameter_seed(self.config.seed, name))
                std = 0.02
                if name.endswith(("attention.o_proj.weight", "mlp.down_proj.weight")):
                    std /= math.sqrt(2 * self.config.num_layers)
                parameter.normal_(mean=0.0, std=std, generator=generator)

    def _get_causal_blocked_mask(
        self, query_length: int, key_length: int, position_offset: int, device: torch.device
    ) -> Tensor | None:
        """Return a shared prefix/offset view; one-query decoding needs no mask.

        Allocate an immutable upper triangle only when capacity/device changes.
        Geometric growth bounds reallocations and retains at most one buffer in
        the model. Existing autograd graphs may safely retain their old views.
        """
        if (
            query_length < 1
            or position_offset < 0
            or position_offset + query_length != key_length
            or key_length > self.config.max_seq_len
        ):
            raise ValueError("Invalid causal-mask lengths, offset, or inference capacity")
        if query_length == 1:
            return None
        buffer = self._causal_blocked_mask
        if buffer.device != device or buffer.shape[0] < key_length or buffer.is_inference():
            capacity = min(
                self.config.max_seq_len,
                max(buffer.shape[0], 1 << (key_length - 1).bit_length()),
            )
            # Validation and generation often initialize the buffer first. A
            # normal tensor is required when a later training backward saves it.
            with torch.inference_mode(False):
                buffer = torch.ones((capacity, capacity), dtype=torch.bool, device=device).triu_(1)
            self._causal_blocked_mask = buffer
        with torch.inference_mode(False):
            return buffer[position_offset : position_offset + query_length, :key_length]

    def make_cache(self, batch_size: int, capacity: int) -> KVCache:
        if capacity > self.config.max_seq_len:
            raise ValueError("Cache capacity exceeds model inference capacity")
        parameter = self.embedding.weight
        return KVCache(
            num_layers=self.config.num_layers,
            batch_size=batch_size,
            num_kv_heads=self.config.num_kv_heads,
            capacity=capacity,
            head_dim=self.config.head_dim,
            device=parameter.device,
            dtype=parameter.dtype,
            owner=self._cache_owner,
        )

    def forward(
        self,
        input_ids: Tensor,
        cache: KVCache | None = None,
        last_only: bool = False,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        if input_ids.ndim != 2 or min(input_ids.shape) <= 0:
            raise ValueError("input_ids must be a nonempty [batch, tokens] tensor")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have torch.long dtype")
        if input_ids.device != self.embedding.weight.device:
            raise ValueError("Move input_ids to the model device before forward")
        offset = cache.length if cache is not None else 0
        if offset + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("Input exceeds configured inference capacity")
        if cache is not None:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("Cached inference requires eval() and no_grad() or inference_mode()")
            cache.begin_append(
                input_ids.shape[1],
                owner=self._cache_owner,
                batch_size=input_ids.shape[0],
                num_layers=self.config.num_layers,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
                device=self.embedding.weight.device,
                dtype=self.embedding.weight.dtype,
            )
        try:
            x = self.embedding(input_ids)
            blocked_mask = self._get_causal_blocked_mask(
                input_ids.shape[1], offset + input_ids.shape[1], offset, input_ids.device
            )
            attention_weights: list[Tensor] = []
            for index, layer in enumerate(self.layers):
                x, weights = layer(
                    x,
                    position_offset=offset,
                    cache=cache,
                    layer_index=index,
                    return_attention=return_attention,
                    causal_blocked_mask=blocked_mask,
                )
                if return_attention:
                    attention_weights.append(weights)
            if last_only:
                x = x[:, -1:, :]
            logits = self.lm_head(self.final_norm(x))
            if cache is not None:
                cache.commit()
        except BaseException:
            if cache is not None:
                cache.abort()
            raise
        return (logits, attention_weights) if return_attention else logits
