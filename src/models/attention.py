"""Shared grouped-bmm attention for MHA, GQA and MQA."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..cache import KVCache
from ..config import ModelConfig


def causal_mask(
    query_length: int, key_length: int, position_offset: int, device: torch.device | str
) -> Tensor:
    """True means visible, including the cached prefix and each query itself."""
    queries = torch.arange(position_offset, position_offset + query_length, device=device)
    keys = torch.arange(key_length, device=device)
    return keys.unsqueeze(0) <= queries.unsqueeze(1)


def grouped_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    position_offset: int = 0,
    return_attention: bool = False,
    causal_blocked_mask: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Attention on [B,Hq,L,D] queries and compact [B,Hkv,S,D] K/V.

    Fold each KV group's query heads into the bmm row dimension. K/V remain
    [B*Hkv,S,D] views, including when S is a prefix of preallocated storage.
    Scores are explicitly materialized: this backend is not FlashAttention.
    An optional reusable boolean mask has shape [L,S] with True for blocked
    future positions. Its rows must start at position_offset. The decoder
    supplies a view shared by all layers; standalone calls construct a mask.
    """
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Expected rank-four queries and matching compact K/V")
    batch, query_heads, query_length, head_dim = query.shape
    key_batch, kv_heads, key_length, key_dim = key.shape
    if (
        batch != key_batch
        or head_dim != key_dim
        or kv_heads <= 0
        or query_heads % kv_heads
        or min(batch, query_length, key_length) <= 0
    ):
        raise ValueError("Incompatible query/KV shapes")
    if position_offset < 0 or position_offset + query_length != key_length:
        raise ValueError("Keys must contain exactly the prefix plus current query chunk")
    if any(t.device != query.device or t.dtype != query.dtype for t in (key, value)):
        raise ValueError("Query/K/V devices and dtypes must match")
    if causal_blocked_mask is not None:
        if (
            not isinstance(causal_blocked_mask, Tensor)
            or causal_blocked_mask.shape != (query_length, key_length)
            or causal_blocked_mask.dtype != torch.bool
            or causal_blocked_mask.device != query.device
        ):
            raise ValueError(
                "The causal blocked mask must be boolean [query_length,key_length] on the query device"
            )
    group_size = query_heads // kv_heads
    q = query.reshape(batch, kv_heads, group_size, query_length, head_dim)
    q = q.reshape(batch * kv_heads, group_size * query_length, head_dim)
    k = key.view(batch * kv_heads, key_length, head_dim)
    v = value.view(batch * kv_heads, key_length, head_dim)
    scores = torch.bmm(q, k.transpose(1, 2)) * (1.0 / math.sqrt(head_dim))
    scores = scores.reshape(batch, kv_heads, group_size, query_length, key_length)
    # With one query at the end, every stored key is visible; avoid a mask kernel.
    if query_length != 1:
        blocked = causal_blocked_mask
        if blocked is None:
            blocked = ~causal_mask(query_length, key_length, position_offset, query.device)
        elif torch.is_grad_enabled() and blocked.is_inference():
            # Standalone callers may pass a mask created during inference. The
            # backward operation saves its mask, which must be a normal tensor.
            with torch.inference_mode(False):
                blocked = blocked.clone()
        scores = scores.masked_fill(blocked[None, None, None, :, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    grouped = probabilities.reshape(batch * kv_heads, group_size * query_length, key_length)
    output = torch.bmm(grouped, v).reshape(batch, query_heads, query_length, head_dim)
    if return_attention:
        return output, probabilities.reshape(batch, query_heads, query_length, key_length)
    return output


class RotaryEmbedding(nn.Module):
    """Interleaved-pair RoPE with fixed FP32 tables covering inference capacity."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), frequencies)
        self.register_buffer("cosines", angles.cos(), persistent=False)
        self.register_buffer("sines", angles.sin(), persistent=False)

    def forward(self, x: Tensor, position_offset: int) -> Tensor:
        end = position_offset + x.shape[-2]
        if position_offset < 0 or end > self.cosines.shape[0]:
            raise ValueError("RoPE positions exceed configured inference capacity")
        cos = self.cosines[position_offset:end].to(dtype=x.dtype)[None, None]
        sin = self.sines[position_offset:end].to(dtype=x.dtype)[None, None]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


class GroupedSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        kv_width = config.num_kv_heads * config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_width, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_base)

    def forward(
        self,
        x: Tensor,
        *,
        position_offset: int = 0,
        cache: KVCache | None = None,
        layer_index: int = 0,
        return_attention: bool = False,
        causal_blocked_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        batch, length, _ = x.shape
        cfg = self.config
        q = self.q_proj(x).view(batch, length, cfg.num_query_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        q, k = self.rotary(q, position_offset), self.rotary(k, position_offset)
        if cache is not None:
            k, v = cache.write_layer(layer_index, k, v)
        else:
            # Projection outputs are interleaved by token. Materialize once for a
            # full forward; cached prefix views already have bmm-compatible strides.
            k, v = k.contiguous(), v.contiguous()
        result = grouped_attention(
            q,
            k,
            v,
            position_offset=position_offset,
            return_attention=return_attention,
            causal_blocked_mask=causal_blocked_mask,
        )
        if return_attention:
            context, weights = result
        else:
            context, weights = result, None
        merged = context.transpose(1, 2).reshape(batch, length, cfg.d_model)
        return self.o_proj(merged), weights
