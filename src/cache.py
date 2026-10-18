"""Compact, preallocated K/V storage owned by a single inference request."""

from __future__ import annotations

import torch
from torch import Tensor


class KVCache:
    """Each layer stores [batch, kv_heads, capacity, head_dim], without head copies.

    An append is transactional: all layers write at the old length and the model
    commits once. Failed forwards leave the previous prefix valid. Reset hides
    old storage without zeroing it; only the active prefix is ever exposed.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        capacity: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        owner: object | None = None,
    ):
        dimensions = (num_layers, batch_size, num_kv_heads, capacity, head_dim)
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in dimensions):
            raise ValueError("All cache dimensions must be positive integers")
        if not dtype.is_floating_point:
            raise ValueError("K/V cache dtype must be floating point")
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.capacity = capacity
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.owner = owner
        shape = (batch_size, num_kv_heads, capacity, head_dim)
        self.keys = [torch.empty(shape, device=self.device, dtype=dtype) for _ in range(num_layers)]
        self.values = [torch.empty_like(key) for key in self.keys]
        self.device = self.keys[0].device
        self._length = 0
        self._pending_tokens = 0
        self._written_layers: set[int] = set()

    @property
    def length(self) -> int:
        return self._length

    @property
    def logical_bytes(self) -> int:
        return (
            2
            * self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * self.length
            * self.head_dim
            * self.keys[0].element_size()
        )

    @property
    def allocated_bytes(self) -> int:
        return sum(t.untyped_storage().nbytes() for t in (*self.keys, *self.values))

    def reset(self) -> None:
        self._length = 0
        self.abort()

    def begin_append(
        self,
        token_count: int,
        *,
        owner: object,
        batch_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> int:
        if self._pending_tokens:
            raise RuntimeError("The cache already has an unfinished append")
        if self.owner is not owner:
            raise ValueError("This cache belongs to another model instance")
        expected = (self.batch_size, self.num_layers, self.num_kv_heads, self.head_dim)
        if (batch_size, num_layers, num_kv_heads, head_dim) != expected:
            raise ValueError("Cache batch size or model layout does not match")
        if torch.device(device) != self.device or dtype != self.dtype:
            raise ValueError("Cache device or dtype does not match the model")
        if token_count <= 0 or self.length + token_count > self.capacity:
            raise ValueError("Cache append exceeds capacity or has no tokens")
        self._pending_tokens = token_count
        self._written_layers.clear()
        return self.length

    def write_layer(self, layer_index: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        if not self._pending_tokens:
            raise RuntimeError("Call begin_append before writing K/V")
        if not 0 <= layer_index < self.num_layers or layer_index in self._written_layers:
            raise ValueError("Invalid or duplicate cache layer write")
        expected = (self.batch_size, self.num_kv_heads, self._pending_tokens, self.head_dim)
        if key.shape != expected or value.shape != expected:
            raise ValueError("K/V tensor shapes do not match the pending append")
        if any(t.device != self.device or t.dtype != self.dtype for t in (key, value)):
            raise ValueError("K/V tensor device or dtype does not match the cache")
        if key.requires_grad or value.requires_grad:
            raise RuntimeError("The K/V cache is inference-only")
        end = self.length + self._pending_tokens
        self.keys[layer_index][:, :, self.length : end].copy_(key)
        self.values[layer_index][:, :, self.length : end].copy_(value)
        self._written_layers.add(layer_index)
        # Prefix slicing returns a view. In particular, do not call contiguous().
        return self.keys[layer_index][:, :, :end], self.values[layer_index][:, :, :end]

    def commit(self) -> None:
        if not self._pending_tokens or len(self._written_layers) != self.num_layers:
            raise RuntimeError("Cannot commit before every layer has written K/V")
        self._length += self._pending_tokens
        self.abort()

    def abort(self) -> None:
        self._pending_tokens = 0
        self._written_layers.clear()
