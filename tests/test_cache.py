"""Cache correctness, lifecycle, storage and rejection tests."""

from dataclasses import replace

import pytest
import torch

from src.config import ModelConfig
from src.models import Decoder


def small_config():
    return ModelConfig(vocab_size=97, num_layers=2, d_model=64, ffn_dim=96, max_seq_len=64)


def available_device(name):
    if name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(name)


@pytest.mark.parametrize("variant", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_full_token_and_chunked_cache_parity_storage_and_reset(variant, device_name):
    device = available_device(device_name)
    model = Decoder(small_config().for_variant(variant)).to(device).eval()
    tokens = torch.randint(0, 97, (2, 13), generator=torch.Generator().manual_seed(21)).to(device)
    with torch.inference_mode():
        expected = model(tokens)
        cache = model.make_cache(2, 32)
        pointers = [tensor.data_ptr() for tensor in (*cache.keys, *cache.values)]
        allocated = 2 * 2 * 2 * model.config.num_kv_heads * 32 * 8 * 4
        assert cache.allocated_bytes == allocated
        assert cache.logical_bytes == 0
        pieces = []
        for position in range(tokens.shape[1]):
            pieces.append(model(tokens[:, position : position + 1], cache=cache))
            assert cache.length == position + 1
            assert cache.allocated_bytes == allocated
            assert [tensor.data_ptr() for tensor in (*cache.keys, *cache.values)] == pointers
        torch.testing.assert_close(torch.cat(pieces, dim=1), expected, atol=1e-5, rtol=1e-4)
        assert cache.logical_bytes == allocated * 13 // 32
        prefix = cache.keys[0][:, :, :4].clone()
        cache.reset()
        assert cache.length == 0 and cache.logical_bytes == 0
        assert cache.allocated_bytes == allocated
        chunks, offset = [], 0
        for length in (4, 1, 3, 5):
            chunks.append(model(tokens[:, offset : offset + length], cache=cache))
            offset += length
        torch.testing.assert_close(torch.cat(chunks, dim=1), expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(cache.keys[0][:, :, :4], prefix, atol=1e-5, rtol=1e-4)


def test_cache_rejects_training_gradients_wrong_owner_batch_dtype_and_overflow():
    model = Decoder(small_config())
    tokens = torch.ones(1, 3, dtype=torch.long)
    cache = model.make_cache(1, 4)
    with torch.no_grad(), pytest.raises(RuntimeError, match="eval"):
        model(tokens, cache=cache)
    model.eval()
    with pytest.raises(RuntimeError, match="no_grad"):
        model(tokens, cache=cache)
    other = Decoder(small_config()).eval()
    with torch.inference_mode():
        with pytest.raises(ValueError, match="another model"):
            other(tokens, cache=cache)
        with pytest.raises(ValueError, match="batch"):
            model(tokens.expand(2, -1), cache=cache)
        model(tokens, cache=cache)
        assert cache.length == 3
        saved_prefix = cache.keys[0][:, :, :3].clone()
        with pytest.raises(ValueError, match="capacity"):
            model(tokens[:, :2], cache=cache)
        assert cache.length == 3
        torch.testing.assert_close(cache.keys[0][:, :, :3], saved_prefix, atol=0, rtol=0)
    fresh_cache = model.make_cache(1, 4)
    model.double()
    with torch.no_grad(), pytest.raises(ValueError, match="dtype"):
        model(tokens, cache=fresh_cache)


def test_cache_prefix_is_a_view_and_partial_append_is_not_committed():
    model = Decoder(small_config().for_variant("gqa")).eval()
    cache = model.make_cache(2, 16)
    config = model.config
    args = dict(
        owner=model._cache_owner,
        batch_size=2,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache.begin_append(3, **args)
    k = torch.randn(2, 2, 3, 8)
    v = torch.randn(2, 2, 3, 8)
    with torch.no_grad():
        key_prefix, value_prefix = cache.write_layer(0, k, v)
    assert key_prefix.untyped_storage().data_ptr() == cache.keys[0].untyped_storage().data_ptr()
    assert value_prefix.untyped_storage().data_ptr() == cache.values[0].untyped_storage().data_ptr()
    # Folding B and Hkv must remain a view even with unused preallocated capacity.
    folded = key_prefix.view(4, 3, 8)
    assert folded.untyped_storage().data_ptr() == cache.keys[0].untyped_storage().data_ptr()
    with pytest.raises(RuntimeError, match="every layer"):
        cache.commit()
    assert cache.length == 0
    cache.abort()
    assert cache.length == 0
    with torch.inference_mode():
        logits = model(torch.ones(2, 3, dtype=torch.long), cache=cache)
    assert logits.shape == (2, 3, 97) and cache.length == 3


def test_model_enforces_sequence_capacity_and_cache_capacity():
    model = Decoder(replace(small_config(), max_seq_len=8)).eval()
    with pytest.raises(ValueError, match="capacity"):
        model.make_cache(1, 9)
    with pytest.raises(ValueError, match="capacity"):
        model(torch.zeros(1, 9, dtype=torch.long))
    with pytest.raises(ValueError, match="positive"):
        model.make_cache(0, 4)


def test_single_token_decode_does_not_allocate_a_causal_mask():
    model = Decoder(small_config().for_variant("gqa")).eval()
    ids = torch.randint(0, 97, (2, 9), generator=torch.Generator().manual_seed(9))
    empty_buffer = model._causal_blocked_mask
    assert empty_buffer.numel() == 0
    with torch.inference_mode():
        cache = model.make_cache(2, 16)
        outputs = [model(ids[:, pos : pos + 1], cache=cache) for pos in range(ids.shape[1])]
    assert model._causal_blocked_mask is empty_buffer
    assert model._causal_blocked_mask.numel() == 0
    reference = Decoder(model.config).eval()
    with torch.inference_mode():
        expected = reference(ids)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_mask_buffer_follows_module_device_changes_and_remains_boolean(device_name):
    device = available_device(device_name)
    model = Decoder(small_config().for_variant("gqa")).eval()
    cpu_ids = torch.ones(2, 11, dtype=torch.long)
    with torch.inference_mode():
        expected = model(cpu_ids)
    assert model._causal_blocked_mask.shape == (16, 16)
    model.to(device=device)
    assert model._causal_blocked_mask.device.type == device.type
    assert model._causal_blocked_mask.dtype == torch.bool
    with torch.inference_mode():
        actual = model(cpu_ids.to(device)).cpu()
        cache = model.make_cache(2, 32)
        model(cpu_ids[:, :5].to(device), cache=cache)
        chunk = model(cpu_ids[:, 5:].to(device), cache=cache)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(chunk.cpu(), expected[:, 5:], atol=1e-5, rtol=1e-4)
    model.to(device="cpu", dtype=torch.float64).train()
    assert model._causal_blocked_mask.device.type == "cpu" and model._causal_blocked_mask.dtype == torch.bool
    assert not model._causal_blocked_mask.is_inference()
    model(cpu_ids).sum().backward()
