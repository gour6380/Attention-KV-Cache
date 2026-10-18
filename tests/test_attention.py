"""Attention correctness, mask reuse and autograd lifecycle checks."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.config import ModelConfig
from src.models import Decoder, causal_mask, grouped_attention


@pytest.mark.parametrize("kv_heads", [8, 2, 1])
@pytest.mark.parametrize("offset,length", [(0, 5), (3, 2), (6, 1)])
@pytest.mark.parametrize("reuse_mask", [False, True])
def test_grouped_attention_matches_expanded_sdpa_outputs_and_gradients(kv_heads, offset, length, reuse_mask):
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(2, 8, length, 8, generator=generator, dtype=torch.float64, requires_grad=True)
    k = torch.randn(
        2, kv_heads, offset + length, 8, generator=generator, dtype=torch.float64, requires_grad=True
    )
    v = torch.randn(
        2, kv_heads, offset + length, 8, generator=generator, dtype=torch.float64, requires_grad=True
    )
    rq, rk, rv = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    blocked = None
    if reuse_mask:
        capacity = offset + length + 3
        blocked = torch.ones(capacity, capacity, dtype=torch.bool).triu(1)[
            offset : offset + length, : offset + length
        ]
    actual = grouped_attention(q, k, v, position_offset=offset, causal_blocked_mask=blocked)
    expected = F.scaled_dot_product_attention(
        rq,
        rk.repeat_interleave(8 // kv_heads, dim=1),
        rv.repeat_interleave(8 // kv_heads, dim=1),
        attn_mask=causal_mask(length, offset + length, offset, "cpu"),
        dropout_p=0.0,
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    probe = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    actual_grads = torch.autograd.grad((actual * probe).sum(), (q, k, v))
    expected_grads = torch.autograd.grad((expected * probe).sum(), (rq, rk, rv))
    for observed, wanted in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(observed, wanted, atol=1e-10, rtol=1e-8)


def test_offset_mask_allows_cached_prefix_without_future_leakage():
    expected = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
    assert torch.equal(causal_mask(2, 5, 3, "cpu"), expected)


def test_common_initialization_tied_weights_and_rng_preservation():
    config = ModelConfig(vocab_size=97, num_layers=2, d_model=64, ffn_dim=96, max_seq_len=64)
    rng_before = torch.get_rng_state().clone()
    models = [Decoder(config.for_variant(variant)) for variant in ("mha", "gqa", "mqa")]
    assert torch.equal(torch.get_rng_state(), rng_before)
    baseline = dict(models[0].named_parameters())
    for model in models:
        assert model.embedding.weight is model.lm_head.weight
        assert all(p.dtype == torch.float32 for p in model.parameters())
        for name, parameter in model.named_parameters():
            if not name.endswith(("k_proj.weight", "v_proj.weight")):
                assert torch.equal(parameter, baseline[name]), name
    distinct_seed = Decoder(replace(config, seed=config.seed + 1))
    assert not torch.equal(distinct_seed.embedding.weight, models[0].embedding.weight)


@pytest.mark.parametrize("variant", ["mha", "gqa", "mqa"])
def test_model_causal_nonleakage_attention_and_state_dict_reload(variant):
    config = ModelConfig(vocab_size=97, num_layers=2, d_model=64, ffn_dim=96, max_seq_len=64).for_variant(
        variant
    )
    model = Decoder(config).eval()
    tokens = torch.randint(0, 97, (2, 11), generator=torch.Generator().manual_seed(7))
    changed = tokens.clone()
    changed[:, 6:] = (changed[:, 6:] + 1) % 97
    restored = Decoder(config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.inference_mode():
        logits, weights = model(tokens, return_attention=True)
        torch.testing.assert_close(model(changed)[:, :6], logits[:, :6], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(model(tokens, last_only=True), logits[:, -1:], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(restored(tokens), logits, atol=0, rtol=0)
    for attention in weights:
        assert attention.shape == (2, 8, 11, 11)
        assert torch.count_nonzero(torch.triu(attention, diagonal=1)).item() == 0
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 8, 11), atol=1e-5, rtol=1e-4)


def test_attention_rejects_incompatible_shapes_and_offsets():
    q = torch.zeros(1, 8, 2, 8)
    k = torch.zeros(1, 2, 5, 8)
    with pytest.raises(ValueError, match="prefix"):
        grouped_attention(q, k, k, position_offset=0)
    with pytest.raises(ValueError, match="Incompatible"):
        grouped_attention(q, torch.zeros(1, 3, 2, 8), torch.zeros(1, 3, 2, 8))


@pytest.mark.parametrize("bad_mask", [torch.zeros(2, 5), torch.zeros(2, 4, dtype=torch.bool)])
def test_attention_rejects_incompatible_reused_masks(bad_mask):
    q, k = torch.zeros(1, 8, 2, 8), torch.zeros(1, 2, 5, 8)
    with pytest.raises(ValueError, match="causal blocked mask"):
        grouped_attention(q, k, k, position_offset=3, causal_blocked_mask=bad_mask)


def test_direct_attention_accepts_inference_created_mask_for_backward():
    with torch.inference_mode():
        blocked = torch.ones(5, 5, dtype=torch.bool).triu(1)
    assert blocked.is_inference()
    generator = torch.Generator().manual_seed(54)
    q = torch.randn(2, 8, 5, 8, generator=generator, requires_grad=True)
    k = torch.randn(2, 2, 5, 8, generator=generator, requires_grad=True)
    v = torch.randn(2, 2, 5, 8, generator=generator, requires_grad=True)
    output = grouped_attention(q, k, v, causal_blocked_mask=blocked)
    reference = grouped_attention(q, k, v)
    torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-4)
    expected_gradients = torch.autograd.grad(reference.square().sum(), (q, k, v))
    actual_gradients = torch.autograd.grad(output.square().sum(), (q, k, v))
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_decoder_reuses_one_mask_across_layers_and_repeated_forwards():
    config = ModelConfig(vocab_size=97, num_layers=3, d_model=64, ffn_dim=96, max_seq_len=64)
    model = Decoder(config).eval()
    observed = []

    def capture_mask(_module, _args, kwargs):
        observed.append(kwargs["causal_blocked_mask"])

    handles = [
        layer.attention.register_forward_pre_hook(capture_mask, with_kwargs=True) for layer in model.layers
    ]
    ids = torch.ones(2, 11, dtype=torch.long)
    try:
        with torch.inference_mode():
            model(ids)
            model(ids[:, :7])
            model(ids)
    finally:
        for handle in handles:
            handle.remove()
    assert len(observed) == 9
    assert model._causal_blocked_mask.shape == (16, 16)
    storage = model._causal_blocked_mask.untyped_storage().data_ptr()
    for start in (0, 3, 6):
        assert all(mask is observed[start] for mask in observed[start : start + 3])
    assert all(mask.untyped_storage().data_ptr() == storage for mask in observed)
    assert all(not mask.is_inference() for mask in observed)
    assert "_causal_blocked_mask" not in model.state_dict()


def test_decoder_mask_capacity_is_bounded_and_offset_views_are_correct():
    config = ModelConfig(vocab_size=97, num_layers=1, d_model=64, ffn_dim=96, max_seq_len=50)
    model = Decoder(config)
    first = model._get_causal_blocked_mask(5, 8, 3, torch.device("cpu"))
    assert model._causal_blocked_mask.shape == (8, 8)
    torch.testing.assert_close(first, ~causal_mask(5, 8, 3, "cpu"))
    old = first.clone()
    old_storage = first.untyped_storage().data_ptr()
    bigger = model._get_causal_blocked_mask(5, 35, 30, torch.device("cpu"))
    assert model._causal_blocked_mask.shape == (50, 50)
    assert model._causal_blocked_mask.numel() <= config.max_seq_len**2
    assert bigger.untyped_storage().data_ptr() != old_storage
    torch.testing.assert_close(first, old)
    torch.testing.assert_close(bigger, ~causal_mask(5, 35, 30, "cpu"))
    smaller = model._get_causal_blocked_mask(2, 3, 1, torch.device("cpu"))
    assert smaller.untyped_storage().data_ptr() == bigger.untyped_storage().data_ptr()
    torch.testing.assert_close(smaller, ~causal_mask(2, 3, 1, "cpu"))
    with pytest.raises(ValueError, match="Invalid causal-mask"):
        model._get_causal_blocked_mask(2, 4, 1, torch.device("cpu"))


@pytest.mark.parametrize("variant", ["mha", "gqa", "mqa"])
def test_shared_mask_preserves_decoder_outputs_and_all_parameter_gradients(variant):
    config = ModelConfig(vocab_size=97, num_layers=2, d_model=64, ffn_dim=96, max_seq_len=64).for_variant(
        variant
    )
    optimized = Decoder(config)
    reference = Decoder(config)
    # Exercise the original per-attention causal-mask path with identical weights.
    reference._get_causal_blocked_mask = lambda *args: None
    ids = torch.randint(0, 97, (2, 13), generator=torch.Generator().manual_seed(10))
    actual, expected = optimized(ids), reference(ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    targets = torch.randint(0, 97, (2, 13), generator=torch.Generator().manual_seed(14))
    F.cross_entropy(actual.flatten(0, 1), targets.flatten()).backward()
    F.cross_entropy(expected.flatten(0, 1), targets.flatten()).backward()
    for (name, actual_parameter), (expected_name, expected_parameter) in zip(
        optimized.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == expected_name
        assert actual_parameter.grad is not None and expected_parameter.grad is not None
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, atol=1e-5, rtol=1e-4)


def test_inference_first_mask_is_reusable_in_training_and_after_dtype_conversion():
    config = ModelConfig(vocab_size=97, num_layers=2, d_model=64, ffn_dim=96, max_seq_len=64)
    model = Decoder(config).eval()
    ids = torch.ones(2, 13, dtype=torch.long)
    with torch.inference_mode():
        model(ids)
    buffer = model._causal_blocked_mask
    assert not buffer.is_inference() and buffer.dtype == torch.bool
    model.train()
    model(ids).square().mean().backward()
    assert model._causal_blocked_mask is buffer
    assert all(parameter.grad is not None for parameter in model.parameters())
    model.zero_grad(set_to_none=True)
    model.to(dtype=torch.float64)
    assert model._causal_blocked_mask is buffer and buffer.dtype == torch.bool
    output = model(ids)
    assert output.dtype == torch.float64
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_model_repairs_an_inference_buffer_before_training():
    model = Decoder(ModelConfig(vocab_size=97, num_layers=1, d_model=64, ffn_dim=96, max_seq_len=64))
    # Simulate an external buffer/device transformation performed in inference.
    with torch.inference_mode():
        model._causal_blocked_mask = torch.ones(16, 16, dtype=torch.bool).triu(1)
    assert model._causal_blocked_mask.is_inference()
    model(torch.ones(2, 7, dtype=torch.long)).sum().backward()
    assert not model._causal_blocked_mask.is_inference()
    assert model._causal_blocked_mask.dtype == torch.bool
