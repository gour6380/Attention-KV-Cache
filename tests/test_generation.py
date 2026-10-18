"""Sampling, context limits, and checkpoint-only generation contracts."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from src import generate
from src.io import atomic_json, sha256_file


def test_nucleus_keeps_the_first_crossing_token():
    # p is below the leading token's probability, so only that token survives.
    generator = torch.Generator().manual_seed(7)
    logits = torch.tensor([3.0, 1.0, 0.0])
    assert generate.sample_top_p(logits, 1.0, 0.01, generator) == 0


def test_sampling_replays_with_explicit_cpu_seed():
    logits = torch.tensor([1.0, 2.0, 2.5, -1.0])

    def sequence():
        generator = torch.Generator(device="cpu").manual_seed(2027)
        return [generate.sample_top_p(logits, 0.8, 0.9, generator) for _ in range(20)]

    assert sequence() == sequence()


@pytest.mark.parametrize("temperature,top_p", [(0, 0.9), (float("nan"), 0.9), (1, 0), (1, 1.1)])
def test_invalid_sampling_settings_rejected(temperature, top_p):
    with pytest.raises(ValueError):
        generate.sample_top_p(torch.zeros(3), temperature, top_p, torch.Generator())


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([float("nan"), 0.0]),
        torch.tensor([float("inf"), 0.0]),
        torch.tensor([float("-inf"), float("-inf")]),
    ],
)
def test_nonfinite_sampling_distribution_rejected(logits):
    with pytest.raises(FloatingPointError):
        generate.sample_top_p(logits, 1.0, 0.9, torch.Generator())


class FakeTokenizer:
    def token_to_id(self, token):
        return {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2}[token]

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[3, 4])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(value) for value in ids if value > 2)


@pytest.fixture
def generation_setup(tmp_path, monkeypatch):
    tokenizer = tmp_path / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir()
    tokenizer.write_text("{}")
    atomic_json(tmp_path / "data_manifest" / "tokenizer.json", {"tokenizer_sha256": sha256_file(tokenizer)})
    run = SimpleNamespace(
        path=tmp_path,
        config=SimpleNamespace(data=SimpleNamespace(context_length=16), atol=1e-5, rtol=1e-4),
        require_identity=lambda: None,
        stage=lambda _: nullcontext(),
    )
    monkeypatch.setattr(generate, "require_primary_selection", lambda *_: {"sha256": "final"})
    monkeypatch.setattr(generate, "load_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(generate, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(generate, "assert_correctness", lambda *_: None)
    monkeypatch.setattr(generate, "parity_check", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(generate, "release_memory", lambda *_: None)
    return run


def test_generation_rejects_context_overflow_before_loading_model(generation_setup, monkeypatch):
    def no_loading(*args, **kwargs):
        pytest.fail("Overlong generation should fail before loading the model")

    monkeypatch.setattr(generate, "load_model", no_loading)
    with pytest.raises(ValueError, match="trained context"):
        generate.generate_variant(generation_setup, "mha", "prompt", max_new_tokens=14)


def test_eos_stops_generation_and_final_checkpoint_is_used(generation_setup, monkeypatch):
    calls = []

    class Model:
        def eval(self):
            return self

        def make_cache(self, batch_size, capacity):
            return object()

        def __call__(self, ids, **kwargs):
            calls.append(ids.shape[1])
            return torch.zeros(1, 1, 8)

    def load(run, variant, kind, device):
        assert kind == "final"
        return Model()

    monkeypatch.setattr(generate, "load_model", load)
    monkeypatch.setattr(generate, "sample_top_p", lambda *_: 2)
    result = generate.generate_variant(generation_setup, "gqa", "prompt", max_new_tokens=3)
    assert calls == [3]
    assert result["stop_reason"] == "eos"
    assert result["generated_tokens"] == 1
    assert result["checkpoint_kind"] == "final"
    assert result["sampling_device"] == "cpu"
    assert result["suppressed_token_ids"] == [0, 1]
    assert "path" in result
