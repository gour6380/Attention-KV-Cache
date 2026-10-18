"""Small CPU evaluation checks. Written for later execution by the user."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src import evaluate
from src.io import read_json


class FixedLogits(torch.nn.Module):
    def forward(self, ids):
        logits = torch.zeros((*ids.shape, 2), device=ids.device)
        logits[..., 0] = ids.float()
        return logits


def test_token_weighted_loss_ignores_padding_and_preserves_mode():
    ids = torch.tensor([[0, 0], [4, 4]])
    targets = torch.tensor([[0, 1], [1, -100]])
    model = FixedLogits().train()
    result = evaluate.evaluate_loss(
        model, TensorDataset(ids, targets), 1, torch.device("cpu"), progress=False
    )
    expected_sum = 2 * np.log(2) + np.logaddexp(0, 4)
    assert result["predicted_tokens"] == 3
    assert result["padding_targets"] == 1
    assert result["windows"] == 2
    assert result["nll"] == pytest.approx(expected_sum / 3, abs=1e-6)
    assert result["perplexity"] == pytest.approx(np.exp(expected_sum / 3), rel=1e-6)
    assert model.training


def test_loss_is_independent_of_evaluation_batch_size():
    ids = torch.tensor([[0, 1], [2, 3], [4, 5]])
    labels = torch.tensor([[0, 1], [1, 0], [0, -100]])
    dataset = TensorDataset(ids, labels)
    one = evaluate.evaluate_loss(FixedLogits(), dataset, 1, "cpu", progress=False)
    three = evaluate.evaluate_loss(FixedLogits(), dataset, 3, "cpu", progress=False)
    assert one["nll"] == pytest.approx(three["nll"], abs=1e-6)
    assert one["predicted_tokens"] == three["predicted_tokens"] == 5


def test_mode_is_restored_if_forward_fails():
    class FailingModel(FixedLogits):
        def forward(self, ids):
            raise RuntimeError("deliberate")

    model = FailingModel().train()
    dataset = TensorDataset(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long))
    with pytest.raises(RuntimeError, match="deliberate"):
        evaluate.evaluate_loss(model, dataset, 1, "cpu", progress=False)
    assert model.training


def test_empty_loss_and_nonfinite_loss_rejected():
    with pytest.raises(ValueError, match="no predicted"):
        evaluate.loss_summary(0, 0)
    with pytest.raises(FloatingPointError):
        evaluate.loss_summary(float("nan"), 1)
    assert evaluate.loss_summary(1000, 1)["perplexity"] is None
    assert evaluate.loss_summary(1000, 1)["perplexity_overflow"]


def test_unigram_uses_training_targets_and_laplace_smoothing(tmp_path, monkeypatch):
    training = TensorDataset(torch.zeros(2, 2, dtype=torch.long), torch.tensor([[0, 0], [1, -100]]))
    validation = TensorDataset(torch.zeros(1, 2, dtype=torch.long), torch.tensor([[2, -100]]))
    requested = []

    def dataset(split):
        requested.append(split)
        return {"train": training, "validation": validation}[split]

    tokens = SimpleNamespace(dataset=dataset)
    run = SimpleNamespace(
        path=tmp_path,
        config=SimpleNamespace(model=SimpleNamespace(vocab_size=3)),
        stage=lambda name: nullcontext(),
    )
    monkeypatch.setattr(evaluate, "_verify_tokens", lambda *_: "tokens")
    path = evaluate.fit_unigram(run, tokens)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["counts"], [2, 1, 0])
        np.testing.assert_allclose(np.exp(saved["log_probs"]), [3 / 6, 2 / 6, 1 / 6])
    result = evaluate.evaluate_unigram(run, tokens, path, "validation")
    assert requested == ["train", "validation"]
    assert result["predicted_tokens"] == 1
    assert result["perplexity"] == pytest.approx(6)
    assert read_json(path.with_suffix(".json"))["fit_split"] == "train"


def test_unigram_test_split_requires_frozen_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "_verify_tokens", lambda *_: "tokens")

    def not_frozen(*args):
        raise RuntimeError("not frozen")

    monkeypatch.setattr(evaluate, "require_primary_selection", not_frozen)
    with pytest.raises(RuntimeError, match="not frozen"):
        evaluate.evaluate_unigram(SimpleNamespace(), SimpleNamespace(), tmp_path / "missing.npz", "test")


def test_evaluation_never_loads_model_without_parity(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "_verify_tokens", lambda *_: "tokens")
    monkeypatch.setattr(evaluate, "require_primary_selection", lambda *_: {"sha256": "checkpoint"})
    monkeypatch.setattr(evaluate, "select_device", lambda: torch.device("cpu"))

    def fail_parity(*args):
        raise RuntimeError("parity receipt missing")

    def fail_load(*args, **kwargs):
        pytest.fail("Model loaded before passing the correctness gate")

    monkeypatch.setattr(evaluate, "assert_correctness", fail_parity)
    monkeypatch.setattr(evaluate, "load_model", fail_load)
    with pytest.raises(RuntimeError, match="parity"):
        evaluate.evaluate_variant(SimpleNamespace(), SimpleNamespace(), "mha")
