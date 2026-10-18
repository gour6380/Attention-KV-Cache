"""Small, local tokenizer/preparation tests; no downloads or model execution."""

import logging
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import DataConfig
from src.data import build_splits, cache_binding, iter_stories, prepare_tokens
from src.io import atomic_json, read_json, sha256_file
from src.tokenizer import SPECIAL_TOKENS, inspect_tokenization, load_tokenizer, train_tokenizer


class OfflineRun:
    """Minimal run facade: the public data functions also work with real Run."""

    def __init__(self, root):
        self.project_root = root
        self.path = root / "runs" / "run_001"
        self.config = SimpleNamespace(
            data=replace(DataConfig(), train_stories=3, vocab_size=259, context_length=4)
        )
        self.logger = logging.getLogger("test-tokenizer")
        self.bindings = {}

    def require_identity(self):
        pass

    def stage(self, name):
        return nullcontext()

    def bind_data(self, name, payload):
        if name in self.bindings:
            assert self.bindings[name] == payload
        self.bindings[name] = payload
        atomic_json(self.path / "data_manifest" / f"{name}.json", payload)


def prepare_fixture(tmp_path):
    splits = tmp_path / "splits"
    build_splits(
        [
            {"text": "Café in a tree."},
            {"text": "The cat said <|eos|> loudly."},
            {"text": "Small stories are fun!"},
        ],
        [{"text": "Secret heldout sentence A"}, {"text": "Secret heldout sentence B"}],
        splits,
        train_stories=3,
        seed=2027,
        identity={"test": "tokenizer"},
        progress=False,
    )
    run = OfflineRun(tmp_path)
    run.bind_data("splits", cache_binding(splits, read_json(splits / "manifest.json")))
    return run, splits


def test_byte_alphabet_roundtrip_and_special_ids(tmp_path):
    run, splits = prepare_fixture(tmp_path)
    tokenizer_path = train_tokenizer(run, splits)
    tokenizer = load_tokenizer(tokenizer_path)
    assert tokenizer.get_vocab_size() == 259
    assert [tokenizer.token_to_id(name) for name in SPECIAL_TOKENS] == [0, 1, 2]
    text = "Unseen 中文 🐈 Case!\nNext line."
    encoded = tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.decode(encoded.ids, skip_special_tokens=False) == text
    assert not set(encoded.ids) & {0, 1, 2}
    assert inspect_tokenization(tokenizer_path, text)["decoded"] == text


def test_literal_eos_is_ordinary_text(tmp_path):
    run, splits = prepare_fixture(tmp_path)
    tokenizer = load_tokenizer(train_tokenizer(run, splits))
    encoded = tokenizer.encode("A literal <|eos|> marker", add_special_tokens=False)
    assert tokenizer.token_to_id("<|eos|>") not in encoded.ids
    assert tokenizer.decode(encoded.ids) == "A literal <|eos|> marker"


def test_fitted_identity_is_training_only_and_cache_reuse_is_exact(tmp_path):
    run, splits = prepare_fixture(tmp_path)
    first = train_tokenizer(run, splits)
    digest = sha256_file(first)
    second = train_tokenizer(run, splits)
    assert first == second and sha256_file(second) == digest
    manifest = read_json(first.parent / "manifest.json")
    assert manifest["fit_split"] == "train"
    assert manifest["identity"]["training_jsonl_sha256"] == sha256_file(splits / "train.jsonl")
    assert manifest["identity"]["training_stories"] == 3


def test_tampered_portable_tokenizer_is_not_overwritten(tmp_path):
    run, splits = prepare_fixture(tmp_path)
    path = train_tokenizer(run, splits)
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="different run tokenizer"):
        train_tokenizer(run, splits)


def test_prepared_tokens_have_exactly_one_boundary_pair_per_story(tmp_path):
    run, splits = prepare_fixture(tmp_path)
    tokenizer_path = train_tokenizer(run, splits)
    tokens = prepare_tokens(run, splits, tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    for split in ("train", "validation", "test"):
        rows = list(iter_stories(splits / f"{split}.jsonl"))
        expected = [
            value
            for row in rows
            for value in [1, *tokenizer.encode(row["text"], add_special_tokens=False).ids, 2]
        ]
        actual = tokens.array(split)
        assert actual.tolist() == expected
        assert int(np.count_nonzero(actual == 1)) == len(rows)
        assert int(np.count_nonzero(actual == 2)) == len(rows)
        entry = tokens.manifest["splits"][split]
        valid = sum(
            int((tokens.dataset(split)[index][1] != -100).sum())
            for index in range(len(tokens.dataset(split)))
        )
        assert entry["predicted_tokens"] == valid
        assert entry["discarded_target_tokens"] == len(expected) - 1 - valid
    tokens.verify()
    repeated = prepare_tokens(run, splits, tokenizer_path)
    assert repeated.manifest == tokens.manifest


def test_token_preparation_rejects_other_tokenizer_even_when_json_is_valid(tmp_path):
    run, splits = prepare_fixture(tmp_path)
    tokenizer_path = train_tokenizer(run, splits)
    replacement = tmp_path / "other_tokenizer.json"
    replacement.write_text(tokenizer_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="portable tokenizer"):
        prepare_tokens(run, splits, replacement)


def test_tokenizer_training_rejects_unbound_split_cache(tmp_path):
    run, splits = prepare_fixture(tmp_path)
    (run.path / "data_manifest" / "splits.json").unlink()
    with pytest.raises(RuntimeError, match="immutable artifacts bound"):
        train_tokenizer(run, splits)
