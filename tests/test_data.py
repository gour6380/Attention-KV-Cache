"""Offline data protocol tests. Written for later execution; no Hub access."""

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.config import DataConfig
from src.data import (
    TokenData,
    TokenWindowDataset,
    build_splits,
    download_source,
    iter_stories,
    normalize_story,
    story_hash,
    verify_cache,
)
from src.io import atomic_json, read_json


def fixture_splits(directory, *, seed=2027, reverse=False):
    training = [
        {"text": " train alpha "},
        {"text": "train beta"},
        {"text": "train gamma"},
        {"text": "train delta"},
        {"text": "train alpha"},
        {"text": " \r\n "},
        {"text": "Cafe\u0301"},
    ]
    heldout = [
        {"text": "Café"},
        {"text": "Cafe\u0301"},
        {"text": "held-out b"},
        {"text": "held-out c"},
        {"text": "held-out d"},
        {"text": " "},
    ]
    if reverse:
        training.reverse()
        heldout.reverse()
    identity = {"test": "offline", "seed": seed}
    build_splits(training, heldout, directory, train_stories=3, seed=seed, identity=identity, progress=False)
    return directory


def test_normalization_is_explicit_and_case_preserving():
    assert normalize_story(" \r\nCafe\u0301\rTHE End \n") == "Café\nTHE End"
    assert story_hash("Cafe\u0301") == story_hash("Café")
    assert story_hash("Cat") != story_hash("cat")
    with pytest.raises(TypeError):
        normalize_story(None)


def test_exact_dedup_leakage_and_counts(tmp_path):
    splits = fixture_splits(tmp_path / "splits")
    rows = {name: list(iter_stories(splits / f"{name}.jsonl")) for name in ("train", "validation", "test")}
    hashes = {name: {row["content_sha256"] for row in records} for name, records in rows.items()}
    assert len(rows["train"]) == 3
    assert len(rows["validation"]) == len(rows["test"]) == 2
    assert hashes["train"].isdisjoint(hashes["validation"] | hashes["test"])
    assert hashes["validation"].isdisjoint(hashes["test"])
    assert all(row["source_split"] == "validation" for name in ("validation", "test") for row in rows[name])
    manifest = verify_cache(splits)
    assert manifest["counts"]["empty_train"] == 1
    assert manifest["counts"]["duplicate_train"] == 1
    assert manifest["counts"]["duplicate_validation"] == 1
    assert manifest["counts"]["train_heldout_overlap"] == 1
    assert manifest["counts"]["unselected_unique_train"] == 1
    exclusions = list(iter_stories(splits / "exclusions.jsonl"))
    assert len(exclusions) == manifest["exclusions"]["entries"] == 6
    assert {row["reason"] for row in exclusions} == {
        "empty_after_normalization",
        "duplicate_normalized_content",
        "heldout_overlap",
        "outside_seeded_subset",
    }
    assert manifest["test_label"] == "derived from the official validation split"
    assert not (splits / "dedup.sqlite").exists()


def test_selection_depends_on_content_not_input_order(tmp_path):
    first = fixture_splits(tmp_path / "first")
    second = fixture_splits(tmp_path / "second", reverse=True)
    for name in ("train", "validation", "test"):
        assert [row["content_sha256"] for row in iter_stories(first / f"{name}.jsonl")] == [
            row["content_sha256"] for row in iter_stories(second / f"{name}.jsonl")
        ]
    eligible = {story_hash(text) for text in ("train alpha", "train beta", "train gamma", "train delta")}
    expected = sorted(
        eligible, key=lambda value: (hashlib.sha256(f"2027:{value}".encode()).hexdigest(), value)
    )[:3]
    assert [row["content_sha256"] for row in iter_stories(first / "train.jsonl")] == expected


def test_insufficient_nonoverlapping_training_data_fails(tmp_path):
    with pytest.raises(ValueError, match="unique non-overlapping"):
        build_splits(
            [{"text": "heldout"}],
            [{"text": "heldout"}, {"text": "heldout other"}],
            tmp_path,
            train_stories=1,
            seed=2,
            identity={},
            progress=False,
        )


def test_cache_tamper_is_rejected(tmp_path):
    splits = fixture_splits(tmp_path / "splits")
    with (splits / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_cache(splits)


def test_training_windows_drop_tail_without_duplicate_targets(tmp_path):
    path = tmp_path / "train.bin"
    np.arange(14, dtype="<u4").tofile(path)
    dataset = TokenWindowDataset(path, token_count=14, context_length=4, pad_id=0, drop_last=True)
    assert len(dataset) == 3
    targets = torch.cat([dataset[index][1] for index in range(len(dataset))])
    assert targets.tolist() == list(range(1, 13))
    assert dataset[1][0].tolist() == [4, 5, 6, 7]
    assert dataset[-1][1].tolist() == [9, 10, 11, 12]
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_eval_windows_retain_tail_and_mask_only_padding(tmp_path):
    path = tmp_path / "test.bin"
    np.arange(14, dtype="<u4").tofile(path)
    dataset = TokenWindowDataset(path, token_count=14, context_length=4, pad_id=99, drop_last=False)
    assert len(dataset) == 4
    inputs, targets = dataset[3]
    assert inputs.dtype == targets.dtype == torch.int64
    assert inputs.tolist() == [12, 99, 99, 99]
    assert targets.tolist() == [13, -100, -100, -100]
    target_list = [
        token for index in range(len(dataset)) for token in dataset[index][1].tolist() if token != -100
    ]
    assert target_list == list(range(1, 14))


def test_token_manifest_controls_memmap_and_windows(tmp_path):
    np.arange(9, dtype="<u4").tofile(tmp_path / "test.bin")
    tokens = TokenData(
        tmp_path,
        {"context_length": 4, "pad_id": 0, "splits": {"test": {"file": "test.bin", "token_count": 9}}},
        tmp_path / "tokenizer.json",
    )
    assert isinstance(tokens.array("test"), np.memmap)
    assert tokens.array("test").flags.writeable is False
    assert len(tokens.dataset("test")) == 2
    with pytest.raises(ValueError, match="Unknown split"):
        tokens.dataset("other")


def test_truncated_token_file_is_rejected(tmp_path):
    path = tmp_path / "train.bin"
    path.write_bytes(b"\x00\x00")
    with pytest.raises(RuntimeError, match="size"):
        TokenWindowDataset(path, token_count=10, context_length=4, pad_id=0, drop_last=True)


def source_fixture(tmp_path, monkeypatch, *, interrupt_first=False):
    """Fake the remote boundary; this test never contacts a dataset provider."""
    calls = {"resolve": 0, "download": 0, "revisions": []}
    root = tmp_path / "project"
    run_path = root / "runs" / "run_001"

    def bind(name, payload):
        path = run_path / "data_manifest" / f"{name}.json"
        if path.exists():
            assert read_json(path) == payload
        atomic_json(path, payload)

    run = SimpleNamespace(
        project_root=root,
        path=run_path,
        config=SimpleNamespace(data=DataConfig()),
        require_identity=lambda: None,
        stage=lambda name: nullcontext(),
        bind_data=bind,
        logger=SimpleNamespace(info=lambda *args: None),
    )

    class FakeApi:
        def dataset_info(self, dataset, revision):
            calls["resolve"] += 1
            return SimpleNamespace(sha="a" * 40)

    class FakeSource(dict):
        def save_to_disk(self, path):
            path = Path(path)
            path.mkdir(parents=True)
            (path / "source.json").write_text(json.dumps(self))

    def fake_download(dataset, *, revision, **kwargs):
        calls["download"] += 1
        calls["revisions"].append(revision)
        if interrupt_first and calls["download"] == 1:
            raise InterruptedError("Simulated download interruption")
        return FakeSource(train=[{"text": "training"}], validation=[{"text": "a"}, {"text": "b"}])

    monkeypatch.setattr(
        "src.data._configure_hf_cache",
        lambda root: {"datasets": root / "data" / "datasets", "downloads": root / "data" / "downloads"},
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_download, DownloadConfig=lambda **kwargs: kwargs),
    )
    return run, calls


def test_download_freezes_revision_and_reuses_without_network(tmp_path, monkeypatch):
    run, calls = source_fixture(tmp_path, monkeypatch)
    first = download_source(run)
    receipt = read_json(first)
    assert receipt["resolved_revision"] == "a" * 40
    second = download_source(run)
    assert first == second
    assert calls == {"resolve": 1, "download": 1, "revisions": ["a" * 40]}


def test_interrupted_download_resumes_original_resolved_revision(tmp_path, monkeypatch):
    run, calls = source_fixture(tmp_path, monkeypatch, interrupt_first=True)
    with pytest.raises(InterruptedError):
        download_source(run)
    pinned = read_json(run.path / "data_manifest" / "source_receipt.json")
    assert pinned["resolved_revision"] == "a" * 40
    download_source(run)
    assert calls["resolve"] == 1
    assert calls["revisions"] == ["a" * 40, "a" * 40]
