"""Leakage-safe, disk-backed data preparation, invoked explicitly by the notebook.

No download, corpus preparation, or tokenization occurs on module import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .io import atomic_json, canonical_hash, read_json, sha256_file

if TYPE_CHECKING:
    from .runs import RunContext as Run

SPLITS = ("train", "validation", "test")
DATA_PROTOCOL = "tinystories-nfc-exact-dedup-v1"
TOKEN_PROTOCOL = "byte-bpe-bos-eos-contiguous-u32le-v1"


def normalize_story(text: str) -> str:
    """The exact normalization used before hashing, splitting, and tokenization."""
    if not isinstance(text, str):
        raise TypeError("Each source story must be a string")
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def story_hash(text: str) -> str:
    return hashlib.sha256(normalize_story(text).encode("utf-8")).hexdigest()


def iter_stories(path: Path) -> Iterator[dict[str, Any]]:
    """Read prepared JSONL incrementally, retaining source IDs and content hashes."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _file_inventory(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "manifest.json"
    }


def verify_cache(directory: Path, expected_identity: dict | None = None) -> dict:
    """Fail on incomplete, edited, or mismatched immutable caches; never repair silently."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Incomplete cache: {directory}. Remove it explicitly before rebuilding.")
    manifest = read_json(manifest_path)
    if expected_identity is not None and manifest["identity"] != expected_identity:
        raise RuntimeError(f"Cache identity mismatch: {directory}")
    actual = _file_inventory(directory)
    if actual != manifest["files"]:
        raise RuntimeError(f"Cache content checksum mismatch: {directory}")
    return manifest


@contextmanager
def cache_builder(destination: Path):
    """Build privately and publish a complete cache by directory rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        yield temporary
        if destination.exists():
            raise FileExistsError(f"Cache was created concurrently: {destination}; rerun to verify/reuse it")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def seal_cache(directory: Path, manifest: dict) -> None:
    atomic_json(directory / "manifest.json", {**manifest, "files": _file_inventory(directory)})


def require_data_binding(run: Run, name: str, payload: dict) -> None:
    """Reject individually valid artifacts that belong to a different run pipeline."""
    binding = run.path / "data_manifest" / f"{name}.json"
    if not binding.is_file() or read_json(binding) != payload:
        raise RuntimeError(f"Provided {name} artifacts are not the immutable artifacts bound to this run")


def cache_binding(directory: Path, manifest: dict) -> dict:
    return {
        "directory": str(directory),
        "manifest_sha256": sha256_file(directory / "manifest.json"),
        "manifest": manifest,
    }


def _configure_hf_cache(project_root: Path) -> dict[str, Path]:
    """All dataset and Hub files belong to this project's data cache."""
    root = project_root / "data" / "huggingface"
    paths = {
        "home": root,
        "hub": root / "hub",
        "datasets": root / "datasets",
        "downloads": root / "downloads",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(paths["home"])
    os.environ["HF_HUB_CACHE"] = str(paths["hub"])
    os.environ["HF_DATASETS_CACHE"] = str(paths["datasets"])
    # Hugging Face reads these defaults at import. Explicit cache_dir arguments
    # cover datasets; synchronize Hub constants too if a notebook imported it first.
    from huggingface_hub import constants as hub_constants

    hub_constants.HF_HOME = str(paths["home"])
    hub_constants.HF_HUB_CACHE = str(paths["hub"])
    return paths


def download_source(run: Run) -> Path:
    """Resolve the configured dataset revision once, then save an audited Arrow snapshot.

    Returns a run-local receipt. Rerunning a completed cell verifies cached bytes
    without contacting the Hub; an interrupted acquisition keeps its pinned SHA.
    """
    run.require_identity()
    cfg = run.config.data
    receipt_path = run.path / "data_manifest" / "source_receipt.json"
    with run.stage("download_source"):
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["dataset"] != cfg.dataset or receipt["requested_revision"] != cfg.revision:
                raise RuntimeError("Source receipt disagrees with run configuration")
            if (run.path / "data_manifest" / "source.json").exists():
                require_data_binding(run, "source", receipt)
        else:
            _configure_hf_cache(run.project_root)
            from huggingface_hub import HfApi

            revision = HfApi().dataset_info(cfg.dataset, revision=cfg.revision).sha
            if not revision:
                raise RuntimeError("The Hub did not return an immutable dataset revision")
            receipt = {
                "dataset": cfg.dataset,
                "requested_revision": cfg.revision,
                "resolved_revision": revision,
            }
            # Freeze before downloading: interruption cannot advance a mutable branch.
            atomic_json(receipt_path, receipt)
        identity = {
            "dataset": cfg.dataset,
            "resolved_revision": receipt["resolved_revision"],
            "format": "datasets-arrow-v1",
        }
        destination = run.project_root / "data" / "sources" / canonical_hash(identity)
        if not destination.exists():
            paths = _configure_hf_cache(run.project_root)
            from datasets import DownloadConfig, load_dataset

            source = load_dataset(
                cfg.dataset,
                revision=receipt["resolved_revision"],
                cache_dir=str(paths["datasets"]),
                download_config=DownloadConfig(cache_dir=str(paths["downloads"])),
            )
            if "train" not in source or "validation" not in source:
                raise ValueError("Expected original TinyStories train and validation splits")
            with cache_builder(destination) as temporary:
                source.save_to_disk(str(temporary / "dataset"))
                seal_cache(
                    temporary, {"identity": identity, "counts": {name: len(source[name]) for name in source}}
                )
            del source
        manifest = verify_cache(destination, identity)
        completed = {
            **receipt,
            "directory": str(destination),
            "manifest_sha256": sha256_file(destination / "manifest.json"),
        }
        if "manifest_sha256" in receipt and receipt != completed:
            raise RuntimeError("Source manifest changed after acquisition")
        atomic_json(receipt_path, completed)
        run.bind_data("source", completed)
        run.logger.info(
            "Pinned source %s at %s; counts=%s",
            cfg.dataset,
            completed["resolved_revision"],
            manifest["counts"],
        )
        return receipt_path


def _write_record(stream, content_hash: str, text: str, index: int, source_split: str) -> None:
    stream.write(
        json.dumps(
            {
                "source_split": source_split,
                "source_index": index,
                "content_sha256": content_hash,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )


def build_splits(
    train_rows: Iterable[dict],
    validation_rows: Iterable[dict],
    directory: Path,
    *,
    train_stories: int,
    seed: int,
    identity: dict,
    progress: bool = True,
) -> dict:
    """Build canonical JSONL splits with SQLite exact deduplication and on-disk sorting.

    This lower-level function supports small offline test fixtures. Its caller
    owns the private build directory and publishes it only after success.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "dedup.sqlite"
    counts = {
        "source_train": 0,
        "source_validation": 0,
        "empty_train": 0,
        "empty_validation": 0,
        "duplicate_train": 0,
        "duplicate_validation": 0,
        "train_heldout_overlap": 0,
    }
    connection = sqlite3.connect(database)
    exclusions = (directory / "exclusions.jsonl").open("w", encoding="utf-8")

    def exclude(source_split: str, source_index: int, reason: str, content_hash: str | None) -> None:
        exclusions.write(
            json.dumps(
                {
                    "source_split": source_split,
                    "source_index": source_index,
                    "reason": reason,
                    "content_sha256": content_hash,
                },
                sort_keys=True,
            )
            + "\n"
        )

    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-32768")  # 32 MiB page cache, not corpus-sized RAM.
        connection.execute(
            "CREATE TABLE heldout (hash TEXT PRIMARY KEY, text TEXT NOT NULL, source_index INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE training (hash TEXT PRIMARY KEY, rank TEXT NOT NULL, text TEXT NOT NULL, source_index INTEGER NOT NULL)"
        )
        for index, row in enumerate(
            tqdm(validation_rows, desc="Deduplicate held-out stories", disable=not progress)
        ):
            counts["source_validation"] += 1
            text = normalize_story(row["text"])
            if not text:
                counts["empty_validation"] += 1
                exclude("validation", index, "empty_after_normalization", None)
                continue
            content_hash = story_hash(text)
            inserted = connection.execute(
                "INSERT OR IGNORE INTO heldout VALUES (?, ?, ?)", (content_hash, text, index)
            ).rowcount
            counts["duplicate_validation"] += int(not inserted)
            if not inserted:
                exclude("validation", index, "duplicate_normalized_content", content_hash)
        connection.commit()
        heldout_count = connection.execute("SELECT COUNT(*) FROM heldout").fetchone()[0]
        if heldout_count < 2:
            raise ValueError("At least two unique held-out stories are required")
        midpoint = heldout_count // 2
        with (
            (directory / "validation.jsonl").open("w", encoding="utf-8") as validation_stream,
            (directory / "test.jsonl").open("w", encoding="utf-8") as test_stream,
        ):
            for order, (content_hash, text, index) in enumerate(
                connection.execute("SELECT hash, text, source_index FROM heldout ORDER BY hash")
            ):
                _write_record(
                    validation_stream if order < midpoint else test_stream,
                    content_hash,
                    text,
                    index,
                    "validation",
                )
        for index, row in enumerate(
            tqdm(train_rows, desc="Deduplicate training stories", disable=not progress)
        ):
            counts["source_train"] += 1
            text = normalize_story(row["text"])
            if not text:
                counts["empty_train"] += 1
                exclude("train", index, "empty_after_normalization", None)
                continue
            content_hash = story_hash(text)
            if connection.execute("SELECT 1 FROM heldout WHERE hash=?", (content_hash,)).fetchone():
                counts["train_heldout_overlap"] += 1
                exclude("train", index, "heldout_overlap", content_hash)
                continue
            rank = hashlib.sha256(f"{seed}:{content_hash}".encode("ascii")).hexdigest()
            inserted = connection.execute(
                "INSERT OR IGNORE INTO training VALUES (?, ?, ?, ?)", (content_hash, rank, text, index)
            ).rowcount
            counts["duplicate_train"] += int(not inserted)
            if not inserted:
                exclude("train", index, "duplicate_normalized_content", content_hash)
            if index % 10000 == 0:
                connection.commit()
        connection.commit()
        eligible = connection.execute("SELECT COUNT(*) FROM training").fetchone()[0]
        if eligible < train_stories:
            raise ValueError(
                f"Only {eligible:,} unique non-overlapping training stories; requested {train_stories:,}"
            )
        connection.execute("CREATE INDEX training_rank ON training(rank, hash)")
        with (directory / "train.jsonl").open("w", encoding="utf-8") as stream:
            for content_hash, text, index in connection.execute(
                "SELECT hash, text, source_index FROM training ORDER BY rank, hash LIMIT ?", (train_stories,)
            ):
                _write_record(stream, content_hash, text, index, "train")
        for content_hash, index in connection.execute(
            "SELECT hash, source_index FROM training ORDER BY rank, hash LIMIT -1 OFFSET ?", (train_stories,)
        ):
            exclude("train", index, "outside_seeded_subset", content_hash)
        counts.update(
            {
                "eligible_unique_train": eligible,
                "unselected_unique_train": eligible - train_stories,
                "selected_train": train_stories,
                "validation": midpoint,
                "test": heldout_count - midpoint,
            }
        )
    finally:
        exclusions.close()
        connection.close()
        database.unlink(missing_ok=True)
    manifest = {
        "identity": identity,
        "counts": counts,
        "normalization": "Unicode NFC; CRLF/CR to LF; strip leading/trailing whitespace",
        "selection": "lowest SHA256(seed:normalized-content-sha256), ties broken by content hash",
        "heldout_partition": "sort unique hashes; first floor(N/2) validation, remainder test",
        "test_label": "derived from the official validation split",
        "limitations": "Exact deduplication does not detect near-duplicate text.",
        "exclusions": {
            "file": "exclusions.jsonl",
            "entries": sum(
                counts[key]
                for key in (
                    "empty_train",
                    "empty_validation",
                    "duplicate_train",
                    "duplicate_validation",
                    "train_heldout_overlap",
                    "unselected_unique_train",
                )
            ),
        },
        "splits": {
            name: {"file": f"{name}.jsonl", "stories": counts[{"train": "selected_train"}.get(name, name)]}
            for name in SPLITS
        },
    }
    seal_cache(directory, manifest)
    return read_json(directory / "manifest.json")


def prepare_splits(run: Run, source: Path) -> Path:
    run.require_identity()
    with run.stage("prepare_splits"):
        receipt = read_json(source)
        require_data_binding(run, "source", receipt)
        cfg = run.config.data
        if receipt["dataset"] != cfg.dataset or receipt["requested_revision"] != cfg.revision:
            raise RuntimeError("Provided source differs from configured dataset/revision")
        source_directory = Path(receipt["directory"])
        verify_cache(source_directory)
        if sha256_file(source_directory / "manifest.json") != receipt["manifest_sha256"]:
            raise RuntimeError("Source manifest no longer matches its run receipt")
        identity = {
            "protocol": DATA_PROTOCOL,
            "source_manifest_sha256": receipt["manifest_sha256"],
            "train_stories": cfg.train_stories,
            "seed": cfg.split_seed,
        }
        destination = run.project_root / "data" / "splits" / canonical_hash(identity)
        if not destination.exists():
            from datasets import load_from_disk

            source_data = load_from_disk(str(source_directory / "dataset"))
            with cache_builder(destination) as temporary:
                build_splits(
                    source_data["train"],
                    source_data["validation"],
                    temporary,
                    train_stories=cfg.train_stories,
                    seed=cfg.split_seed,
                    identity=identity,
                )
            del source_data
        manifest = verify_cache(destination, identity)
        run.bind_data("splits", cache_binding(destination, manifest))
        run.logger.info("Prepared split counts: %s", manifest["counts"])
        return destination


def inspect_split_audit(splits: Path) -> dict:
    return verify_cache(Path(splits))


class TokenWindowDataset(Dataset):
    """Non-overlapping *targets* with one overlapping input token between windows."""

    def __init__(self, path: Path, token_count: int, context_length: int, pad_id: int, drop_last: bool):
        if token_count < 2 or context_length < 1:
            raise ValueError("Need at least two tokens and a positive context length")
        self.path = Path(path)
        self.token_count = token_count
        self.context_length = context_length
        self.pad_id = pad_id
        self.drop_last = drop_last
        self._tokens = None
        if self.path.stat().st_size != token_count * np.dtype("<u4").itemsize:
            raise RuntimeError(f"Token file size does not match manifest: {self.path}")

    def __len__(self) -> int:
        targets = self.token_count - 1
        return targets // self.context_length if self.drop_last else math.ceil(targets / self.context_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._tokens is None:
            self._tokens = np.memmap(self.path, mode="r", dtype="<u4", shape=(self.token_count,))
        start = index * self.context_length
        count = min(self.context_length, self.token_count - 1 - start)
        inputs = np.full(self.context_length, self.pad_id, dtype=np.int64)
        targets = np.full(self.context_length, -100, dtype=np.int64)
        inputs[:count] = self._tokens[start : start + count]
        targets[:count] = self._tokens[start + 1 : start + count + 1]
        return torch.from_numpy(inputs), torch.from_numpy(targets)

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_tokens": None}


@dataclass(frozen=True)
class TokenData:
    directory: Path
    manifest: dict
    tokenizer_path: Path

    def array(self, split: str) -> np.memmap:
        entry = self.manifest["splits"][split]
        return np.memmap(self.directory / entry["file"], mode="r", dtype="<u4", shape=(entry["token_count"],))

    def dataset(self, split: str) -> TokenWindowDataset:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        entry = self.manifest["splits"][split]
        return TokenWindowDataset(
            self.directory / entry["file"],
            entry["token_count"],
            self.manifest["context_length"],
            self.manifest["pad_id"],
            split == "train",
        )

    def verify(self) -> None:
        disk_manifest = verify_cache(self.directory)
        if (
            disk_manifest != self.manifest
            or sha256_file(self.tokenizer_path) != self.manifest["tokenizer_sha256"]
        ):
            raise RuntimeError("Token data or tokenizer has changed since preparation")


def prepare_tokens(run: Run, splits: Path, tokenizer_path: Path) -> TokenData:
    """Encode each split once; append exactly one BOS/EOS pair per story."""
    from .tokenizer import load_tokenizer

    run.require_identity()
    with run.stage("prepare_tokens"):
        split_manifest = verify_cache(splits)
        require_data_binding(run, "splits", cache_binding(splits, split_manifest))
        tokenizer_path = Path(tokenizer_path)
        portable = run.path / "tokenizer" / "tokenizer.json"
        portable_manifest = read_json(run.path / "tokenizer" / "manifest.json")
        tokenizer_sha256 = sha256_file(tokenizer_path)
        if tokenizer_sha256 != sha256_file(portable):
            raise RuntimeError("Provided tokenizer differs from this run's portable tokenizer")
        require_data_binding(
            run, "tokenizer", {"tokenizer_sha256": tokenizer_sha256, "manifest": portable_manifest}
        )
        tokenizer = load_tokenizer(tokenizer_path)
        if tokenizer.get_vocab_size() != run.config.data.vocab_size:
            raise RuntimeError("Provided tokenizer vocabulary differs from configured model/data vocabulary")
        special_ids = {name: tokenizer.token_to_id(f"<|{name}|>") for name in ("pad", "bos", "eos")}
        if any(value is None for value in special_ids.values()):
            raise ValueError("Tokenizer is missing required special tokens")
        identity = {
            "protocol": TOKEN_PROTOCOL,
            "splits_manifest_sha256": sha256_file(splits / "manifest.json"),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "context_length": run.config.data.context_length,
        }
        destination = run.project_root / "data" / "tokens" / canonical_hash(identity)
        if not destination.exists():
            with cache_builder(destination) as temporary:
                entries = {}
                for split in SPLITS:
                    token_count = 0
                    output = temporary / f"{split}.bin"
                    with output.open("wb") as stream:
                        rows = iter_stories(splits / split_manifest["splits"][split]["file"])
                        for row in tqdm(
                            rows, total=split_manifest["splits"][split]["stories"], desc=f"Encode {split}"
                        ):
                            ids = [
                                special_ids["bos"],
                                *tokenizer.encode(row["text"], add_special_tokens=False).ids,
                                special_ids["eos"],
                            ]
                            np.asarray(ids, dtype="<u4").tofile(stream)
                            token_count += len(ids)
                        stream.flush()
                        os.fsync(stream.fileno())
                    targets = token_count - 1
                    context = run.config.data.context_length
                    windows = targets // context if split == "train" else math.ceil(targets / context)
                    if token_count < 2 or windows < 1:
                        raise ValueError(f"{split} is too short for the configured token-window protocol")
                    predicted = windows * context if split == "train" else targets
                    entries[split] = {
                        "file": output.name,
                        "token_count": token_count,
                        "sha256": sha256_file(output),
                        "stories": split_manifest["splits"][split]["stories"],
                        "windows": windows,
                        "predicted_tokens": predicted,
                        "discarded_target_tokens": targets - predicted,
                        "discarded_tail_tokens": targets - predicted,
                    }
                seal_cache(
                    temporary,
                    {
                        "identity": identity,
                        "context_length": run.config.data.context_length,
                        "dtype": "<u4",
                        "tokenizer_sha256": identity["tokenizer_sha256"],
                        "vocab_size": tokenizer.get_vocab_size(),
                        **{f"{name}_id": value for name, value in special_ids.items()},
                        "splits": entries,
                        "boundary_policy": "Concatenate BOS,text,EOS stories. Attention crosses story boundaries; no cache reset.",
                        "window_policy": "Stride=context; adjacent windows share one input/target boundary token, never a target.",
                        "test_label": split_manifest["test_label"],
                    },
                )
        manifest = verify_cache(destination, identity)
        run.bind_data("tokens", cache_binding(destination, manifest))
        return TokenData(destination, manifest, tokenizer_path)


def inspect_window(tokens: TokenData, index: int = 0) -> dict:
    inputs, targets = tokens.dataset("train")[index]
    return {
        "window_index": index,
        "input_ids": inputs.tolist(),
        "targets": targets.tolist(),
        "valid_targets": int((targets != -100).sum()),
        "context_length": tokens.manifest["context_length"],
    }
