"""One shared byte-level BPE tokenizer, fitted exclusively on training stories."""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm.auto import tqdm

from .data import cache_binding, cache_builder, iter_stories, require_data_binding, seal_cache, verify_cache
from .io import atomic_json, canonical_hash, sha256_file

if TYPE_CHECKING:
    from .runs import RunContext as Run

SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>")
TOKENIZER_PROTOCOL = "byte-bpe-no-prefix-preserve-case-full-alphabet-v1"


def load_tokenizer(path: Path) -> Tokenizer:
    """Load portable JSON, treating literal marker strings in prose as ordinary text.

    Story delimiters are inserted by ID in data.py. Turning on this Tokenizers
    setting prevents a literal '<|eos|>' within a story from injecting EOS IDs.
    """
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.encode_special_tokens = True
    return tokenizer


def _copy_verified(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"Refusing to replace a different run tokenizer: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tokenizer-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train_tokenizer(run: Run, splits: Path) -> Path:
    """Fit/reuse an immutable tokenizer and copy its complete JSON into the run."""
    run.require_identity()
    splits = Path(splits)
    with run.stage("train_tokenizer"):
        split_manifest = verify_cache(splits)
        require_data_binding(run, "splits", cache_binding(splits, split_manifest))
        train_path = splits / split_manifest["splits"]["train"]["file"]
        identity = {
            "protocol": TOKENIZER_PROTOCOL,
            "training_jsonl_sha256": sha256_file(train_path),
            "training_stories": split_manifest["splits"]["train"]["stories"],
            "vocab_size": run.config.data.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "tokenizers_version": importlib.metadata.version("tokenizers"),
            "min_frequency": 2,
            "initial_alphabet": "all 256 bytes",
            "literal_special_markers": "encode as ordinary text; add boundaries explicitly by ID",
        }
        cache = run.project_root / "data" / "tokenizers" / canonical_hash(identity)
        if not cache.exists():
            with cache_builder(cache) as temporary:
                tokenizer = Tokenizer(models.BPE())
                tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
                tokenizer.decoder = decoders.ByteLevel()
                trainer = trainers.BpeTrainer(
                    vocab_size=run.config.data.vocab_size,
                    min_frequency=2,
                    special_tokens=list(SPECIAL_TOKENS),
                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                    show_progress=False,
                )
                records = iter_stories(train_path)
                texts = (
                    record["text"]
                    for record in tqdm(
                        records, total=identity["training_stories"], desc="Train byte-BPE tokenizer"
                    )
                )
                tokenizer.train_from_iterator(texts, trainer=trainer, length=identity["training_stories"])
                if tokenizer.get_vocab_size() != run.config.data.vocab_size:
                    raise ValueError(
                        f"Training data supports {tokenizer.get_vocab_size()} tokens, below configured "
                        f"{run.config.data.vocab_size}; choose a compatible configuration in a new run."
                    )
                tokenizer.encode_special_tokens = True
                tokenizer.save(str(temporary / "tokenizer.json"))
                seal_cache(
                    temporary,
                    {
                        "identity": identity,
                        "vocab_size": tokenizer.get_vocab_size(),
                        "pad_id": tokenizer.token_to_id(SPECIAL_TOKENS[0]),
                        "bos_id": tokenizer.token_to_id(SPECIAL_TOKENS[1]),
                        "eos_id": tokenizer.token_to_id(SPECIAL_TOKENS[2]),
                        "fit_split": "train",
                        "normalization": "No additional normalization; input is already NFC normalized.",
                    },
                )
        manifest = verify_cache(cache, identity)
        destination = run.path / "tokenizer" / "tokenizer.json"
        _copy_verified(cache / "tokenizer.json", destination)
        portable_manifest = run.path / "tokenizer" / "manifest.json"
        if portable_manifest.exists():
            from .io import read_json

            if read_json(portable_manifest) != manifest:
                raise RuntimeError("Run tokenizer manifest changed")
        else:
            atomic_json(portable_manifest, manifest)
        run.bind_data("tokenizer", {"tokenizer_sha256": sha256_file(destination), "manifest": manifest})
        run.logger.info(
            "Tokenizer: %d tokens, fitted only on %d training stories",
            manifest["vocab_size"],
            identity["training_stories"],
        )
        return destination


def inspect_tokenization(tokenizer_path: Path, text: str) -> dict:
    tokenizer = load_tokenizer(tokenizer_path)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return {
        "text": text,
        "ids": encoded.ids,
        "tokens": encoded.tokens,
        "decoded": tokenizer.decode(encoded.ids, skip_special_tokens=False),
        "token_count": len(encoded.ids),
        "special_tokens_added": False,
    }
