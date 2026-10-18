"""Immutable, notebook-editable experiment configuration."""

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

Variant = Literal["mha", "gqa", "mqa"]
VARIANTS: tuple[Variant, ...] = ("mha", "gqa", "mqa")


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "roneneldan/TinyStories"
    revision: str = "main"  # Resolved once to a full immutable SHA at download.
    train_stories: int = 250_000
    split_seed: int = 2027
    vocab_size: int = 8192
    context_length: int = 1024


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    num_layers: int = 6
    d_model: int = 384
    num_query_heads: int = 8
    num_kv_heads: int = 8
    ffn_dim: int = 1024
    norm_eps: float = 1e-5
    rope_base: float = 10000.0
    max_seq_len: int = 4096
    seed: int = 2027

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_query_heads

    def for_variant(self, variant: Variant) -> "ModelConfig":
        return replace(self, num_kv_heads={"mha": self.num_query_heads, "gqa": 2, "mqa": 1}[variant])

    def validate(self) -> None:
        for name in (
            "vocab_size",
            "num_layers",
            "d_model",
            "num_query_heads",
            "num_kv_heads",
            "ffn_dim",
            "max_seq_len",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_query_heads or self.num_query_heads % self.num_kv_heads:
            raise ValueError("Width must divide into query heads, which must divide into KV groups")
        if self.head_dim % 2:
            raise ValueError("RoPE requires even head_dim")
        if self.norm_eps <= 0 or self.rope_base <= 1:
            raise ValueError("Invalid normalization/RoPE configuration")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2027
    epochs: int = 3
    microbatch_size: int = 4
    accumulation_steps: int = 2
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    checkpoint_interval: int = 250
    log_interval: int = 10


@dataclass(frozen=True)
class BenchmarkConfig:
    prompt_lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    batch_sizes: tuple[int, ...] = (1, 4)
    decode_steps: int = 128
    warmup_trials: int = 5
    measured_trials: int = 20


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    atol: float = 1e-5
    rtol: float = 1e-4

    def validate(self) -> None:
        for variant in VARIANTS:
            self.model.for_variant(variant).validate()
        if self.data.vocab_size != self.model.vocab_size:
            raise ValueError("Model and tokenizer vocabularies must match")
        if self.model.seed != self.train.seed:
            raise ValueError("Model and training seeds must match")
        if not 1 < self.data.context_length <= self.model.max_seq_len:
            raise ValueError("Invalid training context")
        if self.data.train_stories < 1 or self.data.vocab_size < 259:
            raise ValueError("Need training stories and space for all bytes plus special tokens")
        for name in (
            "epochs",
            "microbatch_size",
            "accumulation_steps",
            "checkpoint_interval",
            "log_interval",
        ):
            if getattr(self.train, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.train.warmup_fraction < 1:
            raise ValueError("Invalid warm-up fraction")
        if not 0 < self.train.min_learning_rate <= self.train.learning_rate:
            raise ValueError("Invalid learning rates")
        if (
            self.train.gradient_clip <= 0
            or self.train.weight_decay < 0
            or any(not 0 <= b < 1 for b in self.train.betas)
        ):
            raise ValueError("Invalid optimizer settings")
        b = self.benchmark
        if (
            not b.prompt_lengths
            or not b.batch_sizes
            or min(*b.prompt_lengths, *b.batch_sizes, b.decode_steps, b.measured_trials) < 1
            or b.warmup_trials < 0
        ):
            raise ValueError("Invalid benchmark dimensions")
        if max(b.prompt_lengths) + b.decode_steps > self.model.max_seq_len:
            raise ValueError("Benchmark exceeds inference capacity")
        if self.atol <= 0 or self.rtol <= 0:
            raise ValueError("Parity tolerances must be positive")

    def to_dict(self) -> dict:
        return asdict(self)
