"""Numbered runs, immutable provenance, stage state, and explicit resume."""

import importlib.metadata
import logging
import os
import platform
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from src.config import VARIANTS, ExperimentConfig
from src.io import atomic_json, canonical_hash, read_json, sha256_file

DIRECTORIES = (
    "config",
    "source_snapshot",
    "environment",
    "data_manifest",
    "tokenizer",
    "logs",
    "tensorboard",
    "models",
    "evaluation",
    "benchmarks",
    "figures",
    "generation",
    "report",
)


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def environment_identity() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "packages": dict(
            sorted(
                (d.metadata.get("Name", "unknown").lower(), d.version)
                for d in importlib.metadata.distributions()
            )
        ),
    }


def source_identity(root: Path) -> dict:
    files = sorted((root / "src").rglob("*.py")) + [root / "requirements.txt"]
    return {str(p.relative_to(root)): sha256_file(p) for p in files if p.is_file()}


@dataclass(frozen=True)
class RunContext:
    project_root: Path
    path: Path
    config: ExperimentConfig
    mode: str = "new"

    @property
    def identity(self) -> dict:
        return read_json(self.path / "config" / "identity.json")

    @property
    def logger(self) -> logging.Logger:
        logger = logging.getLogger(f"attention.{self.path.resolve()}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(self.path / "logs" / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        return logger

    def model_dir(self, variant: str) -> Path:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        return self.path / "models" / variant

    def require_identity(self) -> None:
        expected = self.identity
        observed = {
            "config": canonical_hash(self.config.to_dict()),
            "source": canonical_hash(source_identity(self.project_root)),
            "environment": canonical_hash(environment_identity()),
        }
        if expected != observed:
            different = [key for key in expected if expected[key] != observed.get(key)]
            raise RuntimeError(
                f"Run identity changed ({different}); restore the recorded environment/source/config or create a fresh run"
            )

    def bind_data(self, name: str, payload: dict) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Invalid manifest name")
        path = self.path / "data_manifest" / f"{name}.json"
        if path.exists() and canonical_hash(read_json(path)) != canonical_hash(payload):
            raise RuntimeError(f"Immutable data binding changed: {name}; create a fresh run")
        atomic_json(path, payload)

    @property
    def data_identity(self) -> dict:
        return {p.name: sha256_file(p) for p in sorted((self.path / "data_manifest").glob("*.json"))}

    def read_status(self) -> dict:
        return read_json(self.path / "status.json")

    def mark_stage(self, name: str, status: str, **details: Any) -> None:
        if status not in {"running", "complete", "failed", "interrupted"}:
            raise ValueError("Invalid stage state")
        state = self.read_status()
        state["stages"][name] = {"status": status, "updated_at": timestamp(), **details}
        atomic_json(self.path / "status.json", state)
        self.logger.info("Stage %s %s", name, status)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        lock = self.path / ".stage.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(
                "Another stage/kernel owns this run, or a crash left .stage.lock. Confirm no process is active before removing that stale lock."
            ) from exc
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(f"pid={os.getpid()} stage={name}\n")
            self.mark_stage(name, "running")
            try:
                yield
            except KeyboardInterrupt:
                self.mark_stage(name, "interrupted")
                self.logger.exception("Interrupted %s", name)
                raise
            except Exception as exc:
                self.mark_stage(name, "failed", error=str(exc))
                self.logger.exception("Failed %s", name)
                raise
            else:
                self.mark_stage(name, "complete")
        finally:
            lock.unlink(missing_ok=True)


def start_run(
    project_root: Path, config: ExperimentConfig, mode: str = "new", run_id: str | None = None
) -> RunContext:
    root = Path(project_root).resolve()
    config.validate()
    if not (root / "src" / "config.py").is_file():
        raise ValueError("project_root must point to attention-kv-cache")
    if mode not in {"new", "resume"}:
        raise ValueError("mode must be new or resume")
    if mode == "resume":
        if run_id is None or not re.fullmatch(r"run_\d+", run_id):
            raise ValueError("Resume requires an explicit run_XXX identifier")
        run = RunContext(root, root / "runs" / run_id, config, mode)
        run.require_identity()
        return run
    if run_id is not None:
        raise ValueError("A fresh run allocates its own number; set run_id=None")
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    numbers = [int(m.group(1)) for p in runs.iterdir() if (m := re.fullmatch(r"run_(\d+)", p.name))]
    number = max(numbers, default=0) + 1
    while True:
        path = runs / f"run_{number:03d}"
        try:
            path.mkdir()
            break
        except FileExistsError:
            number += 1
    for name in DIRECTORIES:
        (path / name).mkdir()
    for variant in VARIANTS:
        (path / "models" / variant).mkdir()
    sources = source_identity(root)
    environment = environment_identity()
    for relative in sources:
        dest = path / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, dest)
    for notebook in (root / "notebooks").glob("*.ipynb"):
        shutil.copy2(notebook, path / "source_snapshot" / notebook.name)
    atomic_json(path / "config" / "experiment.json", config.to_dict())
    atomic_json(
        path / "config" / "identity.json",
        {
            "config": canonical_hash(config.to_dict()),
            "source": canonical_hash(sources),
            "environment": canonical_hash(environment),
        },
    )
    atomic_json(path / "environment" / "packages.json", environment)
    atomic_json(path / "source_snapshot" / "manifest.json", sources)
    atomic_json(path / "status.json", {"created_at": timestamp(), "stages": {}})
    return RunContext(root, path, config, mode)


def close_run(run: RunContext) -> None:
    run.logger.info("Notebook session closed; incomplete stages are not marked complete")
    for handler in list(run.logger.handlers):
        handler.flush()
        handler.close()
        run.logger.removeHandler(handler)
