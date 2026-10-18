"""TensorBoard is a viewer; checkpoint-committed JSONL is the measurement record."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def _atomic_bytes(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".metrics-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class FitTracker:
    def __init__(self, run, variant: str, resume_step: int | None = None, resume_state: dict | None = None):
        self.run = run
        self.directory = run.model_dir(variant)
        self.path = self.directory / "metrics.jsonl"
        kept = []
        if resume_step is not None:
            if resume_state is None:
                raise RuntimeError("Resume requires the checkpoint's durable metric-log cursor")
            content = self.path.read_bytes() if self.path.exists() else b""
            cursor = resume_state["bytes"]
            if not isinstance(cursor, int) or cursor < 0 or len(content) < cursor:
                raise RuntimeError("Metrics are shorter than the durable checkpoint cursor")
            prefix = content[:cursor]
            if hashlib.sha256(prefix).hexdigest() != resume_state["sha256"]:
                raise RuntimeError("Committed metrics changed since the checkpoint")
            kept = [json.loads(line) for line in prefix.decode("utf-8").splitlines() if line]
            if len(kept) != resume_state["records"] or any(row["step"] > resume_step for row in kept):
                raise RuntimeError("Checkpoint metrics cursor is inconsistent")
            # A same-step validation row may belong to uncommitted work too.
            # Truncate by durable bytes, not just optimizer-step comparisons.
            _atomic_bytes(self.path, prefix)
        elif self.path.exists() and self.path.stat().st_size:
            raise RuntimeError("Fresh training would overwrite existing metrics; resume this run")
        self.records = len(kept)
        self.writer = SummaryWriter(str(run.path / "tensorboard" / variant), purge_step=resume_step)
        if resume_step is not None:
            # TensorBoard purges the replay boundary inclusively. Restore only
            # records at that boundary that the checkpoint actually committed.
            for row in kept:
                if row["step"] == resume_step:
                    self._write_scalars(row)

    def _write_scalars(self, record: dict) -> None:
        for key, value in record.items():
            if key not in {"step", "kind"} and isinstance(value, (float, int)):
                self.writer.add_scalar(f"{record['kind']}/{key}", value, record["step"])

    def log(self, step: int, kind: str, values: dict) -> None:
        record = {"step": step, "kind": kind, **values}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.records += 1
        self._write_scalars(record)
        self.run.logger.info("%s step=%s %s", kind, step, values)

    def flush(self) -> None:
        self.writer.flush()
        if self.path.exists():
            with self.path.open("rb") as stream:
                os.fsync(stream.fileno())

    def checkpoint_state(self) -> dict:
        self.flush()
        content = self.path.read_bytes() if self.path.exists() else b""
        return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "records": self.records}

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
        if self.path.exists():
            rows = []
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        raise
                    # A process interruption may leave an uncommitted final
                    # write. Preserve JSONL for cursor-based replay; CSV only
                    # includes complete records and is never the source of truth.
                    self.run.logger.warning("Skipping incomplete final metrics record during CSV export")
            fields = sorted({key for row in rows for key in row})
            with (self.directory / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def training_summary(run) -> list[dict]:
    from src.config import VARIANTS
    from src.io import read_json

    results = []
    for variant in VARIANTS:
        path = run.model_dir(variant) / "result.json"
        results.append(read_json(path) if path.exists() else {"variant": variant, "status": "not_complete"})
    return results
