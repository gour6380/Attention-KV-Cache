# Inspectable evidence for run_001

This folder contains a compact export of the completed 28–29 September 2026
experiment. The [technical report](../../docs/technical-report.md) interprets the
measurements; the [plain-language report](../../docs/non-technical-report.md)
explains their practical meaning. No training, evaluation, or timing workloads
were rerun to produce this export.

## What is included

| Artifact | Purpose |
|---|---|
| [quality.csv](quality.csv) | Test NLL/perplexity, denominators, parameter counts, checkpoint hashes |
| [quality_cost.csv](quality_cost.csv) | All 30 model/workload comparisons |
| [figures/](figures/) | Original PNG/SVG plots, including training, cache growth, prefill, decoding |
| [config/experiment.json](config/experiment.json) | Exact requested configuration |
| [config/selection.json](config/selection.json) | Final checkpoint identities and equal training-token exposure |
| [models/](models/) | Training metrics, architectures, parameter counts, completion records, checkpoint checksum receipts |
| [evaluation/](evaluation/) | Quality, baseline, and correctness measurements |
| [benchmarks/](benchmarks/) | 750 raw trials, per-variant summaries, metadata, 30 parity receipts, fixture identity |
| [generation/](generation/) | All twelve illustrative generation examples |
| [data_manifest/](data_manifest/) | Source revision, preprocessing counts, split/token identities |
| [environment/](environment/) | Recorded package/platform/device metadata |
| [source_manifest.json](source_manifest.json) | Hashes of the matching repository source and requirements |
| [benchmark_audit.json](benchmark_audit.json) | Recomputed statistics, order sensitivity, and memory-state observations |
| [verification.json](verification.json) | Saved-artifact checks and explicitly historical test evidence |
| [artifact_manifest.json](artifact_manifest.json) | Export hashes, original hashes, and transformation notes |

The benchmark attempt IDs in original records are MHA `6f9674787850`, GQA
`42faa0ad390b`, and MQA `10327f1f5c56`. Their selected contents are placed directly
under `benchmarks/mha`, `benchmarks/gqa`, and `benchmarks/mqa` for browsing.
Historical `raw_trials_path`/`csv_path` strings inside summaries still refer to
the original run layout; the adjacent `trials.jsonl`/`summary.csv` files are the
exported counterparts. Their hashes are recorded in the export manifest.

## Preservation and transformations

Original `runs/run_001`, source, dataset caches, model weights, and the executed
notebook were preserved. Source and notebook identity checks are recorded in the
verification receipt. Tables, plots, raw trials, generation examples, and metrics
are copied without changing measurements. In selected JSON metadata, absolute
project paths are replaced by the literal `${PROJECT_ROOT}`. Those JSON files
are reserialized with sorted keys. The manifest records both original and export
SHA-256 values and identifies each transformed file.

Embedded experiment identities continue to identify the original local run.
They are not recalculated to pretend that a path-redacted export can resume that
run. An original hash of a redacted manifest need not equal the redacted file's
hash; use `artifact_manifest.json` to verify the exported file itself.

The executed notebook is preserved byte-for-byte, including local paths already
visible in its output. It is an execution record, not a newly sanitized notebook.

## What is deliberately absent

This is an evidence bundle, **not a resumable run directory**. It omits corpus
text, token arrays, benchmark token fixtures, tokenizer payload, model/optimizer
weights, TensorBoard events, and local review history. Files such as `final.json`
record a checksum for an omitted checkpoint; they do not contain the checkpoint.
Likewise, metadata may name omitted data files for identity purposes.

A fresh clone can inspect the reports, figures, metrics, and notebook outputs
without running anything. To create new models, follow the [setup and execution
instructions](../../README.md). A local resume requires the actual original
`runs/` and `data/` contents plus their matching environment.

## Verify exported file hashes

From the repository root, this standard-library-only command checks every file
listed in the export manifest. It does not execute the notebook or project code.

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("results/run_001")
manifest = json.loads((root / "artifact_manifest.json").read_text())
for name, record in manifest["files"].items():
    digest = hashlib.sha256()
    with (root / name).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    assert actual == record["sha256"], name
print(f"Verified {len(manifest['files'])} exported files")
PY
```

These hashes detect accidental changes to the recorded bundle; they are not an
independent signature or evidence of additional training replications. The
manifest excludes itself to avoid a circular hash dependency.
