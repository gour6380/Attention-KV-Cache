# Attention and KV Cache

**GitHub repository:** [gour6380/attention-kv-cache](https://github.com/gour6380/attention-kv-cache)

[Setup](#install-and-open) · [Run a fresh experiment](#run-the-notebook-in-phases) ·
[Resume a local run](#resume-an-actual-local-run) · [Generated files](#where-your-new-results-are-saved) ·
[Technical report](docs/technical-report.md) · [Non-technical report](docs/non-technical-report.md)

A notebook-led experiment on a practical language-model question: **how much
memory can attention save by sharing keys and values, and what happens to quality
and decoding speed?** Three small decoders learn the same TinyStories data, then
run the same inference workloads on an Apple M2 Pro using PyTorch MPS.

The completed `run_001` compares multi-head attention (**MHA**, 8 KV heads),
grouped-query attention (**GQA**, 2), and multi-query attention (**MQA**, 1). All
three retain 8 query heads. A KV cache stores keys and values from earlier tokens
so generation can reuse them instead of recomputing the entire prefix.

## Results at a glance

Each model completed three epochs, **164,637,696 training targets** and **20,100
optimizer updates**. Final checkpoints were evaluated on the same 2,358,938 test
targets. Lower negative log-likelihood (NLL) and perplexity are better.

| Model | Parameters | Test NLL | Test perplexity | KV storage relative to MHA |
|---|---:|---:|---:|---:|
| MHA | 13,767,552 | 1.61475 | 5.02663 | 100% |
| GQA | 12,440,448 | 1.61436 | 5.02467 | 25% |
| MQA | 12,219,264 | 1.62119 | 5.05913 | 12.5% |
| Unigram reference | — | 5.96020 | 387.68707 | — |

GQA and MHA produced very close quality measurements in this one-seed experiment.
MQA had slightly higher test loss. GQA reduced KV storage by 75%; MQA reduced it
by 87.5%. These percentages describe the cache, not total application memory.

Decoding speed depended on the workload. The values below are median milliseconds
per cached append step; each trial appends exactly 128 tokens per sequence.

| Prompt tokens | Batch | MHA | GQA | MQA |
|---|---:|---:|---:|---:|
| 128 | 1 | 6.291 ms | 6.419 ms | 6.473 ms |
| 512 | 4 | 7.443 ms | 6.811 ms | 6.885 ms |

MHA was fastest in the short, single-sequence example; sharing KV heads helped in
the larger batch example. A smaller cache does not guarantee faster decoding.
These are hardware- and implementation-specific observations, not a universal
architecture ranking or evidence of a statistically significant quality gap.

![Cached and uncached decode latency](results/run_001/figures/decode_latency.png)

Read the [non-technical report](docs/non-technical-report.md) for an accessible
explanation, or the [technical report](docs/technical-report.md) for methodology,
full results, and limitations. Exact numbers are in
[quality.csv](results/run_001/quality.csv) and
[quality_cost.csv](results/run_001/quality_cost.csv).

## Explore without running anything

The [executed notebook](notebooks/attention_kv_cache.ipynb) retains its saved
outputs, figures, diagnostics, and sample generations for browsing on GitHub or
in Jupyter. Opening it does not require retraining. Running cells replaces their
displayed outputs; save a separate notebook copy before experimenting if you
want to retain the recorded view locally.

The notebook's opening Markdown is a preserved pre-run note: its "execution
pending" and "output-free" wording predates the completed outputs. Its local
`reviews/` links are also excluded from the public repository. Use this README,
the two reports, and the exported evidence for the completed run's status.

The compact evidence under [results/run_001](results/run_001/) contains tables,
figures, the experiment configuration, per-model training metrics and result
records, a benchmark audit, and artifact hashes. It is a publication bundle,
not a resumable run. Downloaded data, token arrays, trained weights, optimizer
checkpoints, the original run directory, and local review records are omitted.
A fresh clone therefore cannot load the recorded models or resume `run_001`.

## Repository contents

| Path | Included in the repository |
|---|---|
| `notebooks/attention_kv_cache.ipynb` | Experiment workflow with preserved execution outputs |
| `src/` | Data, model, training, evaluation, cache, benchmark, and reporting code |
| `tests/` | Correctness and notebook checks |
| `requirements.txt` | Pinned Python dependencies for the recorded macOS environment |
| `docs/` | Technical and non-technical reports, design reference, upload instructions |
| `results/run_001/` | Compact measured evidence, figures, configuration, and verification records |
| `.gitignore` | Exclusions for local environments, data, full runs, reviews, and generated caches |

The upload includes the executed notebook and compact evidence. Local data,
virtual environments, full run/checkpoint directories, and review workspaces stay
outside Git. For initial publication from an existing local copy, follow the
[GitHub upload guide](docs/github-upload.md) before running Git commands.

## Install and open

The validated environment is **native macOS arm64, Python 3.13.11, PyTorch 2.14.0,
and MPS on an Apple M2 Pro**. The pinned requirements reflect that environment.
CUDA, Linux, and Windows execution have not been validated. Device selection
supports MPS → CUDA → CPU, but that does not make these pins a tested installation
recipe for other platforms.

Install native Python 3.13 before the commands below. The setup uses standard
[Python virtual environments](https://docs.python.org/3.13/library/venv.html) and
[IPython kernel registration](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

### 1. Get the code

The clone command uses the repository linked above and requires it to be
published there. If you are publishing the existing local project first, use the
[upload guide](docs/github-upload.md). For a fork, replace the URL with your fork's URL.
Keep the directory name **`attention-kv-cache`** because the notebook uses it to
locate the project. If extracting a ZIP, rename the extracted directory to
`attention-kv-cache`.

```bash
git clone https://github.com/gour6380/attention-kv-cache.git attention-kv-cache
cd attention-kv-cache
```

If you already have this project directory, skip cloning and open Terminal there.
Run the following commands from the repository root, alongside `README.md`,
`requirements.txt`, and `src/`.

### 2. Create the environment and install dependencies

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

If a suitable `.venv` already exists, activate it instead of recreating it.
The project itself is local source: no editable package installation or project
CLI is required. Installation needs internet access; the dataset is downloaded
later by its own notebook cell.

### 3. Register the kernel and open JupyterLab

With the environment still activated:

```bash
python -m ipykernel install --user --name attention-kv-cache --display-name "Attention KV Cache — Python 3.13"
python -m jupyterlab notebooks/attention_kv_cache.ipynb
```

Select **Attention KV Cache — Python 3.13** in the notebook's kernel picker.
The notebook imports its local `src` modules. Restart the kernel if another
project's `src` was previously imported. If you deleted and recreated `.venv`,
repeat kernel registration so it points to the recreated environment.

For later sessions, open Terminal in the same repository root and run:

```bash
source .venv/bin/activate
python -m jupyterlab notebooks/attention_kv_cache.ipynb
```

After a kernel restart, Python variables from the previous session are not
restored by saved notebook outputs. Follow fresh-run or resume instructions below
before continuing.

The first experiment downloads the corpus and creates local caches and model
checkpoints. Allow disk space for these artifacts and sustained compute time for
three complete fits and benchmarks. `SHOW_TENSORBOARD` controls an optional local
viewer; set it to `False` if you do not want to start it.

## Run the notebook in phases

**Run All intentionally executes the entire experiment:** data preparation,
correctness checks, three training fits, full held-out evaluation, repeated
benchmarks, diagnostics, and generation. The saved outputs are not instructions
to skip prerequisite cells in a new kernel.

For a fresh experiment, edit these values in the existing
**Choose the immutable experiment configuration** cell:

```python
RUN_MODE = "new"
RESUME_RUN_ID = None
SHOW_TENSORBOARD = True  # False disables the optional viewer, not event logging.
```

Keep the remaining configuration for the documented three-model protocol.
To use exactly the recorded corpus, set the [dataset revision](#pin-the-recorded-dataset-for-a-new-experiment)
before starting the run. Any experiment changes must also be made before
**Start or resume one run** freezes its configuration.

Execute the cells from the top in order, one cell at a time, waiting for each
operation to finish before starting the next:

1. **Setup:** locate the project, import interfaces, choose the configuration,
   start a run, and inspect the device. Rerunning **Start or resume one run** in
   fresh mode allocates another numbered directory; subsequent cells use the
   selected run. In **Inspect the selected environment**, an available Apple
   Silicon GPU should appear as `"device": "mps"`.
2. **Data:** download the source, prepare and audit splits, train/load the shared
   tokenizer, then prepare disk-backed token windows. These cells verify and
   reuse matching project caches when available.
3. **Correctness and training:** inspect the three architectures, run the cache
   checks, fit/evaluate the unigram reference, then execute **Train MHA**,
   **Train GQA**, and **Train MQA** sequentially. Each fit has its own cell.
4. **Quality:** inspect completion, plot training curves, freeze the final
   equal-budget checkpoints, and evaluate the baseline and all models on test.
5. **Systems:** prepare identical benchmark fixtures, benchmark each model, then
   render the cache, prefill, decode, and quality/cost comparisons.
6. **Walkthrough and report:** inspect attention and cache growth, release the
   walkthrough model, generate examples, assemble the saved-artifact report,
   and close the notebook session.

Evaluation cells do not train models. Report assembly builds its report from
existing artifacts and labels missing stages. Generation requires a trained
local checkpoint and its tokenizer; the compact published evidence does not
replace them.

### Where your new results are saved

The first fresh run creates `runs/run_001/`; later fresh runs allocate the next
available number without overwriting existing directories. If no `runs/`
directory exists, it is created automatically. The checked-in
`results/run_001/` publication bundle is separate from these new runs.

| Local path | What to inspect |
|---|---|
| `data/` | Reusable source, split, tokenizer, and disk-backed token caches |
| `runs/run_XXX/config/`, `source_snapshot/`, `environment/` | Exact configuration, code snapshot, and environment identity |
| `runs/run_XXX/data_manifest/`, `tokenizer/` | Bound data identities and saved tokenizer |
| `runs/run_XXX/models/{mha,gqa,mqa}/` | Architecture, metrics, and `latest.pt`, `best.pt`, `final.pt` checkpoints |
| `runs/run_XXX/logs/`, `tensorboard/` | Text logs and TensorBoard events |
| `runs/run_XXX/evaluation/`, `benchmarks/`, `figures/` | Quality results, timing trials, summaries, and plots |
| `runs/run_XXX/generation/`, `report/` | Generated examples and assembled `report/report.md` |
| `runs/run_XXX/status.json` | Recorded stage completion, interruption, or failure |

These paths are relative to the repository root; entries after a comma share
the same `runs/run_XXX/` prefix. Files appear as their corresponding cells finish.
The local environment, data caches, and full runs are ignored by Git.

### Pin the recorded dataset for a new experiment

The recorded configuration requests `revision="main"`. At acquisition it resolved
to the immutable TinyStories revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. A future fresh run with `main` may resolve
to different content.

To use the recorded corpus revision, replace the existing `data=DataConfig(...)`
argument inside the notebook's `CONFIG = ExperimentConfig(...)` cell with:

```python
data=DataConfig(
    dataset="roneneldan/TinyStories",
    revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
    train_stories=250_000,
    split_seed=2027,
    vocab_size=8192,
    context_length=1024,
),
```

Keep `RUN_MODE = "new"` and `RESUME_RUN_ID = None`. This pins source content for
a new experiment; changing `main` to the SHA changes configuration identity and
cannot be used to resume a run created with `main`. Matching data and seeds do
not imply bitwise-identical MPS training.

### Resume an actual local run

Resume is for an existing local run directory and its matching data caches in
the original local environment. A fresh clone contains only the published evidence
bundle, so use fresh mode to train your own models. If a local run's environment
or data caches have been removed, reinstalling dependencies alone does not
restore that state. Use fresh mode unless the complete original state can be
restored and verified.
For an eligible local run, set these values in the configuration cell:

```python
RUN_MODE = "resume"
RESUME_RUN_ID = "run_001"  # Replace with your actual local run ID.
```

Restore that run's exact configuration, restart the kernel, and execute setup and
data-preparation cells to verify and bind the saved artifacts. Run **Check attention
and cache correctness** if its saved receipt is missing, then continue with the
relevant training cell. Completed fits are verified and reused; interrupted fits
resume from the latest complete optimizer-boundary checkpoint. Work after that
checkpoint is replayed. Final/best inference checkpoints are separate from the
latest resumable training checkpoint.

Resume does not skip every completed cell: evaluation and benchmark cells perform
their work again when executed. Continue only the stages you need; **Run All** in
resume mode still includes those potentially lengthy operations.

Identity checks include configuration, all `src` Python files, `requirements.txt`,
Python version, OS/platform, machine architecture, **the absolute interpreter
path**, and **every installed package/version**. Dataset and tokenizer identities
are also verified. Moving a checkout or recreating its environment is not a
portable resume workflow. Adding a package or changing devices can invalidate
recovery; use a fresh run when the recorded identity cannot be restored. Do not
edit identity records to bypass checks.

Only one notebook operation may own a run. If a process crash leaves `.stage.lock`,
confirm its original process is gone before removing that stale lock. Allocation
failures or unsupported operations stop explicitly. Restart the kernel after an
allocation failure to release retained tensors, then resume the unchanged run or
choose a new configuration in fresh mode. Silent MPS fallback is rejected.

### Common setup and execution issues

| Symptom | Action |
|---|---|
| `python3.13` is not found | Install native Python 3.13 and reopen Terminal before creating the environment. |
| Missing packages or an unavailable notebook kernel | Activate this project's `.venv`, install its pinned requirements, register the kernel, and select it in Jupyter. |
| The notebook cannot locate the project or imports another `src` | Keep the folder name `attention-kv-cache`, launch Jupyter from its root, and restart the kernel before rerunning setup. |
| An evaluation or generation cell reports a missing checkpoint | Run the prerequisites and the relevant training cell, or resume a complete local run. The public evidence bundle contains no model weights. |
| `Run identity changed` | Restore the run's exact configuration, source, and environment, or start a fresh run. Reinstalling packages while a run is active can change its identity. |
| MPS fallback is rejected | Unset `PYTORCH_ENABLE_MPS_FALLBACK` (or set it to `0`) before launching Jupyter, then restart the kernel. |

## Experiment design

| Setting | Protocol |
|---|---|
| Dataset | Original `roneneldan/TinyStories`; immutable source revision recorded |
| Train split | 250,000 unique normalized stories, excluding exact held-out matches |
| Validation / test | 10,995 stories each, derived from the official validation split |
| Tokenizer | Shared 8,192-token byte-level BPE, fitted on training only |
| Architecture | 6 layers, width 384, FFN 1,024, 8 query heads; 8 / 2 / 1 KV heads |
| Components | RoPE, RMSNorm, SwiGLU, tied embeddings, bias-free projections |
| Context | 1,024 training positions; 4,096 inference capacity |
| Training | Seed 2027; 3 epochs; FP32; microbatch 4 × accumulation 2 |
| Optimizer | AdamW; LR 3e-4 → 3e-5; 5% warm-up; clipping 1.0 |
| Primary comparison | Final checkpoints after equal processed training tokens |
| Benchmark | Prompts 128 / 256 / 512 / 1,024 / 2,048; batches 1 / 4; 128 append steps |
| Repetition | 5 warm-ups and 20 measured trials per workload and model |

Width, depth, query-head count, data order, tokenizer, and optimizer are held
constant. Reducing KV heads reduces parameter counts; other components are not
widened to compensate. Unicode NFC, line-ending normalization, and outer
whitespace trimming precede exact deduplication. Near duplicates can remain.
BOS/EOS-delimited stories are concatenated within splits; EOS does not reset
attention. Training drops and records the incomplete tail; held-out tails retain
valid targets with ignored padding.

All models use compact grouped matrix multiplication with explicit attention
scores and preallocated request-local KV buffers. This is not FlashAttention or
a comparison of optimized native attention kernels. Logical cache storage is:

```text
2 × layers × batch × KV heads × active tokens × head dimension × bytes per value
```

Timing uses the same saved token values, fixed append counts, alternating
cached/uncached order, and synchronized device boundaries. Allocation,
tokenization, transfers, logging, and parity checks are outside timed regions.
Actual trained-model cache parity must pass before timing is accepted.

## Evidence and limits

The [technical report](docs/technical-report.md),
[benchmark audit](results/run_001/benchmark_audit.json),
[artifact manifest](results/run_001/artifact_manifest.json), and
[verification record](results/run_001/verification.json) describe the distributed
evidence and its checks. Source organization and invariants are documented in
[the design reference](docs/design.md). Small correctness tests can be run from
the repository root with the project environment activated:

```bash
python -m pytest -q
```

These tests do not establish full-model performance. On **28 September 2026**,
pre-experiment validation reported 120
passing tests and four CUDA-only skips on the Mac. On **29 September 2026**, the
three static notebook tests passed during documentation preparation, after the
old output-free assertion was updated to allow preserved execution outputs.
The full runtime suite was not rerun during that preparation or this upload
readiness review. These dated records do not establish a fresh installation in
the current checkout, whose project environment has since been removed.

The findings cover one seed, synthetic simplified-English stories, one small
architecture, one attention implementation, and the recorded MPS environment.
The derived test set is not an official TinyStories test split. Prompts plus
continuations exceeding 1,024 tokens are **systems-only** workloads; they do not
establish long-context language quality. Reported p95 values summarize trials'
mean step times, not individual-token tail latency. RSS, MPS tensor allocation,
and Metal driver allocation are overlapping snapshots, not exact peak memory.
Generation examples are illustrative; attention weights do not explain model
reasoning.

Research foundations: [Transformer](https://arxiv.org/abs/1706.03762),
[MQA](https://arxiv.org/abs/1911.02150),
[GQA](https://aclanthology.org/2023.emnlp-main.298/), and
[TinyStories](https://arxiv.org/abs/2305.07759).
