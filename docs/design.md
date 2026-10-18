# Implemented design

This is the design implemented by the source accompanying the completed experiment.
The [technical report](technical-report.md) analyzes the results; the
[README](../README.md) explains installation and notebook execution.

## Question and controls

With the same training data, tokenizer, decoder width/depth, query-head count,
and training-token exposure, how does reducing key/value heads affect language
model quality, KV storage, and decoding latency?

Three independently trained variants share one implementation. MHA uses eight
KV heads, GQA two, and MQA one. All retain eight query heads. The comparison holds
width and depth fixed, rather than compensating for fewer K/V parameters by
widening other layers. It therefore does not match parameter count or compute.
The GQA/MQA mechanisms are inspired by their papers; this is not a reproduction
of those papers' training or checkpoint-conversion experiments.

## Architecture

| Component | Implemented setting |
|---|---|
| Decoder | Six pre-normalized Transformer blocks |
| Width / query heads / head dimension | 384 / 8 / 48 |
| KV heads | MHA 8; GQA 2; MQA 1 |
| Feed-forward | SwiGLU, hidden width 1,024 |
| Normalization | RMSNorm, epsilon `1e-5` |
| Positions | RoPE, base 10,000 |
| Embedding/output | Shared weights, vocabulary 8,192 |
| Linear biases / dropout | Disabled / 0.0 |
| Precision | FP32; no compilation or mixed precision |
| Training context / inference capacity | 1,024 / 4,096 |
| Parameters | MHA 13,767,552; GQA 12,440,448; MQA 12,219,264 |

```text
token IDs -> embedding
          -> six blocks:
               RMSNorm -> causal attention -> residual
               RMSNorm -> SwiGLU -> residual
          -> RMSNorm -> tied vocabulary projection -> logits
```

Same-shaped parameters use deterministic seeds derived from their parameter
names. This prevents a different K/V shape from shifting the random stream of
unrelated layers. It does not force differently shaped K/V projections to be
identical. The architecture and initialization are defined in
[decoder.py](../src/models/decoder.py) and [config.py](../src/config.py).

## Attention and cache

[attention.py](../src/models/attention.py) groups the query heads against compact
K/V heads and uses explicit grouped matrix multiplication. K/V are not physically
expanded to eight heads in the primary implementation. Attention scores are
materialized; this is not a fused or FlashAttention implementation. Eight query
heads still compute attention even when the number of KV heads is smaller.

A model-local boolean causal mask is reused across layers and compatible
forwards. It grows only when necessary and is not persisted in checkpoints.
Offset-aware views preserve causality for chunked appends. A single-token cached
query attends to the existing prefix and itself without constructing a causal
mask. No global tensor cache keeps a previously released model alive.

[cache.py](../src/cache.py) owns request-local, inference-only, preallocated K/V
buffers. It tracks active length and capacity and validates batch, device, dtype,
and layer shapes. RoPE uses absolute positions when appending tokens. All layers
write at the same token positions. Capacity overflow is an error; no sliding
window or silent truncation is performed.

For FP32, logical KV storage is:

```text
2 × layers × batch × KV_heads × active_tokens × head_dimension × 4 bytes
```

Replace active tokens with capacity for allocated cache bytes. Thus storage
ratios are `1 : 1/4 : 1/8`. They describe the K/V tensors, not total process memory.
Cache allocation precedes timed inference in the benchmark.

## Data and training

[Data preparation](../src/data.py) normalizes Unicode to NFC, line endings, and
outer whitespace. It excludes empty text and exact normalized duplicates, gives
held-out stories precedence over overlapping training text, and selects 250,000
training stories with seeded hash ordering. Unique official validation stories
are divided by deterministic hash ordering into 10,995 validation and 10,995 test
stories. The test set is derived from the official validation split; near
variants of stories may remain.

One [byte-level BPE tokenizer](../src/tokenizer.py) is trained on the selected
training stories only. Its 8,192 entries include padding, BOS, and EOS. Stories
are concatenated within each split with BOS/EOS markers; EOS does not reset
attention. Disk-backed arrays supply next-token windows with stride 1,024.
Training keeps complete windows and records its discarded tail. Held-out splits
keep the final partial window and exclude padding from loss.

The [training loop](../src/train.py) uses seed 2027, three epochs, AdamW with
betas `(0.9, 0.95)`, weight decay 0.1 excluding normalization/bias parameters,
5% linear warm-up, and cosine decay from `3e-4` to `3e-5`. Gradient norm clipping
is 1.0. Microbatch size four and accumulation two yield 8,192 targets per full
update; the final partial group is normalized by its actual target count.
Epoch permutations are shared among variants.

Detached losses remain on the accelerator through accumulation. Mean loss and
gradient norm are read back together before the optimizer step; nonfinite values
stop that step. Training throughput uses synchronized reporting intervals and
includes batch preparation, while logging/checkpoint writes are outside the
interval. Training scalar logs are written every ten updates. Complete validation
runs after each epoch. Checkpoints are saved every 250 updates and every epoch.

The final checkpoints are the primary comparison after equal token exposure.
Validation-best checkpoints are secondary. No early stopping changes the
exposure of one variant relative to another.

## Notebook, provenance, and recovery

The [notebook](../notebooks/attention_kv_cache.ipynb) imports local `src` functions;
it does not install this project as a package or hide all training in one call.
Each preparation step, fit, evaluation, benchmark, and figure has a separate
cell. Run All intentionally executes the configured experiment. The committed
notebook retains the completed run's outputs.

[runs.py](../src/runs.py) creates numbered directories by exclusive creation.
Run identities hash configuration, source/requirements, and environment. The
environment identity includes Python, platform, all installed package versions,
and the interpreter's absolute path. Data and tokenizer bindings have additional
identities. This is strict recovery for the same local experiment, not portable
resume after moving folders or rebuilding a different environment.

[checkpoints.py](../src/checkpoints.py) stores explicit model configuration and
state dictionaries. Resumable checkpoints also store optimizer/scheduler state,
epoch/batch position, updates, tokens, sampler position, and applicable RNG states.
Atomic writes occur at completed update boundaries. Completed training fits are
verified and reused; interrupted fits require explicit resume. Work after the
last completed checkpoint is replayed.

Only one operation may own a run's stage lock. Changing scientific settings,
source, device, or environment requires a new compatible run identity. The code
stops on invalid operations rather than silently reducing batches or falling
back mid-operation. MPS bitwise reproducibility is not claimed.

## Measurement contracts

- [Evaluation](../src/evaluate.py) sums valid-token NLL, divides by the number of
  valid targets, then exponentiates to obtain perplexity. The unigram baseline
  uses training-token counts. Evaluation cannot train a missing model.
- [Generation](../src/generate.py) samples at temperature 0.8/top-p 0.9, up to 128
  new tokens or EOS, within the trained context. Samples illustrate behavior;
  they are not a quality ranking.
- [Benchmarking](../src/benchmark.py) uses fixed held-out tokens, five warm-ups and
  twenty measured trials for each prompt-length/batch case. Both modes execute
  128 forced append-token forwards; EOS never shortens the workload. Timing
  synchronizes the device and excludes input transfer, cache allocation, logging,
  and correctness checks. Mode order alternates within each case.
- Trial medians/p95 summarize repeated workloads, not individual-token tail
  latency. Throughput counts tokens across the whole batch. Memory categories
  are overlapping snapshots, not exact peaks. Cases whose final context exceeds
  1,024 are systems-only measurements.
- [Reporting](../src/reporting.py) reads existing artifacts. It does not train
  missing models or execute missing benchmarks. Reexecuting evaluation,
  generation, or benchmark cells performs those operations again; training
  completion reuse is not a general promise of notebook-wide idempotence.

Correctness checks cover expanded-KV reference agreement, gradients, causality,
full/token/chunk cache parity, offset preservation, reset, storage, and overflow.
FP32 parity uses `atol=1e-5` and `rtol=1e-4` together. Actual benchmark endpoints
are checked after all append steps; bounded representative checks compare
intermediate token/chunk paths. See the [technical report](technical-report.md)
for measured evidence, timing variability, and limitations.
