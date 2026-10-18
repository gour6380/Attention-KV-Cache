# Publish the prepared project

The intended GitHub repository is `gour6380/attention-kv-cache`. Create it before
pushing: public visibility, no template, README off, no generated `.gitignore`,
and no initial license file. The local README and ignore rules are already
prepared. Choose and add a code license separately if desired; third-party data
and dependencies retain their own terms.

This guide contains commands for the repository owner to run. Preparing the guide
did not initialize a repository in this project, stage files, commit, or upload.
GitHub's [existing-project instructions](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
explain the empty-remote workflow and command-line authentication.

## Initialize this project and inspect the staged files

Open Terminal **inside the `attention-kv-cache` directory**, alongside this
project's `README.md`, `src/`, and `notebooks/`. The original workspace has a
parent Git repository: before local initialization, Git can resolve to that
parent. The checks below require the project markers and then verify the exact
repository root. Do not run these commands from the parent workspace.

These commands assume the project has not yet been initialized. They do not need
a Python environment, dataset, or model execution. Copy the complete block; the
parentheses keep the error-stop setting local to these commands.

```bash
(
    set -e
    test -f src/config.py
    test -f notebooks/attention_kv_cache.ipynb
    test ! -e .git
    git init -b main
    test "$(git rev-parse --show-toplevel)" = "$(pwd -P)"
    git add .gitignore README.md requirements.txt ruff.toml src tests notebooks docs results
    git status --short
    git diff --cached --stat
)
```

If the `.git` check fails because you already initialized the project, keep the
existing repository. Verify `git rev-parse --show-toplevel` points to this project
before repeating the `git add` and inspection commands. Do not delete `.git` or
force a push to work around an unexpected state.

The staged files should include:

| Include | Purpose |
|---|---|
| `.gitignore`, `README.md`, `requirements.txt`, `ruff.toml` | Setup, dependency pins, project instructions |
| `src/`, `tests/` | Implementation and correctness checks |
| `notebooks/attention_kv_cache.ipynb` | Executed notebook, including saved outputs |
| `docs/` | Design, technical/non-technical reports, and this guide |
| `results/run_001/` | Compact measured evidence, figures, raw trials, and provenance |

The ignore rules exclude `.venv*`, data caches, original `runs/`, local `reviews/`,
Python/tool caches, model checkpoints, TensorBoard events, local credentials, and
editor/OS files. They deliberately keep notebook outputs and the curated results.
Do not use `git add -f` to bypass these rules. The local environment and dataset
were removed on 30 September; uploading does not require restoring them.

## Commit and push

After reviewing the staged files, run this block from the same project directory:

```bash
(
    set -e
    test -e .git
    test "$(git rev-parse --show-toplevel)" = "$(pwd -P)"
    git commit -m "Add attention and KV cache experiment with results and reports"
    git remote add origin https://github.com/gour6380/attention-kv-cache.git
    git remote -v
    git push -u origin main
)
```

This first-push block assumes no existing `origin` remote and an empty GitHub
repository. If `origin` already exists, inspect `git remote -v` and keep it only
if it points to the intended repository; do not blindly replace another remote.
If authentication is requested, use your configured GitHub command-line login.
If the remote already contains a README, license, or other commit, reconcile that
history before pushing rather than force-pushing.

## Verify the uploaded view

Open the repository and check that:

- README charts and links render.
- Both reports and their evidence links open.
- The notebook shows its saved figures, results, and generation outputs.
- Source, tests, requirements, and compact results are present.
- Original run directories, weights, data, and environments are absent.

Add appropriate topics such as `pytorch`, `transformer`, `attention`, `kv-cache`,
`grouped-query-attention`, `multi-query-attention`, `tinystories`, `apple-silicon`,
and `benchmarking`. Use the actual repository URL in your portfolio or post once
it is live. Public availability is not established by the preparation checks.

For a local integrity check without project execution, use the standard-library
[results verification command](../results/run_001/README.md#verify-exported-file-hashes).
It hashes the exported files and compares them with their manifest; no environment
installation or model loading is needed. The [main README](../README.md) separately
explains how to install dependencies and run a new experiment.
