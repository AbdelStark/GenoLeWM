# RFC-0018: CLI design

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0006, RFC-0007, RFC-0010, RFC-0011, RFC-0012, RFC-0013, RFC-0017
- **Supersedes:** —
- **Implementation status:** Not started

---

## 1. Summary

GenoLeWM ships a CLI that covers training, scoring, rollout, planning,
evaluation, exporting, verification, and data preparation. This RFC
specifies the command shape (subcommands vs separate binaries), the flag
conventions, the exit-code mapping, the help / discovery surface, and
the shell-completion contract.

## 2. Motivation

The CLI is the primary public surface for non-Python users and the
canonical demonstration surface for the desktop app. A consistent design
across commands lowers the learning cost and shields the project from
ad-hoc divergence as new commands land.

## 3. Specification

### 3.1 Command layout

Each top-level operation is a separate console script, configured under
`[project.scripts]` in `pyproject.toml`. All scripts dispatch into the
same internal handler (`geno_lewm/cli/_dispatch.py::main`) so behavior
across commands is identical.

```
geno-lewm-train              # train predictor
geno-lewm-score              # score a single variant or a VCF
geno-lewm-rollout            # multi-edit haplotype rollout
geno-lewm-plan               # CEM planning
geno-lewm-eval               # run a single benchmark
geno-lewm-eval-all           # run the release eval suite
geno-lewm-export             # ONNX / Core ML / GGUF export
geno-lewm-verify             # verify a receipt
geno-lewm-cache-windows      # build / repair / reindex window cache
geno-lewm-prepare-gnomad     # build gnomAD shard
geno-lewm-prepare-clinvar    # build ClinVar shard
geno-lewm-update             # check for model updates
```

Separate binaries are chosen over a single `geno-lewm <subcommand>`
because the entry points become the public surface (pinned in
`pyproject.toml`; renames are MAJOR changes) and because shell auto-
completion treats them as independent.

### 3.2 Shared flags

Every command accepts:

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--config FILE` | path | command default | Hydra YAML; RFC-0017 |
| `--set KEY=VAL` (`-s`) | repeatable | — | Hydra override |
| `--seed INT` | int | from config | overrides `seed` |
| `--deterministic` | flag | false | sets `deterministic=true` |
| `--log-level LEVEL` | enum | `info` | `debug|info|warn|error` |
| `--log-dir PATH` | path | `$GENO_LEWM_LOG_DIR` | logging sink root |
| `--run-id STRING` | string | auto | also used as wandb run id |
| `--wandb-project STRING` | string | unset | enables wandb sink |
| `--no-receipt` | flag | false | disable receipt writing where applicable |
| `--print-config` | flag | false | prints resolved config and exits |
| `--print-config-tree` | flag | false | prints config + source files |
| `--explain KEY` | string | — | prints docstring for a config key |
| `--version` | flag | — | prints version and exits |
| `--help` | flag | — | prints help and exits |

### 3.3 Command-specific surface

#### `geno-lewm-train`

```
geno-lewm-train [shared flags] [config overrides]
```

- Resolves training config from `geno_lewm/config/defaults/train.yaml`
  plus overrides.
- Writes checkpoints to `${run_id}/checkpoints/`.
- Writes resolved config to `${run_id}/config.resolved.yaml`.

#### `geno-lewm-score`

```
geno-lewm-score --variant CHROM:POS:REF:ALT [--window-fasta FASTA] [--model PATH] [--output PATH]
geno-lewm-score --vcf PATH [--fasta PATH] [--output PATH] [--batch-size N]
```

- Output VCF or single variant; format selected by `--output` extension.
- Receipts written adjacent: `${output}.receipt.json`.

#### `geno-lewm-rollout`

```
geno-lewm-rollout --window-fasta REF.fa --edits EDITS.jsonl [--model PATH] [--output PATH]
```

EDITS.jsonl is one JSON `EditSpec` per line.

#### `geno-lewm-plan`

```
geno-lewm-plan --window-fasta REGION.fa --target-fasta TARGET.fa [--horizon K] [--iterations I] [--samples N] [--output PATH]
```

#### `geno-lewm-eval`

```
geno-lewm-eval --model PATH --benchmark BENCH [--head surprise|displacement]
```

`BENCH ∈ {clinvar_coding, clinvar_noncoding, brca2, traitgym_mendelian,
rollout, efficiency}`.

#### `geno-lewm-eval-all`

```
geno-lewm-eval-all --model PATH --output REPORT.md
```

Runs every benchmark, produces release-grade Markdown.

#### `geno-lewm-export`

```
geno-lewm-export --checkpoint PATH --target {coreml,onnx,gguf} --quantization {none,int8,int4} --output DIR
```

Writes the export artifacts plus an updated `manifest.json`.

#### `geno-lewm-verify`

```
geno-lewm-verify RECEIPT.json [--model PATH] [--rerun]
```

Checks manifest hash, input commitment, calibration hash. With
`--rerun`, re-executes inference on a supported backend and bit-matches
the output.

#### `geno-lewm-cache-windows`

```
geno-lewm-cache-windows [--corpus CORPUS_ID] [--repair] [--reindex]
```

Builds, repairs, or rebuilds the SQLite index over the Parquet shards.

#### `geno-lewm-prepare-gnomad` and `geno-lewm-prepare-clinvar`

```
geno-lewm-prepare-gnomad --release v4.1 --output DIR
geno-lewm-prepare-clinvar --release 2026-04-15 --output DIR
```

Network downloads gated by user-visible URLs printed to stderr first.

#### `geno-lewm-update`

```
geno-lewm-update --model-dir PATH [--check-only] [--target-version VERSION] [--yes]
```

Implements the user-initiated update flow (RFC-0010 §3.8). The command
fetches a Hugging Face release index only when invoked, compares the
selected remote manifest to `PATH/manifest.json`, and installs the new
release side by side only after explicit consent.

### 3.4 Exit codes

See [`docs/spec/04-error-model.md`](../docs/spec/04-error-model.md#exit-codes).

### 3.5 Output discipline

- Success: stdout receives the result (where applicable: path, summary).
- Logs go to stderr in pretty mode when stderr is a TTY; otherwise JSONL.
- `--quiet` silences info-level logs to stderr.
- Progress bars use `tqdm` and write to stderr; `--no-progress` disables.

### 3.6 Shell completion

Generated via `argcomplete` (or `click`'s built-in completion if we
adopt `click`). Completion is enabled by:

```
eval "$(geno-lewm-train --completion bash)"
```

Implemented for bash, zsh, and fish.

### 3.7 Banner and disclaimer

Every score / VCF command prints a one-line banner at startup:

```
GenoLeWM v0.1.0 — research tool, not a clinical diagnostic
```

Suppressed only by `--quiet --no-banner` (both required).

### 3.8 Argument parsing library

`typer` (built on `click`) is the chosen library. Reasons:

- First-class Pydantic / dataclass integration.
- Native help generation and shell completion.
- Hydra hand-off for config-heavy commands works cleanly.

Stable; widely used; not abandoned.

## 4. Rationale and alternatives

### 4.1 Why separate binaries rather than `geno-lewm <subcommand>`?

PyPI `console_scripts` entry points are part of the public surface and
are stable identifiers. Pinning them as separate binaries gives a clean
versioning story (renaming `geno-lewm-train` requires a MAJOR bump) and
makes tab completion behave as users expect.

### 4.2 Why `typer` over `argparse`?

`argparse` is in the standard library but does not produce help text we
want, has weaker type integration, and does not support shell completion
natively. `typer` is the most common modern Python CLI library; we
accept the dependency.

### 4.3 Why is the banner non-trivially suppressible?

This is intentional friction for downstream consumers who might be
tempted to script the CLI into clinical workflows. The banner is part of
the safety contract; suppressing it requires opt-in.

### 4.4 Why no `geno-lewm-serve`?

The runtime exposes no HTTP API in v1; a hosted serve mode is explicitly
out of scope (`docs/spec/00-overview.md`). A future hosted variant would
live in a downstream project.

## 5. Unresolved questions

- Whether to ship a single `geno-lewm` umbrella binary in v0.2 that
  dispatches to the subcommands for convenience, while keeping the
  per-command binaries as the stable surface.
- Whether to standardize on TOML, YAML, or JSON for output files
  produced by `--output`. Currently format-by-extension.
- Whether to allow `--config -` (stdin) for scripted use.

## 6. Future work

- Auto-generated man pages.
- An interactive REPL (`geno-lewm-repl`) for exploratory use.
- A `geno-lewm doctor` command that runs environment checks and reports
  the expected vs observed backend, dtype support, and disk space.

## 7. Changelog

- 2026-05-20 — Initial draft.
