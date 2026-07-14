# Production resume equivalence

GenoLeWM can stop a production Carbon training run at step `K` while retaining
its original `N`-step horizon, then continue in a fresh process at `K+1`:

```bash
geno-lewm-train --carbon-train \
  --dataset-dir /path/to/dataset --carbon-model-dir /path/to/carbon \
  --training-config /path/to/train.yaml --run-dir /tmp/prefix \
  --steps 100 --stop-after-step 40

geno-lewm-train --carbon-train \
  --dataset-dir /path/to/dataset --carbon-model-dir /path/to/carbon \
  --training-config /path/to/train.yaml --run-dir /tmp/resumed \
  --steps 100 --resume-from /tmp/prefix/predictor_checkpoint.pt
```

`--stop-after-step` is incompatible with `--package-release-run`. The prefix
metadata is labeled `stopped_early`, with both `steps_completed=K` and
`target_steps=N`; it is not a completed release run.

## Closed checkpoint contract

Each checkpoint is written through a unique same-directory file opened with
exclusive creation and no-follow semantics. A per-target writer lock rejects
concurrent replacement. The public CLI acquires its run-level lock before any
effective-config or preflight publication and retains it through training and
optional release packaging; the direct runner acquires the same lock itself.
Every lock, temporary, replace, cleanup, and directory-`fsync` operation is
anchored to one opened non-symlink parent directory, and the textual parent must
still identify that directory before success. The writer flushes and `fsync`s
the file, atomically replaces the destination, `fsync`s the directory, and
removes only temporary/lock inodes that it created. A production resume restores
and validates all continuation state before fetching or encoding the first
`K+1` batch:

Checkpoint/report publication requires anchored `dir_fd`, no-follow, hard-link,
and rename support. Unsupported platforms or filesystems fail closed before
creating the publication parent or artifacts; standalone Carbon preflight and
fixture smoke remain available because they do not use this publication path.

- predictor and action-encoder parameters;
- AdamW moments, parameter groups, current LR, and base LR;
- the `TorchTrainer` collapse-monitor baseline, policy, thresholds, and alerts;
- Python, NumPy, Torch CPU, and every available Torch CUDA RNG state, with CUDA
  availability and device count bound across continuation;
- exact dataset snapshot/manifest, encoder, config, seed, sample cursor, and
  consumed window order;
- cumulative step metrics and the LR schedule computed against the original
  `N`, not a shortened suffix horizon.

The loader uses PyTorch's weights-only path and rejects missing, extra,
corrupted, cross-source, cross-package-version, cross-data, cross-config,
CUDA-domain/count-drifted, or wrong-horizon state. CUDA model, device UUID, and
driver identity are not bound, so this contract does not establish equivalence
after moving between accelerator identities.

Production source identity comes from the imported `geno_lewm` package, never
the caller's current directory. The CLI requires the canonical tracked package
root with only regular tracked files, a clean `geno_lewm/` source tree, and no
ignored package artifacts such as bytecode. It records full lowercase
40-character COMMIT/TREE values plus the package version and fails closed for
unresolved or sentinel identities. Remove package caches and launch with
`PYTHONDONTWRITEBYTECODE=1` so the immutable package remains free of ignored
artifacts. Wheels currently do not embed build-time COMMIT/TREE provenance, so
`--carbon-train` intentionally rejects a wheel-only installation; use a clean
immutable source checkout. Preflight and non-production fixture commands retain
their existing wheel behavior.

## External three-process verifier

Run evidence outside the source worktree so the checkout remains clean:

```bash
COMMIT=$(git rev-parse HEAD)
TREE=$(git rev-parse 'HEAD^{tree}')

python -m tools.research.production_resume_equivalence run \
  --repo-root . --output-dir /tmp/geno-lewm-resume-evidence \
  --dataset-dir /path/to/dataset --carbon-model-dir /path/to/carbon \
  --training-config /path/to/train.yaml \
  --expected-source-commit "$COMMIT" --expected-source-tree "$TREE" \
  --total-steps 100 --split-step 40
```

The runner invokes the public `geno-lewm-train` console script in three
distinct processes: uninterrupted `1..N`, prefix `1..K`, and resumed `K+1..N`.
It requires the tracked GenoLeWM package at the clean Git top level, places that
source root first on each subprocess import path, records the argv and PID for
each arm, and hashes the raw checkpoints, metrics, logs, metadata, stdout, and
stderr.

Verification requires the expected COMMIT, TREE, N, and K again rather than
trusting values embedded in the report:

```bash
python -m tools.research.production_resume_equivalence verify \
  --repo-root . \
  --report /tmp/geno-lewm-resume-evidence/production_resume_equivalence.json \
  --expected-source-commit "$COMMIT" --expected-source-tree "$TREE" \
  --total-steps 100 --split-step 40
```

The verifier additionally rejects any dirty path in its checkout; rehashes every
raw artifact; safely reloads every checkpoint; checks the exact source,
identities, N/K progress, prefix metrics, and cursor; and independently compares
final model, action, AdamW, trainer-monitor, RNG, metric-history, and order
digests.

## Claim boundary

A passing report establishes bit-equal, single-process production continuation
for that exact software fixture and source identity. It does **not** establish
accelerator or distributed equivalence, hardware-independent floating-point
behavior, model quality, biological utility, or clinical validity.
