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

Each checkpoint is written to a same-directory temporary file, flushed and
`fsync`ed, atomically replaced, and directory-`fsync`ed on POSIX. A production
resume restores and validates all continuation state before encoding the first
`K+1` batch:

- predictor and action-encoder parameters;
- AdamW moments, parameter groups, current LR, and base LR;
- the `TorchTrainer` collapse-monitor baseline, policy, thresholds, and alerts;
- Python, NumPy, Torch CPU, and every available Torch CUDA RNG state;
- exact dataset snapshot/manifest, encoder, config, seed, sample cursor, and
  consumed window order;
- cumulative step metrics and the LR schedule computed against the original
  `N`, not a shortened suffix horizon.

The loader uses PyTorch's weights-only path and rejects missing, extra,
corrupted, cross-source, cross-data, cross-config, cross-device, or wrong-horizon
state.

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

The verifier rejects a dirty checkout; rehashes every raw artifact; safely
reloads every checkpoint; checks the exact source, identities, N/K progress,
prefix metrics, and cursor; and independently compares final model, action,
AdamW, trainer-monitor, RNG, metric-history, and order digests.

## Claim boundary

A passing report establishes bit-equal, single-process production continuation
for that exact software fixture and source identity. It does **not** establish
accelerator or distributed equivalence, hardware-independent floating-point
behavior, model quality, biological utility, or clinical validity.
