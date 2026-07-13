# GenoLeWM

**Action-conditioned latent world models for genomic edits.**

GenoLeWM treats a genomic edit as an action. A frozen DNA encoder maps a
reference window to a latent state, and a trainable predictor estimates
the post-edit latent state from that state plus the structured edit.

The package is alpha research software. It is installable, typed, tested,
benchmarked, and published with public artifacts.

> **Checkpoint validity notice (2026-07-10):** Every published v0.1 and v0.2.1
> checkpoint uses `legacy_raw_v1`. `encoder.normalize: true` was not applied,
> so those runs trained and evaluated on raw pooled Carbon states. Training
> sources used global pooling while targets were edit-centered. Historical
> centered pools were shifted one hidden token left because the leading `<dna>`
> token was omitted from the token offset; rollout and candidate centers also
> differed, and cache v1 omitted the center identity. The pinned Carbon
> `tokenizer.py` also performed an unpinned, network-capable
> `Qwen/Qwen3-4B-Base` lookup, so the prior runtime was not self-contained. In the
> v0.2.1 Phase 2 run, KL was computed from frozen target states and supplied
> no gradient to the trainable parameters. Published metrics are historical
> legacy-implementation outputs. These defects invalidate the checkpoints and
> metrics as evidence for the intended normalized method. Corrected results
> require a new `l2_normalized_v2` lineage. The local pure-DNA tokenizer and
> token-layout-aware centering repairs establish code contracts only, not model
> quality.

## Start Here

- [Quickstart](quickstart.md)
- [Architecture](architecture.md)
- [v0.3 snapshot-lineage assembly](data-v03-snapshot-lineage.md)
- [Public API contract](api/public-surface.md)
- [v0.3 membership-store contract](data-v03-membership-store.md)
- [v0.3 membership split evidence](data-v03-membership-splits.md)
- [API reference](reference/index.md)
- [GenoLeWM-FX contract](research/fx-experiment-contract.md)
- [GenoLeWM-FX feasibility report](research/fx-feasibility-report.md)
- [GenoLeWM-FX Borzoi rescue plan](research/fx-borzoi-rescue-plan.md)
- [GenoLeWM-FX Borzoi alignment report](research/fx-borzoi-overlap-report.md)
- [GenoLeWM-FX Borzoi cache report](research/fx-borzoi-cache-report.md)
- [GenoLeWM-FX Borzoi baseline report](research/fx-borzoi-baseline-report.md)
- [GenoLeWM-FX Borzoi residual report](research/fx-borzoi-residual-report.md)
- [GenoLeWM-FX Borzoi final report](research/fx-borzoi-final-report.md)

## Public Artifacts

| Artifact | Link |
| --- | --- |
| PyPI package | <https://pypi.org/project/geno-lewm/0.2.1/> |
| Source/wheel release | <https://github.com/AbdelStark/GenoLeWM/releases/tag/v0.2.1> |
| Model package | <https://huggingface.co/abdelstark/geno-lewm> |
| Dataset package | <https://huggingface.co/datasets/abdelstark/geno-lewm-data> |
| Verified v0.3 variant-membership candidate | [Exact Hub commit `96e97a7f…`](https://huggingface.co/datasets/abdelstark/geno-lewm-data/tree/96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2/candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/geno-lewm-v03-membership-fd7f4bbde476-r1/success) |
| Verified v0.3 membership split evidence | [Exact Hub commit `6d2ec7dd…`](https://huggingface.co/datasets/abdelstark/geno-lewm-data/tree/6d2ec7dd68af636ba8c594774c3c55a236c0995f/candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/geno-lewm-v03-membership-fd7f4bbde476-r1/splits/geno-lewm-v03-membership-splits-bb24f6344274-r2/success) |
| Historical v0.2.1 benchmark/planning/paper tree | <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1> |
| Historical v0.2.1 generated paper | <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md> |

## Core Commands

```bash
python -m pip install geno-lewm
geno-lewm-verify examples/data/verify_receipt/receipt.json \
  --manifest examples/data/verify_receipt/manifest.json
geno-lewm-train --fixture-smoke --run-dir /tmp/geno-lewm-smoke --steps 50
```

Use `geno-lewm[eval]` for FASTA/VCF evaluation and `geno-lewm[train]`
for Carbon-backed training paths.

## Current Boundaries

- No corrected normalized-method result is published. Existing checkpoints
  and metrics are `legacy_raw_v1` historical outputs.
- The v0.2.1 Phase 2 KL changed the reported scalar loss but had no gradient
  with respect to the optimized predictor and action encoder.
- No clinical utility claim.
- No broad model-quality claim.
- No K=20 rollout-speed closure.
- No useful-planning claim from the current planning demo.
- A real checksum-closed v0.3 variant-membership candidate is published and
  independently verified at exact Hub commit
  `96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2`. It is not a released v0.3
  snapshot or phased-haplotype holdout. Its placed-window and held-role split
  evidence is separately published and independently verified at exact Hub
  commit `6d2ec7dd68af636ba8c594774c3c55a236c0995f`. The canonical schema-`1.1.0`
  assembler is implemented locally, but no assembled snapshot candidate is
  published yet.
- No GenoLeWM-FX model or demo ships; the FX pivot is stopped at the
  feasibility gate.
- The FX precomputed-Borzoi rescue is complete as a no-positive-claim
  result: the residual lift is small and non-significant, no
  model-quality claim is open, and exact fipip overlap is not claimed.
- No runtime or privacy assurance beyond local execution contracts and
  checksum provenance.
