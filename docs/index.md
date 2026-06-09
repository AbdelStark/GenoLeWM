# GenoLeWM

**Action-conditioned latent world models for genomic edits.**

GenoLeWM treats a genomic edit as an action. A frozen DNA encoder maps a
reference window to a latent state, and a trainable predictor estimates
the post-edit latent state from that state plus the structured edit.

The package is alpha research software. It is installable, typed,
tested, benchmarked, and published with public artifacts; the current
model-quality evidence is mixed or negative versus Carbon.

## Start Here

- [Quickstart](quickstart.md)
- [Architecture](spec/01-architecture.md)
- [API reference](reference/index.md)
- [Implementation tracker](roadmap/IMPLEMENTATION.md)
- [Roadmap](https://github.com/AbdelStark/GenoLeWM/blob/main/ROADMAP.md)

## Public Artifacts

| Artifact | Link |
| --- | --- |
| PyPI package | <https://pypi.org/project/geno-lewm/0.2.1/> |
| Source/wheel release | <https://github.com/AbdelStark/GenoLeWM/releases/tag/v0.2.1> |
| Model package | <https://huggingface.co/abdelstark/geno-lewm> |
| Dataset package | <https://huggingface.co/datasets/abdelstark/geno-lewm-data> |
| Benchmark/planning/paper tree | <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1> |
| Generated paper | <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md> |

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

- No clinical utility claim.
- No broad model-quality claim.
- No K=20 rollout-speed closure.
- No useful-planning claim from the current planning demo.
- No runtime or privacy assurance beyond local execution contracts and
  checksum provenance.
