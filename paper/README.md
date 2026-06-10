# GenoLeWM paper

**GenoLeWM: An Action-Conditioned Latent World Model for Genomic Edits. A Reproducible Pipeline and an Honest Negative Result**

A comprehensive preprint covering the GenoLeWM `v0.2.1-r1` experiments: an action-conditioned
Joint-Embedding Predictive Architecture that freezes Carbon-500M as a state encoder and trains a small
cross-attention predictor to estimate post-edit DNA-window embeddings, with surprise scoring, a CEM
latent planner, and a content-addressed train→eval→benchmark→replay pipeline.

The central finding is **negative and honest**: the learned predictor does not beat the encoder's own
zero-shot baseline, and its multi-edit rollout is worse than a trivial "predict-no-change" baseline,
diagnosed as a *latent-residual baseline trap* structural to the frozen-encoder regime.

## Build

```bash
make            # tectonic, two passes -> main.pdf
# or directly:
tectonic --keep-intermediates main.tex && tectonic --keep-intermediates main.tex
```

Requires [`tectonic`](https://tectonic-typesetting.github.io/) (self-contained; downloads TeX packages
on first run). The NeurIPS 2025 preprint style (`neurips.sty`) is vendored.

## Files

| File | Contents |
| --- | --- |
| `main.tex` | manuscript body + preamble + notation macros |
| `figures.tex` | 5 figures (architecture, VEP-vs-Carbon, latent-residual trap, efficiency regimes, AR speedup); TikZ/pgfplots, data-exact |
| `tables.tex` | result tables (VEP, rollout fidelity, efficiency, artifact identity) |
| `refs.bib` | 16 programmatically verified references |
| `neurips.sty` | vendored conference style |
| `OUTLINE.md` | locked outline / claim→evidence map (working doc) |
| `EVIDENCE_DOSSIER.md` | consolidated subsystem evidence + verified citations (working doc) |

## Provenance of numbers

Every reported value derives from the published `geno-lewm-v0.2.1-r1` benchmark readiness report and
model card (content-addressed; `model_id sha256:cddb8f3b…`, `commit d9b06815…`, NVIDIA H200). Carbon
zero-shot columns are recovered as `GenoLeWM − Δ`; rollout source-state baselines as `cosine + |Δ|`.

## Scope

Intended for arXiv (cs.LG + q-bio.GN) and a negative-results / reproducibility venue
(e.g. ICBINB, NeurIPS Datasets & Benchmarks). Not a clinical, privacy, or efficiency claim.
