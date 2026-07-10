# GenoLeWM paper

**GenoLeWM: An Action-Conditioned Latent World Model for Genomic Edits. Post-Release Validity Correction for v0.2.1-r1**

A corrected preprint covering the GenoLeWM `v0.2.1-r1` release: an action-conditioned
Joint-Embedding Predictive Architecture with a frozen Carbon-500M encoder, cross-attention predictor,
surprise scoring, CEM planning, and a content-addressed train→eval→benchmark→replay pipeline.

## Post-release validity correction

The original negative-result interpretation is withdrawn. The released configuration declared
`normalize: true`, but encoder source and target states were not normalized while predictor outputs
were unit norm. Mean source/target/prediction norms were:

| Rollout slice | Source | Target | Prediction |
| --- | ---: | ---: | ---: |
| Phased haplotypes | 33.982 | 33.595 | 1.000 |
| Synthetic edit chains | 29.253 | 29.089 | 1.000 |

Consequences:

- Released L2 loss, surprise/VEP scores, rollout L2, and planning-distance interpretations are invalid
  for the intended normalized-state method.
- Every historical centered pool omitted Carbon's leading `<dna>` control token
  from the coordinate conversion, shifting the intended center one hidden token
  left and sometimes centering on the control token.
- The pinned Carbon `tokenizer.py` performed an unpinned, network-capable
  `Qwen/Qwen3-4B-Base` load. Its local hash therefore did not define a
  self-contained runtime identity.
- Released cosine values remain historical implementation measurements because cosine is
  scale-invariant, but they are confounded by the invalid training objective and train/rollout
  distribution mismatch. They do not support a capability or mechanistic conclusion.
- The `v0.2.1` run used seed `271828`, 10,000 optimizer steps, and 80,000 samples. It was configured as
  `phase2`, but had no LoRA; its KL was computed from frozen targets and had no gradient path to
  trainable parameters.
- The latent-residual-trap attribution and all released model-capability conclusions are withdrawn.
- A corrected end-to-end experiment must publish a new checkpoint, manifests, reports, and receipt
  graph under a new run identity. The `v0.2.1-r1` identity remains an immutable historical record.
- The corrected source identity includes the local pure-DNA tokenizer
  implementation and validates the observed DNA token layout. Those repairs do
  not constitute model-quality evidence.

## Build

```bash
make            # tectonic, two passes -> main.pdf
# or directly:
tectonic --keep-intermediates main.tex && tectonic --keep-intermediates main.tex

# release evidence report:
python -m tools.release.paper_tex --paper-dir paper --output paper/paper_tex_build_report.json
```

Requires [`tectonic`](https://tectonic-typesetting.github.io/) (self-contained; downloads TeX packages
on first run). The NeurIPS 2025 preprint style (`neurips.sty`) is vendored.

## Files

| File | Contents |
| --- | --- |
| `main.tex` | manuscript body + preamble + notation macros |
| `figures.tex` | 5 figures (released computation, historical VEP, historical rollout cosine, efficiency regimes, AR speedup); TikZ/pgfplots |
| `tables.tex` | result tables (VEP, rollout fidelity, efficiency, artifact identity) |
| `refs.bib` | 18 programmatically verified references |
| `neurips.sty` | vendored conference style |
| `OUTLINE.md` | corrected claim boundary and manuscript outline (working doc) |
| `EVIDENCE_DOSSIER.md` | consolidated subsystem evidence + verified citations (working doc) |

## Provenance of numbers

Historical benchmark values derive from the published `geno-lewm-v0.2.1-r1` readiness report and model
card (content-addressed; `model_id sha256:cddb8f3b…`, `commit d9b06815…`, NVIDIA H200). Carbon zero-shot
columns are recovered as `GenoLeWM − Δ`; rollout source-state baselines as `cosine + |Δ|`. The state-norm
values derive from the post-release audit of the published rollout artifacts.
Content addressing proves which included artifacts are attached to those values;
it does not bind the historical unpinned transitive tokenizer or repair the
normalization and token-coordinate semantics.

## Scope

This corrected manuscript is an audit record, not a valid negative-result paper. It makes no clinical,
privacy, efficiency, world-model-capability, variant-effect, planning, or mechanistic claim from
`v0.2.1-r1`. Scientific interpretation is deferred until a corrected run with a new identity exists.
