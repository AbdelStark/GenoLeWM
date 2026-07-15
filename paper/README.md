# GenoLeWM paper

**Registered but Not Anticipated: The Edit-Response Geometry of a Frozen Genomic Foundation Model**

A measurement paper with a negative headline. It characterises the *edit-response geometry* of frozen
DNA foundation models — the displacement `Δ = s(w_alt) − s(w_ref)` that a single-base edit induces in
a pooled state — and reports that the pathogenicity-bearing component of that response cannot be
anticipated from the pooled reference state plus the action. The withdrawn `v0.2.1-r1` release is one
section of this paper, not its thesis.

## The result

A frozen genomic FM's pooled state **registers** a single-base edit, but the registration **cannot be
anticipated**. There is no shortcut through the pooled latent: you must run the encoder on the edited
sequence.

Three findings, which dissociate — and the dissociation is the point:

1. **The predict-no-change trap is universal.** A single edit leaves `cos(s_ref, s_alt) ≥ 0.993` on
   every encoder at every pooling width, reaching `1.00000` for HyenaDNA under global pooling. An
   identity map therefore wins any cosine-supervised objective. Yet on Carbon the AUROC of `‖Δ‖` stays
   flat at 0.92–0.93 while relative displacement varies 9.4×. **Dilution is not information loss; the
   trap is the metric, not the geometry.**
2. **Informativeness is *not* universal — it is encoder-specific.** The identical measurement on the
   identical variants gives AUROC `0.9313` on Carbon-500M, `0.7574` on Nucleotide Transformer v2, and
   `0.5460` — near chance — on HyenaDNA-450k (global pooling). The *interventional* property is
   Carbon's: its edit beats a reference-only probe by `+0.1346` to `+0.3075`, while NT-v2's does not at
   ±48 bp (`−0.0298`: the unedited window predicts better than the edit response).
3. **The core negative result.** Predicting `Δ` from `(s_ref, action)` reaches AUROC `0.6520` against a
   `0.9259` ceiling, barely above the `0.6306` obtainable from the substitution class with no model at
   all. The pooled state is not merely uninformative but *harmful*: ridge scores `0.5854`, below the
   `0.6079` of a lookup table that ignores it. Predictability degrades as pooling widens (`0.5400` at
   global pooling) while the ceiling stays flat — the signature of an information bottleneck.

The geometry is **not** a restatement of likelihood surprise: on identical variants it beats
Δlog-likelihood (`0.9255` vs. `0.8945`), correlates with it only weakly (Spearman `−0.2391`), retains
`0.8529` after the likelihood is residualised out, and ensembles to `0.9482`.

## Scope — read this before quoting any AUROC

- **Every AUROC in the paper is cohort-dependent.** Carbon's own Δlog-likelihood reaches `0.8945` on
  our ClinVar slice against the `0.6`–`0.8` it attains in the published zero-shot DNA-LM literature.
  **Our slice is easy**, and that — not the merit of the geometry — is why the raw AUROCs exceed the
  literature's. None of them are comparable to a benchmark number.
- **This is not a variant-effect-prediction method** and must not be presented as one. Transfer to
  quantitative function is weak: Spearman `−0.265` on the BRCA2 saturation genome-editing assay
  (`−0.155` transferring the ClinVar-learned direction).
- **The claim is about *pooled* states only.** A token-level world model is not shown impossible; it is
  untested and is the natural follow-up.
- **No clinical, diagnostic, or decision-support claim of any kind.**

## Cohort and encoders

14,000 real GRCh38-grounded variants × 4 pool radii = 56,000 rows, 0 skips: ClinVar 1,572 pathogenic /
6,428 benign, plus 6,000 BRCA2 saturation genome-editing variants (Sahu et al. 2025, MaveDB
`urn:mavedb:00001242-a-1`). Headline analyses restrict to SNVs: **12,993 SNVs** (ClinVar 899 pathogenic
/ 6,094 benign), with 1,007 non-SNV rows excluded and counted. All 12,993 matched their recorded
reference base against the FASTA (12,993/12,993).

| Encoder | Objective / tokenizer | `d` | Role |
| --- | --- | ---: | --- |
| Carbon-500M (rev. `5d31d59b`, layer 20) | autoregressive, 6 bp | 1024 | primary; the only one whose response is interventional |
| Nucleotide Transformer v2 100M | masked LM, 6 bp | 512 | generality test (51,972 rows, 0 skips) |
| HyenaDNA-450k | implicit convolution, 1 bp (character) | 256 | generality test (51,972 rows, 0 skips) |

All three are frozen; all see the same 12,993 SNVs in 4,096 bp windows and are processed by
`tools/research/edit_response_analysis.py` *unchanged*. **Pool radius is counted in tokens, so every
cross-encoder comparison is reported in base pairs** — HyenaDNA is run at radii 0/48/384/1536 to match
the others' 0/8/64/256.

## Build

```bash
make            # tectonic, two passes -> main.pdf
# or directly:
tectonic --keep-intermediates main.tex && tectonic --keep-intermediates main.tex

# release evidence report:
python -m tools.release.paper_tex --paper-dir paper --output paper/paper_tex_build_report.json
```

Two passes are required so forward references to floats resolve. Requires
[`tectonic`](https://tectonic-typesetting.github.io/) (self-contained; downloads TeX packages on first
run). The NeurIPS 2025 preprint style (`neurips.sty`) is vendored.

## Files

| File | Contents |
| --- | --- |
| `main.tex` | manuscript body + preamble + notation macros |
| `figures.tex` | 6 figures (architecture, sensitivity ceiling, AUROC vs. controls, predictability, propagation, cross-encoder); TikZ/pgfplots |
| `tables.tex` | 9 tables (controls, trinucleotide control, predictability, scope/dLogLik/BRCA2, encoder trap, encoder informativeness, historical VEP, historical rollout, artifact identity) |
| `refs.bib` | 29 programmatically verified references |
| `neurips.sty` | vendored conference style |
| `EVIDENCE_DOSSIER.md` | consolidated subsystem evidence + verified citations (working doc) |

## Structure

1. **Introduction** — the interventional question; the world-model promise; the predict-no-change trap.
2. **Related Work** — latent world models over frozen features; JEPAs; genomic FMs; the Δlog-likelihood
   and embedding-distance VEP paradigms.
3. **Method** — frozen encoders; the window/pooling/normalization contracts; definitions; the
   predictability protocol and its baseline ladder.
4. **Edit-Response Spectroscopy of Carbon-500M** — the sensitivity ceiling, the interventional controls,
   and propagation.
5. **Generality: The Trap Is Universal, the Signal Is Not** — the three-encoder test.
6. **Predictability: The Core Negative Result** — the paper's headline.
7. **Baselines and Scope** — BRCA2, Δlog-likelihood, and the difficulty calibration of our cohort.
8. **Why the Released Run Failed, and Why a Correct One Would Have Too** — the `v0.2.1-r1` retraction,
   preserved and explained.
9. **Limitations**, **Conclusion**, and appendices (notation, artifact identity, citation caveats).

## The v0.2.1-r1 retraction, in one paragraph

The withdrawn release declared `normalize: true` while encoder states stayed raw and predictor outputs
were unit norm (source/target/prediction norms `33.982/33.595/1.000` phased, `29.253/29.089/1.000`
synthetic); source and target were pooled in different coordinate frames; the `phase2`-labelled KL was
computed from frozen targets and carried no gradient; and the pinned Carbon `tokenizer.py` performed an
unpinned, network-capable `Qwen/Qwen3-4B-Base` load. Those defects were real, and the released L2,
surprise, and rollout interpretations remain withdrawn. **But they are not why the idea could not
work.** A corrected rerun would have failed too: its objective was degenerate (the identity function
attains `cos = 0.99989`), and the conditioning information is absent from the pooled state. The
historical numbers are preserved in the paper's tables for auditability; the interpretations are not.

## Provenance of numbers

Every headline number comes from R1/R3/R5/R6 measurement runs on 14,000 GRCh38-grounded variants, with
data published at
[`abdelstark/geno-lewm-edit-response`](https://huggingface.co/datasets/abdelstark/geno-lewm-edit-response)
and code in `tools/research/` (`edit_response_spectroscopy.py`, `edit_response_analysis.py`,
`delta_predictability.py`). The Δlog-likelihood baseline is scored in **fp32**: computed in `bf16`,
90.7% of differences came back exactly zero over only 18 distinct values, because a 4,096 bp window's
log-likelihood has magnitude ≈5,000 where the `bf16` spacing *is* 16. The reported fp32 run yields 0
exact zeros and 9,115 distinct values. **Do not accumulate or difference log-likelihoods in `bf16`;** no
number from the `bf16` run is cited anywhere. Historical `v0.2.1-r1` values derive from the published
readiness report and model card (content-addressed). See the artifact-identity table for the full
provenance graph.
