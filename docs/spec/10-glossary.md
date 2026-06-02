# 10 — Glossary

- Status: Authoritative for v0.1
- Companion file: [`docs/glossary.md`](../glossary.md) (synonymous; kept
  for the existing docs convention).

This is the canonical term registry across the spec corpus, RFCs, and
implementation. Any term that appears in more than one document belongs
here. When the spec and the glossary disagree, the spec wins; please open
a PR fixing the glossary.

## ML / architecture

**Action.** A structured genetic edit (SNV, INS, DEL, MNV, INDEL, SV)
passed to the predictor as a first-class input. Canonical type:
`EditSpec` (RFC-0003).

**Action embedding.** Fixed-size vector representation of an action,
produced by the action encoder. Dimension `d_action = 512` in v1.

**Action encoder.** Trainable module mapping `EditSpec` (or `RelEdit`) to
an action embedding. See RFC-0003.

**Autoregressive predictor (`ARPredictor`).** Wrapper around the
predictor that unrolls it step-by-step over a sequence of actions, with
KV caching, for multi-edit haplotype rollout and planning. See RFC-0004
§3.3.

**Carbon.** DNA foundation model family released by HuggingFaceBio
(500M / 3B / 8B). Used as GenoLeWM's state encoder. See RFC-0002.

**Cross-attention predictor.** Default predictor architecture: 4
cross-attention blocks (state ↔ action) plus 2 self-attention blocks on
the fused sequence. See RFC-0004 §3.1.

**JEPA.** Joint-Embedding Predictive Architecture. A class of models
that predict in representation space rather than input space.

**LeJEPA regularizer.** Isotropic-Gaussian regularizer on encoder output
distributions that prevents collapse. Live in Phase 2 only. See RFC-0005
§3.2.

**LeWorldModel (LeWM).** Reference architecture and training recipe
(Maes et al., 2026) for stable end-to-end JEPA world models. Basis for
GenoLeWM's training recipe.

**Predictor.** Trainable module mapping `(state, action)` to predicted
next-state in latent space. See RFC-0004.

**State.** Vector embedding of a contiguous DNA window produced by the
state encoder (Carbon). Dimension `d_state = 1024` for Carbon-500M.

**State encoder.** Carbon (frozen by default in v1). See RFC-0002.

**Surprise.** Predictor's residual: distance between predicted post-edit
latent and the encoder's actual post-edit latent. Calibrated to a
per-context percentile to produce a pathogenicity score. See RFC-0009.

## Genomics

**bp.** Base pairs. Unit of DNA length.

**ClinVar.** NCBI's public database of human variants with clinical
significance labels.

**Coding / non-coding.** Regions of DNA that do (coding) or do not
(non-coding) translate to protein.

**Edit / variant.** Change to a reference DNA sequence. Used
interchangeably; "edit" is the action-conditioned framing, "variant" is
the genomics framing.

**EditSpec.** Canonical edit type: `(chrom, pos, ref, alt, edit_type)`.
See RFC-0003.

**EditType.** Enumeration: `{SNV, INS, DEL, MNV, INDEL, SV}`.

**FASTA.** Standard text format for nucleotide sequences. Indexed via
`.fai` files (samtools / pysam convention).

**gnomAD.** Genome Aggregation Database. Reference catalog of human
genetic variation, used both for biological-prior edit sampling during
training and for the surprise calibration distribution.

**Haplotype.** Coordinated set of variants on the same chromosome copy.
In GenoLeWM, a multi-edit input to the predictor.

**Indel.** Insertion or deletion of one or more bases.

**MNV.** Multi-nucleotide variant: substitution of equal-length ref and
alt, both > 1 bp.

**Pathogenic / Likely pathogenic (P/LP).** ClinVar classes for variants
asserted to cause disease (P) or strongly suspected to (LP).

**RelEdit.** Window-relative form of an `EditSpec`, with `rel_pos` as
the offset within the window in bp. See RFC-0003 §3.3.

**SNV.** Single-nucleotide variant: substitution of one base for another.

**SV.** Structural variant: large-scale rearrangement (inversion,
duplication, large indel, translocation). Not supported in v1; deferred
to v2.

**TraitGym.** Curated benchmark of trait-associated variants for
variant-effect prediction.

**VCF.** Variant Call Format. Standard file format for representing
genetic variants.

**Window.** Contiguous DNA region passed to the state encoder. Default
length 12,288 bp (2,048 Carbon 6-mer tokens).

## Operations

**Calibration table.** Per-context empirical CDFs of `σ_raw` over gnomAD
common variants, intended to ship with a released GenoLeWM checkpoint.
No public calibration artifact is released yet. See RFC-0009 §3.4.

**Cache (window cache).** On-disk Parquet store of pre-computed
reference-window embeddings. Content-addressed by
`(window_hash, encoder_hash, state_layer, pool_type, pool_radius, dtype)`.
See RFC-0002 §3.6.

**Encoder hash.** SHA-256 of the encoder weights file. Part of every
cache key and checksum receipt.

**Holdout.** Set of data excluded from training and reserved for
evaluation. Three holdouts: `holdout-chr` (chr21), `holdout-clinvar`,
`holdout-haplotypes`. See RFC-0006 §3.8.

**Manifest.** Canonical JSON document describing a GenoLeWM checkpoint's
identity, configuration, and provenance. Its hash is the `model_id`.
See RFC-0011 §3.7.

**Receipt.** JSON document that release scoring paths can emit to bind
model identity, input commitment, and output commitment. See RFC-0011.

**Tuple builder.** Data-pipeline component that produces `(w_ref, a, w_alt)`
training tuples. See RFC-0006 §3.5.

## Artifact provenance

**Checksum receipt.** Receipt mode currently supported by GenoLeWM. It
binds the model manifest identity, input commitment, output commitment,
and runtime metadata; it is not a model-quality or runtime-assurance
guarantee.

**Content addressing.** Identifying data (weights, inputs, outputs) by
the cryptographic hash of their canonical serialization, not by name or
location.

**Input commitment.** SHA-256 hash of the canonical serialization of an
inference's inputs.

**`model_id`.** SHA-256 of a GenoLeWM checkpoint's manifest. Globally
identifies a specific release once public checkpoint artifacts exist.

## Project & process

Current public status: alpha implementation in Phase 1. Phase names below
describe the roadmap, not completed release evidence.

**Phase 0 (Design, complete).** Spec and RFC bootstrap.

**Phase 1 (MVP).** Carbon-500M frozen, SNVs only, ClinVar coding/non-
coding eval. See ROADMAP.

**Phase 2 (Full edits).** Adds indels, MNVs, LoRA-adapted encoder,
LeJEPA regularizer, planning, calibrated surprise. Full eval suite.

**Phase 3 (On-device).** Export pipeline, quantization, desktop app
skeleton.

**RFC.** Request for Comments. A numbered design document under
[`rfcs/`](../rfcs/). The source of truth for individual design
decisions.

**Tracking issue.** A GitHub issue tagged `type:tracking` that lists
the implementation issues for a subsystem with a checkbox per child.

## Conventions

- RFC numbers are zero-padded to four digits (`RFC-0001`).
- File names are kebab-case after the number (`0001-project-scope-and-goals.md`).
- Mathematical notation: lowercase italic for scalars (`s`, `a`),
  lowercase plain for vectors (`s_t`, `a_emb`), uppercase for sets /
  matrices.
- `→` denotes function mapping; `≜` denotes definition.
- Hashes are SHA-256 unless otherwise noted; rendered in lowercase hex.
