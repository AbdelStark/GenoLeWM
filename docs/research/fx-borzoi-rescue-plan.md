# GenoLeWM-FX Borzoi rescue plan

Status: follow-up trajectory #266 after issue #257.

This plan does not reverse the #257 kill decision. It defines one narrow
way to reopen the FX question without running expensive teacher
inference: use public precomputed Borzoi-derived scores as the
teacher-derived functional substrate, then test whether enough public
TraitGym variants can be aligned to justify the residual-model path.

## Source Fact Pattern

The statgen/fipip repository documents precomputed Borzoi scores for
more than 19 million common and low-frequency variants. Those scores are
based on hg19 and include both variant-effect predictions and principal
components derived from those VEPs. The public score table is available
through Google Cloud Storage under
`seqnn-share/sniff/borzoi_102_annotation_set/`, with the main compressed
annotation table published as `sniff_102_annotations.gz`.

The practical implementation finding is narrower than the original
overlap wording. The full fipip table is public but large enough that a
first-pass rescue should not pretend to have staged or joined it unless a
local copy is explicitly provided. TraitGym also publishes row-aligned
precomputed Borzoi artifacts for its matched slices, including the
compact `complex_traits_matched_9/preds/all/Borzoi_L2_L2.plus.all.parquet`
score vector. That artifact is the default executable substrate for the
rescue trajectory. A full fipip table exact join remains an optional
provenance and coordinate-integrity audit, not a prerequisite for the
first cache/baseline gate.

This changes the blocker from "run a teacher" to "prove row alignment,
source identity, score semantics, split integrity, and reproducibility."
If the TraitGym-native Borzoi artifact is missing, row counts do not
match, labels/splits are unusable, the exact fipip join contradicts the
row-aligned substrate when run, or the score columns do not support the
FX target, the kill decision remains in force.

Relevant public sources:

- fipip precomputed Borzoi score path:
  <https://github.com/statgen/fipip>
- fipip public score prefix:
  <https://console.cloud.google.com/storage/browser/seqnn-share/sniff/borzoi_102_annotation_set>
- Borzoi model repository:
  <https://github.com/calico/borzoi>
- TraitGym regulatory variant benchmark:
  <https://github.com/songlab-cal/TraitGym>

## Hypothesis Adjustment

The original #257 hypothesis required ref/alt teacher deltas generated
or cached for a public 10k-50k slice. The rescue hypothesis is narrower:

- input: TraitGym variant identity, local edit/action metadata, source
  metadata, and optional GenoLeWM/Carbon features;
- primary target: TraitGym-native, row-aligned precomputed Borzoi score
  artifacts matched to the public TraitGym slice by row identity and
  validated against the slice length, labels, split policy, and artifact
  receipts;
- optional audit target: fipip Borzoi VEP or Borzoi-PC score columns
  matched by normalized variant identity when the large score table is
  staged locally;
- objective: predict a residual over zero/source-only, Carbon, direct
  Borzoi score, and linear/logistic probe baselines;
- success: a locked overlap-backed benchmark shows a meaningful gain
  over the strongest simple baseline with no leakage;
- kill: overlap is too small, hg19/hg38 mapping is ambiguous, score
  semantics are unsuitable, baselines saturate, or residual signal is
  too weak.

This is still teacher-derived evidence. It must not be described as
ground truth, clinical evidence, deployment readiness, broad VEP
superiority, or proof of useful planning.

## Stage Gates

### Stage 0 - Contract Update

Lock the rescue-specific contract before building caches:

- exact source URLs and revisions;
- fipip/precomputed-score access path and TraitGym-native Borzoi
  artifact path;
- genome build and liftover rules for any exact fipip table join;
- variant key normalization for `chrom,pos,ref,alt`;
- selected Borzoi VEP/PC score columns;
- row-alignment requirements for TraitGym-native score vectors;
- minimum overlap threshold;
- leakage and split rules;
- claim boundaries.

### Stage 1 - Overlap Audit

Validate TraitGym variants against public precomputed Borzoi artifacts
without training a model. The default go threshold is:

- at least 10,000 matched variants in one primary public task slice,
  where "matched" means the public TraitGym slice and the row-aligned
  Borzoi score vector have identical row counts and stable artifact
  receipts;
- no unresolved row-order, duplicate-key, label, or split mismatch in
  the matched set;
- enough positives and negatives after the locked holdout rule;
- a publishable overlap manifest with checksums.

If a local fipip score table is staged, the audit should also join by
`chrom,pos,ref,alt` with explicit handling for allele flips, strand
issues, multi-allelic rows, and duplicates. If that exact join is not
run, the report must say so directly and must not claim exact fipip
table overlap.

If this fails, stop and publish the overlap no-go report.

### Stage 2 - Score Cache

Materialize only the matched precomputed score columns and metadata
needed for the experiment. The first cache should consume the compact
TraitGym-native Borzoi score vector, not the full fipip table. If a
later exact fipip join is run, the cache may add selected fipip-derived
VEP/PC columns behind the same manifest contract.

The cache must include source revision, checksum, genome-build handling
where applicable, matched/unmatched counts, split identity, and
redaction-safe commands.

### Stage 3 - Baseline Gate

Run source-only, label-prior, Carbon where applicable, direct
TraitGym-native Borzoi score, and linear/logistic probe baselines. The
path continues only if the task is not saturated and the Borzoi-derived
target has enough signal to make a residual model meaningful.

### Stage 4 - Residual Model And Locked Eval

Only after Stages 1-3 pass, train a minimal residual model and evaluate
against the locked benchmark. A positive result must beat the strongest
simple baseline, not merely the label prior.

## Issue Trajectory

Use #266 rather than reopening #257. The #257 artifacts remain the
historical kill report for the teacher-inference path. The follow-up
children are:

- #267 - lock rescue contract and coordinate rules;
- #268 - audit TraitGym-native Borzoi score-vector alignment, and record
  exact fipip join status separately;
- #269 - build the manifest-backed precomputed Borzoi score cache from
  the compact TraitGym-native score vector first;
- #270 - run the leakage-aware baseline and saturation gate against the
  direct TraitGym-native Borzoi score and simple probes;
- #271 - train a residual model only after the baseline gate;
- #272 - publish the locked result or overlap kill report.

#266 should be closed quickly if the overlap audit fails.
