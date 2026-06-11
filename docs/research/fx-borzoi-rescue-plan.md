# GenoLeWM-FX Borzoi rescue plan

Status: follow-up trajectory #266 after issue #257.

This plan does not reverse the #257 kill decision. It defines one narrow
way to reopen the FX question without running expensive teacher
inference: use precomputed Borzoi scores as the teacher-derived
functional substrate, then test whether enough public TraitGym variants
overlap to justify the residual-model path.

## Source Fact Pattern

The statgen/fipip repository documents precomputed Borzoi scores for
more than 19 million common and low-frequency variants. Those scores are
based on hg19 and include both variant-effect predictions and principal
components derived from those VEPs.

That changes the blocker from "run a teacher" to "prove overlap,
coordinate compatibility, score semantics, and reproducibility." If the
overlap is too small or the score columns do not support the FX target,
the kill decision remains in force.

Relevant public sources:

- fipip precomputed Borzoi score path:
  <https://github.com/statgen/fipip>
- Borzoi model repository:
  <https://github.com/calico/borzoi>
- TraitGym regulatory variant benchmark:
  <https://github.com/songlab-cal/TraitGym>

## Hypothesis Adjustment

The original #257 hypothesis required ref/alt teacher deltas generated
or cached for a public 10k-50k slice. The rescue hypothesis is narrower:

- input: TraitGym variant identity, local edit/action metadata, source
  metadata, and optional GenoLeWM/Carbon features;
- target: precomputed Borzoi VEP or Borzoi-PC score columns matched by
  normalized variant identity;
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
- fipip/precomputed-score access path;
- genome build and liftover rules;
- variant key normalization for `chrom,pos,ref,alt`;
- selected Borzoi VEP/PC score columns;
- minimum overlap threshold;
- leakage and split rules;
- claim boundaries.

### Stage 1 - Overlap Audit

Join TraitGym variants to precomputed Borzoi score identities without
training a model. The go threshold is:

- at least 10,000 matched variants in one primary public task slice;
- no unresolved ref/alt flips or build mismatches in the matched set;
- enough positives and negatives after the locked holdout rule;
- a publishable overlap manifest with checksums.

If this fails, stop and publish the overlap no-go report.

### Stage 2 - Score Cache

Materialize only the matched precomputed columns and metadata needed for
the experiment. The cache must include source revision, checksum,
genome-build handling, matched/unmatched counts, split identity, and
redaction-safe commands.

### Stage 3 - Baseline Gate

Run source-only, label-prior, Carbon where applicable, direct Borzoi
score, and linear/logistic probe baselines. The path continues only if
the task is not saturated and the Borzoi-derived target has enough
signal to make a residual model meaningful.

### Stage 4 - Residual Model And Locked Eval

Only after Stages 1-3 pass, train a minimal residual model and evaluate
against the locked benchmark. A positive result must beat the strongest
simple baseline, not merely the label prior.

## Issue Trajectory

Use #266 rather than reopening #257. The #257 artifacts remain the
historical kill report for the teacher-inference path. The follow-up
children are:

- #267 - lock rescue contract and coordinate rules;
- #268 - audit TraitGym coverage by precomputed Borzoi scores;
- #269 - build the manifest-backed precomputed Borzoi score cache;
- #270 - run the leakage-aware baseline and saturation gate;
- #271 - train a residual model only after the baseline gate;
- #272 - publish the locked result or overlap kill report.

#266 should be closed quickly if the overlap audit fails.
