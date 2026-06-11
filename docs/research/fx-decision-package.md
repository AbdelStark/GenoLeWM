# GenoLeWM-FX decision package

Status: final decision package for issues #257 and #265.

The GenoLeWM-FX pivot is stopped at the feasibility gate. No FX teacher
cache, residual JEPA model, Hugging Face sweep, locked benchmark, model
artifact release, paper update, or public demo ships from this
experiment.

Authoritative artifacts:

- [FX experiment contract](fx-experiment-contract.md)
- [FX feasibility and kill report](fx-feasibility-report.md)
- [Machine-readable feasibility report](fx-feasibility-report.json)
- [Borzoi rescue plan](fx-borzoi-rescue-plan.md)
- [Borzoi alignment and overlap report](fx-borzoi-overlap-report.md)

## Decision

The parent epic takes the kill path. The checked-in feasibility report
found a public 11,400-row TraitGym source slice suitable for a cheap
source-only probe, but no reproducible 10k-50k ref/alt teacher-delta
cache from AlphaGenome, Borzoi, Enformer, ChromBPNet, or another
functional teacher.

Continuing would require private credentials, heavyweight teacher setup,
new assay curation, or newly published teacher-cache artifacts before
the actual GenoLeWM-FX hypothesis could be tested. That does not justify
medium or expensive training jobs.

## Demo Decision

No FX demo ships. A demo would imply an available FX model or teacher
residual score that this experiment did not produce. Existing GenoLeWM
demos remain manifest-backed execution examples only and should not be
described as evidence of useful planning behavior.

## Paper Decision

No FX positive-result paper section is added. If referenced in future
paper-facing text, this path should be described as a killed feasibility
probe: public labels were available, but the required teacher-delta
target was not reproducible under the contract.

## Follow-Up Trajectory

The #257 kill decision remains correct for the teacher-inference path.
A separate follow-up trajectory can test whether public precomputed
Borzoi-derived scores rescue the idea without running a teacher. The
trajectory now has two explicit lanes:

- the default executable lane uses TraitGym's row-aligned
  `Borzoi_L2_L2.plus.all` score artifact for the matched complex-trait
  slice;
- the optional provenance lane joins against the large statgen/fipip
  Borzoi table only when that table is explicitly staged locally.

The follow-up must start with an alignment and artifact-receipt audit
against TraitGym identities, labels, splits, and the row-aligned Borzoi
score vector. It must stop quickly if there is not a reproducible,
public, checksum-backed 10k-50k matched slice. If the optional full
fipip table join is not run, reports must say that directly and must not
claim exact fipip table overlap.

That follow-up is tracked in #266 and documented in
[GenoLeWM-FX Borzoi rescue plan](fx-borzoi-rescue-plan.md).

The first follow-up gate now has a source-controlled outcome:
[GenoLeWM-FX Borzoi alignment and overlap report](fx-borzoi-overlap-report.md).
It records a go decision for the compact TraitGym-native row-aligned
Borzoi score path, with 11,400 usable rows and no #268 blockers. That is
an alignment/cache-feasibility result only; it is not a model-quality
claim and it does not claim exact fipip overlap because the full fipip
table was not staged for that run.

## Child Issue Resolution

| Issue | Resolution |
| ---: | --- |
| #258 | Completed by the source-controlled contract. |
| #259 | Completed as a kill decision at feasibility. |
| #260 | Not planned after the kill gate; no teacher cache should be built now. |
| #261 | Not planned after the kill gate; no residual model target is selected. |
| #262 | Not planned after the kill gate; no Stage B or Stage C job should launch. |
| #263 | Not planned because the locked benchmark depends on the killed path. |
| #264 | Completed by publishing the kill report instead of model artifacts. |
| #265 | Completed by this no-demo/no-paper-update decision package. |
