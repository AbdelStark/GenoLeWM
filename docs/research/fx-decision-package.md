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
