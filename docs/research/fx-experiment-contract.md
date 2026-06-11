# GenoLeWM-FX experiment contract

Status: locked for issue #258. Parent epic: #257.

GenoLeWM-FX is a kill-gated research pivot. It is not a roadmap promise
and it is not part of the stable public API. The experiment only
continues if a cheap, reproducible probe shows that edit-conditioned
functional residuals are measurable and not explained by simple
baselines.

## Hypothesis

GenoLeWM-FX should model functional transitions caused by local genomic
edits. The target is not a full pooled Carbon state. The target is a
teacher-derived residual between reference and edited sequence states:

- input: reference sequence context, edit/action, source/task metadata,
  and optional teacher reference summary;
- target: signed and normalized functional delta between reference and
  edited state;
- objective: predict residuals beyond zero-delta, source-only, Carbon,
  direct-teacher, and linear-probe baselines;
- success: a locked held-out benchmark improves over the strongest
  simple baseline with confidence intervals and leakage checks;
- kill: the residual signal is absent, not reproducible, saturated by
  simple baselines, or too weak to justify larger jobs.

## Target Scope

In scope:

- regulatory and functional variant-effect prediction;
- TraitGym-style causal regulatory variant tasks;
- MAVE, MPRA, and fine-mapped eQTL/QTL slices when the assay, genome
  build, windowing, and license are locked;
- teacher-derived pseudo-labels when they are explicitly labeled as
  teacher outputs.

Out of scope:

- clinical utility or diagnostic interpretation;
- deployment readiness;
- broad variant-effect prediction superiority;
- privacy assurance;
- runtime assurance;
- broad biological validity beyond the selected teacher/task surface.

## Candidate Sources

The initial source candidates are:

- TraitGym Mendelian and complex-trait regulatory variant tasks;
- MaveDB or MPRA-style mapped assays with explicit genome build and
  allele mapping;
- fine-mapped eQTL/QTL slices when data access permits reproducible
  redistribution or manifest-backed download;
- synthetic controls only for software checks, never for a go decision.

Every source must ship a deterministic manifest with source URL, version
or revision, license note, genome build, split assignment, variant
identity, source mix, and checksums for any materialized cache.

## Candidate Teachers

The initial teacher candidates are Borzoi, Enformer, ChromBPNet, and
AlphaGenome. Teacher outputs are not ground truth. They are
teacher-derived targets and must remain described that way in public
text.

AlphaGenome is allowed only as a bounded sampled calibration/API
baseline unless the terms, rate limits, model weights, hardware receipt,
and artifact layout make larger use reproducible. The API path must not
be used to create million-scale training data without a separate
documented terms/rate review.

## Required Baselines

Every feasibility or benchmark report must include:

- zero-delta/no-edit baseline;
- source-only or label-prior baseline;
- Carbon likelihood/log-odds where applicable;
- direct teacher ref-alt score where applicable;
- linear or logistic probe on teacher features when teacher features are
  available;
- feasible public model predictions when they are relevant to the
  selected task.

A GenoLeWM-FX result cannot be considered positive if it only beats a
weak baseline while losing to a simple source-only, direct-teacher, or
linear-probe baseline.

## Splits And Leakage

The cheap probe and locked benchmark must use held-out identity rules
before model fitting:

- chromosome holdout when enough variants remain;
- trait, assay, or gene-family holdout when chromosome holdout would
  collapse the task;
- no variant identity overlap between teacher-cache training and final
  eval;
- no duplicated ref/alt examples across splits;
- no overlapping genomic windows across splits unless the overlap is
  explicitly audited and excluded from positive claims;
- no post hoc slice selection after the locked manifest is created.

Reports must include split identity, source mix, duplicate counts,
genomic-overlap checks, and leakage decisions.

## Metrics

Binary tasks must report AUROC, AUPRC, balanced accuracy, prevalence,
and bootstrap confidence intervals. Continuous assay or QTL tasks must
report Spearman, Pearson, residual magnitude summaries, and bootstrap
confidence intervals. Ranking-native tasks must report the benchmark's
native top-k or ranking metric plus a paired comparison against the
strongest simple baseline.

Teacher-delta reports must include signed delta, absolute delta,
rank-normalized delta, and a delta-magnitude distribution. If teacher
outputs barely move under edits, the experiment is killed before model
training.

## Go And No-Go Thresholds

The feasibility gate is a go only if all of these are true:

- a public or publishable 10k-50k example slice exists with ref/alt
  teacher deltas and checksums;
- each source has a deterministic split manifest and no known leakage;
- the teacher-delta target has nontrivial movement under edits;
- required zero-delta/source-only and linear/logistic-probe baselines are
  present;
- the selected primary metric is not saturated by the strongest simple
  baseline;
- a cheap source-only/probe run identifies one default downstream task
  and primary metric for later work.

The feasibility gate is a kill if any of these are true:

- no reproducible public teacher-delta slice exists;
- the teacher path requires private credentials or machine-local state
  that cannot be published as a manifest-backed artifact;
- teacher deltas are noisy, near-zero, or not label-relevant;
- zero-delta, source-only, direct-teacher, or linear-probe baselines
  saturate the task;
- leakage cannot be ruled out;
- the signal is too weak to justify Stage B or Stage C compute.

The final benchmark is positive only if GenoLeWM-FX improves over the
strongest simple baseline by a meaningful margin on a held-out task:

- binary primary metrics: AUROC and AUPRC improve by at least 0.03
  absolute, with paired bootstrap lower bound above zero;
- continuous primary metrics: Spearman or Pearson improves by at least
  0.05 absolute, with paired bootstrap lower bound above zero;
- ranking metrics: top-k or ranking improvement is visible under the
  benchmark-native metric and robust to bootstrap or seed variance;
- collapse diagnostics show predicted residual variance rather than
  no-edit behavior.

## Compute Stages

Stage A is the only stage allowed before the feasibility gate: a cheap
source/teacher probe and fixture software checks. Stage B may train on
100k-500k examples only after #259 records a go decision. Stage C may
scale to million-example or larger-context runs only after Stage B
clears its predefined metric and collapse gates.

No medium or expensive Hugging Face job should launch before the #259
go decision is recorded in a report.

## Publication Rule

Failure is publishable as a kill report. A negative outcome should keep
the manifests, commands, and baseline table visible. Public docs must
not turn a narrow positive result into clinical, deployment, broad VEP,
privacy, runtime, or broad Carbon-outperformance language.
