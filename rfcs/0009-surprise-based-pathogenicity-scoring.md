# RFC-0009: Surprise-based pathogenicity scoring

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-06-11
- **Depends on:** RFC-0002, RFC-0003, RFC-0004, RFC-0005, RFC-0007
- **Supersedes:** —
- **Implementation status:** Partial — context stratification,
  calibration table helpers, raw surprise scoring, local `score_variant`
  / `score_vcf`, `geno-lewm-score`, manifest-backed runtime loading,
  single-variant plus VCF receipt emission, public model/data artifact
  scoring evidence, and released terminal-demo replay evidence exist.
  Broader scoring quality, population-stratified calibration, and new
  runtime surfaces still need artifact-backed validation.

---

## 1. Summary

The predictor's residual error on a variant — the gap between what it
predicts the post-edit latent will be and what the encoder actually
produces — is a real-valued signal we call **surprise**. This RFC
specifies how surprise is computed, how it is calibrated to be
comparable across genomic contexts, and how it is used as an
unsupervised pathogenicity score. The output is a per-variant percentile
in a context-aware empirical surprise distribution, plus a confidence
indicator that flags when the calibration itself is unreliable.

## 2. Motivation

Existing pathogenicity scorers (CADD, REVEL, AlphaMissense, ESM-1b, the
Carbon variant-effect head) require either labeled training data or a
specific scoring rule baked into the encoder. Surprise is different: it
is a side effect of having trained the predictor. No labels are needed
to compute it.

The intuition: the predictor was trained on a mix of natural variants
(gnomAD common variants, biology's "background") and synthetic edits
(uniform coverage). It learns to predict the latent transitions that
this mixture induces. A variant that produces an *unusually large
predictor residual* is, by definition, one the model did not expect.
For common-variant residuals, the predictor is well-calibrated and
residuals are small. For variants that disrupt protein function in
unusual ways — exactly the variants ClinVar labels pathogenic — the
residuals are larger.

This is the same epistemic move that makes language-model perplexity a
proxy for "weirdness" in text, ported to a world-model latent space.
The difference: we measure the model's *predictive* surprise (over a
transition), not its *likelihood* surprise (over a single observation).
The predictive variant is what makes the score discriminative for
edits, which is what we care about.

## 3. Specification

### 3.1 Raw surprise

For a variant `v` applied to a reference window `w_ref`:

```
σ_raw(v) = ||g(s_t, a_v) − enc(apply(v, w_ref))||₂
```

where:
- `s_t = enc(w_ref)` is the reference state.
- `a_v` is the action embedding of `v` (RFC-0003).
- `g` is the predictor (RFC-0004).
- `enc` is the Carbon-based state encoder (RFC-0002).
- `||·||₂` is L2 distance in normalized latent space.

`σ_raw` is non-negative, unbounded above, and depends on:

1. The variant's "weirdness" (the signal we want).
2. The genomic context (the noise we need to factor out).
3. The encoder's own variance in that region (an additional noise).

The next sections deal with (2) and (3).

### 3.2 Why raw surprise alone is not enough

The raw residual distribution depends strongly on context. In
GC-rich regions, in repeats, in poorly-conserved non-coding sequence,
both Carbon and the predictor have more uncertainty, and σ_raw is
systematically larger — *regardless of variant pathogenicity*. If we
ranked variants by raw surprise across the whole genome, the top
percentiles would be dominated by variants in noisy regions, not by
pathogenic variants.

This is the standard problem in evolutionary-conservation pathogenicity
scores (CADD has the same issue) and the same solution applies:
**calibrate within strata of comparable context.**

### 3.3 Context stratification

Each genomic position is assigned a context label drawn from the
Cartesian product:

```
context = (region_class, gc_bin, repeat_class)
```

| Factor | Values |
|--------|--------|
| `region_class` | `{coding_synonymous, coding_missense, coding_nonsense, splice, utr5, utr3, intron, promoter, enhancer, intergenic, other}` |
| `gc_bin` | `{low, mid, high}` (terciles of windowed GC%) |
| `repeat_class` | `{none, simple, low_complexity, transposon, segmental_dup}` |

There are nominally 11 × 3 × 5 = 165 buckets, but many are empty (e.g.,
`splice` ∩ `transposon` is rare). Empty or under-populated buckets
collapse to the next-coarser bucket along a fixed back-off order:
`(region_class, gc_bin, repeat_class)` → `(region_class, gc_bin)` →
`(region_class)` → `(*)`.

The stable bucket ID is the ASCII pipe-joined full context:
`{region_class}|{gc_bin}|{repeat_class}`. Parent bucket IDs omit
rightmost factors (`{region_class}|{gc_bin}`, then `{region_class}`),
and the catch-all bucket is `*`. At calibration/scoring time, sparse
buckets are resolved by selecting the first bucket in this chain with
enough calibration rows; if all specific parents are sparse, the
catch-all bucket is used and confidence reflects its row count.

A bucket is considered "well-populated" if it contains ≥ 1,000 reference
calibration variants (§3.4).

### 3.4 Calibration distribution

For each bucket, we build a calibration distribution by:

1. Sampling ~10,000 gnomAD common variants (AF ≥ 1%) per bucket.
2. Computing `σ_raw` for each.
3. Storing the resulting empirical CDF as a 1,001-point grid (0.0 to
   1.0 in 0.001 increments).

The result is an array `F_bucket[bucket_id] → CDF`. The full
calibration table is computed once, distributed with the GenoLeWM
checkpoint as `calibration.parquet` (~2 MB), and loaded at inference.

The table builder takes pre-scored common-variant rows
(`bucket_id`, `sigma_raw`) as input. Computing `σ_raw` belongs to the
scorer path (§3.1, §3.10); the builder owns deterministic sampling,
parent-bucket aggregation, empirical CDF construction, sparse-bucket
warnings, and Parquet schema validation.

This is built **on the holdout-clinvar-free** set: the gnomAD common
variants used for calibration are drawn so that none of them
overlap with the ClinVar evaluation set. Otherwise we would be
data-leaking the eval.

### 3.5 Calibrated surprise

The calibrated surprise is the percentile of `σ_raw` in its bucket:

```
σ(v) = F_{bucket(v)}(σ_raw(v))    ∈ [0, 1]
```

Values near 1.0 mean the variant's residual is unusually large
*relative to common variants in the same context*. Values near 0.0
mean the variant looks typical.

This is the score we report. It is unitless, bounded, and comparable
across contexts.

### 3.6 Confidence indicator

For each variant, we also report a confidence indicator:

```
conf(v) = min(N_bucket(v) / 1000, 1.0)
```

i.e., 1.0 when the bucket has ≥ 1,000 calibration variants, less when
the calibration is built on a smaller sample (after back-off). A
variant in a sparsely-populated bucket gets a calibrated surprise
score but with a low confidence value, which downstream consumers can
use to weight or filter.

Variants in buckets that even after back-off contain fewer than 100
calibration variants are flagged as `low_confidence=True` in the
output and are explicitly not recommended for downstream use.

### 3.7 Output schema

For each scored variant, the surprise scorer outputs:

```json
{
  "chrom": "17",
  "pos": 43071077,
  "ref": "C",
  "alt": "T",
  "sigma_raw": 0.347,
  "sigma_calibrated": 0.92,
  "bucket_id": "coding_missense|mid|none",
  "confidence": 1.0,
  "low_confidence": false,
  "model_version": "geno-lewm-v0.1.0-carbon-500m-r1"
}
```

### 3.8 Aggregation modes

For variants where multiple windows overlap the variant position
(RFC-0006 §3.2 specifies ~3× window coverage on average), surprise is
computed in each window independently and aggregated:

```
aggregation ∈ {mean, max, median}
```

Default: `mean`. The `max` aggregation is appropriate when sensitivity
to any signal is the priority (e.g., screening); `median` is robust to
a single anomalous window.

### 3.9 Comparison to Carbon's likelihood scoring

Carbon scores variants via `ΔlogLik(alt, ref)`. This is fundamentally
different from surprise:

- `ΔlogLik` measures how much *less likely* the alternative allele is
  under the autoregressive model. It depends on the local sequence's
  baseline likelihood.
- Surprise measures how much the *predicted post-edit latent* differs
  from the *actual post-edit latent*. It depends on whether the
  predictor has seen similar transitions.

Both are valid signals. They are not redundant; we expect them to be
correlated but to have different failure modes. Phase 2 will report
the Spearman correlation between calibrated surprise and Carbon-500M
`ΔlogLik` on a held-out variant set, both as a sanity check and to
allow ensembling.

### 3.10 Scorer API

```python
# geno_lewm.surprise.score

@dataclass
class SurpriseResult:
    sigma_raw: float
    sigma_calibrated: float
    bucket_id: str
    confidence: float
    low_confidence: bool

def score_variant(
    variant: EditSpec,
    encoder: CarbonStateEncoder,
    action_encoder: ActionEncoder,
    predictor: Predictor,
    calibration: CalibrationTable,
    aggregation: str = "mean",
) -> SurpriseResult: ...

def score_vcf(
    vcf_path: Path,
    encoder: CarbonStateEncoder,
    action_encoder: ActionEncoder,
    predictor: Predictor,
    calibration: CalibrationTable,
    output_path: Path,
    show_progress: bool = True,
) -> None: ...
```

The `score_vcf` entry point is the on-device CLI's primary workflow
(RFC-0010).

## 4. Rationale and alternatives

### 4.1 Why predictor residual and not encoder likelihood?

We chose predictor residual because it is the signal that the predictor
is uniquely positioned to provide. The encoder's likelihood (Carbon's
`ΔlogLik`) is already a public scoring head and we do not want to
re-invent it. The residual is the one signal that requires the
world-model framing; it is the genuinely new contribution.

### 4.2 Why calibrate against gnomAD common variants?

These variants are, by selection, the ones biology has tolerated. They
serve as a per-context "null model" of variants that the predictor
should find unsurprising. Any variant that produces a substantially
larger residual than the gnomAD distribution is "unexpected" relative
to a biological background.

Alternative null models considered:

- **Synthetic random SNVs.** Uniform action coverage, but no biological
  selection. Calibration against synthetic would be uncalibrated against
  what's actually rare in real life.
- **The training-set distribution of all variants** (40% gnomAD, 30%
  synthetic SNV, etc.). Would mix biological and synthetic; harder to
  interpret.

gnomAD-only calibration is the cleanest reference.

### 4.3 Why stratify by context?

Without stratification, ranking variants by surprise systematically
biases toward high-entropy regions. CADD and other established scorers
ran into exactly this and ended up with context-aware calibration.
Adopting the same pattern keeps our scores interpretable.

The specific choice of `(region_class, gc_bin, repeat_class)` is a
trade-off: more factors → finer calibration → smaller buckets → less
reliable percentile estimates. Three factors hit the sweet spot in
preliminary analysis (to be confirmed in Phase 2).

### 4.4 Why expose `sigma_raw` in addition to `sigma_calibrated`?

Two reasons:

1. **Debugging.** Investigators can see whether a high calibrated
   surprise comes from a large raw residual or from a tight calibration
   distribution. Both are valid; both have different downstream
   implications.
2. **Recalibration.** Users who disagree with our calibration choice
   can recompute percentiles in their own preferred buckets, given
   `sigma_raw`.

### 4.5 Why a fixed calibration distribution rather than computing it
online?

Distributing a fixed calibration table makes scores reproducible across
runs and across machines. Online calibration would tie the score to the
specific batch the user happens to be processing. Reproducibility wins.

Periodic recalibration (e.g., when a new gnomAD release is published)
is a release-time activity, not a runtime activity.

### 4.6 Why not train a supervised classifier on top of surprise?

This would defeat the unsupervised property. Phase 2 explicitly
considers ensembling surprise with Carbon's likelihood and possibly
external scores into a downstream classifier, but the standalone
surprise score is intentionally unsupervised in v1.

## 5. Unresolved questions

- The right number of GC bins (terciles vs deciles).
- Whether to add per-population calibration distributions (each population
  has its own common-variant distribution; should we use the population
  most similar to the user's, if known?). Privacy-fraught; deferred.
- Whether to expose a per-window surprise breakdown for multi-window
  variants, or only the aggregate. v1 reports only aggregate; debugging
  mode exposes breakdown.
- How to handle calibration drift as Carbon updates: pin or recompute?
  v1 pins to the calibration shipped with the checkpoint.

## 6. Future work

- A Bayesian variant: replace the predictor's point estimate with a
  predictive distribution (e.g., Monte-Carlo dropout) and report
  surprise as a function of the predictive variance.
- Ensemble surprise across multiple GenoLeWM checkpoints trained with
  different seeds; the variance across checkpoints is itself
  informative.
- A "directional surprise" that decomposes the residual into latent
  axes correlated with specific functional features (e.g., a latent
  direction associated with loss-of-function).
- A per-gene surprise calibration for users running screens on a single
  gene; the per-gene distribution may differ meaningfully from the
  global per-context calibration.

## 7. Changelog

- 2026-06-02 — Updated implementation status for scoring APIs, CLI
  paths, calibration helpers, and receipt emission.
- 2026-05-31 — Implemented deterministic context bucket IDs and
  sparse-bucket back-off helpers.
- 2026-05-31 — Implemented deterministic calibration table building
  and Parquet schema validation from pre-scored reference rows.
- 2026-05-20 — Initial draft.
