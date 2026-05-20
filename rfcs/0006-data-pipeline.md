# RFC-0006: Data pipeline

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0002, RFC-0003
- **Supersedes:** —
- **Implementation status:** Not started

---

## 1. Summary

This RFC specifies the training data pipeline: corpus selection, window
sampling, edit sources, edit sampling, the `(w_ref, a, w_alt)` tuple
builder, the encoder caching layer, and the holdout protocol. The goal
is a reproducible stream of training tuples that gives the predictor
broad action coverage with a realistic backbone of natural variation.

## 2. Motivation

The predictor learns a transition function. The data must therefore
expose it to:

- **Sufficient reference diversity:** windows from many genomic
  contexts (coding, non-coding, regulatory, repetitive).
- **Realistic edit distributions:** real-world variants matter for
  generalization to biologically relevant inputs.
- **Sufficient action-space coverage:** synthetic edits fill the long
  tail that real variant catalogs underrepresent (rare positions,
  unusual edit types).
- **Hard negatives:** pathogenic variants from ClinVar serve as
  challenging anchors that force the predictor to learn fine-grained
  latent transitions.

The pipeline must also be efficient: re-encoding the same window for
every edit at every epoch is prohibitive. Caching is essential.

## 3. Specification

### 3.1 Reference corpus

Primary source:

```
HuggingFaceBio/carbon-pretraining-corpus
```

This is Carbon's published pretraining corpus, ~180M sequences,
predominantly eukaryotic with some prokaryotic content. Using Carbon's
own corpus is intentional: it guarantees that the encoder is
"in-distribution" on every window we train on.

**Sub-mix used by GenoLeWM** (matches Carbon's Phase-2 fine-tuning mix):

| Source | Fraction |
|--------|---------:|
| Eukaryotic genes (Generator-style annotated) | 50% |
| mRNA transcripts | 25% |
| Splice-enriched mRNA | 10% |
| GTDB bacterial genomes | 15% |

We do **not** use the full 180M sequences in Phase 1; instead we
sample a 10% slice (~18M sequences) for the baseline run. Phase 2
uses the full corpus.

### 3.2 Window sampling

For each sequence in the corpus:

1. Determine the sequence length in base pairs.
2. If `len < window_bp + 2 × margin`, skip the sequence
   (`window_bp = 12,288`, `margin = 256`).
3. Otherwise, sample window start positions uniformly in
   `[margin, len - window_bp - margin]`.
4. Stride between consecutive windows in the same sequence:
   `stride_bp = 8,192` (i.e., 67% overlap between consecutive windows).

The 67% overlap is chosen so that any genomic position is covered by
approximately three windows on average, giving the predictor multiple
local contexts for the same variant.

**Multiplicity per window:** for each window, we sample `N_edits` edits
to produce `N_edits` training tuples. Default `N_edits = 8`, drawn
from the per-window edit sampler (§3.4).

### 3.3 Edit sources

Four sources, with the following default mix:

| Source | Mix | Notes |
|--------|----:|-------|
| **gnomAD common variants** (AF ≥ 1%) | 40% | realistic edit distribution, biological prior |
| **Synthetic uniform SNVs** | 30% | uniform action-space coverage for SNVs |
| **Synthetic indels** (length ∈ [1, 16]) | 20% | indel coverage |
| **ClinVar P/LP variants** | 10% | hard-negative anchor |

The mix is enforced at the per-window level: of the `N_edits = 8`
edits per window, 3 are from gnomAD, 3 are synthetic SNVs, 1 is a
synthetic indel, and 1 is a ClinVar P/LP variant (if available for that
window; otherwise replaced by an additional synthetic SNV).

### 3.4 Edit samplers

**gnomAD common variants.** Pre-filtered to AF ≥ 1% in any population,
keyed by chromosome+position+ref+alt. The sampler restricts to
variants whose position falls within the current window (plus the
inset margin). If no qualifying variant exists, the slot is filled
with a synthetic SNV.

Dataset source:
- `gnomad.broadinstitute.org` v4.1 release, processed into a Parquet
  file (one row per `(chrom, pos, ref, alt)`) with population AFs.
  Cached locally; not loaded from HF directly because the gnomAD
  release files are large (terabytes raw).
- We provide a `geno-lewm prepare-gnomad` CLI command that downloads
  and processes a minimal subset (only AF ≥ 1% variants), producing a
  ~20 GB Parquet file usable for training.

**Synthetic uniform SNVs.** Sample positions uniformly within the
window (respecting the 64 bp edge margin), then pick a uniform
non-reference base.

**Synthetic indels.** Sample positions uniformly, then:
- 50% chance INS: pick a uniform random base sequence of length
  drawn from `geometric(p=0.5)` truncated to `[1, 16]`.
- 50% chance DEL: delete a contiguous segment of length drawn from
  `geometric(p=0.5)` truncated to `[1, 16]`.

**ClinVar P/LP variants.** Pre-filtered to "Pathogenic" and "Likely
pathogenic" assertions in the latest ClinVar release. Indexed by
position. Sampler restricts to variants in the current window.

ClinVar dataset source: official monthly VCF from NCBI, processed
into a Parquet file with clinical-significance labels. Cached locally;
prepared by the `geno-lewm prepare-clinvar` CLI command.

### 3.5 Tuple builder

For each `(window, edit)` pair, the builder produces:

```python
@dataclass
class TrainingTuple:
    window_id: str          # window hash (for cache lookup)
    rel_edit: RelEdit       # window-relative edit
    target_window: str      # edited window string (for on-the-fly encoding)
    edit_source: str        # "gnomad", "synthetic_snv", "synthetic_indel", "clinvar"
```

The `window_id` lets the data loader look up the cached `s_t` instead
of re-encoding. The `target_window` is encoded on-the-fly (since it
varies per edit) to produce `s_{t+1}`.

For multi-edit training samples (10% of the batch), the builder
produces a tuple with a *list* of edits and an edited window that has
all edits applied.

### 3.6 Encoder caching

See RFC-0002 §3.6 for the cache schema. Operationally:

- **Pre-build:** before training starts, an entry-point command
  `geno-lewm cache-windows` iterates over the corpus and produces the
  reference-window cache. This run is encoder-bound (Carbon-500M
  inference); ~24 hours on a single H100 for the 10% slice.
- **Cache lifetime:** cached embeddings are valid only for a specific
  `(encoder_id, encoder_hash, state_layer, pool_type, pool_radius)`
  tuple. Changing any of these invalidates the cache.
- **Cache index:** an on-disk SQLite database maps `window_hash` →
  Parquet shard file + row offset. Queries are O(1).

### 3.7 Training-time data loader

The PyTorch `Dataset` and `DataLoader` are wired as follows:

```python
class GenoLeWMDataset(IterableDataset):
    def __iter__(self) -> Iterator[TrainingBatch]:
        # 1. Sample a window from the corpus
        # 2. Look up s_t in the cache (or encode if missing)
        # 3. Sample N_edits edits per the mix
        # 4. For each edit: apply, look up or encode s_{t+1}
        # 5. Yield (s_t, action_specs, s_{t+1}_list) tuples
        ...
```

Throughput target: ≥ 5,000 tuples per second per worker on an H100
with the cache warm. With 8 workers and 256 batch size, ~150 steps/sec.

### 3.8 Holdouts

Three holdout sets, never used for training:

| Holdout | Purpose | Definition |
|---------|---------|------------|
| `holdout-chr` | spatial generalization | all windows on chr21 |
| `holdout-clinvar` | known-pathogenic generalization | all ClinVar P/LP variants in the eval bench (§RFC-0007) |
| `holdout-haplotypes` | multi-edit generalization | all gnomAD haplotype blocks with ≥ 2 variants in a 1 kbp window |

The holdouts are enforced by the corpus loader: any window whose
genomic coordinates intersect a holdout region is dropped from the
training stream. Sample tuples from holdouts are never produced by the
training-time data loader.

### 3.9 Validation streams

A separate `GenoLeWMValDataset` produces validation batches from
the holdout sets, sampled at the same rate as the training stream
(default: 500 windows per holdout). Validation runs every 500 training
steps and reports per-holdout metrics.

## 4. Rationale and alternatives

### 4.1 Why use Carbon's own corpus rather than the raw GRCh38 + RNA-seq?

Three reasons.

1. **In-distribution targets.** Carbon's encoder produces the most
   reliable embeddings on sequences from its training distribution.
   Using Carbon's corpus minimizes encoder uncertainty.
2. **Pre-processing already done.** Carbon's corpus has been quality-
   filtered, tokenized-checked (multiple-of-6 alignment, ACGT-only),
   and structured into HF Datasets-friendly shards.
3. **Reproducibility.** A widely-used public corpus makes the GenoLeWM
   training reproducible by anyone with HF Hub access.

### 4.2 Why a 40 / 30 / 20 / 10 edit-source mix?

The mix balances:
- **Realism** (gnomAD = 40% biases the model toward natural variants).
- **Action coverage** (synthetic = 50% guarantees the predictor sees
  edits at all positions and of all types).
- **Hard signal** (ClinVar = 10% gives the predictor exposure to
  variants the eval cares about).

A pure realism mix (100% gnomAD) would leave large parts of the action
space unsampled. A pure synthetic mix would over-train on edits that
do not occur in real biology. The mix is a compromise; we will ablate
in Phase 2.

### 4.3 Why 67% window overlap?

We want any genomic position to appear in multiple training windows so
that the predictor learns context-invariant edit effects. At 67%
overlap, each position is covered by ~3 windows on average. Less
overlap risks under-coverage of edge-positioned variants; more overlap
wastes compute.

### 4.4 Why on-the-fly encoding of edited windows?

The edited-window distribution is much larger than the reference
distribution (every reference window has thousands of possible edits).
Caching all of them is infeasible. On-the-fly encoding ensures we
don't cache things we'll only see once. An LRU at inference time gives
us the speedup for hot variants (e.g., during interactive use of the
on-device app) without the storage cost at training.

### 4.5 Why hold out an entire chromosome (chr21)?

Position-level holdouts inside a chromosome are not enough: linkage
disequilibrium and conservation patterns mean nearby positions in the
training set can leak signal. Holding out an entire chromosome gives
clean spatial generalization measurement. chr21 is chosen because it
is the smallest autosome (~48 Mbp), so the cost of the holdout is
modest.

### 4.6 Why not include CRISPR screen outcomes in the training data?

CRISPR screens (e.g., from RxRx, Tahoe-100M) are downstream functional
readouts of edits, not the edits themselves. They are not the right
training signal for a *latent transition* model; they are the right
signal for a downstream supervised head, which is out of v1 scope.

## 5. Unresolved questions

- The exact `N_edits` per window. 8 is a guess; values from 4 to 16
  will be ablated in Phase 1.
- How to handle multi-allelic sites in gnomAD: as separate single-edit
  tuples, or as a multi-edit tuple with all alleles? v1 decomposes into
  separate tuples; v2 may revisit.
- How to schedule the curriculum: should we start with SNVs only,
  add indels later, or always interleave? v1 always interleaves.
- Whether to include synthetic data from de-novo simulation models
  (e.g., neutral evolution simulators). Could broaden coverage but
  adds dependencies.

## 6. Future work

- A "personal-genome continual pretraining" mode where a user's own
  variant catalog (e.g., 23andMe export, WGS) is mixed in at a small
  rate to specialize the predictor to their genome. Privacy-preserving
  variant of this would be a v3 RFC.
- An adversarial edit sampler that learns to produce edits the
  predictor handles poorly, focusing training compute on weaknesses.
- Cross-species pretraining (mouse, fly, yeast) to test
  transfer-learning hypotheses.

## 7. Changelog

- 2026-05-20 — Initial draft.
