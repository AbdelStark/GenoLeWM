# RFC-0003: Action representation — genomic edits

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0001, RFC-0002
- **Supersedes:** —
- **Implementation status:** Not started

---

## 1. Summary

A genomic edit (action) is a structured object the predictor sees
explicitly. This RFC specifies the edit schema, the encoding pipeline
that maps an edit to a fixed-size action embedding, the validation rules,
the synthetic-edit samplers used during training, and the multi-edit
(haplotype) sequence interface.

## 2. Motivation

In LeWorldModel and CodeLeWM, the action is what makes the model a
world model rather than a representation learner. For genomics, the
"natural" action is a genetic edit: a position, an edit type, the
reference bases, and the alternative bases.

Two non-obvious design pressures shape this RFC.

1. **Edits are highly variable in length.** SNVs are 1 bp; common
   indels are 1–16 bp; structural variants are kilobases to megabases.
   A single fixed encoding cannot do all of these well. We split:
   short edits get a uniform encoding; long structural edits get a
   separate adapter.

2. **Edit distribution in training data is non-uniform.** SNVs dominate
   real variant catalogs. If we sample edits proportional to their
   real frequency, the model learns SNVs well and never sees enough
   indels to generalize. The data pipeline (RFC-0006) addresses
   sampling; the encoder addresses *capacity*: each edit type gets its
   own learned type embedding.

## 3. Specification

### 3.1 EditSpec

The canonical edit type:

```python
@dataclass(frozen=True, slots=True)
class EditSpec:
    chrom: str               # chromosome / contig name
    pos: int                 # 1-based, VCF convention
    ref: str                 # reference bases (uppercase ACGT, ≥ 1 bp)
    alt: str                 # alternative bases (uppercase ACGT, ≥ 1 bp)
    edit_type: EditType      # derived; see §3.2

    def relative_to(self, window_start_bp: int, window_end_bp: int) -> "RelEdit": ...
```

Validation rules (raised as `ValueError`):
- `chrom` non-empty.
- `pos >= 1`.
- `ref` and `alt` non-empty, uppercase, only in `{A, C, G, T}`.
- `ref != alt`.
- `len(ref) <= V1_MAX_LEN` and `len(alt) <= V1_MAX_LEN`, where
  `V1_MAX_LEN = 16` in v1. Edits longer than this are routed through
  the SV adapter (§3.5).

The on-disk representation matches VCF semantics: `pos` is 1-based,
both `ref` and `alt` are explicit bases (no `<DEL>` or `<INS>` symbolic
alleles in v1).

### 3.2 EditType enumeration

```python
class EditType(IntEnum):
    SNV = 0     # len(ref) == 1 and len(alt) == 1
    INS = 1     # len(ref) == 1 and len(alt) > 1 (VCF anchor convention)
    DEL = 2     # len(ref) > 1 and len(alt) == 1
    MNV = 3     # len(ref) == len(alt) > 1
    INDEL = 4  # len(ref) != len(alt), both > 1
    SV = 5     # any edit with ref or alt length > V1_MAX_LEN
```

Edit type is derived from `(ref, alt)` lengths at construction time and
exposed as `edit_type` for the action encoder.

### 3.3 Window-relative form

For predictor consumption, an `EditSpec` is converted to a window-relative
form:

```python
@dataclass(frozen=True, slots=True)
class RelEdit:
    rel_pos: int              # 0-based offset within window, in bp
    edit_type: EditType
    ref_bases: str
    alt_bases: str
```

The conversion is a thin wrapper around `pos - window_start_bp`. The
predictor never sees absolute chromosomal coordinates; it sees only
window-relative offsets.

### 3.4 Encoding pipeline

Each `RelEdit` is encoded into `a_emb ∈ ℝ^{d_action}` (with
`d_action = 512` in v1) via four sub-encoders whose outputs are
concatenated and projected.

```
┌─────────────┐
│ rel_pos     │ ──► sinusoidal positional embedding ──► p_emb ∈ ℝ^128
└─────────────┘

┌─────────────┐
│ edit_type   │ ──► learned embedding table (6 entries) ──► t_emb ∈ ℝ^64
└─────────────┘

┌─────────────┐
│ ref_bases   │ ──► Carbon 6-mer tokenize (pad to 4 tokens)
└─────────────┘    ──► shared SeqMicroEncoder ──► r_emb ∈ ℝ^256

┌─────────────┐
│ alt_bases   │ ──► Carbon 6-mer tokenize (pad to 4 tokens)
└─────────────┘    ──► shared SeqMicroEncoder ──► v_emb ∈ ℝ^256

concat(p_emb, t_emb, r_emb, v_emb) ∈ ℝ^704 ──► MLP(704 → 1024 → 512) ──► a_emb ∈ ℝ^512
```

**Positional encoding.** Sinusoidal at base-pair resolution, with
maximum position = window length (12,288 bp default). We use
sinusoidal rather than learned because the positional space is large
(thousands of bp) and we want the encoding to extrapolate naturally
to longer windows in future versions.

**Edit-type embedding.** A 6-entry learned table; one row per
`EditType`. ~400 parameters.

**SeqMicroEncoder.** A small 2-layer Transformer encoder:
- Input: 4 tokens (= 24 bp at 6-mer tokenization, padded with
  Carbon's 6-mer `<oov>` token for shorter sequences).
- Hidden: 256.
- Heads: 4.
- Output: mean-pooled over the 4 tokens.

This module is shared between the `ref_bases` and `alt_bases` paths.
Sharing is intentional: the function "embed a short DNA snippet" is the
same regardless of which side of the edit it comes from.

**Projection MLP.** 2-layer with GELU activation and LayerNorm. Output
is **not** normalized (the predictor applies its own normalization).

Total action encoder parameters: ~2.5M.

### 3.5 Structural-variant adapter (v2)

Edits with `len(ref) > 16` or `len(alt) > 16` (and thus type `SV`) are
**not** supported in v1. The system raises `UnsupportedEditError` with
a clear message pointing to v2.

The planned v2 adapter encodes SVs by:
- Encoding `rel_pos` as above.
- Setting `edit_type = SV`.
- Encoding the SV through a separate `SVAdapter` that takes:
  - SV subtype (deletion, insertion, inversion, duplication,
    translocation),
  - SV length in base pairs (binned: 16–100, 100–1k, 1k–10k, 10k–100k,
    100k+),
  - For sequence-resolved SVs: a Carbon embedding of a 200 bp region
    around each breakpoint.

The v2 RFC will live as RFC-0003a once we get there.

### 3.6 Multi-edit (haplotype) sequences

For multi-edit inputs, the predictor consumes a **sequence** of
`a_emb` vectors, applied autoregressively. The sequence is constructed
by sorting edits by `rel_pos` ascending, breaking ties by edit type
(SNV < INS < DEL < MNV < INDEL).

The predictor (RFC-0004) takes a list of action embeddings and emits
one predicted state per step. The final state `ŝ_K` is the prediction
for the cumulative haplotype.

A haplotype is **invalid** (raises `OverlappingEditsError`) if any two
edits overlap in their reference span. Detecting overlap:

```
edits_overlap(e1, e2) ⇔ [e1.rel_pos, e1.rel_pos + len(e1.ref))
                        ∩ [e2.rel_pos, e2.rel_pos + len(e2.ref)) ≠ ∅
```

### 3.7 Apply-edit operation

The pure-Python `apply_edit(window, edit)` function produces the
edited window string used to encode the training target `s_{t+1}`.

```python
def apply_edit(window: str, edit: RelEdit) -> str:
    end = edit.rel_pos + len(edit.ref_bases)
    assert window[edit.rel_pos:end].upper() == edit.ref_bases.upper(), (
        "reference bases at edit locus do not match window content"
    )
    return window[:edit.rel_pos] + edit.alt_bases + window[end:]
```

For multi-edit haplotypes, `apply_edits(window, edits)` applies edits
right-to-left (by descending `rel_pos`) so that each edit's relative
position remains valid through the sequence of mutations.

After applying edits the window length may change (indels). The
training pipeline truncates / pads the post-edit window to the same
length as the pre-edit window before passing it to the encoder. The
truncation removes from the side **opposite** the edit locus to
preserve maximum context around the edit.

### 3.8 Synthetic-edit samplers

For data augmentation (RFC-0006), three synthetic samplers are defined.

- **`uniform_snv(window, n)`:** sample `n` random positions, each with
  a uniform alternative base from the three non-reference bases.
- **`indel(window, length_dist, type_mix)`:** sample positions and
  draw lengths from a configurable distribution (default: truncated
  geometric over [1, 16]) and types (default: 50/50 INS/DEL).
- **`mnv(window, length_dist)`:** length-preserving multi-base substitutions
  with lengths drawn from a configurable distribution (default: uniform
  over [2, 8]).

All samplers respect a minimum distance from window edges (default 64
bp) to ensure pooling has enough context on both sides.

### 3.9 Module API

```python
# geno_lewm.action.encoder

class ActionEncoder(nn.Module):
    def __init__(self,
                 d_action: int = 512,
                 d_pos: int = 128,
                 d_type: int = 64,
                 d_seq: int = 256,
                 max_window_bp: int = 12_288,
                 carbon_tokenizer: PreTrainedTokenizer | None = None) -> None: ...

    def forward(self, edits: list[RelEdit]) -> Tensor:
        """Returns a tensor of shape (B, K, d_action) where K is the
        max sequence length in the batch and B is batch size. Shorter
        sequences are right-padded with a learned padding embedding."""

    @property
    def d_action(self) -> int: ...
```

## 4. Rationale and alternatives

### 4.1 Why four sub-encoders instead of one tokenized sequence?

The "obvious" alternative is to encode an edit as a string like
`"<edit><pos=1234><snv><ref=C><alt=T></edit>"` and feed it through a
small Transformer that learns to parse it. We rejected this for three
reasons.

1. **Inductive bias.** Position, type, ref, and alt are genuinely
   different kinds of information. Giving each its own encoder bakes
   that structure into the architecture and reduces what the model has
   to learn.
2. **Compositional generalization.** A SeqMicroEncoder that shares
   weights between `ref` and `alt` is forced to learn a "DNA snippet"
   representation, which transfers to unseen base combinations.
3. **Interpretability.** Sub-embeddings can be inspected post-hoc:
   "does the edit-type embedding cluster by mutation impact?" Single
   tokenized sequences obscure this.

### 4.2 Why sinusoidal position?

Learned positional embeddings would tie us to a fixed window length.
Sinusoidal positions extrapolate naturally to longer windows in v2 and
they are cheaper to compute. The downside (slightly less expressivity
at training-set window lengths) is small in practice for this kind of
positional information.

### 4.3 Why cap at 16 bp in v1?

Most ClinVar / TraitGym / BRCA2 variants are SNVs or short indels
(< 10 bp). v1 covers > 95% of clinically interesting short variants
with a uniform encoding. SVs require additional sequence-resolved
breakpoint encoding (and additional eval benchmarks) and would push v1
out by months. v2 adds them properly.

### 4.4 Why is the SeqMicroEncoder a Transformer rather than just an
average of base-letter embeddings?

For short snippets (≤ 16 bp), an averaged-letter representation loses
order information ("AT" vs "TA"). A 2-layer Transformer preserves
order with minimal parameter cost (~600k params total).

### 4.5 Why right-to-left application for multi-edit?

Applying edits in ascending position order would cause earlier edits'
indels to shift the relative positions of later edits, requiring
position recomputation. Right-to-left application keeps positions
stable. This is a small but consequential implementation detail; we
ship it as a constant.

## 5. Unresolved questions

- Whether to add a "context window" sub-embedding around the edit
  locus (e.g., 5 bp on each side of the alt) as an additional channel.
  Phase 1 ablation.
- Whether `ref_bases` is actually necessary as a separate input, given
  that the state vector already encodes the reference window. Plausibly
  redundant; will ablate.
- Whether to support multi-allelic VCF records natively (currently
  they decompose into per-allele EditSpecs).
- The exact length distribution for the synthetic indel sampler. Real
  indel length distributions are roughly geometric with mean ~3 bp;
  for action-space coverage we want flatter distributions, but how flat?

## 6. Future work

- The SV adapter (RFC-0003a).
- Multi-locus edits (e.g., a coordinated translocation involving two
  separated loci). v3 territory.
- A learned action embedding lookup table for common variants (gnomAD
  most-frequent N variants), which could speed up scoring at the cost
  of generality. Probably not worth it.

## 7. Changelog

- 2026-05-20 — Initial draft.
