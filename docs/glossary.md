# Glossary

## Architecture

**Action.** A structured genomic edit passed to the predictor as a
first-class input.

**Action encoder.** The trainable module that maps an edit into a vector
embedding.

**Autoregressive rollout.** Repeatedly applying the predictor over a
sequence of edits, feeding each predicted state into the next step.

**Carbon.** The HuggingFaceBio DNA foundation model used as GenoLeWM's
frozen state encoder in the released path.

**Carbon pure-DNA tokenizer.** The self-contained tokenizer used by the
corrected source path for GenoLeWM's restricted `<dna>...</dna>` input. It is
constructed from local Carbon configuration and does not execute the upstream
wrapper's unpinned `Qwen/Qwen3-4B-Base` lookup. This runtime repair is not
model-quality evidence.

**Cross-attention predictor.** The trainable model that combines a
latent state and action embedding to predict the post-edit latent state.

**JEPA.** Joint-Embedding Predictive Architecture: a model family that
predicts in representation space instead of reconstructing raw input.

**Predictor.** The trainable GenoLeWM module that maps `(state, action)`
to a predicted next state.

**State.** A latent vector embedding of a DNA window under an explicit state
contract.

**State encoder.** The frozen DNA encoder that maps a sequence window to
a state. The released path uses Carbon-500M.

**State contract.** The transformation, scale, and pooling coordinate shared by
source, target, candidate, and predicted states. `legacy_raw_v1` preserves
historical raw pooled Carbon states; `l2_normalized_v2` applies L2 normalization
after coordinate-matched pooling and is the intended contract for new training.

**DNA content start.** The hidden-state index of the first six-mer after the
leading `<dna>` control token. Correct centered pooling uses
`dna_content_start + edit_locus // 6`. Historical code omitted
`dna_content_start`, shifting every intended center one hidden token left and
sometimes centering on the control token.

**Cache schema 3.** Raw post-pooling, pre-normalization Carbon states keyed by
encoder runtime identity, layer, pooling mode/radius, `center_token`, and
logical compute dtype. New Parquet shards separately declare fixed-size `fp32`
physical storage and use a collision-safe identity namespace. Cache v2 remains
readable for compatibility; cache v1 omitted `center_token` and is intentionally
invalidated.

**Surprise.** A candidate residual between predicted and encoded post-edit
states under the same validated state contract. Published `legacy_raw_v1`
values mixed state scales and are not valid scientific surprise scores.

## Genomics

**bp.** Base pairs.

**ClinVar.** NCBI's public database of human variants with clinical
significance labels.

**Coding / non-coding.** DNA regions that do or do not translate to
protein.

**Edit / variant.** A change to a reference DNA sequence.

**EditSpec.** GenoLeWM's chromosome-position-reference-alternate edit
object.

**RelEdit.** Window-relative edit object used after placing an edit
inside a sequence window.

**gnomAD.** Genome Aggregation Database, used by GenoLeWM data and
calibration tooling.

**Haplotype.** A coordinated set of variants on the same chromosome
copy.

**Indel.** Insertion or deletion of one or more bases.

**MNV.** Multi-nucleotide variant.

**SNV.** Single-nucleotide variant.

**SV.** Structural variant. Large SV support is not established by the
public release.

**TraitGym.** A benchmark of trait-associated variants for
variant-effect prediction.

**VCF.** Variant Call Format.

**Window.** A contiguous DNA sequence region passed to the state
encoder.

## Scores And Artifacts

**Calibration table.** Table that maps raw residuals to contextual calibrated
values. The published legacy table was fitted to invalid mixed-contract
residuals and is retained only for artifact replay.

**Checksum receipt.** JSON document binding model identity, input
commitment, output commitment, and runtime metadata. It is not a model
quality or runtime-assurance certificate.

**Content addressing.** Identifying an artifact by the cryptographic
hash of its canonical bytes. It binds only included files and dependencies;
hashing historical Carbon `tokenizer.py` did not bind its unpinned transitive
Qwen tokenizer load.

**Input commitment.** Hash of an inference input payload.

**Manifest.** JSON document describing a model package's artifact
identity and provenance.

**Model id.** The manifest hash for a specific released checkpoint
artifact set.

**Output commitment.** Hash of a score output payload.

**`sigma_raw`.** Uncalibrated latent residual. Published `legacy_raw_v1`
values are invalid as edit-effect or surprise scores.

**`sigma_calibrated`.** Calibration-table mapping of `sigma_raw`. Published
legacy values are historical implementation outputs, not validated scores.

**Tuple builder.** Data-pipeline component that produces
`(reference window, edit, edited window)` training examples.
