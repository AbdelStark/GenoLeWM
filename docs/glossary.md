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

**Cross-attention predictor.** The trainable model that combines a
latent state and action embedding to predict the post-edit latent state.

**JEPA.** Joint-Embedding Predictive Architecture: a model family that
predicts in representation space instead of reconstructing raw input.

**Predictor.** The trainable GenoLeWM module that maps `(state, action)`
to a predicted next state.

**State.** A latent vector embedding of a DNA window.

**State encoder.** The frozen DNA encoder that maps a sequence window to
a state. The released path uses Carbon-500M.

**Surprise.** The residual between the predicted post-edit state and the
encoded edited state.

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

**Calibration table.** Released table that maps raw residuals to
contextual calibrated surprise values.

**Checksum receipt.** JSON document binding model identity, input
commitment, output commitment, and runtime metadata. It is not a model
quality or runtime-assurance certificate.

**Content addressing.** Identifying an artifact by the cryptographic
hash of its canonical bytes.

**Input commitment.** Hash of an inference input payload.

**Manifest.** JSON document describing a model package's artifact
identity and provenance.

**Model id.** The manifest hash for a specific released checkpoint
artifact set.

**Output commitment.** Hash of a score output payload.

**`sigma_raw`.** Uncalibrated latent residual.

**`sigma_calibrated`.** Calibrated surprise value from the release
calibration table.

**Tuple builder.** Data-pipeline component that produces
`(reference window, edit, edited window)` training examples.
