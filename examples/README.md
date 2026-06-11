# Examples

Reference notebooks and scripts demonstrating GenoLeWM's main use cases.
The fixture-backed scoring notebooks and the receipt-verification
notebook are implemented. Broader benchmark, BRCA2, rollout, and
planning notebooks remain planned until their measured evidence is
available.

---

## Implemented notebooks

### `01_score_single_variant.ipynb`
Scores a single ClinVar-like SNV through the local runtime API with a
tiny deterministic fixture scorer, writes a checksum receipt, and
validates that receipt. This is a fixture smoke tutorial, not learned
model evidence.

### `03_score_vcf.ipynb`
Batch-scores a one-row fixture VCF against a local FASTA, writes one
score JSONL row and one checksum receipt JSONL row, and validates the
first receipt. This is a fixture smoke tutorial, not throughput or
model-quality evidence.

### `07_verify_receipt.ipynb`
Verifies a committed checksum-only fixture receipt against its manifest,
recomputes the input commitment from the original edit and reference
window, and recomputes the output commitment. This does not rerun
scoring and does not claim model-quality assurance beyond checksum
provenance.

## Planned notebooks

### `02_score_brca2_saturation.ipynb`
Phase 1. Score every possible SNV across a BRCA2 exon (saturation
mutagenesis), produce a heatmap of calibrated surprise, and compare to
the published Findlay et al. functional scores. The headline visual
demo.

### `04_multi_edit_rollout.ipynb`
Phase 2. Roll out a phased multi-edit haplotype from gnomAD, compare
predicted vs encoder ground truth latent at each step, plot the
divergence curve. Demonstrates the world-model claim.

### `05_planning_minimal_edits.ipynb`
Phase 2. Given a pathogenic variant and a "benign latent neighborhood"
target, run CEM to find the minimal compensatory edit set. Demonstrates
the planner.

### `06_on_device_desktop.md` (not a notebook)
Phase 3. Walkthrough of installing the desktop app, dropping in a VCF,
viewing scored variants, exporting a receipt. Demonstrates the
freedom-tech flow.

---

## When implementations land

Each notebook will land in its own PR, with a corresponding entry in
the changelog. Notebook PRs require:

- The notebook itself, in clean executed state.
- A short README section in this file describing what the notebook
  shows and what hardware it was tested on.
- A `requirements-example-NN.txt` if it needs anything beyond the
  base GenoLeWM install.

---

## Conventions

- Notebooks are committed in executed state (outputs present), so users
  can read them without running.
- Inputs (small VCFs, FASTA snippets) are committed under
  `examples/data/` and stay under 1 MB each.
- Larger inputs are pulled from the Hugging Face Hub in the first cell.
- Random seeds are pinned so the outputs in the committed notebook are
  reproducible.
