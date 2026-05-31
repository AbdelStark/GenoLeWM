# Examples

Reference notebooks and scripts demonstrating GenoLeWM's main use cases.
Most scoring and planning notebooks are still placeholders because the
runtime scorer and planner are not implemented yet. Implemented examples
state their current fixture scope explicitly.

---

## Planned notebooks

### `01_score_single_variant.ipynb`
Phase 1. Score a single ClinVar variant end-to-end, showing the
prediction, the surprise score, and the receipt. Smallest possible
demo; the "hello world" of GenoLeWM.

### `02_score_brca2_saturation.ipynb`
Phase 1. Score every possible SNV across a BRCA2 exon (saturation
mutagenesis), produce a heatmap of calibrated surprise, and compare to
the published Findlay et al. functional scores. The headline visual
demo.

### `03_score_vcf.ipynb`
Phase 1. Batch-score a VCF (we provide a small toy VCF; users can swap
in their own). Demonstrates the batched throughput path and the
per-variant receipt aggregation.

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

### `07_verify_receipt.ipynb`
Implemented. Verifies a committed checksum-only fixture receipt against
its manifest, recomputes the input commitment from the original edit and
reference window, and recomputes the output commitment. This does not
rerun scoring and does not verify a STARK proof; those paths land with
the runtime/prover issues.

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
