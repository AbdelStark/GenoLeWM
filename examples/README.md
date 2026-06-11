# Examples

Reference notebooks and scripts demonstrating GenoLeWM's main use cases.
The fixture-backed scoring notebooks, BRCA2 fixture mechanics notebook,
and receipt-verification notebook are implemented. Rollout and planning
notebooks remain blocked until measured release evidence exists: #97
requires rollout-fidelity rows with documented cosine-similarity targets,
and #98 requires planner latency and useful-planning boundary evidence.
Fixture smoke outputs are test evidence, not model results, and are not
used as substitutes for those notebooks.

---

## Implemented notebooks

### `01_score_single_variant.ipynb`
Scores a single ClinVar-like SNV through the local runtime API with a
tiny deterministic fixture scorer, writes a checksum receipt, and
validates that receipt. This is a fixture smoke tutorial, not learned
model evidence.

### `02_score_brca2_saturation.ipynb`
Partial. Enumerates every possible SNV across a small BRCA2 exon-scale
fixture, produces a calibrated-surprise heatmap, and compares against a
deterministic fixture functional-score column. It does not use the
released scorer or Findlay et al. rows, so the published-data Spearman
acceptance criterion remains open.

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

### `04_multi_edit_rollout.ipynb`
Phase 2. Blocked on #97. This notebook should roll out a phased
multi-edit haplotype from gnomAD, compare predicted vs encoder
ground-truth latent at each step, and plot the divergence curve. It
lands only after release-backed rollout-state examples, measured
encoder-ground-truth comparisons, and documented cosine-similarity
targets are available.

### `05_planning_minimal_edits.ipynb`
Phase 2. Blocked on #98. This notebook should run CEM from an initial
variant state toward a target latent neighborhood and visualize the edit
sequence. It lands only after planner latency evidence and the
useful-planning boundary are documented; the current released planning
demo exercises the manifest-backed path but does not prove useful
planning behavior.

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
