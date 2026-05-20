# RFC-0010: On-device personal-genome deployment

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0001, RFC-0002, RFC-0004, RFC-0007, RFC-0009
- **Supersedes:** —
- **Implementation status:** Not started

---

## 1. Summary

GenoLeWM's freedom-tech promise is that personal-genome inference
happens entirely on the user's device. This RFC specifies the export
pipeline (ONNX / Core ML / GGUF), the quantization strategy, the
runtime contract (what the user installs and how it behaves), the
performance targets, and the reference desktop app skeleton that
demonstrates the full local-first stack.

## 2. Motivation

Genomic data is permanent, identifying, and family-implicating. Sending
a personal VCF to a cloud API — even a "trusted" one — is a category
of decision that should be the user's, not the application's. The
default behavior must be on-device.

Carbon-500M is the first DNA foundation model that makes this credible.
Quoting the model card: "Fast enough to run locally on your laptop.
Powerful enough to process a whole human genome on a single GPU in
less than 2 days." A ~25M-parameter GenoLeWM head on top adds
negligible cost. The remaining engineering work is mostly in
packaging: making it easy for a non-ML user to install a binary, drop
in a VCF, and get scored variants out.

This RFC is therefore as much about the user experience as it is about
the deployment pipeline. The architecture only matters if the end-to-end
flow is something a quantified-self enthusiast can actually run.

## 3. Specification

### 3.1 Deployment targets

| Target | Runtime | Quantization | Use case |
|--------|---------|--------------|----------|
| **Apple Silicon (M1/M2/M3/M4)** | Core ML + MLX | predictor int8, Carbon int4 | primary on-device target |
| **CUDA workstation** (RTX 4090/5090) | PyTorch + bf16 | predictor bf16, Carbon bf16 | researcher use |
| **CUDA server** (H100) | PyTorch + bf16 + vLLM | none (full precision) | high-throughput batched |
| **CPU-only laptop** | ONNX Runtime | predictor int8, Carbon int4 (llama.cpp) | accessibility fallback |

The primary on-device target is Apple Silicon. This is a deliberate
choice driven by two facts: (a) the personal-genomics enthusiast
community runs disproportionately on Mac, and (b) Apple Silicon
unified memory and Neural Engine give the best practical
single-machine inference experience for models in this size range.

CUDA workstation support is the second-priority target, for users with
a gaming GPU or workstation. CPU-only is a last-resort accessibility
fallback; performance will be substantially slower.

### 3.2 Export pipeline

The export pipeline takes a trained `predictor + action_encoder`
checkpoint (Carbon weights are pulled from the Hugging Face Hub at
runtime, not bundled) and produces deployment artifacts.

```
geno-lewm export \
    --checkpoint geno-lewm-v0.1.0-carbon-500m-r1 \
    --target {coreml,onnx,gguf} \
    --quantization {none,int8,int4} \
    --output ./dist/
```

The export produces:

- `predictor.{mlpackage,onnx,gguf}`: the predictor itself.
- `action_encoder.{mlpackage,onnx,gguf}`: the action encoder.
- `calibration.parquet`: the surprise calibration table (RFC-0009).
- `manifest.json`: metadata including content hashes, encoder identity,
  GenoLeWM version, export-time configuration, and target runtime.

Carbon-500M is **not bundled** with the export. It is pulled from the
Hugging Face Hub on first run (~1 GB download). This keeps the
GenoLeWM-specific artifact small (~120 MB at bf16, ~40 MB at int8) and
ensures the user pulls the canonical, audited Carbon weights.

### 3.3 Quantization

Three quantization levels are supported:

| Level | Predictor | Action encoder | Carbon | Memory | Quality penalty |
|-------|-----------|----------------|--------|-------:|------:|
| `none` | bf16 | bf16 | bf16 | ~2 GB | 0 |
| `int8` | int8 (per-tensor) | int8 (per-tensor) | bf16 | ~1.4 GB | < 1 AUROC point |
| `int4` (on-device default) | int8 | int8 | int4 (llama.cpp Q4_K_M) | ~600 MB | 1–2 AUROC points |

Quantization quality is measured on the full eval suite (RFC-0007). A
quantized checkpoint is "shippable" if its degradation on ClinVar
coding AUROC is ≤ 2 points relative to the bf16 reference.

The predictor is quantized via per-tensor symmetric int8 with an
activation calibration pass over 1,000 reference windows. The action
encoder uses the same scheme. Both are small enough that quantization
overhead is negligible (< 1 minute on a laptop).

Carbon int4 quantization reuses the established `llama.cpp` GGUF
quantization (Q4_K_M variant), which has well-characterized quality
properties on Llama-family architectures. Carbon-500M is a Llama-style
model architecturally, so this transfers cleanly.

### 3.4 Runtime contract

The runtime exposes:

```python
class GenoLeWMRuntime:
    def __init__(self, model_dir: Path, backend: str = "auto") -> None: ...

    def score_variant(self, variant: EditSpec, window: str | None = None) -> SurpriseResult:
        """Score a single variant. If `window` is None, the reference
        window is fetched from a local FASTA index."""

    def score_vcf(self, vcf_path: Path, fasta_path: Path,
                  output_path: Path, batch_size: int = 64,
                  progress: bool = True) -> None:
        """Score all variants in a VCF, writing results to a Parquet file."""

    def encode_window(self, window: str, edit_locus: int | None = None) -> Tensor:
        """Encode a window to a state vector (calls Carbon-500M)."""

    def predict(self, state: Tensor, edits: list[RelEdit]) -> Tensor:
        """Run the predictor on a precomputed state + edits."""
```

`backend="auto"` selects the best available runtime in order:
Core ML (on Apple Silicon) → CUDA (when available) → ONNX → CPU.

### 3.5 Performance targets

| Metric | Apple M3 Max | RTX 4090 | H100 |
|--------|-------------:|---------:|-----:|
| Single-variant scoring, warm cache | < 200 ms | < 20 ms | < 5 ms |
| Single-variant scoring, cold (Carbon call) | < 800 ms | < 100 ms | < 50 ms |
| Whole-VCF scoring (100k variants) | < 30 min | < 5 min | < 1 min |
| Whole-genome reference encoding | < 8 h | < 2 h | < 30 min |
| Peak resident memory (loaded model) | < 8 GB | < 3 GB | < 3 GB |
| Disk footprint (model + cache) | < 5 GB | < 5 GB | < 5 GB |

These are commitments for v1; benchmarks reported per release.

### 3.6 Reference desktop app skeleton

The reference desktop app is a Tauri application (Rust + a small
web frontend) that demonstrates the full on-device flow. It is
intentionally **skeleton-grade**: a workable demo, not a polished
product. Polished products are downstream community work.

App features (v1):

- File-drop: drag a VCF file onto the window.
- Reference build: drop a FASTA (or auto-download from a configured
  source, e.g., 1000 Genomes reference index).
- Live progress bar during scoring.
- Result table: sortable by `sigma_calibrated`, filterable by
  chromosome / gene / region class.
- Per-variant detail view: the variant, its calibrated surprise, the
  bucket it was calibrated in, the confidence indicator, and a link
  to ClinVar (where applicable).
- Local attestation receipt for each scoring session (see RFC-0011).
- A prominent banner stating: "This is a research tool. Not a clinical
  diagnostic. If a variant concerns you, talk to a genetic counselor."

App non-features (v1):

- No cloud sync. All state is local.
- No accounts. No telemetry.
- No PHI handling beyond what the user explicitly drops in.
- No clinical decision support, treatment recommendations, or
  reproductive-use features.

The skeleton is intentionally minimal; the safety banner is permanent
and non-dismissible.

### 3.7 Privacy contract

The runtime makes the following enforceable guarantees:

1. **No network calls after first-run setup.** Carbon weights are
   downloaded once; reference FASTA is downloaded once. All subsequent
   inference is offline. The runtime fails closed: if any inference
   path attempts a network call, it raises an error rather than
   silently degrading to online mode.
2. **No telemetry.** No usage data is reported anywhere.
3. **No persistent caches of personal variants.** The window-embedding
   cache stores *reference* embeddings (which are not personal). Personal
   variants are scored and the results are stored only at the
   user-specified output path.
4. **Crash logs sanitized.** Any crash log written by the runtime
   excludes user variant data; only model identifiers and stack traces.

These are documented in `PRIVACY.md` and enforced by integration tests.

### 3.8 Update mechanism

Model updates are pulled explicitly by the user via:

```
geno-lewm update
```

This compares the local `manifest.json`'s model version against the
latest published GenoLeWM release on the Hugging Face Hub, displays the
diff, and applies the update only with user consent. No automatic
background updates.

Each update preserves the previous model version as a side-by-side
install, so the user can roll back if a new release produces different
scores on previously-scored variants. This is essential: published
results that depend on a specific model version must be reproducible
months later.

### 3.9 Compatibility with personal-genome data formats

The runtime accepts:

- **VCF / VCF.gz**: standard variant call format.
- **23andMe raw data**: tab-separated, converted internally to VCF.
- **AncestryDNA raw data**: converted internally.
- **MyHeritage raw data**: converted internally.
- **Sequencing.com WGS JSON**: where available.

Conversion utilities are bundled with the runtime; conversions are
local-only (no third-party services called).

### 3.10 Distribution

The runtime is distributed via:

- **Hugging Face Hub model release**: `AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1-deploy`
  (the model artifacts).
- **GitHub release**: the desktop app binary (signed; notarized for
  macOS).
- **Homebrew**: `brew install geno-lewm` (planned, post-v1).
- **PyPI**: `pip install geno-lewm[deploy]` for the Python API.

## 4. Rationale and alternatives

### 4.1 Why Apple Silicon as the primary target?

Three reasons:

1. **User overlap.** Personal-genomics enthusiasts skew Mac. This is
   anecdotal but consistent across the community.
2. **Hardware quality for this size.** M3 Max / M4 / future Apple
   Silicon has the right combination of unified memory, Neural Engine,
   and battery efficiency for a model in this size range to feel
   instant rather than slow.
3. **Distribution maturity.** Signed, notarized .app bundles are a
   shipping path that works for non-technical users. Linux desktop
   distribution is still more friction.

We treat CUDA workstation as a strong secondary target so we do not
lose the researcher audience.

### 4.2 Why not bundle Carbon with the GenoLeWM artifact?

Two reasons:

1. **Size.** Carbon-500M at bf16 is ~1 GB. Bundling it would 8×
   the GenoLeWM artifact size. Pulling it once at first run is a one-time
   cost.
2. **Provenance.** The user pulling Carbon from the Hugging Face Hub
   gets the canonical, audited weights. If we bundled them, we'd be
   introducing a re-distribution step that adds an auditable surface.
   The attestation story (RFC-0011) benefits from the user obtaining
   Carbon weights directly from the source.

### 4.3 Why a Tauri app rather than an Electron app, a web app, or a CLI?

- Electron: heavyweight, bundled Chromium, large RAM footprint.
- Web app: would require cloud; defeats the purpose.
- CLI: necessary, but not sufficient — the personal-genomics audience
  is not all developers.

Tauri sits at the right point: small native binary, system webview,
Rust backend for the runtime wiring. It also aligns with the
freedom-tech ecosystem's general preference for Tauri over Electron.
The CLI is shipped alongside; the GUI is the demo.

### 4.4 Why int4 for Carbon but int8 for the predictor?

Carbon is large and has been studied at int4 quantization (via the
established Q4_K_M scheme); the quality penalty is well-characterized.
The predictor is small (40 MB at bf16), so the marginal memory savings
from int4 are negligible, and int8 has a cleaner quality story for
custom architectures that have not been quantization-studied.

### 4.5 Why is automatic update disabled?

Reproducibility. A user who reports a finding based on
`geno-lewm-v0.1.0-carbon-500m-r1` must be able to re-score the same
variant months later and get the same answer. Auto-updates would
silently invalidate that contract.

## 5. Unresolved questions

- Whether the desktop app should provide an export to PDF / printable
  report for users who want to share results with a genetic counselor.
  Probably yes, but adds surface area; defer to v1.1.
- Whether to integrate any third-party variant databases (gnomAD
  population frequencies, ClinVar annotations) into the app, and how to
  do so without leaking personal data.
- Whether to provide a "show me my surprise distribution" view: the
  user's personal histogram of calibrated surprises across their genome,
  to give a global picture. Interesting but probably not v1.
- Mobile deployment (iOS / Android) — the model size is borderline
  feasible on modern phones, but distribution and UX are substantial
  work. Deferred.

## 6. Future work

- iOS deployment (Apple Neural Engine + Core ML). Plausibly feasible
  with int4 Carbon and int8 predictor; v2.
- A plugin architecture for community-contributed scoring heads
  (e.g., an AlphaMissense ensemble plugin, a CADD ensemble plugin).
- WASM / browser deployment for the predictor only, where the user
  uploads reference embeddings (which are not personal) and scoring
  happens in-browser. Interesting niche.
- Offline VCF importers for newer formats as they emerge (e.g., the
  GA4GH gVCF format).

## 7. Changelog

- 2026-05-20 — Initial draft.
