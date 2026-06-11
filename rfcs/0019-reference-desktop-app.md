# RFC-0019: Reference desktop app skeleton

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-06-11
- **Depends on:** RFC-0010, RFC-0011, RFC-0013, RFC-0018
- **Supersedes:** —
- **Implementation status:** Partial scaffold in `desktop/`: Tauri 2
  layout, default-deny network capability/CSP wiring, Rust/PyO3
  `runtime_probe`, static VCF/FASTA drop targets, and a permanent
  safety banner are present with static scaffold tests. Local scoring
  execution, result table/detail views, receipt export, richer file
  picker workflow, and signed/notarized release artifacts remain open
  under #81 and #82.

---

## 1. Summary

The reference desktop app is a Tauri-based local-first application that
demonstrates GenoLeWM's full on-device flow: drop a VCF, score it
locally, view results, export a receipt. The app is **skeleton-grade**
on purpose — it exists to demonstrate the workflow, not to be a
polished product. Polished products are downstream community work.

## 2. Motivation

A CLI alone is insufficient evidence that the on-device contract works.
The personal-genomics audience includes people who will never use a
shell, and the freedom-tech narrative (local inference, transparent
provenance) only lands when there is a working binary they can install
on their laptop. The skeleton app is the demonstration artifact.

## 3. Specification

### 3.1 Stack

| Layer | Choice |
|-------|--------|
| Shell | Tauri 2 |
| Backend | Rust (Tauri's host) |
| Inference | embedded Python (PyO3) running `geno_lewm` |
| Frontend | minimal HTML + vanilla TypeScript or Svelte (no React) |
| Bundling | macOS `.app` (notarized), Linux AppImage, Windows MSI |

Tauri is chosen over Electron for footprint (small native binary,
system webview), security (no Node.js runtime, no bundled Chromium),
and ecosystem alignment.

### 3.2 Features (v0.1)

- **File-drop.** Drag a VCF (or VCF.gz) onto the window.
- **Reference selection.** Drop a FASTA; if none selected, prompt to
  download the configured reference (e.g., 1000 Genomes GRCh38). The
  download is user-initiated, never silent.
- **Run scoring.** Calls `geno_lewm.deploy.GenoLeWMRuntime.score_vcf`
  via PyO3.
- **Progress bar.** Streams the inference progress event from the
  embedded process.
- **Result table.** Sortable by `sigma_calibrated`; filterable by
  chromosome, gene, and region class.
- **Per-variant detail view.** Variant string, calibrated surprise,
  bucket id, confidence, link to the ClinVar record where applicable.
- **Receipt export.** Save the session receipt to a user-chosen path.
- **Permanent safety banner.** Non-dismissible. Reads:
  *"GenoLeWM is a research tool, not a clinical diagnostic. Talk to a
  qualified genetic counselor."*

### 3.3 Non-features (v0.1)

- No cloud sync.
- No accounts.
- No telemetry.
- No PHI handling beyond what the user drops in.
- No clinical decision-support features (no treatment recommendations,
  no "risk score" framing, no "should you have this baby" framing).
- No reproductive-use prompts or features.
- No background updates.

### 3.4 Privacy contract enforcement

- The Tauri-allowed-host list contains only `huggingface.co` and the
  configured FASTA host. Any other URL is refused at the system level
  via Tauri's allowlist mechanism.
- The embedded runtime runs with `GENO_LEWM_REDACTION_STRICT=1`.
- Crash logs are written to a local path; the app provides a "view
  crash log" button and an explicit "share crash log" action that
  opens a redacted version in the user's default editor.

### 3.5 Update flow

- App updates: signed delta updates with explicit user consent. Tauri's
  built-in updater is used. The update endpoint is the project's
  GitHub releases.
- Model updates: an explicit "check for model updates" button calls
  `geno-lewm-update --check-only` and displays the diff. No automatic
  apply.

### 3.6 Distribution

- macOS: `.dmg` containing the notarized `.app`, downloadable from
  GitHub releases.
- Linux: `.AppImage`, signed.
- Windows: `.msi`, signed.

### 3.7 Source location

```
desktop/
├── README.md
├── package.json            # frontend deps
├── pnpm-lock.yaml
├── src-tauri/
│   ├── Cargo.toml
│   ├── capabilities/
│   │   └── default.json    # HTTP allowlist
│   ├── src/
│   │   ├── main.rs
│   │   ├── lib.rs
│   │   └── runtime.rs      # PyO3 bridge
│   └── tauri.conf.json
└── src/                    # frontend
    ├── main.ts
    └── styles.css
```

### 3.8 Reference workflow (canonical demo)

1. Install the macOS `.app`. First launch downloads Carbon-500M (~1 GB)
   and a published GenoLeWM checkpoint with explicit user consent.
2. Drop a VCF onto the window.
3. Drop the matching reference FASTA, or accept the default download.
4. Press "Score." A progress bar advances. Scoring runs entirely
   locally; network calls are blocked.
5. The result table appears, sorted by calibrated surprise.
6. Click a variant for detail.
7. Click "Export receipt" to save the session receipt.

The whole flow is local-only; the app's only post-setup network calls
are explicit "check for model updates" presses.

## 4. Rationale and alternatives

### 4.1 Why Tauri over Electron?

Electron bundles Chromium and Node.js; the resulting binary is large
(100+ MB) and exposes both runtimes' attack surface. Tauri uses the
system webview (tens of MB binary) and a Rust host. For a privacy-first
local app, smaller and audit-friendlier wins.

### 4.2 Why PyO3-embedded Python rather than a subprocess CLI?

- Lifecycle: an embedded Python process can keep the model warm across
  multiple scoring runs without re-loading Carbon weights.
- Memory: a subprocess would double the memory footprint during the
  cross-process call.
- Determinism: in-process Python lets the GUI bind to the same
  randomness / config state.

### 4.3 Why no React / Vue / Svelte by default?

The UI is intentionally minimal (≤ 10 screens). A framework adds a
build pipeline and dependency surface that the project does not need.
Vanilla TS + a tiny utility library is sufficient.

### 4.4 Why a permanent non-dismissible banner?

The safety story requires it. Banner-dismissal would let users forget
they're looking at research output. The friction is intentional.

### 4.5 Why is the app called "skeleton"?

Polishing UIs is a 10× ops cost compared to writing them. The project
optimizes for "the workflow exists" not "the workflow is beautiful."
Beauty is downstream.

## 5. Unresolved questions

- Whether to ship a PDF / printable report export for genetic-counselor
  conversations. Useful but adds surface; deferred.
- Whether to integrate a local gnomAD frequency lookup for richer
  variant context in the detail view. Probably yes; impl in v0.2.
- Whether to provide a "personal surprise distribution" histogram
  view. Interesting; deferred.
- iOS / iPad: the model is borderline feasible at int4; distribution
  and UX are substantial work. Deferred.

## 6. Future work

- Plugin architecture for community-contributed scoring heads (AlphaMissense
  ensemble, CADD ensemble, etc.).
- Local family-genetics view: opt-in linking of multiple personal
  genome files for shared-variant exploration.
- Offline VCF importers for newer formats as they emerge (e.g., GA4GH
  gVCF).
- WASM-only deployment for the predictor (cache pre-populated server-
  side), where the user uploads reference embeddings (not personal) and
  scoring happens in-browser.

## 7. Changelog

- 2026-06-11 — Updated implementation status for the current Tauri
  scaffold, runtime probe, network policy, drop targets, and safety
  banner; scoring and signed releases remain open.
- 2026-05-20 — Initial draft.
