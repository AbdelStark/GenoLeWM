# 06 — Security

- Status: Authoritative for v0.1
- Companion RFCs: [RFC-0010](../rfcs/0010-on-device-personal-genome-deployment.md),
  [RFC-0011](../rfcs/0011-artifact-provenance-receipts.md),
  [RFC-0014](../rfcs/0014-public-api-and-stability.md)

GenoLeWM operates over the most sensitive consumer data category there
is: personal genome data. The security model is therefore conservative,
local-first, and built around explicit trust boundaries. Threats are
enumerated, not implied. Defenses are mechanical, not policy.

Current public release status: model, dataset, demo/replay, and
benchmark/paper artifacts are published. The released demo paths record
runtime-preflight, artifact identity, redaction, and checksum receipt
evidence for the published artifacts. That evidence does not establish a
general privacy assurance, clinical safety claim, or runtime assurance
mode beyond checksum provenance.

## Threat model

### Assets

| Asset | Why it matters | Where it lives |
|-------|----------------|----------------|
| User VCF / variant data | Permanent, identifying, family-implicating | User filesystem only |
| Reference embeddings cache | Public-domain (reference genome derivatives) | User filesystem |
| Trained model weights | Public; signed by maintainers | Hugging Face Hub + user filesystem |
| Receipts | Checksum provenance | User filesystem; shareable |
| Manifest | Trust anchor for model identity | Inside checkpoint |
| Calibration table | Affects score interpretation | Inside checkpoint |
| Surprise scores | User-derived analysis; user-owned | User filesystem only |

### Attackers

| Attacker | Capability | Goal |
|----------|------------|------|
| Network attacker | passive eavesdropping or active MitM during first-run downloads | exfiltrate variant data; substitute model weights |
| Local malware | arbitrary process on the user's machine | exfiltrate variant data; tamper with model |
| Supply-chain attacker | publishes a backdoored release upstream | poisoned weights produce manipulated scores |
| Curious recipient of a shared result | wants to learn the input that produced a published score | reverse-engineer the input from receipt |
| Misconfigured operator | mis-uses runtime in cloud context | accidentally leaks variants |

### Out of attacker model (explicit)

- Physical-access attackers with control of the hardware.
- Side-channel attackers on shared hardware (mitigated only when running
  on dedicated user hardware, which is the deployment context).
- Cryptographic break of SHA-256.

## Trust boundaries

```
┌─────────────────────────────────────────────────────────────┐
│ User device                                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ GenoLeWM runtime (untrusted to the user beyond manifest │ │
│  │ verification; trusted to itself once verified)         │ │
│  │                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │ │
│  │  │ Carbon       │  │ Predictor    │  │ Provenance   │ │ │
│  │  │ (frozen, by  │  │ (small       │  │ (manifest +  │ │ │
│  │  │  public      │  │  trainable   │  │  receipts)   │ │ │
│  │  │  commitment) │  │  head)       │  │              │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  User VCF ──► runtime (in-process only) ──► scores + receipt │
└──────────────────────────────────────────────────────────────┘
            │
            │ explicit user-initiated network calls only:
            │  - first-run weight download (Hugging Face Hub, pinned hash)
            │  - explicit `geno-lewm-update`
            ▼
   Hugging Face Hub (untrusted; verified by content hash)
```

The released runtime must fail closed on any non-allowlisted network
call. The allowlist is a single configuration file shipped with the
build, and the loader verifies its hash on startup.

## Defenses

### 1. Local-first release requirement

- The released runtime must make no network call after first-run setup.
- A guard at the HTTP-client construction site raises
  `NetworkCallProhibitedError` if any call is attempted outside the
  documented allowlist.
- The desktop app release target has no cloud sync, no accounts, no
  telemetry. See [RFC-0010 §3.7](../rfcs/0010-on-device-personal-genome-deployment.md#37-privacy-contract).

### 2. Content-addressed weights

- Every weight file is identified by SHA-256 of its `safetensors` bytes.
- The manifest binds weights, calibration, tokenizer, and configuration
  into one canonical-JSON document whose hash is the `model_id`.
- On load, the runtime recomputes every hash and refuses to start if any
  fails.
- Pinning to `model_id` is the published-results contract.

### 3. Input commitments and receipts

- Single-variant scoring can produce a receipt binding
  `(model_id, input_commitment, output, runtime metadata)`
  ([`03-data-model.md`](03-data-model.md)).
- VCF scoring can produce a JSONL sidecar with one v1 receipt per scored
  alternate. Receipts contain commitments and runtime metadata, not raw
  DNA sequence or VCF fields.
- Receipts are checksum-only in the active schema.
- `batch_receipt_report.json` is a release artifact that verifies the
  score JSONL and per-row receipt JSONL streams as one batch. It does
  not collapse a multi-row VCF output into one single-output v1 receipt.

### 4. Redaction by default

- The observability layer (see [`05-observability.md`](05-observability.md))
  rejects DNA strings ≥ 20 bp, drops unallowlisted keys, and fails closed
  under `GENO_LEWM_REDACTION_STRICT=1` (the default).
- Crash dumps sanitize locals before write.
- VCF parsing never logs variant content.

### 5. Reproducible builds

- Release artifacts must be built from pinned dependency lockfiles. The
  CI release workflow records the build environment hash and embeds it
  in the manifest.
- A `--reproducible` build flag uses a deterministic source date and a
  fixed PYTHONHASHSEED.

### 6. Signed releases

- GitHub releases must publish Sigstore-backed build provenance.
- macOS binaries are notarized and Linux binaries are signed with the
  project's GPG key when those packaged binaries are cut.
- The signing key fingerprints are published in [`SECURITY.md`](../security.md)
  at the repo root.

### 7. Dependency hygiene

- Direct dependencies are pinned to a minimum version with a justified
  upper bound only when required.
- A nightly CI job runs `pip-audit` / `safety check`. Any new advisory
  blocks a release until triaged.
- Transitive lockfile is committed for the release process.

### 8. Failure containment

- Network code paths are confined to `geno_lewm/deploy/runtime.py` and
  `geno_lewm/cli/update.py`. A custom AST check fails CI if any other
  module imports `urllib`, `httpx`, `requests`, or similar.
- Disk-write paths are similarly confined.

## Secrets handling

- The released runtime must require no API keys. Hugging Face Hub
  downloads use anonymous tokens unless the user explicitly
  authenticates.
- Authentication tokens, when supplied, are read from `HF_TOKEN` and never
  logged or stored in receipts.
- No GenoLeWM artifact contains credentials.

## Cryptographic primitives

| Use | Primitive | Notes |
|-----|-----------|-------|
| Content hashes (weights, windows, manifest) | SHA-256 | from `hashlib` |
| Canonical JSON serialization | sorted-keys UTF-8 no-whitespace | helper in `provenance/hashing.py` |

No custom crypto is implemented.

## Disclosure policy

- Security issues are reported privately via GitHub Security Advisories
  or `security@<project domain>` (see [`SECURITY.md`](../security.md)).
- Acknowledgement target: 72 hours.
- Triaged-fix target: 30 days for high; 90 days for medium.
- Coordinated disclosure with an embargo window of up to 90 days.
- Public advisories follow CVE numbering when applicable.

## Safety boundaries

Independent of confidentiality, the project enforces safety boundaries
on use:

- The released runtime, CLI, and desktop app must surface a permanent,
  non-dismissible "research tool — not clinical" banner.
- The CLI emits a warning when ClinVar P/LP variants are detected, pointing
  to clinical follow-up.
- The license addendum (if adopted; see OQ-OVR-2) restricts germline-edit
  reproductive use.

## Invariants

| ID | Invariant | Enforced by |
|----|-----------|-------------|
| INV-SEC-1 | No inference path calls the network after first-run setup | runtime guard + CI integration test |
| INV-SEC-2 | Every loaded weight file matches its hash in the manifest | loader call-site in `deploy/runtime.py` |
| INV-SEC-3 | Receipts produced for the same `(model_id, input)` agree on supported backends | conformance test |
| INV-SEC-4 | DNA strings of length ≥ 20 bp never appear in logs | property test |
| INV-SEC-5 | Network paths exist only in `deploy/runtime.py` and `cli/update.py` | AST check |
| INV-SEC-6 | No credentials are written to disk by the runtime | dedicated test |
| INV-SEC-7 | Personal-data fields are dropped before any non-local sink | redaction unit tests |

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-SEC-1 | Resolved: `geno_lewm.provenance` is the preferred public path; the legacy import path is removed | core | done |
| OQ-SEC-2 | License addendum forbidding germline-edit reproductive use vs README-only disclaimer | core | before v0.1 tag |
| OQ-SEC-3 | Whether to provide a `--audit-log` mode that records all file accesses for forensic review | core | post-v1 |
| OQ-SEC-4 | Mechanism for revoking a published model_id when a serious bug is found (revocation list?) | core | before first model release |
