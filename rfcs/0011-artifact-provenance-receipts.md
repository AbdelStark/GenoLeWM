# RFC-0011: Artifact provenance and receipts

- **Status:** Accepted
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-06-11
- **Depends on:** RFC-0002, RFC-0004, RFC-0009, RFC-0010
- **Supersedes:** the original RFC-0011 scope that included runtime
  assurance mechanisms beyond checksum provenance.

---

## 1. Summary

GenoLeWM needs lightweight, reproducible artifact provenance for model
releases and demos. This RFC specifies:

- content-addressed model manifests;
- input commitments;
- output receipts;
- checksum-mode receipt verification.

The scope is deliberately narrow. Receipts help detect tampering and
reproduce a score with the same artifacts and inputs. They do not claim
execution guarantees beyond checksum reproducibility.

## 2. Motivation

The first GenoLeWM paper/demo release needs auditable artifacts:

- a model identity that binds weights, calibration, config, and eval
  report;
- a receipt format that records which model and input produced a score;
- a command-line verifier that can catch mismatched manifests, altered
  outputs, and input-commitment mismatches.

This is enough for reproducibility hygiene while the project focuses on
the first real model, dataset release, evaluation report, and terminal
demo.

## 3. Manifest

Every checkpoint directory includes `manifest.json`. The model identifier
is:

```text
model_id = SHA-256(canonical_json(manifest))
```

The manifest records:

- model name, version, release id;
- encoder id, revision, and hash;
- predictor, action encoder, calibration, training config, and eval
  report artifacts;
- data snapshot identifiers for the training/eval inputs.

Canonical JSON uses sorted keys, UTF-8, no extra whitespace, and finite
JSON-native values only.

## 4. Input Commitment

For every score call, callers can derive:

```text
input_commitment = SHA-256(
    canonical_json(reference_window, edit_spec, pooling_config, dtype_config)
)
```

The commitment binds the receipt to the exact input fields used by the
score path without requiring the receipt itself to expose personal
genome sequence content.

## 5. Output Receipt

Receipt schema v1.0.0:

```json
{
  "schema_version": "1.0.0",
  "model_id": "sha256:...",
  "input_commitment": "sha256:...",
  "output": {
    "sigma_raw": 0.347,
    "sigma_calibrated": 0.92,
    "bucket_id": "coding_missense|mid|none",
    "confidence": 1.0,
    "low_confidence": false
  },
  "output_commitment": "sha256:...",
  "calibration_hash": "sha256:...",
  "runtime": {
    "backend": "coreml",
    "device": "Apple M3 Max",
    "geno_lewm_version": "0.1.0",
    "carbon_revision": "main@<sha>"
  },
  "timestamp": "2026-MM-DDTHH:MM:SSZ",
  "provenance": {
    "kind": "checksum_only",
    "details": null
  }
}
```

`provenance.kind` is the checksum-provenance mode. The only valid value
is `checksum_only`.

## 6. Verification Protocol

`geno-lewm-verify`:

1. parses and validates receipt JSON;
2. loads a local manifest;
3. recomputes the manifest hash and compares it with `receipt.model_id`;
4. optionally recomputes `input_commitment` from supplied input flags;
5. recomputes `output_commitment` from the receipt output block.

Exit codes follow RFC-0012 and `docs/spec/04-error-model.md`.

## 7. Security and Privacy Boundaries

Receipts are safe to share only if the user accepts that model identity,
runtime metadata, score fields, and input commitment are disclosed. The
receipt must not contain raw DNA sequence, sample identifiers, or VCF
content.

Hashes are for integrity and reproducibility. They are not a substitute
for privacy review or clinical validation.

## 8. Implementation Status

Implemented:

- manifest dataclasses and canonical JSON I/O;
- SHA-256 helpers;
- input and output commitments;
- receipt dataclasses and canonical JSON I/O;
- checksum-mode verifier CLI;
- single-variant runtime and CLI receipt emission for manifest-verified
  local scorer components;
- per-row VCF receipt JSONL sidecars using the same v1 single-output
  receipt schema;
- `tools.release.batch_receipt_report` verification for score/receipt
  JSONL streams;
- terminal-demo runtime preflight and release-package checks that bind
  manifest identity, score/receipt artifacts, batch receipt summaries,
  and portable command inputs;
- tutorial fixture and tests.

Open:

- keep downstream docs and examples aligned with `geno_lewm.provenance`
  as public artifacts evolve;
- keep validating receipt emission for each new published checkpoint and
  terminal-demo artifact set;
- runtime-assurance modes beyond checksum provenance remain out of
  scope unless a separate RFC reintroduces them with evidence.

## 9. Changelog

- 2026-06-11: updated implementation status for batch receipt reports,
  terminal-demo runtime preflight binding, and repeated per-release
  receipt-validation work.
- 2026-06-01: narrowed RFC-0011 to artifact provenance and checksum
  receipts; removed unsupported runtime assurance modes from the active
  schema and roadmap.
- 2026-06-01: made `geno_lewm.provenance` the only active public namespace
  for manifest, receipt, hashing, and commitment helpers.
