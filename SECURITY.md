# Security policy

GenoLeWM operates over personal genome data — permanent, identifying,
family-implicating. This policy is more conservative than typical
open-source norms; the threat model is in
[`docs/spec/06-security.md`](docs/spec/06-security.md) and the
artifact provenance primitives are in
[`docs/spec/06-security.md`](docs/spec/06-security.md) plus
[RFC-0011](rfcs/0011-artifact-provenance-receipts.md).

## Reporting a vulnerability

Please **do not** open public issues for security vulnerabilities.

- Preferred: GitHub Security Advisories on this repository.
- Alternative: email `security@<project domain>` (replace with current
  maintainer address; see the bottom of this file).

Include:

- A description of the issue and the affected component (preferably with
  a link into [`docs/spec/`](docs/spec/) or a specific RFC).
- A minimal reproducer if you have one. **Do not include personal genome
  data.** Synthetic VCFs and FASTAs are fine; redact any real input.
- The version, commit, OS, runtime backend, and whether you ran the
  desktop app or the Python library.
- Your preferred contact and acknowledgement preferences.

## Response targets

| Step | Target |
|------|--------|
| Acknowledgement | within 72 hours |
| Triage decision (accept / decline / need-more-info) | within 7 days |
| Fix for high-severity issues | within 30 days |
| Fix for medium-severity issues | within 90 days |
| Coordinated disclosure embargo | up to 90 days from triage |

## Severity bands

| Band | Meaning |
|------|---------|
| **Critical** | personal data exfiltration, signed-build forgery, provenance-check bypass |
| **High** | bypass of network-fail-closed guard, manifest hash collision, weight substitution |
| **Medium** | denial of service, redaction-filter bypass without exfiltration, deterministic-build break |
| **Low** | parser crashes on malformed input that does not leak data |

## Supported versions

The active MAJOR receives security fixes immediately. The previous
MAJOR receives security fixes for 6 months after the next MAJOR's
release. Older versions are best-effort.

## Out-of-scope

- Issues that require root / administrator on the user's machine.
- Side-channel attacks on shared hardware (we assume dedicated user
  hardware).
- Cryptographic break of SHA-256.
- Issues in third-party dependencies that we cannot fix and that have
  no GenoLeWM-side mitigation; we will document and forward upstream.

## Trust anchors

- Maintainer GPG keys for release signing are published in
  [`docs/release/signing-keys.md`](docs/release/signing-keys.md) once the
  release infrastructure lands; until then, no signed binaries exist and
  this file says so explicitly.
- The Hugging Face Hub repository for the project is the canonical
  weight host.
- Sigstore / GitHub build provenance applies to release artifacts
  once published.

## Privacy-related issues

See [`PRIVACY.md`](PRIVACY.md) for the user-data contract. Privacy
issues that constitute data exfiltration are treated as Critical
severity.

## Maintainer contact

The current security contact list is maintained in
[`docs/maintainers.md`](docs/maintainers.md). Maintainers are encouraged
to also use the project's GitHub Security Advisory inbox for triage so
multiple eyes see incoming reports.
