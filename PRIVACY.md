# Privacy

GenoLeWM is designed to keep your genome data on your device. This file
documents the guarantees the runtime makes about your data — what it
does, what it never does, and how those properties are enforced.

A fuller technical treatment is in
[`docs/spec/06-security.md`](docs/spec/06-security.md) and in
[RFC-0010](rfcs/0010-on-device-personal-genome-deployment.md).

## What we do with your data

- Your VCF / 23andMe / WGS / FASTA inputs stay on your local filesystem.
- Scoring runs entirely in-process on your device.
- Outputs (scores, receipts, plans) are written to the path you specify
  and nowhere else.

## What we never do

- We never upload your genome data anywhere.
- We never collect telemetry about what you score.
- We never log variant content. Logs record variant identifiers (hashes
  of `chrom+pos+ref+alt`) and categorical outputs — not the bases.
- We never embed credentials in any released artifact.
- We never silently fall back to a hosted API if local inference fails.

## How those guarantees are enforced

1. **Fail-closed network guard.** The runtime refuses any HTTP call that
   is not on a documented allowlist (only Hugging Face Hub for
   first-run weight download; only the FASTA host you configured).
2. **Redaction by default.** The observability layer drops any DNA
   string ≥ 20 bp and any field outside a per-event allowlist. The
   default mode (`GENO_LEWM_REDACTION_STRICT=1`) raises an internal
   error if the rule is bypassed rather than silently dropping.
3. **Local-only sinks.** wandb and OpenTelemetry sinks are opt-in.
   They are off by default and require explicit env vars or CLI flags
   to enable.
4. **Crash sanitization.** Stack frames in crash dumps strip locals;
   only model identifiers and code paths survive.
5. **Receipts contain identifiers, not data.** A receipt records the
   `model_id`, the `input_commitment` (a hash), and the categorical
   output — never the raw variant.
6. **Personal data never appears in tests.** Test fixtures are
   synthetic; any fixture under `tests/fixtures/` requires a reviewer
   confirmation.

## What we ask your network for

Only when you explicitly initiate it:

| Action | Endpoint |
|--------|----------|
| First-run weight download | `https://huggingface.co/HuggingFaceBio/Carbon-500M` and the GenoLeWM checkpoint repo |
| Reference FASTA download (optional) | the host you configured (e.g., 1000 Genomes) |
| `geno-lewm-update --check-only` | GitHub releases endpoint for the project |
| `geno-lewm-update` (apply) | the corresponding release asset URL |

All four are gated on explicit user action and surface the URL to stderr
before issuing the request. None of these endpoints receive variant data.

## Sharing scores

If you share a surprise score with a third party (a genetic counselor,
a collaborator), the recommended channel is the receipt:

- The receipt binds the `model_id`, the input commitment, and the
  output.
- It does **not** include the variant bases.
- The recipient can verify the receipt against the public model using
  `geno-lewm-verify`.

If you want to share the input as well, that is your choice; the receipt
format does not embed it.

## Special-purpose modes

- **`--per-variant-receipts`** writes one receipt per variant when
  scoring a VCF. This is opt-in because it produces large outputs.
- **`--audit-log`** (planned post-v1) records every file access during
  scoring. Useful for reproducibility audits. Off by default.

## What you can do

- Use the desktop app's "view crash log" before "share crash log" — both
  open redacted versions and the latter is gated on a click.
- Set `GENO_LEWM_LOG_LEVEL=warn` or `error` to reduce log volume.
- Disable the prometheus textfile exporter via `--metrics-interval-s 0`.
- Use the project's CLI from an air-gapped environment after the first-
  run download. All subsequent operations are offline.

## Compliance posture

GenoLeWM is a research tool. We make no specific HIPAA / GDPR /
clinical-regulation claims. The local-first architecture is the
foundation on which compliant downstream tools can be built; that
compliance is the responsibility of the downstream tool.

The license does not include a data-rights clause because we never
collect data; nothing to grant. If you redistribute scoring outputs,
the data-rights questions are between you and your recipients.

## Reporting privacy concerns

Privacy issues that constitute data exfiltration are treated as
Critical-severity security issues. See [`SECURITY.md`](SECURITY.md) for
the reporting channel.

## Changes to this policy

Changes to the privacy posture are MAJOR changes. The CHANGELOG calls
out any change under the `Security` section. We do not silently tighten
or loosen this policy.
