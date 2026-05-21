# Release signing keys

GenoLeWM releases will be signed with maintainer GPG keys once the
release infrastructure is wired into the CI workflow. The keys
themselves are published here when they're issued.

## Status

No signed binaries exist yet — Phase 0 / Phase 1 are pre-release. This
page exists so [`SECURITY`](../security.md) can reference a stable
URL.

## Planned posture

- Maintainer key fingerprints listed here, one per row, with
  `[Maintainer name] — [PGP fingerprint] — [valid from] — [revoked at]`.
- Release artifacts on PyPI signed via PyPI trusted publishing (OIDC,
  no long-lived API tokens).
- Release artifacts on GitHub attached to a signed tag and a signed
  Sigstore attestation (`cosign verify-blob`).
- Hugging Face Hub model weights signed via the `safetensors`
  manifest; the manifest hash is the trust anchor (RFC-0011 §3.7).

## Until then

- The PyPI project is published from the `Release` workflow at
  `.github/workflows/release.yml`. Verify the workflow's OIDC claim
  against the trusted-publisher configuration at
  <https://pypi.org/manage/project/geno-lewm/settings/publishing/>.
- The repository's tags are GPG-signed by the project lead. Verify
  with `git tag -v vX.Y.Z` after importing the lead's GPG key.

See also: [SECURITY](../security.md), [supply-chain notes](../privacy.md).
