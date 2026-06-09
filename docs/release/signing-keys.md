# Release signing keys

GenoLeWM releases will be signed with maintainer GPG keys once the
release infrastructure is wired into the CI workflow. The keys
themselves are published here when they're issued.

## Status

No signed binaries exist yet. The first package-release target is
`0.2.1`, published through the protected PyPI workflow. This page exists
so [`SECURITY`](../security.md) can reference a stable URL.

## Planned posture

- Maintainer key fingerprints listed here, one per row, with
  `[Maintainer name] — [PGP fingerprint] — [valid from] — [revoked at]`.
- Release artifacts on PyPI published via PyPI trusted publishing
  (OIDC, no long-lived API tokens).
- Release artifacts on GitHub attached to a signed tag and
  Sigstore-backed build provenance.
- Hugging Face Hub model weights signed via the `safetensors`
  manifest; the manifest hash is the trust anchor (RFC-0011 §3.7).

## Until then

- PyPI project releases are published from the `Release PyPI` workflow
  at `.github/workflows/release-pypi.yml` through trusted publishing.
  Verify the workflow's OIDC claim against the trusted-publisher
  configuration before each protected tag.
- The repository's tags are GPG-signed by the project lead. Verify
  with `git tag -v vX.Y.Z` after importing the lead's GPG key.
- Release assets should also verify with the GitHub CLI
  build-provenance verification command for the published artifact.

See also: [SECURITY](../security.md), [supply-chain notes](../privacy.md).
