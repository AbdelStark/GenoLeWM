# Release signing keys

GenoLeWM releases will be signed with maintainer GPG keys once the
release infrastructure is wired into the CI workflow. The keys
themselves are published here when they're issued.

## Status

No signed binaries exist yet. The first package release is `0.2.1`.
The tag workflow built and validated the release distributions, but PyPI
rejected the trusted-publisher exchange with `invalid-publisher`; the
same validated distributions were then published with a maintainer token
and recorded in the release tracker. This page exists so
[`SECURITY`](../security.md) can reference a stable URL.

## Planned posture

- Maintainer key fingerprints listed here, one per row, with
  `[Maintainer name] — [PGP fingerprint] — [valid from] — [revoked at]`.
- Release artifacts on PyPI published via PyPI trusted publishing
  (OIDC) once the account-side publisher mapping is configured; until
  then, any maintainer-token fallback must be recorded in the release
  tracker and public release notes.
- Release artifacts on GitHub attached to a signed tag and
  Sigstore-backed build provenance.
- Hugging Face Hub model weights signed via the `safetensors`
  manifest; the manifest hash is the trust anchor (RFC-0011 §3.7).

## Until then

- PyPI project releases should publish from the `Release PyPI` workflow
  at `.github/workflows/release-pypi.yml` through trusted publishing.
  Verify the workflow's OIDC claim against the trusted-publisher
  configuration before each protected tag; if PyPI rejects the claim,
  use only a scoped maintainer-token fallback and record the exception.
- The repository's tags are GPG-signed by the project lead. Verify
  with `git tag -v vX.Y.Z` after importing the lead's GPG key.
- Release assets should also verify with the GitHub CLI
  build-provenance verification command for the published artifact.

See also: [SECURITY](../security.md), [supply-chain notes](../privacy.md).
