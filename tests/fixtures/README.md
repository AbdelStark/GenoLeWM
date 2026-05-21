# `tests/fixtures/`

Small, committed test data — JSON blobs, sample receipts, sample
manifests — used by the unit / integration / api test layers.

Convention:

- Every fixture is **canonical-JSON** when it represents a hashable
  payload (so receipt / manifest fixtures are byte-stable).
- Each fixture file is kept under 4 kB. Anything larger should live in
  a fixture loader that re-derives the bytes from a recipe (avoids
  bloating the repo for marginal coverage).
- Tests reach the directory via the ``fixtures_dir`` pytest fixture
  (see [`tests/conftest.py`](../conftest.py)) or via
  ``tests.fixtures.load_json(name)``.

The current corpus:

| File | Purpose |
|------|---------|
| `sample_window.fa` | Synthetic 256 bp ``ACGT``-repeating window for commitment tests. |
| `sample_receipt.json` | Canonical Phase-1 receipt JSON for round-trip / verifier tests. |
