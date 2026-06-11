# `bench/results/baseline/`

This directory holds the canonical benchmark baseline that the nightly
performance regression detector
([`tools/ci/perf_regression.py`](../../../tools/ci/perf_regression.py))
diffs against.

- Updated by the release engineer when a budget or benchmark contract is
  intentionally changed. Record the change in `CHANGELOG.md`.
- New `<benchmark>.json` files are accepted automatically; missing
  files are accepted (a new benchmark has no baseline yet).
- Existing files cannot be silently overwritten by a PR; the detector
  reports a `regression` finding when the new median exceeds the
  baseline median by more than the configured threshold (default 5 %).

Locally run benchmarks under `bench/results/<your-machine>/` are
gitignored to keep noisy per-machine numbers out of the tree.
