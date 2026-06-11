# Public API Contract

GenoLeWM exposes a compact alpha API for edit specs, edit application,
training/evaluation helpers, planning primitives, surprise scoring,
runtime loading, provenance receipts, and command-line entry points.

The exhaustive public symbol set is committed at
`tests/api/public_surface.json`. CI checks that snapshot with:

```bash
uv run python tools/api/snapshot.py check
```

Any public addition, removal, or rename must update the snapshot and
explain the compatibility impact in the changelog or pull request.

## Stability Decorators

`geno_lewm.api.experimental` marks symbols that may change during the
alpha period. `geno_lewm.api.deprecated` marks symbols scheduled for a
future breaking release. Both decorators preserve normal Python metadata
so downstream tools can introspect wrapped callables.

## Top-Level Imports

The stable top-level package exports are the entries in
`tests/api/public_surface.json`. Planning solver result types such as
`PlanningConfig`, `PlanningResult`, and `plan` are not stable top-level exports yet; import planning internals from
`geno_lewm.planning` when you need them.

## CLI Surface

The public command-line entry points are declared in `pyproject.toml`.
CLI scaffold factory helpers such as `build_stub_app` and
`make_cli_main` are internal test/development helpers and are not part
of the public API snapshot.
