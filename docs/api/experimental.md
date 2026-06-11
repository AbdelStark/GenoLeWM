# Stability decorators

The `geno_lewm.api` module provides the two decorators that govern
public-surface lifetime during the alpha period.

::: geno_lewm.api
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - experimental
        - deprecated

## Behavioural contract

- `@experimental` emits a single `FutureWarning` **per process** on the
  first call (function) or instantiation (class). Later calls are
  silent. The decorated callable retains `__name__`, `__doc__`,
  `__module__`, `__qualname__`, and `__wrapped__`.
- `@deprecated(reason)` emits a single `DeprecationWarning` **per call
  site** — that is, once per `(filename, lineno)` of the calling code.
  Distinct call sites each receive one warning.

The warnings module sees the project-defined filters (`pyproject.toml
[tool.pytest.ini_options].filterwarnings`) so tests opt-in via
`warnings.simplefilter("always")` blocks.

## When to apply

- **`@experimental`** — symbols that are not yet locked into the
  stability contract. New public surface lands experimental by
  default; promotion to stable is a separate PR with a CHANGELOG
  entry.
- **`@deprecated`** — symbols scheduled for removal in the next MAJOR.
  Pair with a removal target in the CHANGELOG.

See the [public API contract](public-surface.md) for the enforced
snapshot.
