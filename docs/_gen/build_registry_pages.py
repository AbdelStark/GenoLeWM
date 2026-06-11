"""Generate per-registry reference pages.

Two pages drive the human-readable contract:

- ``docs/api/error-codes.md`` — every entry in
  :data:`geno_lewm.errors.ERROR_CODES`.
- ``docs/api/log-events.md`` — every entry in
  :data:`geno_lewm.observability.EVENTS`.

Built at ``mkdocs build`` time so the docs page cannot drift from the
runtime registry.
"""

from __future__ import annotations

import mkdocs_gen_files  # type: ignore[import-not-found]

from geno_lewm.errors import _EXIT_CODE_BY_FAMILY, ERROR_CODES, GenoLeWMError, exit_code_for
from geno_lewm.observability import EVENTS


def _error_family(cls: type[GenoLeWMError]) -> str:
    return cls.code.split(".", 1)[0]


def _exit_code(cls: type[GenoLeWMError]) -> int:
    return exit_code_for(cls())


def build_error_codes() -> None:
    with mkdocs_gen_files.open("api/error-codes.md", "w") as fd:
        fd.write("# Error codes\n\n")
        fd.write(
            "Stable error codes raised by GenoLeWM. The table is generated from\n"
            "`geno_lewm.errors.ERROR_CODES` at docs-build time; renaming any\n"
            "code is a breaking public contract change.\n\n"
        )
        fd.write("| Code | Exception class | Family | Exit code | Summary |\n")
        fd.write("|------|------------------|--------|-----------|---------|\n")
        for entry in ERROR_CODES:
            family = _error_family(entry.exception_class)
            exit_code = _exit_code(entry.exception_class)
            fd.write(
                f"| `{entry.code}` "
                f"| `{entry.exception_class.__name__}` "
                f"| {family} "
                f"| {exit_code} "
                f"| {entry.summary} |\n"
            )

        fd.write("\n## Exit code mapping\n\n")
        fd.write(
            "Family-level mapping applied by `exit_code_for(exc)`. Non-`GenoLeWMError`\n"
            "exceptions map to exit code 1; `KeyboardInterrupt` maps to 130.\n\n"
        )
        fd.write("| Family | Exit code |\n|--------|-----------|\n")
        for family_cls, code in _EXIT_CODE_BY_FAMILY:
            fd.write(f"| `{family_cls.__name__}` | {code} |\n")


def build_log_events() -> None:
    with mkdocs_gen_files.open("api/log-events.md", "w") as fd:
        fd.write("# Log events\n\n")
        fd.write(
            "Every event accepted by `geno_lewm.observability.GenoLeWMLogger`.\n"
            "Generated from `geno_lewm.observability.EVENTS` at docs-build time;\n"
            "renaming an event is a breaking public contract change.\n\n"
            "Payload keys not in `allowed_keys` are dropped by the redaction\n"
            "filter. Standardized fields (`step`, `epoch`,\n"
            "`phase`, `duration_ms`, `trace_id`, `span_id`, `error_code`) are\n"
            "promoted out of `data` and are always allowed at the top level.\n\n"
        )
        fd.write("| Event | Severity | Allowed `data` keys | Summary |\n")
        fd.write("|-------|----------|--------------------|---------|\n")
        for ev in EVENTS:
            keys = (
                ", ".join(f"`{k}`" for k in sorted(ev.allowed_keys))
                if ev.allowed_keys
                else "_(none)_"
            )
            fd.write(f"| `{ev.name}` | `{ev.severity}` | {keys} | {ev.summary} |\n")


def main() -> None:
    build_error_codes()
    build_log_events()


main()
