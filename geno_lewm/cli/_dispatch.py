# SPDX-License-Identifier: Apache-2.0
"""Shared CLI dispatch helpers (RFC-0018 §3.2, §3.4, §3.7).

Every console script in :mod:`geno_lewm.cli` that lands as a Typer app
imports the helpers in this module so that:

* The shared flag set (``--config``, ``--seed``, ``--log-level``, …) is
  defined exactly once.
* The non-dismissible safety banner (RFC-0018 §3.7) is printed by the
  same code path for every command.
* Exceptions are caught at exactly one place and mapped to the exit
  codes specified by ``docs/spec/04-error-model.md`` via
  :func:`geno_lewm.errors.exit_code_for`.

The verify command predates Typer and keeps its argparse-based
implementation; everything else dispatches through here.

Public surface:

* :class:`SharedOptions` — typed dataclass of the shared flag values.
* :func:`shared_option_decls` — builders that return the Typer option
  declarations so every command spells its flags the same way.
* :func:`finalize_shared` — validates inputs, prints the banner, and
  returns a :class:`SharedOptions`.
* :func:`run_app` — wraps a ``typer.Typer`` invocation so uncaught
  :class:`geno_lewm.errors.GenoLeWMError` instances map to the
  documented exit codes.
* :data:`SHARED_FLAG_HELP` — help-text table reused in tests and docs.
* :func:`not_yet_implemented` — Phase 1 stub helper for commands whose
  subsystem has not yet landed.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import IO

import typer

from geno_lewm import __version__
from geno_lewm.errors import GenoLeWMError, InputError, InternalError, exit_code_for

# ---------------------------------------------------------------------------
# Banner (RFC-0018 §3.7)
# ---------------------------------------------------------------------------

#: The non-dismissible safety banner. Suppressed only by ``--quiet
#: --no-banner`` (both required); RFC-0018 §3.7 explains why.
BANNER: str = f"GenoLeWM v{__version__} — research tool, not a clinical diagnostic"


def print_banner(*, quiet: bool, no_banner: bool, stream: IO[str] | None = None) -> None:
    """Print :data:`BANNER` to ``stream`` unless both suppression flags are set."""
    if quiet and no_banner:
        return
    if stream is None:
        stream = sys.stderr
    stream.write(BANNER + "\n")
    stream.flush()


# ---------------------------------------------------------------------------
# Shared flags (RFC-0018 §3.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedOptions:
    """Resolved values for the flags every command accepts."""

    config: str | None
    set_overrides: tuple[str, ...]
    seed: int | None
    deterministic: bool
    log_level: str
    log_dir: str | None
    run_id: str | None
    wandb_project: str | None
    no_receipt: bool
    print_config: bool
    print_config_tree: bool
    explain: str | None
    quiet: bool
    no_banner: bool


#: Help text for every shared flag, surfaced in tests so the contract
#: cannot drift between commands.
SHARED_FLAG_HELP: dict[str, str] = {
    "config": "Path to a Hydra YAML config (RFC-0017).",
    "set": "Hydra-style key=value override (repeatable).",
    "seed": "Override the resolved config seed.",
    "deterministic": "Force deterministic backends (RFC-0005).",
    "log_level": "Logging severity (debug, info, warn, error).",
    "log_dir": "Logging sink root (default: $GENO_LEWM_LOG_DIR).",
    "run_id": "Run identifier; also used as the wandb run id when wandb is enabled.",
    "wandb_project": "Enable the wandb sink in the named project.",
    "no_receipt": "Disable receipt writing where applicable.",
    "print_config": "Print the resolved config and exit (RFC-0017 §3.8).",
    "print_config_tree": "Print the resolved config including each value's source file.",
    "explain": "Print the schema docstring + default + type for a config key.",
    "quiet": "Silence info-level logs to stderr (RFC-0018 §3.5).",
    "no_banner": "Suppress the safety banner (must be combined with --quiet).",
    "version": "Print the package version and exit.",
}


VALID_LOG_LEVELS: tuple[str, ...] = ("debug", "info", "warn", "error")

_PANEL = "Shared (RFC-0018 §3.2)"


def shared_option_decls() -> dict[str, typer.models.OptionInfo]:
    """Return Typer ``OptionInfo`` objects keyed by parameter name.

    Each command's callback unpacks this dict into its parameter
    annotations so every command spells its shared flags identically.
    The annotations are consumed with ``Annotated[T, OptionInfo]`` —
    the parameter default lives on the function signature, not on the
    ``OptionInfo`` itself.
    """
    return {
        "config": typer.Option("--config", help=SHARED_FLAG_HELP["config"], rich_help_panel=_PANEL),
        "set_overrides": typer.Option(
            "--set", "-s", help=SHARED_FLAG_HELP["set"], rich_help_panel=_PANEL
        ),
        "seed": typer.Option("--seed", help=SHARED_FLAG_HELP["seed"], rich_help_panel=_PANEL),
        "deterministic": typer.Option(
            "--deterministic",
            help=SHARED_FLAG_HELP["deterministic"],
            rich_help_panel=_PANEL,
        ),
        "log_level": typer.Option(
            "--log-level", help=SHARED_FLAG_HELP["log_level"], rich_help_panel=_PANEL
        ),
        "log_dir": typer.Option(
            "--log-dir", help=SHARED_FLAG_HELP["log_dir"], rich_help_panel=_PANEL
        ),
        "run_id": typer.Option("--run-id", help=SHARED_FLAG_HELP["run_id"], rich_help_panel=_PANEL),
        "wandb_project": typer.Option(
            "--wandb-project",
            help=SHARED_FLAG_HELP["wandb_project"],
            rich_help_panel=_PANEL,
        ),
        "no_receipt": typer.Option(
            "--no-receipt", help=SHARED_FLAG_HELP["no_receipt"], rich_help_panel=_PANEL
        ),
        "print_config": typer.Option(
            "--print-config",
            help=SHARED_FLAG_HELP["print_config"],
            rich_help_panel=_PANEL,
        ),
        "print_config_tree": typer.Option(
            "--print-config-tree",
            help=SHARED_FLAG_HELP["print_config_tree"],
            rich_help_panel=_PANEL,
        ),
        "explain": typer.Option(
            "--explain", help=SHARED_FLAG_HELP["explain"], rich_help_panel=_PANEL
        ),
        "quiet": typer.Option("--quiet", help=SHARED_FLAG_HELP["quiet"], rich_help_panel=_PANEL),
        "no_banner": typer.Option(
            "--no-banner", help=SHARED_FLAG_HELP["no_banner"], rich_help_panel=_PANEL
        ),
        "version": typer.Option(
            "--version",
            help=SHARED_FLAG_HELP["version"],
            is_eager=True,
            rich_help_panel=_PANEL,
        ),
    }


def finalize_shared(
    *,
    config: str | None,
    set_overrides: list[str] | None,
    seed: int | None,
    deterministic: bool,
    log_level: str,
    log_dir: str | None,
    run_id: str | None,
    wandb_project: str | None,
    no_receipt: bool,
    print_config: bool,
    print_config_tree: bool,
    explain: str | None,
    quiet: bool,
    no_banner: bool,
    version: bool,
    default_config_name: str = "train",
) -> SharedOptions | None:
    """Validate inputs, handle discovery flags, print the banner, return options.

    The function returns ``None`` for every flag that documents itself
    as "print and exit" (``--version``, ``--print-config``,
    ``--print-config-tree``, ``--explain``). Typer treats the empty
    return as exit code 0.

    ``default_config_name`` selects the canonical YAML template
    (``train`` / ``score`` / ``eval`` / ``plan``) loaded for the
    discovery flags. Per-command stubs override this.

    Raises:
        InputError: if ``--log-level`` is outside :data:`VALID_LOG_LEVELS`.
    """
    if version:
        sys.stdout.write(f"geno-lewm {__version__}\n")
        return None

    if log_level not in VALID_LOG_LEVELS:
        raise InputError(
            "invalid --log-level",
            details={"got": log_level, "allowed": list(VALID_LOG_LEVELS)},
        )

    # Discovery flags: handled before banner so they exit cleanly under
    # `--quiet --no-banner` regardless. ``--explain`` is the cheapest
    # and runs without loading a config.
    if explain is not None:
        _emit_explain(explain)
        return None

    if print_config or print_config_tree:
        _emit_resolved_config(
            default_name=default_config_name,
            config_path=config,
            include_source=print_config_tree,
        )
        return None

    print_banner(quiet=quiet, no_banner=no_banner)

    resolved_wandb_project = wandb_project or os.environ.get("WANDB_PROJECT")
    if wandb_project is not None:
        from geno_lewm import observability

        observability._set_wandb_project(wandb_project)

    return SharedOptions(
        config=config,
        set_overrides=tuple(set_overrides or ()),
        seed=seed,
        deterministic=deterministic,
        log_level=log_level,
        log_dir=log_dir,
        run_id=run_id,
        wandb_project=resolved_wandb_project,
        no_receipt=no_receipt,
        print_config=print_config,
        print_config_tree=print_config_tree,
        explain=explain,
        quiet=quiet,
        no_banner=no_banner,
    )


# ---------------------------------------------------------------------------
# Discovery flag implementations (RFC-0017 §3.8)
# ---------------------------------------------------------------------------


def _emit_explain(key: str) -> None:
    """Render the schema docstring + default + type for a dotted key."""
    from geno_lewm.config import describe_field

    info = describe_field(key)
    sys.stdout.write(f"{key}:\n")
    sys.stdout.write(f"  type:    {info['type']}\n")
    sys.stdout.write(f"  default: {info['default']!r}\n")
    sys.stdout.write(f"  doc:     {info['doc']}\n")


def _emit_resolved_config(
    *,
    default_name: str,
    config_path: str | None,
    include_source: bool,
) -> None:
    """Print the resolved config as YAML to stdout (``--print-config[-tree]``)."""
    import yaml

    from geno_lewm.config import config_to_dict, load_config, load_default
    from geno_lewm.config.loader import DEFAULTS_DIR

    if config_path is not None:
        cfg = load_config(config_path)
        source = config_path
    else:
        cfg = load_default(default_name)
        source = str(DEFAULTS_DIR / f"{default_name}.yaml")

    payload = config_to_dict(cfg)
    if include_source:
        # ``--print-config-tree`` annotates every value with its source.
        # Phase 1 has exactly one source per value (the loaded YAML);
        # the Hydra-style multi-source composition (#) lands when the
        # loader gains a ``defaults:`` block.
        sys.stdout.write(f"# resolved from: {source}\n")
    sys.stdout.write(yaml.safe_dump(payload, sort_keys=True, default_flow_style=False))


# ---------------------------------------------------------------------------
# Exit-code wrapper
# ---------------------------------------------------------------------------


def run_app(app: typer.Typer, argv: Sequence[str] | None = None) -> int:
    """Invoke a Typer app and map errors to documented exit codes.

    Resolves the argv from ``sys.argv`` when not provided. Catches
    :class:`GenoLeWMError`, :class:`KeyboardInterrupt`, and ``SystemExit``;
    everything else propagates so internal bugs stay visible in tracebacks.
    """
    args = list(argv) if argv is not None else None
    try:
        app(args=args, standalone_mode=False)
        return 0
    except typer.Exit as exc:
        return int(exc.exit_code)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 1 if exc.code is not None else 0
    except KeyboardInterrupt:
        return 130
    except GenoLeWMError as exc:
        _emit_error(exc)
        return exit_code_for(exc)


def _emit_error(exc: GenoLeWMError) -> None:
    """Single-place stderr write for typed errors (RFC-0012)."""
    details = ""
    if exc.details:
        details = " " + ", ".join(f"{k}={v!r}" for k, v in exc.details.items())
    sys.stderr.write(f"{type(exc).__name__}: {exc}{details}\n")


# ---------------------------------------------------------------------------
# Stub helper (used by every Phase 1 command)
# ---------------------------------------------------------------------------


def not_yet_implemented(
    *,
    command: str,
    issue: str,
    detail: str | None = None,
) -> None:
    """Raise an :class:`InternalError` advertising the tracking issue.

    Phase 1 stub. The dispatcher's :func:`run_app` catches the error
    and maps it to exit code 9 (``InternalError`` family) per
    ``docs/spec/04-error-model.md``.
    """
    msg = (
        f"geno-lewm {command}: not yet implemented; tracks {issue}."
        if detail is None
        else f"geno-lewm {command}: {detail} (tracks {issue})."
    )
    raise InternalError(msg, details={"command": command, "tracks": issue})
