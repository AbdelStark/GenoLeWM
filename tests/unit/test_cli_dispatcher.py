# SPDX-License-Identifier: Apache-2.0
"""Tests for ``geno_lewm.cli._dispatch`` and the stub command surface.

Covers Acceptance Criteria from issue #30:

- Every console script in ``pyproject.toml`` resolves and prints ``--help``.
- Exit codes match the table in ``docs/spec/04-error-model.md``.
- Banner is suppressed only by ``--quiet --no-banner`` (both required).

The console-script discovery uses ``importlib.metadata`` so the test
also catches drift between ``pyproject.toml`` and the package layout.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import os

import pytest
import typer

from geno_lewm import __version__, observability as obs
from geno_lewm.cli import _dispatch
from geno_lewm.errors import InputError, InternalError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_wandb_project_override() -> None:
    obs._set_wandb_project(None)
    yield
    obs._set_wandb_project(None)


#: Console scripts registered in ``pyproject.toml``. Verify keeps its
#: argparse-based dispatcher; everything else goes through the Typer
#: stub factory in this PR.
TYPER_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("geno-lewm-train", "geno_lewm.cli.train"),
    ("geno-lewm-score", "geno_lewm.cli.score"),
    ("geno-lewm-rollout", "geno_lewm.cli.rollout"),
    ("geno-lewm-plan", "geno_lewm.cli.plan"),
    ("geno-lewm-eval", "geno_lewm.cli.eval"),
    ("geno-lewm-eval-all", "geno_lewm.cli.eval_all"),
    ("geno-lewm-export", "geno_lewm.cli.export"),
    ("geno-lewm-cache-windows", "geno_lewm.cli.cache_windows"),
    ("geno-lewm-prepare-gnomad", "geno_lewm.cli.prepare_gnomad"),
    ("geno-lewm-prepare-clinvar", "geno_lewm.cli.prepare_clinvar"),
    ("geno-lewm-update", "geno_lewm.cli.update"),
)

TYPER_STUB_SCRIPTS: tuple[tuple[str, str], ...] = tuple(
    item for item in TYPER_SCRIPTS if item[1] != "geno_lewm.cli.update"
)


# ---------------------------------------------------------------------------
# print_banner
# ---------------------------------------------------------------------------


def test_banner_printed_by_default() -> None:
    buf = io.StringIO()
    _dispatch.print_banner(quiet=False, no_banner=False, stream=buf)
    assert "GenoLeWM" in buf.getvalue()
    assert "not a clinical diagnostic" in buf.getvalue()


def test_banner_suppressed_only_when_both_flags_set() -> None:
    for quiet, no_banner in [(False, False), (True, False), (False, True)]:
        buf = io.StringIO()
        _dispatch.print_banner(quiet=quiet, no_banner=no_banner, stream=buf)
        assert buf.getvalue(), f"banner missing for quiet={quiet} no_banner={no_banner}"
    buf = io.StringIO()
    _dispatch.print_banner(quiet=True, no_banner=True, stream=buf)
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# finalize_shared
# ---------------------------------------------------------------------------


def _finalize(**overrides: object) -> _dispatch.SharedOptions | None:
    """Call ``finalize_shared`` with sensible defaults; override per-test."""
    base: dict[str, object] = {
        "config": None,
        "set_overrides": None,
        "seed": None,
        "deterministic": False,
        "log_level": "info",
        "log_dir": None,
        "run_id": None,
        "wandb_project": None,
        "no_receipt": False,
        "print_config": False,
        "print_config_tree": False,
        "explain": None,
        "quiet": False,
        "no_banner": False,
        "version": False,
    }
    base.update(overrides)
    return _dispatch.finalize_shared(**base)  # type: ignore[arg-type]


def test_finalize_shared_returns_dataclass() -> None:
    opts = _finalize(seed=42, deterministic=True)
    assert isinstance(opts, _dispatch.SharedOptions)
    assert opts.seed == 42
    assert opts.deterministic is True
    assert opts.set_overrides == ()


def test_finalize_shared_collects_set_overrides() -> None:
    opts = _finalize(set_overrides=["a=1", "b=2"])
    assert opts.set_overrides == ("a=1", "b=2")


def test_finalize_shared_resolves_wandb_project_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "env-project")
    opts = _finalize()
    assert opts.wandb_project == "env-project"


def test_finalize_shared_wandb_flag_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "env-project")
    opts = _finalize(wandb_project="flag-project")
    assert opts.wandb_project == "flag-project"
    assert os.environ["WANDB_PROJECT"] == "env-project"
    assert obs._resolve_wandb_project(None) == "flag-project"


def test_finalize_shared_rejects_unknown_log_level() -> None:
    with pytest.raises(InputError, match="invalid --log-level"):
        _finalize(log_level="trace")


@pytest.mark.parametrize("level", _dispatch.VALID_LOG_LEVELS)
def test_finalize_shared_accepts_documented_log_levels(level: str) -> None:
    assert _finalize(log_level=level).log_level == level


def test_finalize_shared_version_flag_returns_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--version`` prints the version and returns ``None`` (clean exit signal)."""
    assert _finalize(version=True) is None
    assert __version__ in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run_app — exit-code mapping (RFC-0018 §3.4 / docs/spec/04-error-model.md)
# ---------------------------------------------------------------------------


def _make_app(raise_exc: BaseException | None) -> typer.Typer:
    app = typer.Typer(no_args_is_help=False, pretty_exceptions_enable=False)

    @app.callback(invoke_without_command=True)
    def main() -> None:
        if raise_exc is not None:
            raise raise_exc

    return app


def test_run_app_zero_on_success() -> None:
    assert _dispatch.run_app(_make_app(None), argv=[]) == 0


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (InputError("x"), 2),
        (InternalError("x"), 9),
        (RuntimeError("x"), None),  # propagates — internal bug surface
    ],
)
def test_run_app_maps_known_errors(exc: BaseException, expected: int | None) -> None:
    if expected is None:
        with pytest.raises(RuntimeError):
            _dispatch.run_app(_make_app(exc), argv=[])
        return
    assert _dispatch.run_app(_make_app(exc), argv=[]) == expected


def test_run_app_keyboard_interrupt_returns_130() -> None:
    """The defensive KeyboardInterrupt path returns 130 even if Typer is bypassed.

    Click swallows ``KeyboardInterrupt`` when it reaches the framework,
    so the documented exit code only fires when the interrupt happens
    before Typer takes over (e.g., a stray ``raise`` at module-import
    time). Simulate that path with a fake ``app`` callable.
    """

    class _RaisingApp:
        def __call__(self, *_: object, **__: object) -> None:
            raise KeyboardInterrupt

    assert _dispatch.run_app(_RaisingApp(), argv=[]) == 130  # type: ignore[arg-type]


def test_run_app_preserves_systemexit_code() -> None:
    assert _dispatch.run_app(_make_app(SystemExit(7)), argv=[]) == 7


# ---------------------------------------------------------------------------
# not_yet_implemented
# ---------------------------------------------------------------------------


def test_not_yet_implemented_raises_internal_error() -> None:
    with pytest.raises(InternalError) as excinfo:
        _dispatch.not_yet_implemented(command="train", issue="#44")
    assert "tracks #44" in str(excinfo.value)
    assert excinfo.value.details["command"] == "train"


# ---------------------------------------------------------------------------
# Stub commands — registered console scripts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("entry", "module"), TYPER_SCRIPTS)
def test_every_console_script_imports(entry: str, module: str) -> None:
    """Each entry in ``[project.scripts]`` resolves to a callable."""
    mod = importlib.import_module(module)
    cli_main = getattr(mod, "cli_main", None)
    assert callable(cli_main), f"{module}.cli_main is not callable"
    # Sanity-check the Typer app's name matches the entry point name.
    assert mod.app.info.name == entry


@pytest.mark.parametrize(("entry", "module"), TYPER_SCRIPTS)
def test_every_console_script_prints_help(
    entry: str,
    module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``cli_main(['--help'])`` exits cleanly with rendered help text."""
    mod = importlib.import_module(module)
    rc = _dispatch.run_app(mod.app, argv=["--help"])
    rendered = capsys.readouterr().out
    assert rc == 0
    assert entry in rendered or "Usage:" in rendered, f"no help text from {entry}"


@pytest.mark.parametrize(("entry", "module"), TYPER_STUB_SCRIPTS)
def test_every_console_script_returns_internal_error_when_invoked(
    entry: str,
    module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Phase 1 stubs map to exit code 9 (`InternalError` family)."""
    mod = importlib.import_module(module)
    rc = _dispatch.run_app(mod.app, argv=[])
    captured = capsys.readouterr()
    assert rc == 9
    assert "not yet implemented" in captured.err
    assert entry.removeprefix("geno-lewm-") in captured.err


@pytest.mark.parametrize(("entry", "module"), TYPER_SCRIPTS)
def test_every_console_script_supports_version(
    entry: str,
    module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--version`` prints the package version and exits 0."""
    mod = importlib.import_module(module)
    rc = _dispatch.run_app(mod.app, argv=["--version"])
    captured = capsys.readouterr()
    assert rc == 0
    assert __version__ in captured.out


@pytest.mark.parametrize(("entry", "module"), TYPER_STUB_SCRIPTS)
def test_every_console_script_suppresses_banner_with_both_flags(
    entry: str,
    module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Banner is suppressed only when both --quiet AND --no-banner are set."""
    mod = importlib.import_module(module)
    rc = _dispatch.run_app(mod.app, argv=["--quiet", "--no-banner"])
    captured = capsys.readouterr()
    assert rc == 9  # stub still fires
    assert "research tool" not in captured.err, (
        f"banner leaked under --quiet --no-banner for {entry}"
    )


def test_update_requires_model_manifest_when_invoked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``geno-lewm-update`` is implemented, so no-arg invocation validates local state."""
    from geno_lewm.cli.update import app

    rc = _dispatch.run_app(app, argv=["--quiet", "--no-banner"])
    captured = capsys.readouterr()

    assert rc == 4
    assert "manifest.json" in captured.err
    assert "research tool" not in captured.err


@pytest.mark.parametrize(("entry", "module"), TYPER_SCRIPTS)
def test_banner_appears_with_single_suppression_flag(
    entry: str,
    module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Either flag alone must NOT suppress the banner (RFC-0018 §3.7)."""
    mod = importlib.import_module(module)
    _dispatch.run_app(mod.app, argv=["--quiet"])
    captured = capsys.readouterr()
    assert "research tool" in captured.err, f"banner missing under --quiet alone for {entry}"


# ---------------------------------------------------------------------------
# Discovery flags (#29; RFC-0017 §3.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("entry", "module"), TYPER_SCRIPTS)
def test_print_config_emits_resolved_yaml(
    entry: str,
    module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--print-config` dumps the resolved config to stdout and exits 0."""
    mod = importlib.import_module(module)
    rc = _dispatch.run_app(mod.app, argv=["--print-config"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "run_id:" in captured.out
    assert "encoder:" in captured.out
    assert "schema_version: 1.0.0" in captured.out


@pytest.mark.parametrize(("entry", "module"), TYPER_SCRIPTS)
def test_print_config_tree_includes_source_comment(
    entry: str,
    module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--print-config-tree` adds a ``# resolved from:`` provenance line."""
    mod = importlib.import_module(module)
    rc = _dispatch.run_app(mod.app, argv=["--print-config-tree"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "# resolved from:" in captured.out
    assert "encoder:" in captured.out


@pytest.mark.parametrize(("entry", "module"), TYPER_SCRIPTS)
def test_explain_emits_type_default_doc(
    entry: str,
    module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--explain encoder.dtype` returns the schema docstring + default + type."""
    mod = importlib.import_module(module)
    rc = _dispatch.run_app(mod.app, argv=["--explain", "encoder.dtype"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "encoder.dtype:" in captured.out
    assert "default:" in captured.out
    assert "type:" in captured.out


def test_explain_unknown_key_returns_config_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bad ``--explain`` key surfaces a CONFIG.MISSING_FIELD exit code (3)."""
    from geno_lewm.cli.train import app

    rc = _dispatch.run_app(app, argv=["--explain", "nope.nada"])
    captured = capsys.readouterr()
    assert rc == 3  # ConfigError family
    assert "not found" in captured.err


# ---------------------------------------------------------------------------
# Console-script entry-point metadata (pyproject ↔ package layout)
# ---------------------------------------------------------------------------


def test_pyproject_entry_points_match_module_layout() -> None:
    """Every Typer stub in :data:`TYPER_SCRIPTS` is also a real entry point."""
    eps = importlib.metadata.entry_points(group="console_scripts")
    registered = {ep.name: ep.value for ep in eps}
    for entry, module in TYPER_SCRIPTS:
        target = f"{module}:cli_main"
        assert registered.get(entry) == target, (
            f"entry point {entry} missing or points to {registered.get(entry)} instead of {target}"
        )
