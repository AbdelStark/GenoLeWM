# SPDX-License-Identifier: Apache-2.0
"""Phase 1 Typer-app factory shared by every stub command.

Building a Typer app for each of the 12 console scripts requires
declaring the same 15-flag shared option set every time. The factory
here folds that boilerplate into a single function so the per-command
modules stay short and focused on their own help text.

The factory returns a ready-to-go ``typer.Typer`` whose single callback
parses the shared options, prints the banner, and raises
:class:`geno_lewm.errors.InternalError` advertising the tracking issue
for the per-subsystem implementation.
"""

from __future__ import annotations

from typing import Annotated

import typer

from geno_lewm.cli._dispatch import (
    SharedOptions,
    finalize_shared,
    not_yet_implemented,
    run_app,
    shared_option_decls,
)

_S = shared_option_decls()


def build_stub_app(
    *,
    name: str,
    help_text: str,
    command: str,
    issue: str,
    default_config_name: str = "train",
) -> typer.Typer:
    """Construct a fully-wired Typer app for a Phase 1 stub command.

    ``name`` and ``help_text`` populate the Typer metadata. ``command``
    is the bare command name (e.g., ``"train"``) and ``issue`` is the
    GitHub tracking issue (``"#44"``) where the real implementation is
    being built. ``default_config_name`` selects which canonical YAML
    template (``train`` / ``score`` / ``eval`` / ``plan``) the
    discovery flags load when invoked.
    """

    app = typer.Typer(
        name=name,
        help=help_text,
        no_args_is_help=False,
        add_completion=True,
        pretty_exceptions_enable=False,
    )

    @app.callback(invoke_without_command=True)
    def main(
        config: Annotated[str | None, _S["config"]] = None,
        set_overrides: Annotated[list[str] | None, _S["set_overrides"]] = None,
        seed: Annotated[int | None, _S["seed"]] = None,
        deterministic: Annotated[bool, _S["deterministic"]] = False,
        log_level: Annotated[str, _S["log_level"]] = "info",
        log_dir: Annotated[str | None, _S["log_dir"]] = None,
        run_id: Annotated[str | None, _S["run_id"]] = None,
        wandb_project: Annotated[str | None, _S["wandb_project"]] = None,
        no_receipt: Annotated[bool, _S["no_receipt"]] = False,
        print_config: Annotated[bool, _S["print_config"]] = False,
        print_config_tree: Annotated[bool, _S["print_config_tree"]] = False,
        explain: Annotated[str | None, _S["explain"]] = None,
        quiet: Annotated[bool, _S["quiet"]] = False,
        no_banner: Annotated[bool, _S["no_banner"]] = False,
        version: Annotated[bool, _S["version"]] = False,
    ) -> None:
        opts: SharedOptions | None = finalize_shared(
            config=config,
            set_overrides=set_overrides,
            seed=seed,
            deterministic=deterministic,
            log_level=log_level,
            log_dir=log_dir,
            run_id=run_id,
            wandb_project=wandb_project,
            no_receipt=no_receipt,
            print_config=print_config,
            print_config_tree=print_config_tree,
            explain=explain,
            quiet=quiet,
            no_banner=no_banner,
            version=version,
            default_config_name=default_config_name,
        )
        # ``finalize_shared`` returns None when --version was passed; the
        # callback then returns silently and Typer treats it as exit 0.
        if opts is None:
            return
        # ``opts`` is intentionally unused in the stub — the per-subsystem
        # implementation will read from it once it lands. Reference it
        # here so the local stays visible to readers.
        del opts
        not_yet_implemented(command=command, issue=issue)

    return app


def make_cli_main(app: typer.Typer) -> object:
    """Return a ``cli_main() -> int`` thunk for ``app`` — used by pyproject."""

    def cli_main() -> int:
        return run_app(app)

    return cli_main
