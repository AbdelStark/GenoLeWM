"""Integration coverage for the opt-in wandb observability sink."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

from geno_lewm import observability as obs
from geno_lewm.cli import _dispatch


class _Run:
    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, int | float], int | None]] = []
        self.finished = False

    def log(self, payload: dict[str, int | float], *, step: int | None = None) -> None:
        self.logs.append((payload, step))

    def finish(self) -> None:
        self.finished = True


def _finalize_with_wandb(project: str) -> _dispatch.SharedOptions:
    opts = _dispatch.finalize_shared(
        config=None,
        set_overrides=None,
        seed=None,
        deterministic=False,
        log_level="info",
        log_dir=None,
        run_id="wandb-integration",
        wandb_project=project,
        no_receipt=False,
        print_config=False,
        print_config_tree=False,
        explain=None,
        quiet=True,
        no_banner=True,
        version=False,
    )
    assert isinstance(opts, _dispatch.SharedOptions)
    return opts


def test_wandb_project_flag_enables_logger_sink(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    obs._set_wandb_project(None)
    run = _Run()
    init_calls: list[dict[str, Any]] = []

    def init(**kwargs: Any) -> _Run:
        init_calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))

    opts = _finalize_with_wandb("project-from-flag")
    logger = obs.get_logger("training", run_id=opts.run_id, log_dir=tmp_path)
    logger.info("training.metric", step=12, name="loss", value=0.125)
    obs.shutdown_run("wandb-integration", tmp_path)

    assert init_calls == [
        {
            "project": "project-from-flag",
            "id": "wandb-integration",
            "resume": "allow",
            "anonymous": "never",
        }
    ]
    assert run.logs == [({"loss": 0.125}, 12)]
    assert run.finished is True
    obs._set_wandb_project(None)
