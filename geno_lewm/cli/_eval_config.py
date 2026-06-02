# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for writing effective evaluation config artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml

from geno_lewm.cli._dispatch import SharedOptions
from geno_lewm.config import (
    GenoLeWMConfig,
    config_to_dict,
    load_config,
    load_default,
    write_resolved_config,
)
from geno_lewm.errors import InputError

EFFECTIVE_EVAL_CONFIG_NAME: Final = "eval_config.effective.yaml"


def write_effective_eval_config(metrics_output: Path, opts: SharedOptions) -> Path:
    """Write the resolved eval config beside ``metrics_output``."""
    config_path = metrics_output.with_name(EFFECTIVE_EVAL_CONFIG_NAME)
    resolved = _resolve_eval_config(
        config_path=opts.config,
        set_overrides=opts.set_overrides,
        seed=opts.seed,
        deterministic=opts.deterministic,
        run_id=opts.run_id,
    )
    return write_resolved_config(resolved, config_path)


def _resolve_eval_config(
    *,
    config_path: str | None,
    set_overrides: tuple[str, ...],
    seed: int | None,
    deterministic: bool,
    run_id: str | None,
) -> GenoLeWMConfig:
    cfg = load_config(Path(config_path)) if config_path is not None else load_default("eval")
    payload = config_to_dict(cfg)
    for raw in set_overrides:
        _apply_set_override(payload, raw)
    if seed is not None:
        payload["seed"] = seed
    if deterministic:
        payload["deterministic"] = True
    if run_id is not None:
        payload["run_id"] = run_id
    return load_config(payload)


def _apply_set_override(payload: dict[str, Any], raw: str) -> None:
    if "=" not in raw:
        raise InputError("--set override must have the form key=value", details={"override": raw})
    key, value_text = raw.split("=", maxsplit=1)
    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise InputError("--set override key must be a non-empty dotted path", details={"key": key})
    try:
        value = yaml.safe_load(value_text)
    except yaml.YAMLError as exc:
        raise InputError(
            "--set override value is not valid YAML",
            details={"override": raw, "error": str(exc)},
        ) from exc
    target: dict[str, Any] = payload
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise InputError("--set override path does not resolve to a config block")
        target = child
    target[parts[-1]] = value
