# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the RFC-0017 implementation-status boundary."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC = REPO_ROOT / "rfcs" / "0017-configuration-system.md"
LOADER = REPO_ROOT / "geno_lewm" / "config" / "loader.py"
SCHEMA = REPO_ROOT / "geno_lewm" / "config" / "schema.py"
DISPATCH = REPO_ROOT / "geno_lewm" / "cli" / "_dispatch.py"
TRAIN_CLI = REPO_ROOT / "geno_lewm" / "cli" / "train.py"
EVAL_CONFIG = REPO_ROOT / "geno_lewm" / "cli" / "_eval_config.py"


def test_rfc_config_status_tracks_current_loader_boundary() -> None:
    text = RFC.read_text(encoding="utf-8")
    required = (
        "- **Updated:** 2026-06-11",
        "closed dataclass/YAML config\n  schema",
        "command defaults for `train`, `score`, `eval`, and `plan`",
        "strict unknown top-level and sub-field rejection",
        "canonical\n  resolved-config writing",
        "`describe_field` / `--explain`\n  discovery exist",
        "Train/eval paths apply `--set` overrides",
        "Hydra defaults-block composition,\n  multi-run sweeps",
        "editor schema export remain\n  open",
    )

    for fragment in required:
        assert fragment in text


def test_rfc_config_open_questions_do_not_defer_dataclass_choice() -> None:
    text = RFC.read_text(encoding="utf-8")

    assert "implementation choice deferred to the first\n  config PR" not in text
    assert "current closed dataclass/YAML loader" in text


def test_config_status_tracks_live_loader_and_cli_paths() -> None:
    loader = LOADER.read_text(encoding="utf-8")
    schema = SCHEMA.read_text(encoding="utf-8")
    dispatch = DISPATCH.read_text(encoding="utf-8")
    train_cli = TRAIN_CLI.read_text(encoding="utf-8")
    eval_config = EVAL_CONFIG.read_text(encoding="utf-8")

    assert "DEFAULTS_DIR" in loader
    assert "load_default" in loader
    assert "UnknownTopLevelKeyError" in loader
    assert "write_resolved_config" in loader
    assert "describe_field" in loader
    assert "dataclass(frozen=True, slots=True)" in schema
    assert "--print-config" in dispatch
    assert "--print-config-tree" in dispatch
    assert "--explain" in dispatch
    assert "_apply_set_override" in train_cli
    assert "_apply_set_override" in eval_config
