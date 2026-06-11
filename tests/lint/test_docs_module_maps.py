# SPDX-License-Identifier: Apache-2.0
"""Regression tests for public documentation module maps."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_WITH_MODULE_MAPS = (
    REPO_ROOT / "ARCHITECTURE.md",
    REPO_ROOT / "docs" / "api" / "public-surface.md",
    REPO_ROOT / "README.md",
)


def _doc_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOCS_WITH_MODULE_MAPS)


def test_public_module_maps_do_not_list_removed_or_absent_paths() -> None:
    text = _doc_text()

    stale_paths = (
        "├── eval/",
        "├── " + "att" + "estation/",
        "geno_lewm." + "att" + "estation",
        "holdouts.py",
        "onnx.py",
        "coreml.py",
        "ggml.py",
        "provenance.py",
        "planning/cem",
        "0011-" + "veri" + "fiable-" + "inference-" + "att" + "estation",
    )

    for stale_path in stale_paths:
        assert stale_path not in text


def test_public_module_maps_name_current_provenance_and_eval_modules() -> None:
    text = _doc_text()

    required_paths = (
        "├── provenance/",
        "commitment.py",
        "manifest.py",
        "receipt.py",
        "evaluation.py",
        "carbon_zero_shot.py",
    )

    for required_path in required_paths:
        assert required_path in text
