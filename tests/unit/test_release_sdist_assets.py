# SPDX-License-Identifier: Apache-2.0
"""Tests for source-distribution release asset checks."""

from __future__ import annotations

import tarfile
from pathlib import Path

from tools.release.check_sdist_assets import (
    REQUIRED_SDIST_ASSETS,
    check_sdist_assets,
    main,
    missing_sdist_assets,
    sdist_members,
)


def test_sdist_members_strip_archive_root(tmp_path: Path) -> None:
    sdist = _write_sdist(tmp_path, ("README.md", "bench/inference.py"))

    assert sdist_members(sdist) == frozenset({"README.md", "bench/inference.py"})


def test_check_sdist_assets_accepts_complete_inventory(tmp_path: Path) -> None:
    sdist = _write_sdist(tmp_path, REQUIRED_SDIST_ASSETS)

    check_sdist_assets(sdist)
    assert missing_sdist_assets(sdist) == ()


def test_required_sdist_assets_cover_first_publication_release_path() -> None:
    """The sdist must carry the tools needed to reproduce release evidence."""
    assets = set(REQUIRED_SDIST_ASSETS)

    expected = {
        "bench/inference.py",
        "configs/first_experiment/dataset-snapshot-snv.json",
        "configs/first_experiment/eval-clinvar-snv.yaml",
        "configs/first_experiment/train-carbon-500m-snv.yaml",
        "examples/data/verify_receipt/manifest.json",
        "examples/data/verify_receipt/receipt.json",
        "tools/demo/terminal_inference.py",
        "tools/lint/check_scope_language.py",
        "tools/release/batch_receipt_report.py",
        "tools/release/clean_machine_demo.py",
        "tools/release/dataset_integrity.py",
        "tools/release/dataset_package.py",
        "tools/release/dataset_snapshot.py",
        "tools/release/efficiency_report.py",
        "tools/release/eval_report.py",
        "tools/release/hub_publish.py",
        "tools/release/hub_release.py",
        "tools/release/model_package.py",
        "tools/release/paper_draft.py",
        "tools/release/paper_package.py",
        "tools/release/publication_assets.py",
        "tools/release/publication_report.py",
        "tools/release/release_candidate.py",
        "tools/release/runtime_preflight.py",
        "tools/release/rollout_state_examples.py",
        "tools/release/rollout_state_rows.py",
        "tools/release/rollout_speed_scope.py",
        "tools/release/training_run.py",
        "tools/release/v02_benchmark_readiness.py",
        "tools/release/v02_benchmark_suite.py",
    }

    assert expected <= assets


def test_missing_sdist_assets_reports_required_files(tmp_path: Path) -> None:
    present = tuple(asset for asset in REQUIRED_SDIST_ASSETS if asset != "bench/inference.py")
    sdist = _write_sdist(tmp_path, present)

    assert missing_sdist_assets(sdist) == ("bench/inference.py",)


def test_main_returns_two_for_missing_asset(tmp_path: Path, capsys) -> None:
    sdist = _write_sdist(tmp_path, ("README.md",))

    assert main([str(sdist)]) == 2
    captured = capsys.readouterr()
    assert "missing release-critical assets" in captured.err


def _write_sdist(tmp_path: Path, members: tuple[str, ...]) -> Path:
    root = "geno_lewm-0.1.0.dev0"
    source = tmp_path / "source"
    archive_path = tmp_path / "geno_lewm-0.1.0.dev0.tar.gz"
    for member in members:
        path = source / root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{member}\n", encoding="utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source / root, arcname=root)
    return archive_path
