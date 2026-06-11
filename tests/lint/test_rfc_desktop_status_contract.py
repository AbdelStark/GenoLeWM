# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the RFC-0019 implementation-status boundary."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC = REPO_ROOT / "rfcs" / "0019-reference-desktop-app.md"
DESKTOP = REPO_ROOT / "desktop"
DESKTOP_TEST = REPO_ROOT / "tests" / "unit" / "test_desktop_scaffold.py"


def test_rfc_desktop_status_tracks_current_scaffold() -> None:
    text = RFC.read_text(encoding="utf-8")
    required = (
        "- **Updated:** 2026-06-11",
        "Tauri 2\n  layout",
        "default-deny network capability/CSP wiring",
        "Rust/PyO3\n  `runtime_probe`",
        "static VCF/FASTA drop targets",
        "permanent\n  safety banner",
        "static scaffold tests",
    )

    for fragment in required:
        assert fragment in text


def test_rfc_desktop_status_preserves_open_work() -> None:
    text = RFC.read_text(encoding="utf-8")
    required = (
        "Local scoring\n  execution",
        "result table/detail views",
        "receipt export",
        "richer file\n  picker workflow",
        "signed/notarized release artifacts remain open",
        "under #81 and #82",
    )

    for fragment in required:
        assert fragment in text


def test_desktop_status_tracks_live_scaffold_files() -> None:
    index = (DESKTOP / "index.html").read_text(encoding="utf-8")
    main_ts = (DESKTOP / "src" / "main.ts").read_text(encoding="utf-8")
    tauri_conf = (DESKTOP / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
    capabilities = (DESKTOP / "src-tauri" / "capabilities" / "default.json").read_text(
        encoding="utf-8"
    )
    runtime_rs = (DESKTOP / "src-tauri" / "src" / "runtime.rs").read_text(encoding="utf-8")
    scaffold_tests = DESKTOP_TEST.read_text(encoding="utf-8")

    assert "safety-banner" in index
    assert "vcf-drop" in index
    assert "fasta-drop" in index
    assert "runtime_probe" in main_ts
    assert "connect-src 'self' ipc: http://ipc.localhost" in tauri_conf
    assert "huggingface.co" in capabilities
    assert 'py.import("geno_lewm.deploy")' in runtime_rs
    assert "test_desktop_network_policy_is_default_deny" in scaffold_tests
    assert "test_desktop_rust_host_registers_pyo3_runtime_probe" in scaffold_tests
