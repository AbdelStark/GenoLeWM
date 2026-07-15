# SPDX-License-Identifier: Apache-2.0
"""Contracts for the exact Carbon runtime-hash CPU probe."""

from __future__ import annotations

from pathlib import Path

from geno_lewm.encoder._identity import encoder_runtime_hash
from tools.research import v03_carbon_runtime_hash_probe as probe


def _write_runtime(root: Path) -> Path:
    root.mkdir()
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.py",
        "dna_config.json",
    ):
        (root / name).write_text(f"{name}\n", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"real-weight-bytes")
    return root


def test_cli_hashes_runtime_and_prints_one_fully_bound_terminal_marker(
    tmp_path: Path,
    capsys,
) -> None:
    runtime = _write_runtime(tmp_path / "carbon")
    expected = encoder_runtime_hash(runtime)

    assert (
        probe.main(
            [
                "--carbon-dir",
                str(runtime),
                "--source-commit",
                "a" * 40,
                "--container-image",
                "registry.example/uv@sha256:" + "b" * 64,
                "--carbon-repository",
                "HuggingFaceBio/Carbon-500M",
                "--carbon-revision",
                "5d31d59b3c845b288a13aedb1358934196852eec",
                "--expected-runtime-hash",
                expected,
                "--flavor",
                "cpu-basic",
                "--namespace",
                "abdelstark",
                "--purpose",
                "geno-lewm-v03-carbon-runtime-hash-probe",
                "--timeout",
                "30m",
                "--run-attempt",
                "7",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    marker = lines[0]
    assert marker.startswith("GENO_LEWM_V03_CARBON_RUNTIME_HASH_PROBE_OK ")
    assert f"source_commit={'a' * 40}" in marker
    assert "carbon_repository=HuggingFaceBio/Carbon-500M" in marker
    assert "carbon_revision=5d31d59b3c845b288a13aedb1358934196852eec" in marker
    assert f"carbon_mount_path={runtime}" in marker
    assert f"encoder_runtime_hash={expected}" in marker
    assert "flavor=cpu-basic" in marker
    assert "namespace=abdelstark" in marker
    assert "purpose=geno-lewm-v03-carbon-runtime-hash-probe" in marker
    assert "timeout=30m" in marker
    assert "run_attempt=7" in marker


def test_cli_fails_without_success_marker_when_weight_bytes_drift(
    tmp_path: Path,
    capsys,
) -> None:
    runtime = _write_runtime(tmp_path / "carbon")
    expected = encoder_runtime_hash(runtime)
    (runtime / "model.safetensors").write_bytes(b"different-one-gigabyte-mount-content")

    assert (
        probe.main(
            [
                "--carbon-dir",
                str(runtime),
                "--source-commit",
                "a" * 40,
                "--container-image",
                "registry.example/uv@sha256:" + "b" * 64,
                "--carbon-repository",
                "HuggingFaceBio/Carbon-500M",
                "--carbon-revision",
                "5d31d59b3c845b288a13aedb1358934196852eec",
                "--expected-runtime-hash",
                expected,
                "--flavor",
                "cpu-basic",
                "--namespace",
                "abdelstark",
                "--purpose",
                "geno-lewm-v03-carbon-runtime-hash-probe",
                "--timeout",
                "30m",
                "--run-attempt",
                "1",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert "mounted Carbon runtime hash differs" in captured.err
    assert "GENO_LEWM_V03_CARBON_RUNTIME_HASH_PROBE_OK" not in captured.out


def test_cli_reports_incomplete_runtime_without_a_traceback(tmp_path: Path, capsys) -> None:
    runtime = _write_runtime(tmp_path / "carbon")
    (runtime / "dna_config.json").unlink()

    assert (
        probe.main(
            [
                "--carbon-dir",
                str(runtime),
                "--source-commit",
                "a" * 40,
                "--container-image",
                "registry.example/uv@sha256:" + "b" * 64,
                "--carbon-repository",
                "HuggingFaceBio/Carbon-500M",
                "--carbon-revision",
                "5d31d59b3c845b288a13aedb1358934196852eec",
                "--expected-runtime-hash",
                "sha256:" + "c" * 64,
                "--flavor",
                "cpu-basic",
                "--namespace",
                "abdelstark",
                "--purpose",
                "geno-lewm-v03-carbon-runtime-hash-probe",
                "--timeout",
                "30m",
                "--run-attempt",
                "1",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert "missing required identity files" in captured.err
    assert "GENO_LEWM_V03_CARBON_RUNTIME_HASH_PROBE_OK" not in captured.out
