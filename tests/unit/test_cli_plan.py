# SPDX-License-Identifier: Apache-2.0
"""Tests for the implemented ``geno-lewm-plan`` CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

from geno_lewm.cli import _dispatch, plan as plan_cli
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    write_manifest,
)


def test_plan_requires_window_fasta(capsys) -> None:
    rc = _dispatch.run_app(plan_cli.app, argv=["--quiet", "--no-banner"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "requires --window-fasta" in captured.err


def test_plan_requires_explicit_proxy_without_model_dir(
    tmp_path: Path,
    capsys,
) -> None:
    window = tmp_path / "window.fa"
    target = tmp_path / "target.fa"
    _write_fasta(window, "ACGTACGT")
    _write_fasta(target, "ACGTTCGT")

    rc = _dispatch.run_app(
        plan_cli.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--window-fasta",
            str(window),
            "--target-fasta",
            str(target),
            "--edge-margin",
            "0",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 4
    assert "requires --allow-sequence-proxy" in captured.err


def test_plan_writes_sequence_proxy_plan_json_from_fixture_fasta(
    tmp_path: Path,
    capsys,
) -> None:
    window = tmp_path / "window.fa"
    target = tmp_path / "target.fa"
    output = tmp_path / "plan.json"
    _write_fasta(window, "ACGTACGTACGTACGT")
    _write_fasta(target, "ACGTATGTACGTACGT")

    rc = _dispatch.run_app(
        plan_cli.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--window-fasta",
            str(window),
            "--target-fasta",
            str(target),
            "--allow-sequence-proxy",
            "--horizon",
            "1",
            "--iterations",
            "4",
            "--samples",
            "256",
            "--elite",
            "8",
            "--edge-margin",
            "0",
            "--position-bin-bp",
            "1",
            "--seed",
            "13",
            "--output",
            str(output),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["evaluation_mode"] == "sequence_proxy"
    assert summary["output_path"] == str(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generated_by"] == "geno-lewm-plan"
    assert payload["evaluation_mode"] == "sequence_proxy"
    assert "not learned predictor evidence" in payload["negative_findings"][0]
    assert "clinical" in payload["claim_boundary"]
    assert payload["result"]["best_distance"] == 0.0
    assert payload["result"]["reproducibility_sha256"].startswith("sha256:")
    assert payload["result"]["best_edits"] == [
        {
            "alt_bases": "T",
            "edit_type": "SNV",
            "edit_type_id": 0,
            "ref_bases": "C",
            "rel_pos": 5,
        }
    ]


def test_plan_uses_manifest_runtime_for_precomputed_states(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    manifest = _write_manifest(model_dir / "manifest.json")
    window = tmp_path / "window.fa"
    initial = tmp_path / "initial.json"
    target = tmp_path / "target.json"
    output = tmp_path / "runtime-plan.json"
    _write_fasta(window, "ACGTACGT")
    initial.write_text("[0.0, 0.0]", encoding="utf-8")
    target.write_text("[1.0, 0.0]", encoding="utf-8")

    class FakeRuntime:
        def __init__(self, model_dir: Path, backend: str) -> None:
            self.model_dir = model_dir
            self.backend = backend

        def predict(self, state, edits):
            del state, edits
            return [1.0, 0.0]

    monkeypatch.setattr(plan_cli, "GenoLeWMRuntime", FakeRuntime)

    rc = _dispatch.run_app(
        plan_cli.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--window-fasta",
            str(window),
            "--initial-state",
            str(initial),
            "--target-state",
            str(target),
            "--model-dir",
            str(model_dir),
            "--backend",
            "cpu",
            "--horizon",
            "1",
            "--iterations",
            "1",
            "--samples",
            "4",
            "--elite",
            "1",
            "--edge-margin",
            "0",
            "--seed",
            "3",
            "--output",
            str(output),
        ],
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["evaluation_mode"] == "manifest_runtime"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evaluation_mode"] == "manifest_runtime"
    assert payload["runtime"]["model_id"] == manifest.model_id()
    assert payload["negative_findings"] == []
    assert payload["result"]["best_distance"] == 0.0


def _write_manifest(path: Path) -> Manifest:
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm-test",
        model_version="0.0.0",
        release_id="test-release",
        encoder=ManifestEncoder(
            id="HuggingFaceBio/Carbon-500M",
            revision="test-revision",
            hash="sha256:" + "a" * 64,
        ),
        predictor=ManifestArtifact(file="predictor.safetensors", hash="sha256:" + "b" * 64),
        action_encoder=ManifestArtifact(
            file="action_encoder.safetensors",
            hash="sha256:" + "c" * 64,
        ),
        calibration=ManifestArtifact(file="calibration.parquet", hash="sha256:" + "d" * 64),
        training=ManifestTraining(
            config_file="training_config.effective.yaml",
            hash="sha256:" + "e" * 64,
        ),
        eval=ManifestArtifact(file="eval_metrics.json", hash="sha256:" + "f" * 64),
    )
    write_manifest(manifest, path)
    return manifest


def _write_fasta(path: Path, sequence: str) -> None:
    path.write_text(f">fixture\n{sequence}\n", encoding="utf-8")
