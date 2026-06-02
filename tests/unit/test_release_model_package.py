"""Tests for the release model package builder."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from geno_lewm._artifact_sources import SCORE_JSONL_GENERATED_BY, SCORE_JSONL_SCHEMA_VERSION
from geno_lewm.errors import InputError
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    load_manifest,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from geno_lewm.training.preflight import REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME
from tests.unit.test_release_training_run import _write_training_run_inputs
from tools.release.efficiency_report import (
    GENERATED_BY as EFFICIENCY_REPORT_GENERATED_BY,
    REPORT_NAME as EFFICIENCY_REPORT_NAME,
    parse_efficiency_report,
)
from tools.release.eval_report import parse_report_input, render_report
from tools.release.model_package import (
    EVAL_METRICS_NAME,
    GENERATED_BY as MODEL_PACKAGE_GENERATED_BY,
    MODEL_PACKAGE_NAME,
    build_model_package,
    main,
    parse_model_package,
)
from tools.release.paper_package import PackageIssue, _verify_model_dir
from tools.release.training_run import build_training_run_package


def test_build_model_package_writes_verifier_compatible_artifacts(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path)

    report = build_model_package(tmp_path, metadata_path)

    assert report.model_id.startswith("sha256:")
    assert report.metadata_path == tmp_path / MODEL_PACKAGE_NAME
    assert (tmp_path / MODEL_PACKAGE_NAME).is_file()
    assert (tmp_path / "model_card.md").is_file()
    assert (tmp_path / "SHA256SUMS").is_file()
    assert "manifest.json" in report.files
    assert MODEL_PACKAGE_NAME in report.files
    assert "model_card.md" in report.files
    assert EVAL_METRICS_NAME in report.files
    assert EFFICIENCY_REPORT_NAME in report.files
    assert "eval_config.effective.yaml" in report.files
    assert "eval/scores.jsonl" in report.files
    assert TRAINING_PREFLIGHT_REPORT_NAME in report.files
    assert "training_run_manifest.json" in report.files
    assert "training_run_card.md" in report.files
    assert "training_run_SHA256SUMS" in report.files
    checksums = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"  {MODEL_PACKAGE_NAME}\n" in checksums
    assert "  eval_config.effective.yaml\n" in checksums
    assert "  eval/scores.jsonl\n" in checksums
    normalized_metadata = json.loads((tmp_path / MODEL_PACKAGE_NAME).read_text(encoding="utf-8"))
    assert normalized_metadata["summary"].startswith("SNV predictor checkpoint")

    model_card = (tmp_path / "model_card.md").read_text(encoding="utf-8")
    assert "## Data" in model_card
    assert "## Hardware" in model_card
    assert "## License" in model_card
    assert "## Intended Use" in model_card
    assert "## Limitations" in model_card
    assert "- Model id: sha256:" in model_card

    issues: list[PackageIssue] = []
    _verify_model_dir(tmp_path, tmp_path / "dataset", issues, allow_fixture_manifest=False)
    assert issues == []


def test_model_package_main_outputs_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_path = _write_model_inputs(tmp_path)

    rc = main(["--model-dir", str(tmp_path), "--metadata-json", str(metadata_path)])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "1.0.0"
    assert payload["generated_by"] == MODEL_PACKAGE_GENERATED_BY
    assert payload["model_id"].startswith("sha256:")
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["metadata_path"] == MODEL_PACKAGE_NAME
    assert payload["model_card_path"] == "model_card.md"
    assert payload["checksums_path"] == "SHA256SUMS"
    assert "model_card.md" in payload["files"]


def test_parse_model_package_rejects_placeholder_text() -> None:
    payload = _metadata()
    payload["limitations"] = ["TODO"]

    with pytest.raises(InputError, match="placeholder text is not allowed"):
        parse_model_package(payload)


def test_parse_model_package_rejects_unexpected_generator() -> None:
    payload = _metadata()
    payload["generated_by"] = "manual-editor"

    with pytest.raises(InputError, match="generated_by"):
        parse_model_package(payload)


def test_build_model_package_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path)
    (tmp_path / "predictor.safetensors").write_bytes(b"tampered predictor")

    with pytest.raises(InputError, match="manifest artifact hash mismatch"):
        build_model_package(tmp_path, metadata_path)


def test_build_model_package_rejects_missing_eval_metrics_json(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path)
    (tmp_path / EVAL_METRICS_NAME).unlink()

    with pytest.raises(InputError, match=r"eval_metrics\.json is required"):
        build_model_package(tmp_path, metadata_path)


def test_build_model_package_rejects_missing_eval_score_artifact(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path)
    (tmp_path / "eval" / "scores.jsonl").unlink()

    with pytest.raises(InputError, match="eval metrics artifact is missing"):
        build_model_package(tmp_path, metadata_path)


def test_build_model_package_rejects_missing_efficiency_report(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path)
    (tmp_path / EFFICIENCY_REPORT_NAME).unlink()

    with pytest.raises(InputError, match=r"efficiency_report\.json is required"):
        build_model_package(tmp_path, metadata_path)


def test_build_model_package_rejects_stale_eval_report(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path)
    payload = json.loads((tmp_path / EVAL_METRICS_NAME).read_text(encoding="utf-8"))
    payload["metrics"][0]["value"] = 0.74
    payload["conclusions"] = ["The auroc metric value 0.74 on clinvar_coding was measured."]
    (tmp_path / EVAL_METRICS_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InputError, match=r"eval_report\.md does not match"):
        build_model_package(tmp_path, metadata_path)


def test_build_model_package_rejects_eval_metrics_release_mismatch(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path)
    payload = json.loads((tmp_path / EVAL_METRICS_NAME).read_text(encoding="utf-8"))
    payload["model_release"] = "other-release"
    (tmp_path / EVAL_METRICS_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_report.md").write_text(
        render_report(parse_report_input(payload)),
        encoding="utf-8",
    )
    _refresh_manifest_eval_hash(tmp_path)

    with pytest.raises(InputError, match="model_release must match"):
        build_model_package(tmp_path, metadata_path)


def test_build_model_package_rejects_efficiency_dataset_mismatch(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path)
    payload = json.loads((tmp_path / EFFICIENCY_REPORT_NAME).read_text(encoding="utf-8"))
    payload["dataset_snapshot"] = "other-dataset"
    (tmp_path / EFFICIENCY_REPORT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="dataset_snapshot must match"):
        build_model_package(tmp_path, metadata_path)


def test_build_model_package_rejects_fixture_manifest_by_default(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path, release_id="geno-lewm-fixture-r1")

    with pytest.raises(InputError, match="fixture/test manifests cannot back"):
        build_model_package(tmp_path, metadata_path)


def test_build_model_package_can_allow_fixture_manifest_for_local_tests(tmp_path: Path) -> None:
    metadata_path = _write_model_inputs(tmp_path, release_id="geno-lewm-fixture-r1")

    report = build_model_package(tmp_path, metadata_path, allow_fixture_manifest=True)

    assert report.model_id.startswith("sha256:")


def test_parse_model_package_rejects_generated_file_as_extra() -> None:
    payload = _metadata()
    payload["extra_files"] = ["model_card.md"]

    with pytest.raises(InputError, match="generated model package files cannot be listed"):
        parse_model_package(payload)


def test_parse_model_package_rejects_unsafe_extra_path() -> None:
    payload = _metadata()
    payload["extra_files"] = ["../outside.txt"]

    with pytest.raises(InputError, match="model package paths must be relative"):
        parse_model_package(payload)


def test_parse_model_package_requires_training_run_evidence_extra_files() -> None:
    payload = _metadata()
    payload["extra_files"] = ["training_preflight_report.json", "training_run_manifest.json"]

    with pytest.raises(InputError, match="release training-run evidence") as exc_info:
        parse_model_package(payload)

    assert exc_info.value.details == {
        "missing": ["training_run_SHA256SUMS", "training_run_card.md"]
    }


def _write_model_inputs(
    root: Path,
    *,
    release_id: str = "geno-lewm-v0.1.0-r1",
) -> Path:
    _write_model_dir(root, release_id=release_id)
    metadata_path = root / "model_package.json"
    metadata_path.write_text(json.dumps(_metadata(), indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def _write_model_dir(root: Path, *, release_id: str) -> None:
    training_metadata = _write_training_run_inputs(root)
    _write_eval_report_artifacts(root, release_id=release_id)
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm",
        model_version="0.1.0",
        release_id=release_id,
        encoder=ManifestEncoder(
            id="HuggingFaceBio/Carbon-500M",
            revision="main",
            hash=sha256_bytes(b"encoder"),
        ),
        predictor=ManifestArtifact(
            file="predictor.safetensors",
            hash=sha256_file(root / "predictor.safetensors"),
            dtype="bf16",
        ),
        action_encoder=ManifestArtifact(
            file="action_encoder.safetensors",
            hash=sha256_file(root / "action_encoder.safetensors"),
            dtype="bf16",
        ),
        calibration=ManifestArtifact(
            file="calibration.parquet",
            hash=sha256_file(root / "calibration.parquet"),
            version="1.0.0",
        ),
        training=ManifestTraining(
            config_file="train_config.yaml",
            hash=sha256_file(root / "train_config.yaml"),
            data_snapshot={"snapshot": "geno-lewm-data-v0.1.0-r1"},
        ),
        eval=ManifestArtifact(file="eval_report.md", hash=sha256_file(root / "eval_report.md")),
    )
    write_manifest(manifest, root / "manifest.json")
    build_training_run_package(root, training_metadata)


def _metadata() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": MODEL_PACKAGE_GENERATED_BY,
        "generated_at": "2026-06-01T00:00:00Z",
        "summary": "SNV predictor checkpoint for the first GenoLeWM paper/demo release.",
        "data": [
            "Training data snapshot: geno-lewm-data-v0.1.0-r1.",
            "Evaluation data is documented in eval_report.md.",
        ],
        "hardware": ["Apple M3 Max CPU smoke package; final release records training accelerator."],
        "license": "Apache-2.0 for GenoLeWM metadata; upstream encoder and data terms apply.",
        "intended_use": "Research-only GenoLeWM SNV scoring and reproducibility experiments.",
        "limitations": [
            "Not a clinical diagnostic model.",
            "Performance claims are limited to the generated evaluation report.",
        ],
        "training": [
            "Configuration is recorded in train_config.yaml.",
            "Manifest records all checkpoint artifact hashes.",
        ],
        "evaluation": [
            "eval_report.md is generated from measured metrics JSON.",
            "The model card does not restate benchmark claims.",
        ],
        "runtime": [
            "Load through GenoLeWM runtime with manifest verification.",
            "Requires compatible Carbon encoder revision from the manifest.",
        ],
        "release_notes": [
            "Publish with dataset snapshot and terminal demo transcript links.",
            "Run tools.release.paper_package before uploading artifacts.",
        ],
        "extra_files": [
            TRAINING_PREFLIGHT_REPORT_NAME,
            "training_run_manifest.json",
            "training_run_card.md",
            "training_run_SHA256SUMS",
        ],
    }


def _write_eval_report_artifacts(root: Path, *, release_id: str) -> None:
    _write_efficiency_report(root, release_id=release_id)
    _write_eval_score_artifacts(root)
    (root / "eval_config.effective.yaml").write_text(
        "metrics:\n  aggregate: true\n",
        encoding="utf-8",
    )
    payload = _eval_metrics_payload(release_id=release_id)
    (root / EVAL_METRICS_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_report(parse_report_input(payload))
    (root / "eval_report.md").write_text(report, encoding="utf-8")


def _write_eval_score_artifacts(root: Path) -> None:
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        eval_dir / "scores.jsonl",
        [
            {
                "schema_version": SCORE_JSONL_SCHEMA_VERSION,
                "generated_by": SCORE_JSONL_GENERATED_BY,
                "chrom": "1",
                "pos": 10,
                "ref": "A",
                "alt": "T",
                "sigma_calibrated": 0.73,
            }
        ],
    )
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_by": "test.fixture",
                "snapshot": "geno-lewm-data-v0.1.0-r1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _eval_metrics_payload(*, release_id: str = "geno-lewm-v0.1.0-r1") -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": "geno-lewm-eval-all",
        "generated_at": "2026-06-01T00:00:00Z",
        "model_id": sha256_bytes(b"eval-report-model"),
        "model_release": release_id,
        "dataset_snapshot": "geno-lewm-data-v0.1.0-r1",
        "commit": "abcdef1234567890",
        "hardware": "local CPU fixture",
        "metrics": [
            {
                "name": "auroc",
                "value": 0.73,
                "split": "clinvar_coding",
                "unit": "area",
                "higher_is_better": True,
                "ci_low": 0.70,
                "ci_high": 0.76,
                "n": 1200,
                "notes": "measured",
            }
        ],
        "artifacts": {
            "checkpoint": "model/predictor.safetensors",
            "config": "model/train_config.yaml",
            "dataset_manifest": "dataset/dataset_manifest.json",
            "eval_config": "eval_config.effective.yaml",
            "efficiency_report": f"model/{EFFICIENCY_REPORT_NAME}",
            "scores": "eval/scores.jsonl",
        },
        "limitations": ["Known limitations."],
        "negative_findings": ["No clinical utility claim is measured by this fixture report."],
        "conclusions": ["The auroc metric value 0.73 on clinvar_coding was measured."],
    }


def _write_efficiency_report(root: Path, *, release_id: str) -> None:
    report = parse_efficiency_report(_efficiency_payload(root, release_id=release_id))
    (root / EFFICIENCY_REPORT_NAME).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _efficiency_payload(
    root: Path,
    *,
    release_id: str = "geno-lewm-v0.1.0-r1",
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": EFFICIENCY_REPORT_GENERATED_BY,
        "generated_at": "2026-06-01T00:00:00Z",
        "model_id": sha256_bytes(b"eval-report-model"),
        "model_release": release_id,
        "dataset_snapshot": "geno-lewm-data-v0.1.0-r1",
        "commit": "abcdef1234567890",
        "command": [
            "python",
            "-m",
            "bench.inference",
            "--model-dir",
            "model",
            "--batch-size",
            "64",
        ],
        "hardware": "local CPU fixture",
        "runtime": "Python fixture runtime",
        "warmup_batches": 1,
        "samples": 8,
        "measurements": {
            "single_variant_latency_ms": 12.5,
            "batched_throughput_variants_per_s": 512.0,
            "peak_memory_bytes": 123456789,
        },
        "inputs": {
            "checkpoint": {
                "path": "model/predictor.safetensors",
                "sha256": sha256_file(root / "predictor.safetensors"),
                "size_bytes": (root / "predictor.safetensors").stat().st_size,
            },
            "dataset_manifest": {
                "path": "dataset/dataset_manifest.json",
                "sha256": sha256_bytes(b"dataset-manifest"),
                "size_bytes": 1024,
            },
        },
        "limitations": ["Local fixture run; release numbers require the published checkpoint."],
    }


def _refresh_manifest_eval_hash(root: Path) -> None:
    manifest = load_manifest(root / "manifest.json")
    write_manifest(
        dataclasses.replace(
            manifest,
            eval=dataclasses.replace(
                manifest.eval,
                hash=sha256_file(root / "eval_report.md"),
            ),
        ),
        root / "manifest.json",
    )
