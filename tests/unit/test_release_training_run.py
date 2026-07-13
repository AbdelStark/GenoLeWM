"""Tests for training-run release evidence packaging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.data import MEMBERSHIP_STORE_SCHEMA_VERSION
from geno_lewm.errors import InputError
from geno_lewm.provenance import canonical_json_sha256, sha256_file
from geno_lewm.training.preflight import (
    GENERATED_BY as TRAINING_PREFLIGHT_GENERATED_BY,
    REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME,
    SCHEMA_VERSION as TRAINING_PREFLIGHT_SCHEMA_VERSION,
)
from tools.release.training_run import (
    GENERATED_BY as TRAINING_RUN_GENERATED_BY,
    REQUIRED_PREFLIGHT_DATASET_CORE_FILES,
    TRAINING_PREFLIGHT_KIND,
    build_training_run_package,
    main,
    parse_training_run_metadata,
    verify_training_run_manifest,
)


def test_training_run_tool_delegates_to_installed_package() -> None:
    from geno_lewm import _training_run_package
    from tools.release import training_run

    assert training_run.TrainingArtifact is _training_run_package.TrainingArtifact
    assert training_run.TrainingRunManifest is _training_run_package.TrainingRunManifest
    assert training_run.TrainingRunPackageReport is _training_run_package.TrainingRunPackageReport
    assert training_run.build_training_run_package is build_training_run_package
    assert training_run.parse_training_run_metadata is parse_training_run_metadata
    assert training_run.verify_training_run_manifest is verify_training_run_manifest


def test_build_training_run_package_writes_release_evidence(tmp_path: Path) -> None:
    metadata_path = _write_training_run_inputs(tmp_path)

    report = build_training_run_package(tmp_path, metadata_path)

    assert report.run_id == "geno-lewm-snv-v0.1.0-r1-seed0"
    assert (tmp_path / "training_run_manifest.json").is_file()
    assert (tmp_path / "training_run_card.md").is_file()
    assert (tmp_path / "training_run_SHA256SUMS").is_file()
    loaded = verify_training_run_manifest(tmp_path)
    assert loaded.run_id == report.run_id
    assert {artifact.kind for artifact in loaded.artifacts} >= {
        "dataset_manifest",
        "training_config",
        "metrics",
        TRAINING_PREFLIGHT_KIND,
        "log",
        "checkpoint",
    }
    assert loaded.schema_version == "1.0.0"
    assert loaded.membership_and_split_evidence is None
    manifest_payload = json.loads(
        (tmp_path / "training_run_manifest.json").read_text(encoding="utf-8")
    )
    assert "membership_and_split_evidence" not in manifest_payload
    assert "Membership and Split Evidence" not in (tmp_path / "training_run_card.md").read_text(
        encoding="utf-8"
    )


def test_bound_training_run_round_trips_membership_evidence(tmp_path: Path) -> None:
    binding = _membership_runtime_binding()
    metadata_path = _write_bound_training_run_inputs(tmp_path, binding=binding)

    report = build_training_run_package(tmp_path, metadata_path)
    loaded = verify_training_run_manifest(tmp_path)

    assert report.to_dict()["schema_version"] == "1.1.0"
    assert loaded.schema_version == "1.1.0"
    assert loaded.membership_and_split_evidence == binding
    manifest_payload = json.loads(
        (tmp_path / "training_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_payload["membership_and_split_evidence"] == binding
    card = (tmp_path / "training_run_card.md").read_text(encoding="utf-8")
    assert "## Membership and Split Evidence" in card
    assert _MEMBERSHIP_STORE_BINDING["content_identity"] in card
    assert binding["holdout_policy_identity"] in card


def test_bound_training_run_rejects_metrics_membership_mismatch(tmp_path: Path) -> None:
    binding = _membership_runtime_binding()
    metadata_path = _write_bound_training_run_inputs(tmp_path, binding=binding)
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["membership_and_split_evidence"]["report"]["artifact_id"] = "wrong-report"
    metrics_path.write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="metrics membership and split evidence"):
        build_training_run_package(tmp_path, metadata_path)


@pytest.mark.parametrize(
    ("artifact_name", "expected_error"),
    [
        ("metrics.json", "metrics membership and split evidence"),
        ("dataset_manifest.json", "dataset manifest membership and split evidence"),
    ],
)
def test_verify_bound_training_run_rejects_semantic_membership_mismatch(
    tmp_path: Path,
    artifact_name: str,
    expected_error: str,
) -> None:
    binding = _membership_runtime_binding()
    metadata_path = _write_bound_training_run_inputs(tmp_path, binding=binding)
    build_training_run_package(tmp_path, metadata_path)
    artifact_path = tmp_path / artifact_name
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["membership_and_split_evidence"]["report"]["artifact_id"] = "wrong-report"
    artifact_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_manifest_artifact_identity(tmp_path, artifact_name)

    with pytest.raises(InputError, match=expected_error):
        verify_training_run_manifest(tmp_path)


@pytest.mark.parametrize("section", ["membership_store", "report"])
def test_bound_training_run_rejects_dataset_manifest_membership_mismatch(
    tmp_path: Path,
    section: str,
) -> None:
    binding = _membership_runtime_binding()
    metadata_path = _write_bound_training_run_inputs(tmp_path, binding=binding)
    dataset_path = tmp_path / "dataset_manifest.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset["membership_and_split_evidence"][section]["artifact_id"] = "wrong-artifact"
    dataset_path.write_text(json.dumps(dataset, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="dataset manifest membership and split evidence"):
        build_training_run_package(tmp_path, metadata_path)


def test_bound_training_run_rejects_noncanonical_policy_digest(tmp_path: Path) -> None:
    binding = _membership_runtime_binding()
    binding["holdout_policy_identity"] = "sha256:" + ("f" * 64)
    metadata_path = _write_bound_training_run_inputs(tmp_path, binding=binding)

    with pytest.raises(InputError, match="holdout policy identity does not match"):
        build_training_run_package(tmp_path, metadata_path)


def test_bound_training_run_rejects_policy_content_mismatch(tmp_path: Path) -> None:
    binding = _membership_runtime_binding()
    policy = binding["holdout_policy"]
    assert isinstance(policy, dict)
    policy["membership_content_identity"] = "sha256:" + ("f" * 64)
    binding["holdout_policy_identity"] = canonical_json_sha256(policy)
    metadata_path = _write_bound_training_run_inputs(tmp_path, binding=binding)

    with pytest.raises(InputError, match="policy content identity"):
        build_training_run_package(tmp_path, metadata_path)


def test_bound_training_run_rejects_noncanonical_holdout_chromosomes(tmp_path: Path) -> None:
    binding = _membership_runtime_binding()
    policy = binding["holdout_policy"]
    assert isinstance(policy, dict)
    policy["excluded_chromosomes"] = ["1"]
    binding["holdout_policy_identity"] = canonical_json_sha256(policy)
    metadata_path = _write_bound_training_run_inputs(tmp_path, binding=binding)

    with pytest.raises(InputError, match=r"do not match the v0\.3 split"):
        build_training_run_package(tmp_path, metadata_path)


def test_verify_bound_training_run_rejects_preflight_membership_mismatch(
    tmp_path: Path,
) -> None:
    binding = _membership_runtime_binding()
    metadata_path = _write_bound_training_run_inputs(tmp_path, binding=binding)
    build_training_run_package(tmp_path, metadata_path)
    preflight_path = tmp_path / TRAINING_PREFLIGHT_REPORT_NAME
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    del preflight["dataset"]["membership_and_split_evidence"]
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_manifest_artifact_identity(tmp_path, TRAINING_PREFLIGHT_REPORT_NAME)

    with pytest.raises(InputError, match="preflight membership and split evidence"):
        verify_training_run_manifest(tmp_path, require_preflight=True)


def test_legacy_schema_rejects_membership_binding(tmp_path: Path) -> None:
    _write_training_files(tmp_path)
    payload = _metadata()
    payload["membership_and_split_evidence"] = _membership_runtime_binding()

    with pytest.raises(InputError, match=r"schema 1\.0\.0 cannot bind"):
        parse_training_run_metadata(payload, run_dir=tmp_path)


def test_bound_schema_requires_membership_binding(tmp_path: Path) -> None:
    _write_training_files(tmp_path)
    payload = _metadata()
    payload["schema_version"] = "1.1.0"

    with pytest.raises(InputError, match=r"schema 1\.1\.0 requires"):
        parse_training_run_metadata(payload, run_dir=tmp_path)


def test_training_run_main_outputs_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_path = _write_training_run_inputs(tmp_path)

    rc = main(["--run-dir", str(tmp_path), "--metadata-json", str(metadata_path)])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "1.0.0"
    assert payload["generated_by"] == TRAINING_RUN_GENERATED_BY
    assert payload["run_id"] == "geno-lewm-snv-v0.1.0-r1-seed0"
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["manifest_path"] == "training_run_manifest.json"
    assert payload["card_path"] == "training_run_card.md"
    assert payload["checksums_path"] == "training_run_SHA256SUMS"


def test_parse_training_run_metadata_rejects_placeholder_text(tmp_path: Path) -> None:
    _write_training_files(tmp_path)
    payload = _metadata()
    payload["limitations"] = ["TODO"]

    with pytest.raises(InputError, match="placeholder text is not allowed"):
        parse_training_run_metadata(payload, run_dir=tmp_path)


def test_parse_training_run_metadata_requires_monitoring(tmp_path: Path) -> None:
    _write_training_files(tmp_path)
    payload = _metadata()
    payload["monitoring"] = {"collapse_monitoring": True, "nan_monitoring": False}

    with pytest.raises(InputError, match="monitoring must be enabled"):
        parse_training_run_metadata(payload, run_dir=tmp_path)


def test_parse_training_run_metadata_rejects_invalid_commit_sha(tmp_path: Path) -> None:
    _write_training_files(tmp_path)
    payload = _metadata()
    payload["commit_sha"] = "not-a-sha"

    with pytest.raises(InputError, match="commit_sha must be"):
        parse_training_run_metadata(payload, run_dir=tmp_path)


def test_parse_training_run_metadata_rejects_unexpected_generator(tmp_path: Path) -> None:
    _write_training_files(tmp_path)
    payload = _metadata()
    payload["generated_by"] = "manual-editor"

    with pytest.raises(InputError, match="generated_by"):
        parse_training_run_metadata(payload, run_dir=tmp_path)


def test_verify_training_run_manifest_rejects_tampered_artifact(tmp_path: Path) -> None:
    metadata_path = _write_training_run_inputs(tmp_path)
    build_training_run_package(tmp_path, metadata_path)
    (tmp_path / "metrics.json").write_text('{"sample_count":1,"metrics":{"loss":9.0}}\n')

    with pytest.raises(InputError, match="artifact hash mismatch"):
        verify_training_run_manifest(tmp_path)


def test_verify_training_run_manifest_rejects_unexpected_generator(tmp_path: Path) -> None:
    metadata_path = _write_training_run_inputs(tmp_path)
    build_training_run_package(tmp_path, metadata_path)
    payload = json.loads((tmp_path / "training_run_manifest.json").read_text(encoding="utf-8"))
    payload["generated_by"] = "manual-editor"
    (tmp_path / "training_run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="generated_by"):
        verify_training_run_manifest(tmp_path)


def test_parse_training_run_metadata_requires_positive_sample_count(tmp_path: Path) -> None:
    _write_training_files(tmp_path)
    (tmp_path / "metrics.json").write_text(
        json.dumps({"sample_count": 0, "metrics": {"loss": 0.42}}),
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="positive sample_count"):
        parse_training_run_metadata(_metadata(), run_dir=tmp_path)


def test_verify_training_run_manifest_requires_preflight_for_release(tmp_path: Path) -> None:
    metadata_path = _write_training_run_inputs(tmp_path, include_preflight=False)
    build_training_run_package(tmp_path, metadata_path)

    with pytest.raises(InputError, match="missing required artifact kinds"):
        verify_training_run_manifest(tmp_path, require_preflight=True)


def test_verify_training_run_manifest_rejects_private_preflight_path(tmp_path: Path) -> None:
    metadata_path = _write_training_run_inputs(tmp_path)
    payload = json.loads((tmp_path / TRAINING_PREFLIGHT_REPORT_NAME).read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(tmp_path / "private-dataset")
    (tmp_path / TRAINING_PREFLIGHT_REPORT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_training_run_package(tmp_path, metadata_path)

    with pytest.raises(InputError, match="public-relative"):
        verify_training_run_manifest(tmp_path, require_preflight=True)


def test_verify_training_run_manifest_rejects_stale_preflight_config_hash(
    tmp_path: Path,
) -> None:
    metadata_path = _write_training_run_inputs(tmp_path)
    payload = json.loads((tmp_path / TRAINING_PREFLIGHT_REPORT_NAME).read_text(encoding="utf-8"))
    payload["training_config"]["sha256"] = "sha256:" + ("0" * 64)
    (tmp_path / TRAINING_PREFLIGHT_REPORT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_training_run_package(tmp_path, metadata_path)

    with pytest.raises(InputError, match="preflight config hash"):
        verify_training_run_manifest(tmp_path, require_preflight=True)


def test_verify_training_run_manifest_requires_preflight_dataset_evidence(
    tmp_path: Path,
) -> None:
    metadata_path = _write_training_run_inputs(tmp_path)
    payload = json.loads((tmp_path / TRAINING_PREFLIGHT_REPORT_NAME).read_text(encoding="utf-8"))
    payload["dataset"]["core_files"].pop("dataset_input_check_report.json")
    (tmp_path / TRAINING_PREFLIGHT_REPORT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_training_run_package(tmp_path, metadata_path)

    with pytest.raises(InputError, match="dataset core file evidence missing"):
        verify_training_run_manifest(tmp_path, require_preflight=True)


def test_verify_training_run_manifest_rejects_invalid_preflight_dataset_evidence(
    tmp_path: Path,
) -> None:
    metadata_path = _write_training_run_inputs(tmp_path)
    payload = json.loads((tmp_path / TRAINING_PREFLIGHT_REPORT_NAME).read_text(encoding="utf-8"))
    payload["dataset"]["core_files"]["dataset_input_check_report.json"]["sha256"] = "not-sha"
    (tmp_path / TRAINING_PREFLIGHT_REPORT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_training_run_package(tmp_path, metadata_path)

    with pytest.raises(InputError, match="dataset core file sha256"):
        verify_training_run_manifest(tmp_path, require_preflight=True)


def _write_training_run_inputs(root: Path, *, include_preflight: bool = True) -> Path:
    _write_training_files(root, include_preflight=include_preflight)
    metadata_path = root / "training_run.json"
    metadata_path.write_text(
        json.dumps(_metadata(include_preflight=include_preflight), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


_MEMBERSHIP_STORE_BINDING = {
    "path": "evidence/membership-store",
    "artifact_id": "geno-lewm-v03-membership-fixture-r1",
    "content_identity": "sha256:" + ("b" * 64),
    "physical_identity": "sha256:" + ("c" * 64),
    "rowset_sha256": "sha256:" + ("d" * 64),
}
_MEMBERSHIP_REPORT_BINDING = {
    "path": "evidence/membership-split-evidence.json",
    "schema_path": "contract/membership-split-evidence.schema.json",
    "artifact_id": "geno-lewm-v03-membership-splits-fixture-r1",
    "schema_version": "geno-lewm.membership-split-evidence.v1",
}


def _membership_runtime_binding() -> dict[str, object]:
    policy = {
        "schema_version": MEMBERSHIP_STORE_SCHEMA_VERSION,
        "membership_content_identity": _MEMBERSHIP_STORE_BINDING["content_identity"],
        "excluded_chromosomes": ["20", "21"],
        "selection": "chromosome_roles",
        "lookup": "lookup.sqlite",
    }
    return {
        "membership_store": dict(_MEMBERSHIP_STORE_BINDING),
        "report": dict(_MEMBERSHIP_REPORT_BINDING),
        "holdout_policy": policy,
        "holdout_policy_identity": canonical_json_sha256(policy),
    }


def _write_bound_training_run_inputs(
    root: Path,
    *,
    binding: dict[str, object],
) -> Path:
    _write_training_files(root)
    dataset_binding = {
        "membership_store": binding["membership_store"],
        "report": binding["report"],
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1.0",
                "snapshot_id": "geno-lewm-data-v0.1.0-r1",
                "membership_and_split_evidence": dataset_binding,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["membership_and_split_evidence"] = binding
    metrics_path.write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    preflight_path = root / TRAINING_PREFLIGHT_REPORT_NAME
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["dataset"]["membership_and_split_evidence"] = dataset_binding
    dataset_manifest_path = root / "dataset_manifest.json"
    preflight["dataset"]["core_files"]["dataset_manifest.json"] = {
        "path": "dataset_manifest.json",
        "sha256": sha256_file(dataset_manifest_path),
        "size_bytes": dataset_manifest_path.stat().st_size,
    }
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = _metadata()
    metadata["schema_version"] = "1.1.0"
    metadata["membership_and_split_evidence"] = binding
    metadata_path = root / "training_run.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _rewrite_manifest_artifact_identity(root: Path, artifact_name: str) -> None:
    manifest_path = root / "training_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = root / artifact_name
    for artifact in manifest["artifacts"]:
        if artifact["path"] == artifact_name:
            artifact["sha256"] = sha256_file(artifact_path)
            artifact["size_bytes"] = artifact_path.stat().st_size
            break
    else:  # pragma: no cover - fixture contract guard.
        raise AssertionError(f"missing artifact {artifact_name}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_training_files(root: Path, *, include_preflight: bool = True) -> None:
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "snapshot_id": "geno-lewm-data-v0.1.0-r1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "train_config.yaml").write_text("seed: 0\nbatch_size: 2\n", encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "sample_count": 128,
                "metrics": {"train_loss": 0.42, "collapse_var_min": {"value": 0.11}},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "train.log").write_text(
        "step=1 loss=0.42 collapse_var_min=0.11 nan_loss=false\n",
        encoding="utf-8",
    )
    (root / "predictor.safetensors").write_bytes(b"predictor")
    (root / "action_encoder.safetensors").write_bytes(b"action")
    (root / "calibration.parquet").write_bytes(b"calibration")
    if include_preflight:
        _write_training_preflight_report(root)


def _metadata(*, include_preflight: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "run_id": "geno-lewm-snv-v0.1.0-r1-seed0",
        "generated_by": TRAINING_RUN_GENERATED_BY,
        "generated_at": "2026-06-01T00:00:00Z",
        "command": "uv run geno-lewm-train --config train_config.yaml",
        "commit_sha": "abcdef1234567890",
        "package_version": "0.1.0.dev0",
        "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "dataset_manifest": "dataset_manifest.json",
        "training_config": "train_config.yaml",
        "metrics": "metrics.json",
        "logs": ["train.log"],
        "checkpoint_files": [
            "predictor.safetensors",
            "action_encoder.safetensors",
            "calibration.parquet",
        ],
        "status": "completed",
        "hardware": ["Apple M3 Max fixture run; release run records accelerator details."],
        "runtime": ["Python 3.10; GenoLeWM 0.1.0.dev0; deterministic fixture backend."],
        "seeds": {"python": 0, "numpy": 0, "torch": 0},
        "determinism": "Deterministic seeds are recorded; GPU kernels may vary by backend.",
        "monitoring": {"collapse_monitoring": True, "nan_monitoring": True},
        "result_summary": "Completed run archive for the first SNV predictor release path.",
        "limitations": [
            "Fixture-scale evidence only; paper claims require the real Carbon-backed run.",
            "Negative results are acceptable when the completed run archive is preserved.",
        ],
    }
    if include_preflight:
        payload["training_preflight_report"] = TRAINING_PREFLIGHT_REPORT_NAME
    return payload


def _write_training_preflight_report(root: Path) -> None:
    training_config = root / "train_config.yaml"
    payload = {
        "schema_version": TRAINING_PREFLIGHT_SCHEMA_VERSION,
        "generated_by": TRAINING_PREFLIGHT_GENERATED_BY,
        "generated_at": "2026-06-01T00:00:00Z",
        "ok": True,
        "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "training_config": {
            "path": "train_config.yaml",
            "sha256": sha256_file(training_config),
            "size_bytes": training_config.stat().st_size,
            "top_level_keys": ["batch_size", "seed"],
            "resolved": {"run_id": "geno-lewm-snv-v0.1.0-r1-seed0"},
        },
        "run_dir": {
            "path": "training-run",
            "exists": True,
            "preflight_report_path": TRAINING_PREFLIGHT_REPORT_NAME,
        },
        "dataset": {
            "path": "dataset",
            "snapshot_id": "geno-lewm-data-v0.1.0-r1",
            "core_files": {
                relative: _preflight_dataset_file_identity(relative)
                for relative in REQUIRED_PREFLIGHT_DATASET_CORE_FILES
            },
            "files": [
                {
                    "path": "carbon/windows.jsonl",
                    "split": "train",
                    "records": 120,
                    "sha256": "sha256:" + ("4" * 64),
                    "size_bytes": 128,
                }
            ],
            "splits": {"train": {"records": 120}, "validation": {"records": 8}},
        },
        "carbon": {
            "path": "carbon-model",
            "local_files_only": True,
            "artifacts": {
                "config": {
                    "path": "config.json",
                    "sha256": "sha256:" + ("1" * 64),
                    "size_bytes": 2,
                },
                "tokenizer": {
                    "path": "tokenizer.json",
                    "sha256": "sha256:" + ("2" * 64),
                    "size_bytes": 2,
                },
                "weights": {
                    "path": "model.safetensors",
                    "sha256": "sha256:" + ("3" * 64),
                    "size_bytes": 2,
                },
            },
        },
        "dependencies": [
            {
                "import_name": "torch",
                "package": "torch",
                "required": True,
                "available": True,
                "version": "2.0.0",
                "reason": "available in release fixture",
            }
        ],
        "issues": [],
    }
    (root / TRAINING_PREFLIGHT_REPORT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _preflight_dataset_file_identity(relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": "sha256:" + ("5" * 64),
        "size_bytes": 128,
    }
