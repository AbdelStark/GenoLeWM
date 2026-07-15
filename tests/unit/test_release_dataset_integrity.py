"""Tests for dataset split-integrity release evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file
from tests.unit.test_release_dataset_package import _write_dataset_inputs
from tools.release.dataset_integrity import GENERATED_BY, build_dataset_integrity_report, main
from tools.release.dataset_package import build_dataset_package


def test_build_dataset_integrity_report_checks_counts_hashes_and_leakage(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)

    report = build_dataset_integrity_report(
        tmp_path,
        generated_at="2026-06-01T00:00:00Z",
    )

    assert report.generated_by == GENERATED_BY
    assert report.snapshot_id == "geno-lewm-data-v0.1.0-r1"
    assert report.splits["train"]["observed_records"] == 1
    assert report.splits["train"]["label_counts"] == {}
    assert report.splits["train"]["labelled_records"] == 0
    assert report.splits["train"]["unlabelled_records"] == 1
    assert report.splits["eval_clinvar_coding"]["observed_records"] == 1
    assert report.splits["eval_clinvar_coding"]["label_counts"] == {"P": 1}
    assert report.splits["eval_clinvar_coding"]["labelled_records"] == 1
    assert report.splits["eval_clinvar_coding"]["unlabelled_records"] == 0
    assert report.files[0].comparable_keys == 1
    assert report.files[0].genomic_regions == 0
    assert report.files[0].label_counts == {}
    assert report.files[1].genomic_regions == 1
    assert report.files[1].label_counts == {"P": 1}
    assert "artifact_role" not in report.files[0].to_dict()
    assert "companion_of" not in report.files[0].to_dict()
    assert "included_in_split_totals" not in report.files[0].to_dict()
    assert report.leakage_checks[0]["status"] == "passed"


def test_dataset_integrity_main_writes_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    output = tmp_path / "integrity-copy.json"

    rc = main(["--dataset-dir", str(tmp_path), "--output", str(output)])
    captured = capsys.readouterr()

    assert rc == 0
    assert output.is_file()
    payload = json.loads(captured.out)
    assert payload["generated_by"] == GENERATED_BY
    assert payload["snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert payload["files"][0]["records"] == 1
    assert payload["splits"]["eval_clinvar_coding"]["label_counts"] == {"P": 1}


def test_schema_1_1_integrity_excludes_companions_and_evidence_from_split_totals(
    tmp_path: Path,
) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    companion_path = tmp_path / "carbon" / "windows.labels.jsonl"
    companion_path.write_text(
        '{"record_id":"r1","source":"mrna","start_bp":0,"end_bp":12}\n',
        encoding="utf-8",
    )
    evidence_path = tmp_path / "membership-evidence.json"
    evidence_path.write_text('{"verified":true}\n', encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "1.1.0"
    for file in metadata["files"]:
        file["artifact_role"] = "split_data"
    metadata["files"].extend(
        [
            {
                "path": "carbon/windows.labels.jsonl",
                "artifact_role": "split_companion",
                "split": "train",
                "records": 1,
                "companion_of": "carbon/windows.jsonl",
            },
            {
                "path": "membership-evidence.json",
                "artifact_role": "evidence",
            },
        ]
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    build_dataset_package(tmp_path, metadata_path)
    report = build_dataset_integrity_report(tmp_path)

    assert report.schema_version == "1.1.0"
    assert report.splits["train"]["observed_records"] == 1
    assert report.splits["train"]["files"] == ["carbon/windows.jsonl"]
    assert [file.to_dict()["included_in_split_totals"] for file in report.files] == [
        True,
        True,
        False,
        False,
    ]
    assert report.files[2].to_dict()["artifact_role"] == "split_companion"
    assert report.files[2].to_dict()["companion_of"] == "carbon/windows.jsonl"
    assert report.files[3].to_dict()["artifact_role"] == "evidence"


def test_schema_1_1_integrity_rejects_companion_comparable_key_mismatch(
    tmp_path: Path,
) -> None:
    metadata_path = _write_schema_1_1_companion_inputs(
        tmp_path,
        canonical_row='{"record_id":"canonical"}\n',
        companion_row='{"record_id":"different"}\n',
    )

    with pytest.raises(InputError, match="split_companion comparable key set mismatch"):
        build_dataset_package(tmp_path, metadata_path)


def test_schema_1_1_integrity_rejects_companion_label_count_mismatch(
    tmp_path: Path,
) -> None:
    metadata_path = _write_schema_1_1_companion_inputs(
        tmp_path,
        canonical_row='{"record_id":"r1","label":true}\n',
        companion_row='{"record_id":"r1","label":false}\n',
    )

    with pytest.raises(InputError, match="split_companion label counts mismatch"):
        build_dataset_package(tmp_path, metadata_path)


def test_schema_1_1_integrity_identity_checks_evidence_files(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text('{"verified":true}\n', encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "1.1.0"
    for file in metadata["files"]:
        file["artifact_role"] = "split_data"
    metadata["files"].append({"path": evidence_path.name, "artifact_role": "evidence"})
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    build_dataset_package(tmp_path, metadata_path)
    evidence_path.write_text('{"verified":false}\n', encoding="utf-8")

    with pytest.raises(InputError, match="dataset file hash mismatch"):
        build_dataset_integrity_report(tmp_path)


def test_schema_1_1_integrity_rejects_symlinked_artifacts(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    alias = tmp_path / "carbon" / "windows-alias.jsonl"
    try:
        alias.symlink_to("windows.jsonl")
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.1.0"
    for file in manifest["files"]:
        file["artifact_role"] = "split_data"
    manifest["files"][0]["path"] = "carbon/windows-alias.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match="must not traverse symbolic links"):
        build_dataset_integrity_report(tmp_path)


@pytest.mark.parametrize("field", ["artifact_role", "companion_of"])
def test_schema_1_0_integrity_rejects_role_fields_even_when_null(
    tmp_path: Path,
    field: str,
) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0][field] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match=r"schema 1\.0\.0 forbids"):
        build_dataset_integrity_report(tmp_path)


def test_build_dataset_integrity_report_rejects_split_count_mismatch(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["records"] = 2
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(InputError, match="dataset split record count mismatch"):
        build_dataset_integrity_report(tmp_path)


def test_build_dataset_package_rejects_train_eval_key_overlap(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    (tmp_path / "carbon" / "windows.jsonl").write_text(
        '{"chrom":"1","pos":10,"ref":"A","alt":"T"}\n',
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="dataset split leakage check failed"):
        build_dataset_package(tmp_path, metadata_path)


def test_build_dataset_integrity_report_inspects_parquet_variant_keys(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    schema = pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("clinical_significance", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"chrom": "1", "pos": 10, "ref": "A", "alt": "C", "clinical_significance": "B"}],
            schema=schema,
        ),
        train_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "chrom": "1",
                    "pos": 20,
                    "ref": "G",
                    "alt": "T",
                    "clinical_significance": "Likely pathogenic",
                }
            ],
            schema=schema,
        ),
        eval_path,
    )
    manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": "snapshot",
        "splits": {
            "train": {"records": 1},
            "eval_clinvar": {"records": 1},
        },
        "files": [
            {
                "path": "train.parquet",
                "split": "train",
                "records": 1,
                "sha256": sha256_file(train_path),
                "size_bytes": train_path.stat().st_size,
            },
            {
                "path": "eval.parquet",
                "split": "eval_clinvar",
                "records": 1,
                "sha256": sha256_file(eval_path),
                "size_bytes": eval_path.stat().st_size,
            },
        ],
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = build_dataset_integrity_report(tmp_path)

    assert [file.records for file in report.files] == [1, 1]
    assert [file.comparable_keys for file in report.files] == [1, 1]
    assert [file.label_counts for file in report.files] == [{"B": 1}, {"LP": 1}]
    assert report.splits["train"]["label_counts"] == {"B": 1}
    assert report.splits["eval_clinvar"]["label_counts"] == {"LP": 1}
    assert report.leakage_checks[0]["status"] == "passed"


def test_build_dataset_integrity_report_rejects_parquet_key_overlap(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    row = {"chrom": "1", "pos": 10, "ref": "A", "alt": "C"}
    table = pa.Table.from_pylist([row])
    pq.write_table(table, train_path)
    pq.write_table(table, eval_path)
    manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": "snapshot",
        "splits": {
            "train": {"records": 1},
            "eval_clinvar": {"records": 1},
        },
        "files": [
            {
                "path": "train.parquet",
                "split": "train",
                "records": 1,
                "sha256": sha256_file(train_path),
                "size_bytes": train_path.stat().st_size,
            },
            {
                "path": "eval.parquet",
                "split": "eval_clinvar",
                "records": 1,
                "sha256": sha256_file(eval_path),
                "size_bytes": eval_path.stat().st_size,
            },
        ],
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match="dataset split leakage check failed"):
        build_dataset_integrity_report(tmp_path)


def test_build_dataset_integrity_report_rejects_missing_comparable_keys(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.txt"
    eval_path = tmp_path / "eval.txt"
    train_path.write_text("train row without variant key\n", encoding="utf-8")
    eval_path.write_text("eval row without variant key\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": "snapshot",
        "splits": {
            "train": {"records": 1},
            "eval_clinvar": {"records": 1},
        },
        "files": [
            {
                "path": "train.txt",
                "split": "train",
                "records": 1,
                "sha256": sha256_file(train_path),
                "size_bytes": train_path.stat().st_size,
            },
            {
                "path": "eval.txt",
                "split": "eval_clinvar",
                "records": 1,
                "sha256": sha256_file(eval_path),
                "size_bytes": eval_path.stat().st_size,
            },
        ],
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match="dataset split leakage check failed"):
        build_dataset_integrity_report(tmp_path)


def test_build_dataset_integrity_report_accepts_region_separated_holdout(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train_windows.jsonl"
    holdout_path = tmp_path / "holdout_chr.jsonl"
    train_path.write_text('{"chrom":"1","start_bp":0,"end_bp":100}\n', encoding="utf-8")
    holdout_path.write_text(
        '{"chrom":"chr1","start_bp":200,"end_bp":300}\n',
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": "snapshot",
        "splits": {
            "train": {"records": 1},
            "holdout_chr": {"records": 1},
        },
        "files": [
            {
                "path": "train_windows.jsonl",
                "split": "train",
                "records": 1,
                "sha256": sha256_file(train_path),
                "size_bytes": train_path.stat().st_size,
            },
            {
                "path": "holdout_chr.jsonl",
                "split": "holdout_chr",
                "records": 1,
                "sha256": sha256_file(holdout_path),
                "size_bytes": holdout_path.stat().st_size,
            },
        ],
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = build_dataset_integrity_report(tmp_path)

    assert [file.comparable_keys for file in report.files] == [0, 0]
    assert [file.genomic_regions for file in report.files] == [1, 1]
    assert report.splits["train"]["genomic_regions"] == 1
    assert report.splits["holdout_chr"]["genomic_regions"] == 1
    assert report.leakage_checks[0]["status"] == "passed"
    assert report.leakage_checks[0]["failure_reason"] == ""


def test_build_dataset_integrity_report_rejects_region_holdout_intersection(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train_windows.jsonl"
    holdout_path = tmp_path / "holdout_chr.jsonl"
    train_path.write_text('{"chrom":"1","start_bp":100,"end_bp":250}\n', encoding="utf-8")
    holdout_path.write_text('{"chrom":"1","start_bp":200,"end_bp":300}\n', encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": "snapshot",
        "splits": {
            "train": {"records": 1},
            "holdout_chr": {"records": 1},
        },
        "files": [
            {
                "path": "train_windows.jsonl",
                "split": "train",
                "records": 1,
                "sha256": sha256_file(train_path),
                "size_bytes": train_path.stat().st_size,
            },
            {
                "path": "holdout_chr.jsonl",
                "split": "holdout_chr",
                "records": 1,
                "sha256": sha256_file(holdout_path),
                "size_bytes": holdout_path.stat().st_size,
            },
        ],
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match="dataset split leakage check failed") as excinfo:
        build_dataset_integrity_report(tmp_path)

    assert excinfo.value.details["failure_reason"] == "intersecting_genomic_regions"
    assert excinfo.value.details["region_overlap_count"] == 1
    assert excinfo.value.details["region_examples"] == ["1:100-250 intersects 1:200-300"]


@pytest.mark.parametrize("held_split", ["validation", "evaluation"])
def test_build_dataset_integrity_report_rejects_held_window_intersection(
    tmp_path: Path, held_split: str
) -> None:
    train_path = tmp_path / "train_windows.jsonl"
    held_path = tmp_path / f"{held_split}.jsonl"
    train_path.write_text(
        '{"record_id":"window-r1","chrom":"20","start_bp":100,"end_bp":250}\n',
        encoding="utf-8",
    )
    held_path.write_text(
        '{"variant_key":"20:201:A:C","chrom":"20","pos":201,"ref":"A","alt":"C"}\n',
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": "snapshot",
        "splits": {
            "train": {"records": 1},
            held_split: {"records": 1},
        },
        "files": [
            {
                "path": train_path.name,
                "split": "train",
                "records": 1,
                "sha256": sha256_file(train_path),
                "size_bytes": train_path.stat().st_size,
            },
            {
                "path": held_path.name,
                "split": held_split,
                "records": 1,
                "sha256": sha256_file(held_path),
                "size_bytes": held_path.stat().st_size,
            },
        ],
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match="dataset split leakage check failed") as excinfo:
        build_dataset_integrity_report(tmp_path)

    assert excinfo.value.details["failure_reason"] == "intersecting_genomic_regions"
    assert excinfo.value.details["region_overlap_count"] == 1
    assert excinfo.value.details["region_examples"] == ["20:100-250 intersects 20:200-201"]


def test_build_dataset_integrity_report_rejects_missing_train_eval_comparison(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text('{"record_id":"train-r1"}\n', encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": "snapshot",
        "splits": {"train": {"records": 1}},
        "files": [
            {
                "path": "train.jsonl",
                "split": "train",
                "records": 1,
                "sha256": sha256_file(train_path),
                "size_bytes": train_path.stat().st_size,
            }
        ],
    }
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match="dataset split leakage check failed"):
        build_dataset_integrity_report(tmp_path)


def test_build_dataset_integrity_report_requires_records_for_unparsed_format(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "train.bin"
    data_path.write_bytes(b"opaque")
    manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": "snapshot",
        "splits": {"train": {"records": 1}},
        "files": [
            {
                "path": "train.bin",
                "split": "train",
                "sha256": "sha256:" + "a" * 64,
                "size_bytes": data_path.stat().st_size,
            }
        ],
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match="dataset file hash mismatch"):
        build_dataset_integrity_report(tmp_path)

    observed_hash = sha256_file(data_path)
    manifest["files"][0]["sha256"] = observed_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(InputError, match="format is not record-countable"):
        build_dataset_integrity_report(tmp_path)


def _write_schema_1_1_companion_inputs(
    root: Path,
    *,
    canonical_row: str,
    companion_row: str,
) -> Path:
    metadata_path = _write_dataset_inputs(root)
    (root / "carbon" / "windows.jsonl").write_text(canonical_row, encoding="utf-8")
    companion_path = root / "carbon" / "windows.labels.jsonl"
    companion_path.write_text(companion_row, encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "1.1.0"
    for file in metadata["files"]:
        file["artifact_role"] = "split_data"
    metadata["files"].append(
        {
            "path": "carbon/windows.labels.jsonl",
            "artifact_role": "split_companion",
            "split": "train",
            "records": 1,
            "companion_of": "carbon/windows.jsonl",
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path
