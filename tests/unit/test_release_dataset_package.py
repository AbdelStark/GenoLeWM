"""Tests for the release dataset package builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file
from tools.release.dataset_integrity import DEFAULT_REPORT_NAME
from tools.release.dataset_package import (
    GENERATED_BY as DATASET_PACKAGE_GENERATED_BY,
    build_dataset_package,
    main,
    parse_dataset_package,
)
from tools.release.dataset_snapshot import (
    GENERATED_BY as DATASET_SNAPSHOT_GENERATED_BY,
    INPUT_CHECK_GENERATED_BY as DATASET_INPUT_CHECK_GENERATED_BY,
    INPUT_CHECK_REPORT_NAME as DATASET_INPUT_CHECK_REPORT_NAME,
    REPORT_NAME as DATASET_SNAPSHOT_REPORT_NAME,
)
from tools.release.paper_package import PackageIssue, _verify_dataset_dir


def test_build_dataset_package_writes_verifier_compatible_artifacts(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)

    report = build_dataset_package(tmp_path, metadata_path)

    assert report.snapshot_id == "geno-lewm-data-v0.1.0-r1"
    assert (tmp_path / "dataset_package.json").is_file()
    assert (tmp_path / "dataset_manifest.json").is_file()
    assert (tmp_path / "data_card.md").is_file()
    assert (tmp_path / "split_integrity.json").is_file()
    assert (tmp_path / "SHA256SUMS").is_file()

    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert manifest["files"][0]["path"] == "carbon/windows.jsonl"
    assert manifest["files"][0]["sha256"].startswith("sha256:")
    assert manifest["files"][0]["size_bytes"] > 0
    assert manifest["files"][0]["records"] == 1

    integrity = json.loads((tmp_path / "split_integrity.json").read_text(encoding="utf-8"))
    assert integrity["splits"]["train"]["observed_records"] == 1
    assert integrity["splits"]["train"]["label_counts"] == {}
    assert integrity["splits"]["eval_clinvar_coding"]["label_counts"] == {"P": 1}
    assert integrity["splits"]["eval_clinvar_coding"]["labelled_records"] == 1

    data_card = (tmp_path / "data_card.md").read_text(encoding="utf-8")
    assert "## Sources" in data_card
    assert "## License" in data_card
    assert "## Preprocessing" in data_card
    assert "## Splits" in data_card
    assert "## Class Balance" in data_card
    assert "| train | 1 | 0 | 1 | none observed |" in data_card
    assert "| eval_clinvar_coding | 1 | 1 | 0 | P=1 |" in data_card
    assert "split_integrity.json" in data_card
    assert "## Limitations" in data_card

    _write_dataset_snapshot_report(tmp_path)
    issues: list[PackageIssue] = []
    _verify_dataset_dir(tmp_path, issues)
    assert issues == []


def test_dataset_verifier_requires_metadata_json(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    (tmp_path / "dataset_package.json").unlink()
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.metadata_missing" in {issue.code for issue in issues}
    assert "dataset.checksums.file_missing" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_stale_manifest_from_metadata(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    manifest_path = tmp_path / "dataset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["snapshot_id"] = "stale-snapshot"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.manifest_stale" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_stale_data_card_from_metadata(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    card_path = tmp_path / "data_card.md"
    card_path.write_text(
        card_path.read_text(encoding="utf-8").replace("First GenoLeWM", "Changed GenoLeWM", 1),
        encoding="utf-8",
    )
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.card.stale" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_stale_data_card_class_balance(tmp_path: Path) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    card_path = tmp_path / "data_card.md"
    card_path.write_text(
        card_path.read_text(encoding="utf-8").replace(
            "| eval_clinvar_coding | 1 | 1 | 0 | P=1 |",
            "| eval_clinvar_coding | 1 | 0 | 1 | none observed |",
        ),
        encoding="utf-8",
    )
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.card.stale" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_hand_authored_split_integrity_report(
    tmp_path: Path,
) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)
    build_dataset_package(tmp_path, metadata_path)
    _write_dataset_snapshot_report(tmp_path)
    integrity_path = tmp_path / DEFAULT_REPORT_NAME
    payload = json.loads(integrity_path.read_text(encoding="utf-8"))
    payload["generated_by"] = "manual-editor"
    integrity_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    issues: list[PackageIssue] = []

    _verify_dataset_dir(tmp_path, issues)

    assert "dataset.integrity.generated_by" in {issue.code for issue in issues}


def test_dataset_package_main_outputs_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata_path = _write_dataset_inputs(tmp_path)

    rc = main(["--dataset-dir", str(tmp_path), "--metadata-json", str(metadata_path)])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "1.0.0"
    assert payload["generated_by"] == DATASET_PACKAGE_GENERATED_BY
    assert payload["snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["manifest_path"] == "dataset_manifest.json"
    assert payload["data_card_path"] == "data_card.md"
    assert payload["integrity_path"] == DEFAULT_REPORT_NAME
    assert payload["checksums_path"] == "SHA256SUMS"
    assert len(payload["files"]) == 2


def test_parse_dataset_package_rejects_missing_file(tmp_path: Path) -> None:
    payload = _metadata()
    payload["files"][0]["path"] = "missing.jsonl"

    with pytest.raises(InputError, match="dataset file is missing"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_unsafe_file_path(tmp_path: Path) -> None:
    payload = _metadata()
    payload["files"][0]["path"] = "../outside.jsonl"

    with pytest.raises(InputError, match="dataset paths must be relative"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_placeholder_text(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["limitations"] = ["TODO"]

    with pytest.raises(InputError, match="placeholder text is not allowed"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_unexpected_generator(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["generated_by"] = "manual-editor"

    with pytest.raises(InputError, match="generated_by"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_undeclared_file_split(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    payload = _metadata()
    payload["files"][0]["split"] = "holdout"

    with pytest.raises(InputError, match="file split must be declared"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def test_parse_dataset_package_rejects_generated_file_as_input(tmp_path: Path) -> None:
    _write_data_files(tmp_path)
    (tmp_path / "data_card.md").write_text("# Data Card\n", encoding="utf-8")
    payload = _metadata()
    payload["files"].append({"path": "data_card.md", "split": "train"})

    with pytest.raises(InputError, match="generated dataset package files cannot be listed"):
        parse_dataset_package(payload, dataset_dir=tmp_path)


def _write_dataset_inputs(root: Path) -> Path:
    _write_data_files(root)
    metadata_path = root / "dataset_package.json"
    metadata_path.write_text(json.dumps(_metadata(), indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def _write_data_files(root: Path) -> None:
    (root / "carbon").mkdir()
    (root / "clinvar").mkdir()
    (root / "carbon" / "windows.jsonl").write_text(
        '{"record_id":"r1","source":"mrna","start_bp":0,"end_bp":12}\n',
        encoding="utf-8",
    )
    (root / "clinvar" / "eval.vcf").write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\tCLNSIG=Pathogenic\n",
        encoding="utf-8",
    )


def _write_dataset_snapshot_report(root: Path) -> Path:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    files = []
    for item in manifest["files"]:
        path = root / item["path"]
        files.append(
            {
                "path": item["path"],
                "source_path": f"sources/{Path(item['path']).name}",
                "source_sha256": sha256_file(path),
                "source_size_bytes": path.stat().st_size,
                "split": item["split"],
                "records": item["records"],
                "description": item["description"],
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "already_exists": False,
            }
        )
    payload = {
        "schema_version": "1.0.0",
        "generated_by": DATASET_SNAPSHOT_GENERATED_BY,
        "generated_at": manifest["generated_at"],
        "snapshot_id": manifest["snapshot_id"],
        "report_path": DATASET_SNAPSHOT_REPORT_NAME,
        "snapshot_spec": {
            "path": "dataset_snapshot.json",
            "sha256": "sha256:" + "1" * 64,
            "size_bytes": 1024,
        },
        "input_check_path": DATASET_INPUT_CHECK_REPORT_NAME,
        "input_check": _file_identity(root, _write_dataset_input_check_report(root, files)),
        "metadata_path": "dataset_package.json",
        "package": {
            "snapshot_id": manifest["snapshot_id"],
            "metadata": _file_identity(root, "dataset_package.json"),
            "manifest_path": "dataset_manifest.json",
            "manifest": _file_identity(root, "dataset_manifest.json"),
            "data_card_path": "data_card.md",
            "data_card": _file_identity(root, "data_card.md"),
            "integrity_path": DEFAULT_REPORT_NAME,
            "integrity": _file_identity(root, DEFAULT_REPORT_NAME),
            "checksums_path": "SHA256SUMS",
            "files": [
                {
                    key: file[key]
                    for key in ("path", "sha256", "size_bytes", "split", "records", "description")
                    if key in file
                }
                for file in files
            ],
        },
        "files": files,
    }
    report_path = root / DATASET_SNAPSHOT_REPORT_NAME
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sha256sums(
        root,
        (
            "data_card.md",
            "dataset_package.json",
            "dataset_manifest.json",
            DEFAULT_REPORT_NAME,
            DATASET_INPUT_CHECK_REPORT_NAME,
            DATASET_SNAPSHOT_REPORT_NAME,
            *(file["path"] for file in manifest["files"]),
        ),
    )
    return report_path


def _write_dataset_input_check_report(root: Path, files: list[dict[str, object]]) -> str:
    inputs = []
    total_size_bytes = 0
    for item in files:
        size_bytes = item["source_size_bytes"]
        assert isinstance(size_bytes, int)
        total_size_bytes += size_bytes
        inputs.append(
            {
                "kind": "carbon" if str(item["path"]).startswith("carbon/") else "clinvar",
                "source_path": item["source_path"],
                "staged_path": item["path"],
                "split": item["split"],
                "description": item["description"],
                "sha256": item["source_sha256"],
                "size_bytes": size_bytes,
            }
        )
    payload = {
        "ok": True,
        "schema_version": "1.0.0",
        "generated_by": DATASET_INPUT_CHECK_GENERATED_BY,
        "snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "snapshot_spec": {
            "path": "dataset_snapshot.json",
            "sha256": "sha256:" + "1" * 64,
            "size_bytes": 1024,
        },
        "source_count": len(inputs),
        "total_size_bytes": total_size_bytes,
        "inputs": inputs,
    }
    (root / DATASET_INPUT_CHECK_REPORT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return DATASET_INPUT_CHECK_REPORT_NAME


def _file_identity(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_sha256sums(root: Path, files: tuple[str, ...]) -> None:
    lines = []
    for relative in files:
        digest = sha256_file(root / relative).removeprefix("sha256:")
        lines.append(f"{digest}  {relative}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metadata() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "generated_by": DATASET_PACKAGE_GENERATED_BY,
        "generated_at": "2026-06-01T00:00:00Z",
        "sources": [
            {
                "name": "Carbon pretraining corpus",
                "revision": "2026-04-15",
                "url": "https://huggingface.co/collections/HuggingFaceBio/carbon",
                "license": "upstream Carbon corpus terms",
                "notes": "deterministic Phase 1 source mix",
            },
            {
                "name": "ClinVar",
                "revision": "2026-04-15",
                "url": "https://www.ncbi.nlm.nih.gov/clinvar/",
                "license": "NCBI public data terms",
            },
        ],
        "license": "Apache-2.0 for package metadata; upstream data licenses apply.",
        "preprocessing": [
            "Select the pinned Carbon source mix and deterministic subset seed.",
            "Normalize VCF alleles to uppercase ACGT and split ClinVar coding/non-coding rows.",
        ],
        "split_policy": "Training windows exclude all ClinVar evaluation variants by genomic locus.",
        "splits": {
            "train": {"records": 1, "description": "Carbon corpus source-mix windows"},
            "eval_clinvar_coding": {"records": 1, "description": "held-out ClinVar coding SNVs"},
        },
        "leakage_checks": [
            "No ClinVar evaluation locus may appear in training-window positive labels.",
            "Split manifests are checked before publishing the dataset package.",
        ],
        "intended_use": "First GenoLeWM paper/demo release experiments and reproducibility checks.",
        "limitations": [
            "This package records the selected snapshot; it is not a clinical dataset.",
            "ClinVar labels may change after the pinned upstream release.",
        ],
        "files": [
            {
                "path": "carbon/windows.jsonl",
                "split": "train",
                "records": 1,
                "description": "sampled Carbon source-mix windows",
            },
            {
                "path": "clinvar/eval.vcf",
                "split": "eval_clinvar_coding",
                "records": 1,
                "description": "held-out ClinVar coding variants",
            },
        ],
    }
