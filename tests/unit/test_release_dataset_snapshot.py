"""Tests for the first-experiment dataset snapshot orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file
from tools.release.dataset_package import GENERATED_BY as DATASET_PACKAGE_GENERATED_BY
from tools.release.dataset_snapshot import (
    INPUT_CHECK_GENERATED_BY,
    INPUT_CHECK_REPORT_NAME,
    REPORT_NAME,
    SPEC_CHECK_GENERATED_BY,
    build_dataset_snapshot,
    check_dataset_snapshot_inputs,
    check_dataset_snapshot_spec,
    main,
)
from tools.release.paper_package import PackageIssue, _verify_dataset_dir

FIRST_EXPERIMENT_SPEC = Path("configs/first_experiment/dataset-snapshot-snv.json")
SERIOUS_COMPLETION_SPEC = Path("configs/serious_completion/dataset-snapshot-snv-post-v02.json")


def test_checked_first_experiment_snapshot_spec_is_valid() -> None:
    report = check_dataset_snapshot_spec(FIRST_EXPERIMENT_SPEC)

    assert report.snapshot_id == "geno-lewm-data-v0.1.0-r1"
    assert report.generated_by == SPEC_CHECK_GENERATED_BY
    assert report.staged_paths == (
        "carbon/source-mix-windows.jsonl",
        "gnomad/v4.1/variants.parquet",
        "placed/gnomad-common-windows.jsonl",
        "clinvar/2026-04-15/variants.parquet",
    )
    assert report.source_paths == (
        "inputs/carbon/source-mix-windows.jsonl",
        "inputs/gnomad/gnomad-v4.1-snv.vcf.gz",
        "inputs/reference/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz",
        "inputs/clinvar/clinvar-2026-04-15-snv.vcf.gz",
    )
    assert report.splits == (
        "eval_clinvar",
        "train_carbon",
        "train_placed_gnomad_common",
        "train_gnomad_common",
    )
    assert report.sources == (
        "Carbon pretraining corpus",
        "gnomAD",
        "Ensembl GRCh38 chromosome FASTA",
        "ClinVar",
    )
    assert all(not Path(path).is_absolute() for path in report.source_paths)
    assert all(".." not in Path(path).parts for path in report.source_paths)


def test_checked_serious_completion_snapshot_spec_is_valid() -> None:
    report = check_dataset_snapshot_spec(SERIOUS_COMPLETION_SPEC)

    assert report.snapshot_id == "geno-lewm-data-v0.2.1-r1"
    assert report.generated_by == SPEC_CHECK_GENERATED_BY
    assert report.staged_paths == (
        "carbon/source-mix-windows.jsonl",
        "gnomad/v4.1/variants.parquet",
        "placed/gnomad-common-windows.jsonl",
        "clinvar/2026-04-15/variants.parquet",
    )
    assert report.source_paths == (
        "inputs/carbon/source-mix-windows.jsonl",
        "inputs/gnomad/gnomad-v4.1-snv.vcf.gz",
        "inputs/reference/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz",
        "inputs/clinvar/clinvar-2026-04-15-snv.vcf.gz",
    )
    assert report.splits == (
        "eval_clinvar",
        "train_carbon",
        "train_placed_gnomad_common",
        "train_gnomad_common",
    )
    assert report.sources == (
        "Carbon pretraining corpus",
        "gnomAD",
        "Ensembl GRCh38 chromosome FASTA",
        "ClinVar",
    )
    assert all(not Path(path).is_absolute() for path in report.source_paths)
    assert all(".." not in Path(path).parts for path in report.source_paths)


def test_dataset_snapshot_main_check_spec_outputs_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["--spec-json", str(FIRST_EXPERIMENT_SPEC), "--check-spec"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["generated_by"] == SPEC_CHECK_GENERATED_BY
    assert payload["snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert payload["snapshot_spec"]["path"] == "configs/first_experiment/dataset-snapshot-snv.json"
    assert payload["snapshot_spec"]["sha256"] == sha256_file(FIRST_EXPERIMENT_SPEC)
    assert payload["staged_paths"] == [
        "carbon/source-mix-windows.jsonl",
        "gnomad/v4.1/variants.parquet",
        "placed/gnomad-common-windows.jsonl",
        "clinvar/2026-04-15/variants.parquet",
    ]


def test_dataset_snapshot_input_check_records_source_identities(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)

    report = check_dataset_snapshot_inputs(spec_path)

    assert report.snapshot_id == "geno-lewm-data-v0.1.0-r1"
    assert report.generated_by == INPUT_CHECK_GENERATED_BY
    assert report.total_size_bytes == sum(file.size_bytes for file in report.inputs)
    assert [file.kind for file in report.inputs] == [
        "carbon",
        "gnomad",
        "placed_windows",
        "clinvar",
    ]
    assert [file.source_path for file in report.inputs] == [
        "inputs/carbon_windows.jsonl",
        "inputs/gnomad.vcf",
        "inputs/reference.fa",
        "inputs/clinvar.vcf",
    ]
    assert [file.staged_path for file in report.inputs] == [
        "carbon/windows.jsonl",
        "gnomad/v4.1/variants.parquet",
        "placed/gnomad-windows.jsonl",
        "clinvar/2026-04-15/variants.parquet",
    ]
    assert [file.sha256 for file in report.inputs] == [
        sha256_file(tmp_path / "inputs" / "carbon_windows.jsonl"),
        sha256_file(tmp_path / "inputs" / "gnomad.vcf"),
        sha256_file(tmp_path / "inputs" / "reference.fa"),
        sha256_file(tmp_path / "inputs" / "clinvar.vcf"),
    ]


def test_dataset_snapshot_main_check_inputs_outputs_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = _write_spec(tmp_path)

    rc = main(["--spec-json", str(spec_path), "--check-inputs"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["generated_by"] == INPUT_CHECK_GENERATED_BY
    assert payload["snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert payload["snapshot_spec"]["sha256"] == sha256_file(spec_path)
    assert payload["source_count"] == 4
    assert [item["kind"] for item in payload["inputs"]] == [
        "carbon",
        "gnomad",
        "placed_windows",
        "clinvar",
    ]
    assert [item["source_path"] for item in payload["inputs"]] == [
        "inputs/carbon_windows.jsonl",
        "inputs/gnomad.vcf",
        "inputs/reference.fa",
        "inputs/clinvar.vcf",
    ]


def test_build_dataset_snapshot_stages_inputs_and_packages_dataset(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"

    report = build_dataset_snapshot(spec_path, dataset_dir)

    assert report.snapshot_id == "geno-lewm-data-v0.1.0-r1"
    assert (dataset_dir / "carbon" / "windows.jsonl").is_file()
    assert (dataset_dir / "gnomad" / "v4.1" / "variants.parquet").is_file()
    assert (dataset_dir / "placed" / "gnomad-windows.jsonl").is_file()
    assert (dataset_dir / "clinvar" / "2026-04-15" / "variants.parquet").is_file()
    assert (dataset_dir / "dataset_package.json").is_file()
    assert (dataset_dir / "dataset_manifest.json").is_file()
    assert (dataset_dir / "data_card.md").is_file()
    assert (dataset_dir / "split_integrity.json").is_file()
    assert (dataset_dir / INPUT_CHECK_REPORT_NAME).is_file()
    assert (dataset_dir / "SHA256SUMS").is_file()
    assert report.report_path == dataset_dir / REPORT_NAME
    assert (dataset_dir / REPORT_NAME).is_file()

    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert [file["path"] for file in manifest["files"]] == [
        "carbon/windows.jsonl",
        "gnomad/v4.1/variants.parquet",
        "placed/gnomad-windows.jsonl",
        "clinvar/2026-04-15/variants.parquet",
    ]
    assert manifest["splits"]["train_carbon"]["records"] == 1
    assert manifest["splits"]["train_gnomad_common"]["records"] == 3
    assert manifest["splits"]["train_placed_gnomad_common"]["records"] == 1
    assert manifest["splits"]["eval_clinvar"]["records"] == 1
    snapshot_report = json.loads((dataset_dir / REPORT_NAME).read_text(encoding="utf-8"))
    assert snapshot_report["schema_version"] == "1.0.0"
    assert snapshot_report["generated_by"] == "tools.release.dataset_snapshot"
    assert snapshot_report["report_path"] == REPORT_NAME
    assert snapshot_report["snapshot_spec"]["path"] == "dataset_snapshot.json"
    assert snapshot_report["snapshot_spec"]["sha256"] == sha256_file(spec_path)
    assert snapshot_report["input_check_path"] == INPUT_CHECK_REPORT_NAME
    assert snapshot_report["input_check"] == _file_identity(dataset_dir, INPUT_CHECK_REPORT_NAME)
    assert snapshot_report["metadata_path"] == "dataset_package.json"
    assert snapshot_report["package"]["schema_version"] == "1.0.0"
    assert snapshot_report["package"]["generated_by"] == DATASET_PACKAGE_GENERATED_BY
    assert snapshot_report["package"]["metadata"] == _file_identity(
        dataset_dir, "dataset_package.json"
    )
    assert snapshot_report["package"]["manifest_path"] == "dataset_manifest.json"
    assert snapshot_report["package"]["manifest"] == _file_identity(
        dataset_dir, "dataset_manifest.json"
    )
    assert snapshot_report["package"]["data_card"] == _file_identity(dataset_dir, "data_card.md")
    assert snapshot_report["package"]["integrity"] == _file_identity(
        dataset_dir, "split_integrity.json"
    )
    package_metadata = json.loads(
        (dataset_dir / "dataset_package.json").read_text(encoding="utf-8")
    )
    assert package_metadata["generated_by"] == DATASET_PACKAGE_GENERATED_BY
    assert manifest["generated_by"] == DATASET_PACKAGE_GENERATED_BY
    assert [file["source_sha256"] for file in snapshot_report["files"]] == [
        sha256_file(tmp_path / "inputs" / "carbon_windows.jsonl"),
        sha256_file(tmp_path / "inputs" / "gnomad.vcf"),
        sha256_file(tmp_path / "inputs" / "reference.fa"),
        sha256_file(tmp_path / "inputs" / "clinvar.vcf"),
    ]
    assert all(str(tmp_path) not in file["source_path"] for file in snapshot_report["files"])
    checksums = (dataset_dir / "SHA256SUMS").read_text(encoding="utf-8")
    assert f"  {INPUT_CHECK_REPORT_NAME}\n" in checksums
    assert f"  {REPORT_NAME}\n" in checksums

    issues: list[PackageIssue] = []
    _verify_dataset_dir(dataset_dir, issues)
    assert issues == []


def test_dataset_snapshot_main_outputs_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"

    rc = main(["--spec-json", str(spec_path), "--dataset-dir", str(dataset_dir)])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert payload["report_path"] == REPORT_NAME
    assert payload["snapshot_spec"]["sha256"] == sha256_file(spec_path)
    assert payload["package"]["generated_by"] == DATASET_PACKAGE_GENERATED_BY
    assert payload["package"]["snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    assert len(payload["files"]) == 4


def test_dataset_snapshot_rejects_missing_local_input(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["carbon_files"][0]["source_path"] = "missing.jsonl"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="dataset snapshot input file is missing"):
        build_dataset_snapshot(spec_path, tmp_path / "release-dataset")


def test_dataset_snapshot_input_check_rejects_missing_local_input(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["gnomad"]["input_vcf"] = "inputs/missing-gnomad.vcf"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="dataset snapshot input file is missing"):
        check_dataset_snapshot_inputs(spec_path)


def test_dataset_snapshot_rejects_split_record_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["splits"]["train_carbon"]["records"] = 2
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="split declared records do not match"):
        build_dataset_snapshot(spec_path, tmp_path / "release-dataset")


def test_dataset_snapshot_rejects_unsafe_carbon_target(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["carbon_files"][0]["path"] = "../windows.jsonl"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="dataset snapshot spec paths must be public relative"):
        build_dataset_snapshot(spec_path, tmp_path / "release-dataset")


def test_dataset_snapshot_spec_check_rejects_private_source_paths(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["carbon_files"][0]["source_path"] = "/private/inputs/carbon_windows.jsonl"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="public relative paths"):
        check_dataset_snapshot_spec(spec_path)


def test_dataset_snapshot_spec_check_rejects_duplicate_staged_paths(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["carbon_files"][0]["path"] = "gnomad/v4.1/variants.parquet"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="duplicate staged paths"):
        check_dataset_snapshot_spec(spec_path)


def test_dataset_snapshot_spec_check_rejects_undeclared_splits(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["gnomad"]["split"] = "train_gnomad_rare"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="staged file split is not declared"):
        check_dataset_snapshot_spec(spec_path)


def test_dataset_verifier_requires_snapshot_report(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    (dataset_dir / REPORT_NAME).unlink()
    issues: list[PackageIssue] = []

    _verify_dataset_dir(dataset_dir, issues)

    codes = {issue.code for issue in issues}
    assert "dataset.snapshot_report.missing" in codes
    assert "dataset.checksums.file_missing" in codes


def test_dataset_verifier_requires_input_check_report(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    (dataset_dir / INPUT_CHECK_REPORT_NAME).unlink()
    issues: list[PackageIssue] = []

    _verify_dataset_dir(dataset_dir, issues)

    codes = {issue.code for issue in issues}
    assert "dataset.snapshot_report.input_check.missing" in codes
    assert "dataset.checksums.file_missing" in codes


def test_dataset_verifier_rejects_stale_input_check_report(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    input_check_path = dataset_dir / INPUT_CHECK_REPORT_NAME
    payload = json.loads(input_check_path.read_text(encoding="utf-8"))
    payload["inputs"][0]["sha256"] = "sha256:" + "0" * 64
    input_check_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = dataset_dir / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["input_check"] = _file_identity(dataset_dir, INPUT_CHECK_REPORT_NAME)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_snapshot_checksum(dataset_dir)
    issues: list[PackageIssue] = []

    _verify_dataset_dir(dataset_dir, issues)

    assert "dataset.snapshot_report.input_check.stale" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_private_snapshot_report_source_path(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    report_path = dataset_dir / REPORT_NAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["files"][0]["source_path"] = str(tmp_path / "inputs" / "carbon_windows.jsonl")
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_snapshot_checksum(dataset_dir)
    issues: list[PackageIssue] = []

    _verify_dataset_dir(dataset_dir, issues)

    assert "dataset.snapshot_report.file.source_path" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_stale_snapshot_report_file_hash(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    report_path = dataset_dir / REPORT_NAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "sha256:" + "0" * 64
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_snapshot_checksum(dataset_dir)
    issues: list[PackageIssue] = []

    _verify_dataset_dir(dataset_dir, issues)

    assert "dataset.snapshot_report.file.hash_mismatch" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_stale_snapshot_report_package_identity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    report_path = dataset_dir / REPORT_NAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["package"]["manifest"]["sha256"] = "sha256:" + "0" * 64
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_snapshot_checksum(dataset_dir)
    issues: list[PackageIssue] = []

    _verify_dataset_dir(dataset_dir, issues)

    assert "dataset.snapshot_report.package.manifest.hash_mismatch" in {
        issue.code for issue in issues
    }


def test_dataset_verifier_rejects_stale_snapshot_report_metadata_identity(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    report_path = dataset_dir / REPORT_NAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["package"]["metadata"]["sha256"] = "sha256:" + "0" * 64
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_snapshot_checksum(dataset_dir)
    issues: list[PackageIssue] = []

    _verify_dataset_dir(dataset_dir, issues)

    assert "dataset.snapshot_report.package.metadata.hash_mismatch" in {
        issue.code for issue in issues
    }


def test_dataset_verifier_rejects_stale_snapshot_report_package_files(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    report_path = dataset_dir / REPORT_NAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["package"]["files"][0]["sha256"] = "sha256:" + "0" * 64
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_snapshot_checksum(dataset_dir)
    issues: list[PackageIssue] = []

    _verify_dataset_dir(dataset_dir, issues)

    assert "dataset.snapshot_report.package.files_stale" in {issue.code for issue in issues}


def test_dataset_verifier_rejects_duplicate_snapshot_report_file_entries(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    spec_path = _write_spec(tmp_path)
    dataset_dir = tmp_path / "release-dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    report_path = dataset_dir / REPORT_NAME
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["files"].append(dict(payload["files"][0]))
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_snapshot_checksum(dataset_dir)
    issues: list[PackageIssue] = []

    _verify_dataset_dir(dataset_dir, issues)

    assert "dataset.snapshot_report.file.duplicate" in {issue.code for issue in issues}


def _file_identity(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_spec(root: Path) -> Path:
    inputs = root / "inputs"
    inputs.mkdir()
    carbon = inputs / "carbon_windows.jsonl"
    carbon.write_text(
        '{"record_id":"carbon-r1","source":"mrna","chrom":"1","start_bp":0,"end_bp":128,"sequence":"'
        + ("A" * 128)
        + '"}\n',
        encoding="utf-8",
    )
    gnomad = _write_gnomad_vcf(inputs / "gnomad.vcf")
    clinvar = _write_clinvar_vcf(inputs / "clinvar.vcf")
    reference = _write_reference_fasta(inputs / "reference.fa")
    spec = {
        "schema_version": "1.0.0",
        "snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "generated_at": "2026-06-01T00:00:00Z",
        "sources": [
            {
                "name": "Carbon pretraining corpus",
                "revision": "carbon-local-snapshot-2026-04-15",
                "url": "https://huggingface.co/collections/HuggingFaceBio/carbon",
                "license": "upstream Carbon corpus terms",
                "notes": "local staged shard selected for the first experiment snapshot",
            },
            {
                "name": "gnomAD",
                "revision": "v4.1",
                "url": "https://gnomad.broadinstitute.org/",
                "license": "gnomAD terms of use",
            },
            {
                "name": "ClinVar",
                "revision": "2026-04-15",
                "url": "https://www.ncbi.nlm.nih.gov/clinvar/",
                "license": "NCBI public data terms",
            },
            {
                "name": "Ensembl GRCh38 chromosome FASTA",
                "revision": "release-110 fixture",
                "url": "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/",
                "license": "Ensembl genome browser data terms",
            },
        ],
        "license": "Apache-2.0 for package metadata; upstream data licenses apply.",
        "preprocessing": [
            "Stage pinned Carbon source-mix windows from local upstream release files.",
            "Filter local gnomAD VCF rows to PASS alleles above the global AF threshold.",
            "Generate placed windows from the gnomAD shard and a staged reference FASTA.",
            "Normalize local ClinVar VCF rows and preserve labelled and unlabelled classes.",
        ],
        "split_policy": "Training splits exclude held-out ClinVar evaluation variant keys.",
        "splits": {
            "train_carbon": {"description": "Carbon source-mix windows for training"},
            "train_gnomad_common": {"description": "gnomAD common variants for edit sampling"},
            "train_placed_gnomad_common": {
                "description": "reference-placed gnomAD windows for training"
            },
            "eval_clinvar": {"description": "held-out ClinVar SNV labels"},
        },
        "leakage_checks": [
            "No train split may share comparable variant keys with ClinVar evaluation rows.",
            "Split record counts are recomputed from staged files before publication.",
        ],
        "intended_use": "First GenoLeWM paper/demo release experiments and reproducibility checks.",
        "limitations": [
            "This package records a research snapshot; it is not a clinical dataset.",
            "ClinVar labels may change after the pinned upstream release.",
        ],
        "carbon_files": [
            {
                "source_path": carbon.relative_to(root).as_posix(),
                "path": "carbon/windows.jsonl",
                "split": "train_carbon",
                "description": "Carbon source-mix windows staged from the pinned local snapshot",
            }
        ],
        "gnomad": {
            "input_vcf": gnomad.relative_to(root).as_posix(),
            "release": "v4.1",
            "min_af": 0.01,
            "split": "train_gnomad_common",
            "description": "gnomAD PASS common variants used for edit sampling",
        },
        "placed_windows": {
            "input_fasta": reference.relative_to(root).as_posix(),
            "path": "placed/gnomad-windows.jsonl",
            "split": "train_placed_gnomad_common",
            "description": "reference-placed gnomAD windows generated for training",
            "source": "gnomad_common",
            "variant_source": "gnomad",
            "window_bp": 4096,
            "min_variants_per_window": 1,
            "max_windows": 1,
        },
        "clinvar": {
            "input_vcf": clinvar.relative_to(root).as_posix(),
            "release": "2026-04-15",
            "split": "eval_clinvar",
            "description": "ClinVar variants held out for first experiment evaluation",
        },
    }
    spec_path = root / "dataset_snapshot.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    return spec_path


def _write_gnomad_vcf(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "1\t10\trs1\tA\tC\t.\tPASS\tAF=0.02;AF_afr=0.03",
                "1\t20\trs2\tA\tG\t.\tPASS\tAF=0.03;AF_afr=0.04",
                "1\t30\trs3\tA\tT\t.\tPASS\tAF=0.04;AF_afr=0.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_reference_fasta(path: Path) -> Path:
    path.write_text(
        ">1 dna:chromosome chromosome:GRCh38:1:1:512:1 REF\n" + ("A" * 512) + "\n",
        encoding="utf-8",
    )
    return path


def _rewrite_snapshot_checksum(dataset_dir: Path) -> None:
    entries = []
    for raw_line in (dataset_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        _, relative = raw_line.split()
        digest = sha256_file(dataset_dir / relative).removeprefix("sha256:")
        entries.append(f"{digest}  {relative}")
    (dataset_dir / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _write_clinvar_vcf(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "1\t100\t101\tA\tG\t.\t.\tCLNSIG=Pathogenic;ALLELEID=111",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
