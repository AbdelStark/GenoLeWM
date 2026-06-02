"""Tests for the Hugging Face Hub release dry-run planner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import ReceiptOutput, load_manifest, sha256_file
from geno_lewm.training.preflight import REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME
from tests.unit.test_release_dataset_package import (
    _write_dataset_inputs,
    _write_dataset_snapshot_report,
)
from tests.unit.test_release_model_package import _write_model_inputs
from tests.unit.test_release_paper_package import (
    _jsonl_field_list,
    _receipt_json,
    _write_terminal_demo_manifest,
)
from tools.demo.terminal_inference import DEMO_MANIFEST_NAME
from tools.release import hub_release
from tools.release.batch_receipt_report import write_batch_receipt_report
from tools.release.dataset_package import build_dataset_package
from tools.release.efficiency_report import REPORT_NAME as EFFICIENCY_REPORT_NAME
from tools.release.hub_release import GENERATED_BY, SCHEMA_VERSION, build_hub_release_plan, main
from tools.release.model_package import EVAL_METRICS_NAME, build_model_package
from tools.release.paper_draft import build_paper_draft
from tools.release.runtime_preflight import (
    DependencyProbe,
    RuntimePreflightRequest,
    write_runtime_preflight_report,
)

PAPER_URL = "https://github.com/AbdelStark/GenoLeWM/releases/download/demo-v0.1.0/paper.md"


def test_build_hub_release_plan_verifies_package_and_lists_upload_files(tmp_path: Path) -> None:
    paths = _write_release_candidate(tmp_path)

    plan = build_hub_release_plan(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        paper_url=PAPER_URL,
        commit_sha="abcdef1234567890",
    )

    assert plan.schema_version == SCHEMA_VERSION
    assert plan.generated_by == GENERATED_BY
    assert plan.ready is True
    assert plan.model_id.startswith("sha256:")
    assert plan.release_id == "geno-lewm-v0.1.0-r1"
    assert plan.paper_file is not None
    assert plan.paper_file.source == "paper.md"
    assert plan.paper_file.destination == "paper.md"
    assert plan.paper_file.sha256 == sha256_file(paths["paper_path"])
    assert all(not Path(file.source).is_absolute() for file in plan.files)
    assert all(not Path(file.source).is_absolute() for file in plan.dataset_files)
    assert all(not Path(file.source).is_absolute() for file in plan.demo_files)
    assert {file.destination for file in plan.files} >= {
        "manifest.json",
        "model_card.md",
        EVAL_METRICS_NAME,
        EFFICIENCY_REPORT_NAME,
        "eval_report.md",
        "eval/scores.jsonl",
        "metrics.json",
        "train_config.yaml",
        "train.log",
        "predictor.safetensors",
        TRAINING_PREFLIGHT_REPORT_NAME,
        "SHA256SUMS",
    }
    assert {file.destination for file in plan.dataset_files} >= {
        "data_card.md",
        "dataset_package.json",
        "dataset_manifest.json",
        "split_integrity.json",
        "dataset_snapshot_report.json",
        "carbon/windows.jsonl",
        "clinvar/eval.vcf",
        "SHA256SUMS",
    }
    assert {file.destination for file in plan.demo_files} >= {
        DEMO_MANIFEST_NAME,
        "terminal-demo-transcript.md",
        "scores.jsonl",
        "receipts.jsonl",
        "runtime_preflight_report.json",
        "batch_receipt_report.json",
        "input.vcf",
        "ref.fa",
    }
    assert "tools.release.paper_package" in plan.commands[0]
    assert "huggingface-cli upload AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1" in plan.commands[1]
    commands = "\n".join(plan.commands)
    assert f"{paths['model_dir']} ." not in commands
    assert f"{paths['dataset_dir']} ." not in commands
    assert "eval/scores.jsonl" in commands
    assert "huggingface-cli upload AbdelStark/geno-lewm-data-v0.1.0-r1" in commands
    assert "--repo-type dataset" in commands
    assert "gh release upload demo-v0.1.0" in commands
    assert "--repo AbdelStark/GenoLeWM --clobber" in commands
    assert "gh release upload demo-v0.1.0 paper.md" in commands
    assert "--notes-file" not in commands


def test_hub_release_main_outputs_json_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_release_candidate(tmp_path)

    rc = main(
        [
            "--model-dir",
            str(paths["model_dir"]),
            "--dataset-dir",
            str(paths["dataset_dir"]),
            "--demo-dir",
            str(paths["demo_dir"]),
            "--paper-path",
            str(paths["paper_path"]),
            "--repo-id",
            "AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            "--dataset-url",
            "https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            "--demo-url",
            "https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            "--paper-url",
            PAPER_URL,
            "--commit-sha",
            "abcdef1234567890",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["generated_by"] == GENERATED_BY
    assert payload["generated_at"].endswith("Z")
    assert payload["ready"] is True
    assert payload["model_id"].startswith("sha256:")
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["paper_file"]["source"] == "paper.md"
    assert payload["paper_file"]["destination"] == "paper.md"
    assert payload["paper_file"]["sha256"] == sha256_file(paths["paper_path"])
    assert {file["destination"] for file in payload["dataset_files"]} >= {
        "dataset_manifest.json",
        "carbon/windows.jsonl",
    }
    assert {file["destination"] for file in payload["demo_files"]} >= {
        DEMO_MANIFEST_NAME,
        "terminal-demo-transcript.md",
    }


def test_build_hub_release_plan_rejects_invalid_package(tmp_path: Path) -> None:
    paths = _write_release_candidate(tmp_path)
    (paths["demo_dir"] / "terminal-demo-transcript.md").unlink()

    with pytest.raises(InputError, match="paper/demo release package is not valid"):
        build_hub_release_plan(
            model_dir=paths["model_dir"],
            dataset_dir=paths["dataset_dir"],
            demo_dir=paths["demo_dir"],
            paper_path=paths["paper_path"],
            repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            paper_url=PAPER_URL,
            commit_sha="abcdef1234567890",
        )


def test_build_hub_release_plan_rejects_stale_dataset_upload_inventory(
    tmp_path: Path,
) -> None:
    paths = _write_release_candidate(tmp_path)
    (paths["dataset_dir"] / "carbon" / "windows.jsonl").write_text(
        '{"record_id":"stale","source":"mrna","start_bp":1,"end_bp":13}\n',
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="paper/demo release package is not valid"):
        build_hub_release_plan(
            model_dir=paths["model_dir"],
            dataset_dir=paths["dataset_dir"],
            demo_dir=paths["demo_dir"],
            paper_path=paths["paper_path"],
            repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            paper_url=PAPER_URL,
            commit_sha="abcdef1234567890",
        )


def test_hub_release_checksum_parser_rejects_duplicate_destinations() -> None:
    with pytest.raises(InputError, match="duplicate paths"):
        hub_release._parse_sha256sums(f"{'a' * 64}  manifest.json\n{'a' * 64}  manifest.json\n")


def test_build_hub_release_plan_rejects_private_demo_upload_inputs(
    tmp_path: Path,
) -> None:
    paths = _write_release_candidate(tmp_path)
    private_vcf = tmp_path / "private-input.vcf"
    private_vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    manifest_path = paths["demo_dir"] / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["inputs"]["vcf"] = {
        "path": str(private_vcf),
        "sha256": sha256_file(private_vcf),
        "size_bytes": private_vcf.stat().st_size,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="paper/demo release package is not valid"):
        build_hub_release_plan(
            model_dir=paths["model_dir"],
            dataset_dir=paths["dataset_dir"],
            demo_dir=paths["demo_dir"],
            paper_path=paths["paper_path"],
            repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            paper_url=PAPER_URL,
            commit_sha="abcdef1234567890",
        )


def test_build_hub_release_plan_accepts_package_relative_demo_manifest_paths(
    tmp_path: Path,
) -> None:
    paths = _write_release_candidate(tmp_path)
    manifest_path = paths["demo_dir"] / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("vcf", "fasta"):
        payload["inputs"][key]["path"] = f"demo/{Path(payload['inputs'][key]['path']).name}"
    for artifact in payload["artifacts"]:
        artifact["path"] = f"demo/{Path(artifact['path']).name}"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plan = build_hub_release_plan(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        paper_url=PAPER_URL,
        commit_sha="abcdef1234567890",
    )

    assert {file.destination for file in plan.demo_files} >= {"input.vcf", "ref.fa"}


def test_build_hub_release_plan_rejects_duplicate_demo_asset_basenames(
    tmp_path: Path,
) -> None:
    paths = _write_release_candidate(tmp_path)
    extra_scores = paths["demo_dir"] / "extra" / "scores.jsonl"
    extra_scores.parent.mkdir()
    extra_scores.write_text('{"sigma_raw":0.2}\n', encoding="utf-8")
    manifest_path = paths["demo_dir"] / DEMO_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"].append(
        {
            "label": "extra score preview",
            "path": "extra/scores.jsonl",
            "sha256": sha256_file(extra_scores),
            "size_bytes": extra_scores.stat().st_size,
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="asset names must be unique"):
        build_hub_release_plan(
            model_dir=paths["model_dir"],
            dataset_dir=paths["dataset_dir"],
            demo_dir=paths["demo_dir"],
            repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            commit_sha="abcdef1234567890",
        )


def test_build_hub_release_plan_rejects_fixture_manifest_by_default(tmp_path: Path) -> None:
    paths = _write_release_candidate(tmp_path, release_id="geno-lewm-fixture-r1")

    with pytest.raises(InputError, match="paper/demo release package is not valid"):
        build_hub_release_plan(
            model_dir=paths["model_dir"],
            dataset_dir=paths["dataset_dir"],
            demo_dir=paths["demo_dir"],
            paper_path=paths["paper_path"],
            repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            paper_url=PAPER_URL,
            commit_sha="abcdef1234567890",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repo_id": "https://huggingface.co/AbdelStark/model"}, "repo_id must look like"),
        ({"dataset_url": "not-a-url"}, "dataset_url must be an http"),
        ({"commit_sha": "not-sha"}, "commit_sha must be"),
    ],
)
def test_build_hub_release_plan_rejects_invalid_release_metadata(
    tmp_path: Path,
    kwargs: dict[str, str],
    message: str,
) -> None:
    paths = _write_release_candidate(tmp_path)
    args = {
        "model_dir": paths["model_dir"],
        "dataset_dir": paths["dataset_dir"],
        "demo_dir": paths["demo_dir"],
        "paper_path": paths["paper_path"],
        "repo_id": "AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        "dataset_url": "https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        "demo_url": "https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        "paper_url": PAPER_URL,
        "commit_sha": "abcdef1234567890",
        **kwargs,
    }

    with pytest.raises(InputError, match=message):
        build_hub_release_plan(**args)


def test_build_hub_release_plan_requires_paper_url_for_paper_candidates(
    tmp_path: Path,
) -> None:
    paths = _write_release_candidate(tmp_path)

    with pytest.raises(InputError, match="paper_url is required"):
        build_hub_release_plan(
            model_dir=paths["model_dir"],
            dataset_dir=paths["dataset_dir"],
            demo_dir=paths["demo_dir"],
            paper_path=paths["paper_path"],
            repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            commit_sha="abcdef1234567890",
        )


def _write_release_candidate(
    root: Path,
    *,
    release_id: str = "geno-lewm-v0.1.0-r1",
) -> dict[str, Path]:
    model_dir = root / "model"
    dataset_dir = root / "dataset"
    demo_dir = root / "demo"
    model_dir.mkdir()
    dataset_dir.mkdir()
    demo_dir.mkdir()

    model_metadata = _write_model_inputs(model_dir, release_id=release_id)
    build_model_package(
        model_dir,
        model_metadata,
        allow_fixture_manifest="fixture" in release_id,
    )
    dataset_metadata = _write_dataset_inputs(dataset_dir)
    build_dataset_package(dataset_dir, dataset_metadata)
    _write_dataset_snapshot_report(dataset_dir)
    manifest = load_manifest(model_dir / "manifest.json")
    vcf = demo_dir / "input.vcf"
    fasta = demo_dir / "ref.fa"
    _write_demo_inputs(vcf, fasta)
    runtime_preflight = write_runtime_preflight_report(
        RuntimePreflightRequest(
            model_dir=model_dir,
            vcf=vcf,
            fasta=fasta,
            output_dir=demo_dir,
            backend="cpu",
            allow_fixture_manifest="fixture" in release_id,
        ),
        demo_dir / "runtime_preflight_report.json",
        generated_at="2026-06-01T12:00:00Z",
        dependency_probe=_available_dependency,
    )
    assert runtime_preflight.ok is True
    scores = demo_dir / "scores.jsonl"
    receipts = demo_dir / "receipts.jsonl"
    receipt_output = ReceiptOutput(
        sigma_raw=0.1,
        sigma_calibrated=0.2,
        bucket_id="coding_missense|mid|none",
        confidence=0.9,
        low_confidence=False,
    )
    scores.write_text(
        json.dumps(
            {
                "chrom": "1",
                "pos": 10,
                "ref": "A",
                "alt": "T",
                "sigma_raw": receipt_output.sigma_raw,
                "sigma_calibrated": receipt_output.sigma_calibrated,
                "bucket_id": receipt_output.bucket_id,
                "confidence": receipt_output.confidence,
                "low_confidence": receipt_output.low_confidence,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipts.write_text(
        _receipt_json(
            model_id=manifest.model_id(),
            calibration_hash=manifest.calibration.hash,
            output=receipt_output,
            row_index=1,
        )
        + "\n",
        encoding="utf-8",
    )
    batch_report = write_batch_receipt_report(
        scores,
        receipts,
        demo_dir / "batch_receipt_report.json",
        generated_at="2026-06-01T12:00:00Z",
    )
    (demo_dir / "terminal-demo-transcript.md").write_text(
        "\n".join(
            [
                "# GenoLeWM Terminal Inference Transcript",
                "",
                "- Generated: 2026-06-01T12:00:00Z",
                "- Status: passed",
                "- Exit code: 0",
                f"- Model release: {manifest.release_id}",
                f"- Model version: {manifest.model_version}",
                f"- Model id: {manifest.model_id()}",
                "- Input VCF records: 1",
                "- Input alternate alleles: 1",
                "- Input contigs: 1",
                "- First input variants: 1:10:A>T",
                "- Scores: demo/scores.jsonl",
                "- Receipts: demo/receipts.jsonl",
                "- Runtime preflight report: demo/runtime_preflight_report.json",
                "- Batch receipt report: demo/batch_receipt_report.json",
                "- Demo manifest: demo/terminal_demo_manifest.json",
                "",
                "## Command",
                "",
                "```console",
                "$ geno-lewm-score --model-dir model --vcf input.vcf --fasta ref.fa",
                "```",
                "",
                "## Output Artifacts",
                "",
                "| Artifact | Path | SHA-256 | Bytes | JSONL rows |",
                "| --- | --- | --- | ---: | ---: |",
                f"| scores | demo/scores.jsonl | {sha256_file(scores)} | {scores.stat().st_size} | 1 |",
                f"| receipts | demo/receipts.jsonl | {sha256_file(receipts)} | {receipts.stat().st_size} | 1 |",
                (
                    "| runtime preflight report | demo/runtime_preflight_report.json | "
                    f"{sha256_file(demo_dir / 'runtime_preflight_report.json')} | "
                    f"{(demo_dir / 'runtime_preflight_report.json').stat().st_size} | - |"
                ),
                (
                    "| batch receipt report | demo/batch_receipt_report.json | "
                    f"{sha256_file(batch_report)} | {batch_report.stat().st_size} | - |"
                ),
                f"- Scores SHA-256: {sha256_file(scores)}",
                "- Scores JSONL rows: 1",
                f"- Scores JSONL fields: {_jsonl_field_list(scores)}",
                f"- Receipts SHA-256: {sha256_file(receipts)}",
                "- Receipts JSONL rows: 1",
                f"- Receipts JSONL fields: {_jsonl_field_list(receipts)}",
                f"- Runtime Preflight Report SHA-256: {sha256_file(demo_dir / 'runtime_preflight_report.json')}",
                f"- Batch Receipt Report SHA-256: {sha256_file(batch_report)}",
                "",
                "## Score And Receipt Summary",
                "",
                "- Records: 1",
                f"- Score fields: {_jsonl_field_list(scores)}",
                f"- Receipt fields: {_jsonl_field_list(receipts)}",
                "- Checked score fields: sigma_raw, sigma_calibrated, bucket_id, confidence, low_confidence",
                "- Receipt stream: jsonl_per_scored_alternate_v1",
                "- Receipt schema: 1.0.0",
                f"- Receipt model id: {manifest.model_id()}",
                f"- Calibration hash: {manifest.calibration.hash}",
                "- Runtime backend: cpu",
                "- Runtime device: CPU",
                "",
                "## Artifact Inputs",
                "",
                "- Model directory: model",
                "- Manifest: model/manifest.json",
                "- VCF: demo/input.vcf",
                "- FASTA: demo/ref.fa",
                "",
                "This transcript records command behavior only. Model-quality claims require the "
                "published evaluation report linked from the release.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_terminal_demo_manifest(
        model_dir=model_dir,
        demo_dir=demo_dir,
        vcf=vcf,
        fasta=fasta,
        scores=scores,
        receipts=receipts,
        batch_report=batch_report,
        manifest=manifest,
    )
    paper_path = root / "paper.md"
    build_paper_draft(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        demo_dir=demo_dir,
        output=paper_path,
        generated_at="2026-06-01T12:00:00Z",
    )
    return {
        "model_dir": model_dir,
        "dataset_dir": dataset_dir,
        "demo_dir": demo_dir,
        "paper_path": paper_path,
    }


def _write_demo_inputs(vcf: Path, fasta: Path) -> None:
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    fasta.write_text(">1\nAAAAAAAAAAAAAAAAAAAA\n", encoding="utf-8")


def _available_dependency(import_name: str, required: bool) -> DependencyProbe:
    return DependencyProbe(
        import_name=import_name,
        package=import_name.split(".", 1)[0],
        required=required,
        available=True,
        version="1.0.0",
        reason="available in test",
    )
