"""Tests for clean-machine terminal demo replay from public artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import load_manifest, sha256_file
from tests.unit.test_release_hub_release import PAPER_URL, _write_release_candidate
from tools.demo.terminal_inference import DEMO_MANIFEST_NAME, GENERATED_BY as DEMO_GENERATED_BY
from tools.release import clean_machine_demo
from tools.release.clean_machine_demo import replay_public_terminal_demo
from tools.release.paper_package import PackageIssue, PackageReport
from tools.release.release_candidate import (
    PublicArtifactCheck,
    PublicLinkCheck,
    build_release_candidate_report,
)


def test_replay_public_terminal_demo_downloads_artifacts_and_runs_demo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(
            paths["model_dir"],
            paths["dataset_dir"],
            paths["demo_dir"],
            expected_hf_token="hf-token",
            expected_github_token="github-token",
        ),
    )

    def write_replay_transcript(request: clean_machine_demo.DemoRequest) -> Path:
        assert request.model_dir == output_dir / "model"
        assert request.vcf == output_dir / "demo" / "input.vcf"
        assert request.fasta == output_dir / "demo" / "ref.fa"
        assert request.require_native_runtime is False
        request.output_dir.mkdir(parents=True, exist_ok=True)
        transcript = request.output_dir / "terminal-demo-transcript.md"
        transcript.write_text("# replay\n\n- Status: passed\n", encoding="utf-8")
        (request.output_dir / DEMO_MANIFEST_NAME).write_text(
            json.dumps({"status": "passed"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (request.output_dir / "scores.jsonl").write_text('{"sigma_raw":0.1}\n', encoding="utf-8")
        (request.output_dir / "receipts.jsonl").write_text(
            '{"schema_version":"1.0.0"}\n',
            encoding="utf-8",
        )
        _write_runtime_preflight_report(request)
        _write_batch_receipt_report(request)
        _write_replay_manifest(request)
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_replay_transcript)

    report = replay_public_terminal_demo(
        release_candidate_report=report_path,
        output_dir=output_dir,
        require_native_runtime=False,
        hf_token="hf-token",
        github_token="github-token",
        generated_at="2026-06-01T12:00:00Z",
    )

    assert (output_dir / report.transcript_path).is_file()
    assert (output_dir / report.demo_manifest_path).is_file()
    assert (output_dir / "model" / "manifest.json").is_file()
    assert (output_dir / "dataset" / "dataset_manifest.json").is_file()
    assert (output_dir / "dataset" / "clinvar" / "eval.vcf").is_file()
    assert (output_dir / "demo" / DEMO_MANIFEST_NAME).is_file()
    assert {artifact.group for artifact in report.downloaded_artifacts} == {
        "model",
        "dataset",
        "demo",
    }
    payload = json.loads((output_dir / "clean_machine_demo_report.json").read_text())
    assert payload["schema_version"] == "1.0.0"
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["release_candidate_report"] == "release_candidate_report.json"
    assert payload["release_candidate_report_identity"]["path"] == "release_candidate_report.json"
    assert payload["model_dir"] == "model"
    assert payload["dataset_dir"] == "dataset"
    assert payload["demo_dir"] == "demo"
    assert payload["replay_dir"] == "replay"
    assert payload["package"] == {"issues": [], "ok": True}
    assert payload["transcript_path"] == "replay/terminal-demo-transcript.md"
    assert payload["demo_manifest_path"] == f"replay/{DEMO_MANIFEST_NAME}"
    assert payload["release_candidate_report_identity"]["sha256"].startswith("sha256:")
    downloaded_paths = {artifact["path"] for artifact in payload["downloaded_artifacts"]}
    assert all(not Path(path).is_absolute() for path in downloaded_paths)
    assert {
        "model/manifest.json",
        "dataset/dataset_manifest.json",
        f"demo/{DEMO_MANIFEST_NAME}",
    }.issubset(downloaded_paths)
    assert {artifact["label"] for artifact in payload["replay_artifacts"]} == {
        "terminal transcript",
        "terminal demo manifest",
        "scores",
        "receipts",
        "runtime preflight report",
        "batch receipt report",
    }


def test_replay_public_terminal_demo_requires_full_replay_artifact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )

    def write_incomplete_replay(request: clean_machine_demo.DemoRequest) -> Path:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        transcript = request.output_dir / "terminal-demo-transcript.md"
        transcript.write_text("# replay\n", encoding="utf-8")
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_incomplete_replay)

    with pytest.raises(InputError, match="replay artifact is missing"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_not_ready_report(tmp_path: Path) -> None:
    report_path = tmp_path / "release_candidate_report.json"
    report_path.write_text('{"ready": false}\n', encoding="utf-8")

    with pytest.raises(InputError, match="requires a ready release candidate"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
        )


def test_replay_public_terminal_demo_rejects_manual_candidate_report(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["generated_by"] = "manual-release-candidate"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="release candidate report generated_by is invalid"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_manual_hub_plan(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["hub_plan"]["generated_by"] = "manual-hub-plan"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="release candidate Hub plan generated_by is invalid"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_stripped_readiness(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    del payload["readiness"]
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="readiness must be a list"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_failed_candidate_readiness(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for item in payload["readiness"]:
        if item["code"] == "public_artifacts":
            item["ok"] = False
            break
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="passing release-candidate readiness"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_missing_public_artifact_check(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["public_artifacts"]["checks"] = [
        check for check in payload["public_artifacts"]["checks"] if check["name"] != "demo"
    ]
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="passing public artifact checks"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_skipped_public_checks(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["public_links"]["required"] = False
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="requires public link checks"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_candidate_plan_identity_mismatch(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["model_id"] = "sha256:" + "f" * 64
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="identity does not match embedded Hub plan"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_unsafe_hub_plan_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["hub_plan"]["files"][0]["destination"] = "../outside.txt"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )

    with pytest.raises(InputError, match="destinations must be package-relative"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_malformed_hub_plan_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["hub_plan"]["dataset_files"][0]["sha256"] = "not-a-sha"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )

    with pytest.raises(InputError, match="sha256 must be sha256"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_duplicate_demo_asset_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    duplicate = dict(payload["hub_plan"]["demo_files"][0])
    duplicate["destination"] = f"nested/{duplicate['destination']}"
    payload["hub_plan"]["demo_files"].append(duplicate)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )

    with pytest.raises(InputError, match="duplicate demo asset names"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_hash_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(
            paths["model_dir"],
            paths["dataset_dir"],
            paths["demo_dir"],
            corrupt_model_card=True,
        ),
    )

    with pytest.raises(InputError, match="hash mismatch"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_demo_input_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    manifest_path = paths["demo_dir"] / DEMO_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["vcf"]["sha256"] = "sha256:" + "e" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    for file in report_payload["hub_plan"]["demo_files"]:
        if file["destination"] == DEMO_MANIFEST_NAME:
            file["sha256"] = clean_machine_demo.sha256_file(manifest_path)
            file["size_bytes"] = manifest_path.stat().st_size
            break
    report_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )
    monkeypatch.setattr(
        clean_machine_demo,
        "verify_package",
        lambda _paths: PackageReport(ok=True, model_id="sha256:" + "a" * 64, issues=()),
    )
    monkeypatch.setattr(
        clean_machine_demo,
        "run_demo_transcript",
        lambda _request: (_ for _ in ()).throw(AssertionError("replay called")),
    )

    with pytest.raises(InputError, match="downloaded demo input hash mismatch"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_invalid_downloaded_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )
    monkeypatch.setattr(
        clean_machine_demo,
        "verify_package",
        lambda _paths: PackageReport(
            ok=False,
            model_id=None,
            issues=(
                PackageIssue(
                    severity="error",
                    code="demo.transcript_missing",
                    path="demo/terminal-demo-transcript.md",
                    message="terminal transcript is required",
                ),
            ),
        ),
    )

    with pytest.raises(InputError, match="downloaded public release package is invalid"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=tmp_path / "clean-machine",
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_replayed_manifest_model_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )

    def write_bad_manifest(request: clean_machine_demo.DemoRequest) -> Path:
        transcript = _write_replay_artifacts(request)
        _write_replay_manifest(request, model_id="sha256:" + ("f" * 64))
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_bad_manifest)

    with pytest.raises(InputError, match="manifest model_id"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_replayed_manifest_artifact_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )

    def write_bad_manifest(request: clean_machine_demo.DemoRequest) -> Path:
        transcript = _write_replay_artifacts(request)
        _write_replay_manifest(
            request,
            artifact_overrides={"scores": {"sha256": "sha256:" + ("e" * 64)}},
        )
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_bad_manifest)

    with pytest.raises(InputError, match="artifact hash mismatch"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_replayed_manifest_vcf_input_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )

    def write_bad_manifest(request: clean_machine_demo.DemoRequest) -> Path:
        transcript = _write_replay_artifacts(request)
        _write_replay_manifest(
            request,
            input_overrides={"vcf": {"sha256": "sha256:" + ("e" * 64)}},
        )
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_bad_manifest)

    with pytest.raises(InputError, match="artifact hash mismatch"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_replayed_manifest_fasta_input_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )

    def write_bad_manifest(request: clean_machine_demo.DemoRequest) -> Path:
        transcript = _write_replay_artifacts(request)
        _write_replay_manifest(
            request,
            input_overrides={"fasta": {"path": "demo/other.fa"}},
        )
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_bad_manifest)

    with pytest.raises(InputError, match="artifact path mismatch"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_requires_replayed_manifest_runtime_preflight_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )

    def write_bad_manifest(request: clean_machine_demo.DemoRequest) -> Path:
        transcript = _write_replay_artifacts(request)
        _write_replay_manifest(request, include_runtime_preflight=False)
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_bad_manifest)

    with pytest.raises(InputError, match="must include runtime_preflight"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_replayed_manifest_runtime_preflight_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )

    def write_bad_manifest(request: clean_machine_demo.DemoRequest) -> Path:
        transcript = _write_replay_artifacts(request)
        _write_replay_manifest(
            request,
            runtime_preflight_overrides={"command": {"argv": ["geno-lewm-score", "--bad"]}},
        )
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_bad_manifest)

    with pytest.raises(InputError, match="runtime_preflight summary mismatch"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_requires_replayed_manifest_score_receipt_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )

    def write_bad_manifest(request: clean_machine_demo.DemoRequest) -> Path:
        transcript = _write_replay_artifacts(request)
        _write_replay_manifest(request, include_score_receipt_batch=False)
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_bad_manifest)

    with pytest.raises(InputError, match="must include score_receipt_batch"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def test_replay_public_terminal_demo_rejects_replayed_manifest_score_receipt_batch_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    paths = _write_release_candidate(candidate_root)
    report_path = _write_ready_report(tmp_path, paths)
    output_dir = tmp_path / "clean-machine"
    monkeypatch.setattr(
        clean_machine_demo.urllib_request,
        "urlopen",
        _fake_urlopen(paths["model_dir"], paths["dataset_dir"], paths["demo_dir"]),
    )

    def write_bad_manifest(request: clean_machine_demo.DemoRequest) -> Path:
        transcript = _write_replay_artifacts(request)
        _write_replay_manifest(request, score_receipt_overrides={"records": 2})
        return transcript

    monkeypatch.setattr(clean_machine_demo, "run_demo_transcript", write_bad_manifest)

    with pytest.raises(InputError, match="score_receipt_batch summary mismatch"):
        replay_public_terminal_demo(
            release_candidate_report=report_path,
            output_dir=output_dir,
            require_native_runtime=False,
        )


def _write_ready_report(tmp_path: Path, paths: dict[str, Path]) -> Path:
    report = build_release_candidate_report(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        paper_url=PAPER_URL,
        commit_sha="abcdef1234567890",
        generated_at="2026-06-01T12:00:00Z",
        public_link_probe=_ok_public_link,
        public_artifact_probe=_ok_public_artifacts,
    )
    path = tmp_path / "release_candidate_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _ok_public_link(name: str, url: str, timeout_seconds: float) -> PublicLinkCheck:
    assert timeout_seconds > 0
    return PublicLinkCheck(name=name, url=url, ok=True, status_code=200)


def _ok_public_artifacts(
    name: str,
    url: str,
    expected_files: tuple[Any, ...],
    timeout_seconds: float,
) -> PublicArtifactCheck:
    assert timeout_seconds > 0
    return PublicArtifactCheck(
        name=name,
        url=url,
        ok=True,
        expected_count=len(expected_files),
        observed_count=len(expected_files),
        verified_count=len(expected_files),
    )


def _write_replay_artifacts(request: clean_machine_demo.DemoRequest) -> Path:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    transcript = request.output_dir / "terminal-demo-transcript.md"
    transcript.write_text("# replay\n\n- Status: passed\n", encoding="utf-8")
    (request.output_dir / "scores.jsonl").write_text('{"sigma_raw":0.1}\n', encoding="utf-8")
    (request.output_dir / "receipts.jsonl").write_text(
        '{"schema_version":"1.0.0"}\n',
        encoding="utf-8",
    )
    _write_runtime_preflight_report(request)
    _write_batch_receipt_report(request)
    return transcript


def _write_replay_manifest(
    request: clean_machine_demo.DemoRequest,
    *,
    model_id: str | None = None,
    input_overrides: dict[str, dict[str, object]] | None = None,
    artifact_overrides: dict[str, dict[str, object]] | None = None,
    include_runtime_preflight: bool = True,
    runtime_preflight_overrides: dict[str, object] | None = None,
    include_score_receipt_batch: bool = True,
    score_receipt_overrides: dict[str, object] | None = None,
) -> None:
    input_overrides = input_overrides or {}
    artifact_overrides = artifact_overrides or {}
    model_manifest = load_manifest(request.model_dir / "manifest.json")
    vcf_identity = _identity(request.vcf, reported_path="demo/input.vcf")
    vcf_identity.update(input_overrides.get("vcf", {}))
    fasta_identity = _identity(request.fasta, reported_path="demo/ref.fa")
    fasta_identity.update(input_overrides.get("fasta", {}))
    payload = {
        "schema_version": "1.0.0",
        "generated_by": DEMO_GENERATED_BY,
        "generated_at": "2026-06-01T12:00:00Z",
        "status": "passed",
        "model": {
            "model_id": model_id or model_manifest.model_id(),
        },
        "inputs": {
            "model_manifest": _identity(
                request.model_dir / "manifest.json",
                reported_path="model/manifest.json",
            ),
            "vcf": vcf_identity,
            "fasta": fasta_identity,
        },
        "artifacts": [
            _artifact_identity(
                "terminal transcript",
                request.output_dir / "terminal-demo-transcript.md",
                artifact_overrides,
            ),
            _artifact_identity("scores", request.output_dir / "scores.jsonl", artifact_overrides),
            _artifact_identity(
                "receipts",
                request.output_dir / "receipts.jsonl",
                artifact_overrides,
            ),
            _artifact_identity(
                "runtime preflight report",
                request.output_dir / "runtime_preflight_report.json",
                artifact_overrides,
            ),
            _artifact_identity(
                "batch receipt report",
                request.output_dir / "batch_receipt_report.json",
                artifact_overrides,
            ),
        ],
    }
    if include_runtime_preflight:
        summary = _runtime_preflight_summary(request.output_dir / "runtime_preflight_report.json")
        if runtime_preflight_overrides:
            summary.update(runtime_preflight_overrides)
        payload["runtime_preflight"] = summary
    if include_score_receipt_batch:
        summary = _score_receipt_summary(request.output_dir / "batch_receipt_report.json")
        if score_receipt_overrides:
            summary.update(score_receipt_overrides)
        payload["score_receipt_batch"] = summary
    (request.output_dir / DEMO_MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_runtime_preflight_report(request: clean_machine_demo.DemoRequest) -> None:
    model_manifest = load_manifest(request.model_dir / "manifest.json")
    command = [
        "geno-lewm-score",
        "--quiet",
        "--no-banner",
        "--model-dir",
        "model",
        "--backend",
        request.backend,
        "--vcf",
        "demo/input.vcf",
        "--fasta",
        "demo/ref.fa",
        "--output",
        "replay/scores.jsonl",
        "--receipt",
        "replay/receipts.jsonl",
        "--batch-size",
        str(request.batch_size),
        "--no-progress",
    ]
    (request.output_dir / "runtime_preflight_report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_by": "tools.release.runtime_preflight",
                "ok": True,
                "model_id": model_manifest.model_id(),
                "release_id": model_manifest.release_id,
                "requested_backend": request.backend,
                "selected_backend": request.backend,
                "requirements": {
                    "native_runtime": request.require_native_runtime,
                    "carbon_cache": False,
                    "fixture_manifest_allowed": False,
                },
                "command": {
                    "argv": command,
                    "shell": " ".join(command),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_batch_receipt_report(request: clean_machine_demo.DemoRequest) -> None:
    model_manifest = load_manifest(request.model_dir / "manifest.json")
    (request.output_dir / "batch_receipt_report.json").write_text(
        json.dumps(
            {
                "records": 1,
                "model_id": model_manifest.model_id(),
                "calibration_hash": model_manifest.calibration.hash,
                "receipt_schema_version": "1.0.0",
                "receipt_stream": "jsonl_per_scored_alternate_v1",
                "checked_score_fields": [
                    "sigma_raw",
                    "sigma_calibrated",
                    "bucket_id",
                    "confidence",
                    "low_confidence",
                ],
                "runtime": {"backend": request.backend, "device": "CPU"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _runtime_preflight_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    command = payload["command"]
    requirements = payload["requirements"]
    assert isinstance(command, dict)
    assert isinstance(requirements, dict)
    return {
        "schema_version": payload["schema_version"],
        "generated_by": payload["generated_by"],
        "ok": payload["ok"],
        "model_id": payload["model_id"],
        "release_id": payload["release_id"],
        "requested_backend": payload["requested_backend"],
        "selected_backend": payload["selected_backend"],
        "requirements": requirements,
        "command": {
            "argv": command["argv"],
            "shell": command["shell"],
        },
    }


def _score_receipt_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "records",
        "model_id",
        "calibration_hash",
        "receipt_schema_version",
        "receipt_stream",
        "checked_score_fields",
        "runtime",
    )
    return {field: payload[field] for field in required}


def _artifact_identity(
    label: str,
    path: Path,
    overrides: dict[str, dict[str, object]],
) -> dict[str, object]:
    payload = {
        "label": label,
        **_identity(path, reported_path=f"replay/{path.name}"),
    }
    payload.update(overrides.get(label, {}))
    return payload


def _identity(path: Path, *, reported_path: str) -> dict[str, object]:
    return {
        "path": reported_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _fake_urlopen(
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
    *,
    corrupt_model_card: bool = False,
    expected_hf_token: str | None = None,
    expected_github_token: str | None = None,
) -> Any:
    def urlopen(request: Any, timeout: float) -> _Response:
        assert timeout > 0
        url = request.full_url
        if url.startswith("https://api.github.com/repos/"):
            _assert_authorization(request, expected_github_token)
            assets = [
                {
                    "name": path.name,
                    "browser_download_url": f"https://downloads.example/demo/{path.name}",
                }
                for path in sorted(demo_dir.iterdir())
                if path.is_file()
            ]
            return _Response(json.dumps({"assets": assets}).encode("utf-8"))
        if url.startswith("https://huggingface.co/datasets/") and "/resolve/main/" in url:
            _assert_authorization(request, expected_hf_token)
            relative = url.split("/resolve/main/", maxsplit=1)[1]
            return _Response((dataset_dir / relative).read_bytes())
        if "/resolve/main/" in url:
            _assert_authorization(request, expected_hf_token)
            relative = url.split("/resolve/main/", maxsplit=1)[1]
            path = model_dir / relative
            data = path.read_bytes()
            if corrupt_model_card and relative == "model_card.md":
                data = b"corrupt"
            return _Response(data)
        if url.startswith("https://downloads.example/demo/"):
            _assert_authorization(request, None)
            name = url.rsplit("/", maxsplit=1)[1]
            return _Response((demo_dir / name).read_bytes())
        raise AssertionError(f"unexpected URL: {url}")

    return urlopen


def _assert_authorization(request: Any, expected_token: str | None) -> None:
    expected = None if expected_token is None else f"Bearer {expected_token}"
    assert request.get_header("Authorization") == expected


class _Response:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.status = 200

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._data
