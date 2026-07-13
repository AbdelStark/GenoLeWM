"""Tests for the immutable correction-control job preflight."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.research.correction_control_preflight import (
    EXPECTED_CARBON_CONFIG,
    EXPECTED_CARBON_MODEL_DIR,
    EXPECTED_CARBON_SOURCE,
    EXPECTED_CLINVAR_LINES,
    EXPECTED_CLINVAR_URL,
    EXPECTED_CONFIG_PATH,
    EXPECTED_CONTAINER_IMAGE,
    EXPECTED_CORPUS_REVISION,
    EXPECTED_GNOMAD_LINES,
    EXPECTED_GNOMAD_URL,
    EXPECTED_HOLDOUT_CHROM,
    EXPECTED_MAX_WINDOWS,
    EXPECTED_SNAPSHOT_PATH,
    EXPECTED_STEPS,
    EXPECTED_TUPLE_THROUGHPUT_SAMPLES,
    EXPECTED_WINDOW_BP,
    CorrectionControlPreflightReport,
    CorrectionControlRequest,
    build_correction_control_preflight_report,
    main,
)


def test_correction_control_preflight_accepts_exact_contract(tmp_path: Path) -> None:
    request = _write_repo_and_request(tmp_path)

    report = build_correction_control_preflight_report(
        request,
        generated_at="2026-07-10T12:00:00Z",
    )

    assert report.ok is True
    assert report.issues == ()
    assert report.repository["observed_commit_sha"] == request.expected_commit_sha
    assert report.config["run_id"] == "correction-control-l2-p1-smoke-v1"
    assert report.snapshot["snapshot_id"] == ("geno-lewm-data-correction-control-l2-p1-proof-v1")
    assert str(tmp_path) not in json.dumps(report.to_dict(), sort_keys=True)
    claim_boundary = report.to_dict()["claim_boundary"]
    assert isinstance(claim_boundary, str)
    assert claim_boundary.startswith("This preflight validates launch identity")


def test_correction_control_preflight_cli_emits_only_json_and_writes_only_on_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = _write_repo_and_request(tmp_path)

    rc = main(_argv(request))
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert json.loads(captured.out)["ok"] is True
    assert list(tmp_path.rglob("*preflight*.json")) == []

    output = tmp_path / "reports" / "job_contract_preflight.json"
    rc = main([*_argv(request), "--output", str(output)])
    captured = capsys.readouterr()

    assert rc == 0
    assert json.loads(captured.out)["ok"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def test_correction_control_preflight_rejects_every_job_knob_and_source_drift(
    tmp_path: Path,
) -> None:
    request = _write_repo_and_request(tmp_path)
    cases: tuple[tuple[str, object, str], ...] = (
        ("steps", 51, "request.steps_mismatch"),
        ("max_windows", 513, "request.max_windows_mismatch"),
        ("clinvar_lines", 60_001, "request.clinvar_lines_mismatch"),
        ("gnomad_lines", 60_001, "request.gnomad_lines_mismatch"),
        (
            "tuple_throughput_samples",
            401,
            "request.tuple_throughput_samples_mismatch",
        ),
        ("window_bp", 12_288, "request.window_bp_mismatch"),
        ("holdout_chrom", 21, "request.holdout_chrom_mismatch"),
        ("carbon_model_dir", "/tmp/carbon", "request.carbon_model_dir_mismatch"),
        ("carbon_config", "default", "request.carbon_config_mismatch"),
        ("carbon_source", "other", "request.carbon_source_mismatch"),
        ("corpus_revision", "main", "request.corpus_revision_mismatch"),
        ("container_image", "ghcr.io/astral-sh/uv:latest", "request.container_image_mismatch"),
        ("clinvar_url", "https://example.test/latest", "request.clinvar_url_mismatch"),
        ("gnomad_url", EXPECTED_GNOMAD_URL.split("?", 1)[0], "request.gnomad_url_mismatch"),
    )

    for field, value, code in cases:
        changed_request = replace(request, **{field: value})  # type: ignore[arg-type]
        report = build_correction_control_preflight_report(changed_request)
        assert report.ok is False
        assert code in _codes(report), field


def test_correction_control_preflight_rejects_legacy_or_unpinned_commit_identity(
    tmp_path: Path,
) -> None:
    request = _write_repo_and_request(tmp_path)

    legacy = build_correction_control_preflight_report(replace(request, run_name="geno-lewm-proof"))
    malformed = build_correction_control_preflight_report(
        replace(request, expected_commit_sha=request.expected_commit_sha.upper())
    )
    invalid_attempt = build_correction_control_preflight_report(replace(request, run_attempt=0))

    assert "request.run_name_mismatch" in _codes(legacy)
    assert {
        "request.commit_sha_format",
        "repository.head_mismatch",
        "request.run_name_mismatch",
    }.issubset(_codes(malformed))
    assert {
        "request.run_attempt_invalid",
        "request.run_name_mismatch",
    }.issubset(_codes(invalid_attempt))


def test_correction_control_preflight_accepts_distinct_positive_retry_attempt(
    tmp_path: Path,
) -> None:
    request = _write_repo_and_request(tmp_path)
    retry = replace(
        request,
        run_attempt=2,
        run_name=f"geno-lewm-l2-p1-smoke-{request.expected_commit_sha[:12]}-50-r2",
    )

    report = build_correction_control_preflight_report(retry)

    assert report.ok is True
    assert report.job["run_attempt"] == 2


def test_correction_control_preflight_rejects_dirty_worktree(tmp_path: Path) -> None:
    request = _write_repo_and_request(tmp_path)
    (request.repo_root / "uncommitted.txt").write_text("drift\n", encoding="utf-8")

    report = build_correction_control_preflight_report(request)

    assert report.repository["worktree_clean"] is False
    assert report.repository["dirty_paths"] == ["uncommitted.txt"]
    assert "repository.worktree_dirty" in _codes(report)


def test_correction_control_preflight_rejects_config_path_and_all_pinned_fields(
    tmp_path: Path,
) -> None:
    request = _write_repo_and_request(tmp_path)
    payload = _load_yaml(request.config_path)
    cases: tuple[tuple[tuple[str, ...], object], ...] = (
        (("run_id",), "first-snv-carbon-500m-r1"),
        (("seed",), 0),
        (("phase",), "phase2"),
        (("deterministic",), False),
        (("schema_version",), "1.0.0"),
        (("encoder", "model_id"), "HuggingFaceBio/Carbon-500M"),
        (("encoder", "revision"), "main"),
        (("encoder", "dtype"), "fp32"),
        (("encoder", "state_layer"), 19),
        (("encoder", "pool_type"), "mean"),
        (("encoder", "pool_radius"), 7),
        (("encoder", "normalize"), False),
        (("encoder", "state_contract_version"), "legacy_raw_v1"),
        (("encoder", "trust_remote_code"), True),
        (("predictor", "architecture"), "other"),
        (("predictor", "n_layers"), 5),
        (("predictor", "n_heads"), 4),
        (("predictor", "d_state"), 512),
        (("predictor", "d_action"), 32),
        (("predictor", "dtype"), "bf16"),
        (("action", "d_action"), 32),
        (("action", "max_len"), 8),
        (("action", "sub_encoders"), ["snv", "ins"]),
        (("training", "max_steps"), 51),
        (("training", "collapse_log_every_steps"), 50),
        (("optimizer", "name"), "sgd-momentum"),
        (("optimizer", "lr"), 1.0e-3),
        (("optimizer", "beta1"), 0.8),
        (("optimizer", "beta2"), 0.9),
        (("optimizer", "weight_decay"), 0.0),
        (("optimizer", "grad_clip"), 0.5),
        (("optimizer", "warmup_steps"), 11),
        (("optimizer", "schedule"), "cosine"),
        (("data", "corpus_id"), "other/corpus"),
        (("data", "corpus_revision"), "main"),
        (("data", "batch_size"), 16),
        (("data", "num_workers"), 1),
        (("data", "shuffle_buffer"), 512),
        (("runtime", "backend"), "onnx"),
        (("runtime", "device"), "cpu"),
    )

    for keys, value in cases:
        changed = copy.deepcopy(payload)
        _set_nested(changed, keys, value)
        _write_yaml(request.config_path, changed)
        report = build_correction_control_preflight_report(request)
        assert f"config.{'.'.join(keys)}_mismatch" in _codes(report), keys

    wrong_path = request.repo_root / "copy.yaml"
    _write_yaml(wrong_path, payload)
    report = build_correction_control_preflight_report(replace(request, config_path=wrong_path))
    assert "request.config_path_mismatch" in _codes(report)


def test_correction_control_preflight_rejects_snapshot_caps_integrity_and_sources(
    tmp_path: Path,
) -> None:
    request = _write_repo_and_request(tmp_path)
    payload = json.loads(request.snapshot_path.read_text(encoding="utf-8"))
    nested_cases: tuple[tuple[tuple[str, ...], object], ...] = (
        (("schema_version",), "2.0.0"),
        (("snapshot_id",), "geno-lewm-data-v0.1.0-r1"),
        (("clinvar", "release"), "latest"),
        (("clinvar", "max_allele_len"), 17),
        (("gnomad", "release"), "latest"),
        (("gnomad", "max_allele_len"), 17),
        (("gnomad", "min_af"), 0.02),
    )
    for keys, value in nested_cases:
        changed = copy.deepcopy(payload)
        _set_nested(changed, keys, value)
        request.snapshot_path.write_text(json.dumps(changed), encoding="utf-8")
        report = build_correction_control_preflight_report(request)
        assert f"snapshot.{'.'.join(keys)}_mismatch" in _codes(report), keys

    changed = copy.deepcopy(payload)
    changed["source_integrity"]["gnomad_generation"] = "latest"
    request.snapshot_path.write_text(json.dumps(changed), encoding="utf-8")
    report = build_correction_control_preflight_report(request)
    assert "snapshot.source_integrity_mismatch" in _codes(report)

    for index, field in ((0, "revision"), (1, "url"), (2, "url")):
        changed = copy.deepcopy(payload)
        changed["sources"][index][field] = "mutable"
        request.snapshot_path.write_text(json.dumps(changed), encoding="utf-8")
        report = build_correction_control_preflight_report(request)
        source_name = changed["sources"][index]["name"]
        expected_code = f"snapshot.sources.{_slug(source_name)}.{field}_mismatch"
        assert expected_code in _codes(report)

    wrong_path = request.repo_root / "copy.json"
    wrong_path.write_text(json.dumps(payload), encoding="utf-8")
    report = build_correction_control_preflight_report(replace(request, snapshot_path=wrong_path))
    assert "request.snapshot_path_mismatch" in _codes(report)


def test_correction_control_preflight_fails_closed_on_invalid_documents(
    tmp_path: Path,
) -> None:
    request = _write_repo_and_request(tmp_path)
    request.config_path.write_text("encoder: [", encoding="utf-8")
    request.snapshot_path.write_text("[]\n", encoding="utf-8")

    report = build_correction_control_preflight_report(request)

    assert report.ok is False
    assert {"config.parse_failed", "snapshot.root_invalid"}.issubset(_codes(report))
    json.dumps(report.to_dict())


def _write_repo_and_request(root: Path) -> CorrectionControlRequest:
    source_root = Path(__file__).resolve().parents[2]
    config = root / EXPECTED_CONFIG_PATH
    snapshot = root / EXPECTED_SNAPSHOT_PATH
    config.parent.mkdir(parents=True)
    shutil.copy2(source_root / EXPECTED_CONFIG_PATH, config)
    shutil.copy2(source_root / EXPECTED_SNAPSHOT_PATH, snapshot)
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "GenoLeWM Test")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    head = _git(root, "rev-parse", "HEAD")
    return CorrectionControlRequest(
        repo_root=root,
        config_path=config,
        snapshot_path=snapshot,
        expected_commit_sha=head,
        run_name=f"geno-lewm-l2-p1-smoke-{head[:12]}-50-r1",
        run_attempt=1,
        steps=EXPECTED_STEPS,
        max_windows=EXPECTED_MAX_WINDOWS,
        clinvar_lines=EXPECTED_CLINVAR_LINES,
        gnomad_lines=EXPECTED_GNOMAD_LINES,
        tuple_throughput_samples=EXPECTED_TUPLE_THROUGHPUT_SAMPLES,
        window_bp=EXPECTED_WINDOW_BP,
        holdout_chrom=EXPECTED_HOLDOUT_CHROM,
        carbon_model_dir=EXPECTED_CARBON_MODEL_DIR,
        carbon_config=EXPECTED_CARBON_CONFIG,
        carbon_source=EXPECTED_CARBON_SOURCE,
        corpus_revision=EXPECTED_CORPUS_REVISION,
        container_image=EXPECTED_CONTAINER_IMAGE,
        clinvar_url=EXPECTED_CLINVAR_URL,
        gnomad_url=EXPECTED_GNOMAD_URL,
    )


def _argv(request: CorrectionControlRequest) -> list[str]:
    return [
        "--repo-root",
        str(request.repo_root),
        "--config",
        str(request.config_path),
        "--snapshot",
        str(request.snapshot_path),
        "--expected-commit-sha",
        request.expected_commit_sha,
        "--run-name",
        request.run_name,
        "--run-attempt",
        str(request.run_attempt),
        "--steps",
        str(request.steps),
        "--max-windows",
        str(request.max_windows),
        "--clinvar-lines",
        str(request.clinvar_lines),
        "--gnomad-lines",
        str(request.gnomad_lines),
        "--tuple-throughput-samples",
        str(request.tuple_throughput_samples),
        "--window-bp",
        str(request.window_bp),
        "--holdout-chrom",
        str(request.holdout_chrom),
        "--carbon-model-dir",
        request.carbon_model_dir,
        "--carbon-config",
        request.carbon_config,
        "--carbon-source",
        request.carbon_source,
        "--corpus-revision",
        request.corpus_revision,
        "--container-image",
        request.container_image,
        "--clinvar-url",
        request.clinvar_url,
        "--gnomad-url",
        request.gnomad_url,
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _set_nested(payload: dict[str, Any], keys: tuple[str, ...], value: object) -> None:
    target = payload
    for key in keys[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[keys[-1]] = value


def _codes(report: CorrectionControlPreflightReport) -> set[str]:
    return {issue.code for issue in report.issues}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _slug(value: str) -> str:
    return "_".join(value.lower().split())
