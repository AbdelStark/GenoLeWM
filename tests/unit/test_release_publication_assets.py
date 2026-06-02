"""Tests for final publication evidence asset manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file
from tools.release.publication_assets import (
    GENERATED_BY,
    SCHEMA_VERSION,
    build_publication_asset_report,
    main,
    write_publication_asset_report,
)


def test_publication_asset_report_binds_release_target_and_asset_identities(
    tmp_path: Path,
) -> None:
    paths = _write_publication_assets(tmp_path)

    report = build_publication_asset_report(
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        hub_release_plan_path=paths["hub_release_plan"],
        release_candidate_path=paths["release_candidate"],
        publish_report_path=paths["publish_report"],
        clean_machine_report_path=paths["clean_machine_report"],
        publication_report_path=paths["publication_report"],
        replay_dir=paths["replay_dir"],
        generated_at="2026-06-01T12:00:00Z",
    )

    payload = report.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["generated_by"] == GENERATED_BY
    assert payload["generated_at"] == "2026-06-01T12:00:00Z"
    assert payload["target"] == {
        "repo": "AbdelStark/GenoLeWM",
        "tag": "demo-v0.1.0",
        "url": "https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
    }
    assert str(tmp_path) not in json.dumps(payload)
    assets = {asset["label"]: asset for asset in payload["assets"]}
    assert set(assets) == {
        "hub_release_plan",
        "release_candidate",
        "hub_publish",
        "clean_machine_demo",
        "terminal_transcript",
        "terminal_demo_manifest",
        "scores_jsonl",
        "receipts_jsonl",
        "runtime_preflight",
        "batch_receipt_report",
        "publication_evidence",
    }
    assert assets["publication_evidence"]["destination"] == "publication_evidence_report.json"
    assert assets["publication_evidence"]["sha256"] == sha256_file(paths["publication_report"])
    assert payload["upload_command"][:4] == [
        "gh",
        "release",
        "upload",
        "demo-v0.1.0",
    ]
    assert "publication_evidence_assets.json" in payload["upload_command"]
    assert payload["upload_command"][-3:] == [
        "--repo",
        "AbdelStark/GenoLeWM",
        "--clobber",
    ]


def test_write_publication_asset_report_writes_target_env(tmp_path: Path) -> None:
    paths = _write_publication_assets(tmp_path)
    output = tmp_path / "publication_evidence_assets.json"
    env_output = tmp_path / "publication-evidence-target.env"

    report = write_publication_asset_report(
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        hub_release_plan_path=paths["hub_release_plan"],
        release_candidate_path=paths["release_candidate"],
        publish_report_path=paths["publish_report"],
        clean_machine_report_path=paths["clean_machine_report"],
        publication_report_path=paths["publication_report"],
        replay_dir=paths["replay_dir"],
        output=output,
        env_output=env_output,
        generated_at="2026-06-01T12:00:00Z",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == report.to_dict()
    assert "publication_evidence_assets.json" in report.upload_command
    assert env_output.read_text(encoding="utf-8") == (
        "DEMO_REPO=AbdelStark/GenoLeWM\nDEMO_TAG=demo-v0.1.0\n"
    )


def test_write_publication_asset_report_uses_custom_output_in_upload_command(
    tmp_path: Path,
) -> None:
    paths = _write_publication_assets(tmp_path)
    output = tmp_path / "final-publication-assets.json"

    report = write_publication_asset_report(
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        hub_release_plan_path=paths["hub_release_plan"],
        release_candidate_path=paths["release_candidate"],
        publish_report_path=paths["publish_report"],
        clean_machine_report_path=paths["clean_machine_report"],
        publication_report_path=paths["publication_report"],
        replay_dir=paths["replay_dir"],
        output=output,
        generated_at="2026-06-01T12:00:00Z",
    )

    assert "final-publication-assets.json" in report.upload_command


def test_publication_asset_report_rejects_unsupported_demo_url(tmp_path: Path) -> None:
    paths = _write_publication_assets(tmp_path)

    with pytest.raises(InputError, match="demo_url must be a GitHub release tag URL"):
        build_publication_asset_report(
            demo_url="https://example.test/demo-v0.1.0",
            hub_release_plan_path=paths["hub_release_plan"],
            release_candidate_path=paths["release_candidate"],
            publish_report_path=paths["publish_report"],
            clean_machine_report_path=paths["clean_machine_report"],
            publication_report_path=paths["publication_report"],
            replay_dir=paths["replay_dir"],
        )


def test_publication_asset_report_rejects_missing_asset(tmp_path: Path) -> None:
    paths = _write_publication_assets(tmp_path)
    paths["publication_report"].unlink()

    with pytest.raises(InputError, match="publication evidence asset is missing"):
        build_publication_asset_report(
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            hub_release_plan_path=paths["hub_release_plan"],
            release_candidate_path=paths["release_candidate"],
            publish_report_path=paths["publish_report"],
            clean_machine_report_path=paths["clean_machine_report"],
            publication_report_path=paths["publication_report"],
            replay_dir=paths["replay_dir"],
        )


def test_publication_asset_report_rejects_duplicate_release_asset_names(
    tmp_path: Path,
) -> None:
    paths = _write_publication_assets(tmp_path)

    with pytest.raises(InputError, match="destinations must be unique"):
        build_publication_asset_report(
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            hub_release_plan_path=paths["hub_release_plan"],
            release_candidate_path=paths["release_candidate"],
            publish_report_path=paths["publish_report"],
            clean_machine_report_path=paths["clean_machine_report"],
            publication_report_path=paths["hub_release_plan"],
            replay_dir=paths["replay_dir"],
        )


def test_publication_assets_main_writes_report_and_env(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_publication_assets(tmp_path)
    output = tmp_path / "publication_evidence_assets.json"
    env_output = tmp_path / "publication-evidence-target.env"

    rc = main(
        [
            "--demo-url",
            "https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            "--hub-release-plan",
            str(paths["hub_release_plan"]),
            "--release-candidate",
            str(paths["release_candidate"]),
            "--publish-report",
            str(paths["publish_report"]),
            "--clean-machine-demo-report",
            str(paths["clean_machine_report"]),
            "--publication-report",
            str(paths["publication_report"]),
            "--replay-dir",
            str(paths["replay_dir"]),
            "--output",
            str(output),
            "--env-output",
            str(env_output),
        ]
    )

    assert rc == 0
    assert f"wrote {output}" in capsys.readouterr().out
    assert output.is_file()
    assert env_output.is_file()


def _write_publication_assets(tmp_path: Path) -> dict[str, Path]:
    replay_dir = tmp_path / "clean-machine-public-replay" / "replay"
    paths = {
        "hub_release_plan": tmp_path / "hub_release_plan.json",
        "release_candidate": tmp_path / "release_candidate_report.json",
        "publish_report": tmp_path / "hub_publish_report.json",
        "clean_machine_report": tmp_path
        / "clean-machine-public-replay"
        / "clean_machine_demo_report.json",
        "publication_report": tmp_path / "publication_evidence_report.json",
        "replay_dir": replay_dir,
    }
    for label, path in paths.items():
        if label == "replay_dir":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{label}\n", encoding="utf-8")
    for name in (
        "terminal-demo-transcript.md",
        "terminal_demo_manifest.json",
        "scores.jsonl",
        "receipts.jsonl",
        "runtime_preflight_report.json",
        "batch_receipt_report.json",
    ):
        path = replay_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    return paths
