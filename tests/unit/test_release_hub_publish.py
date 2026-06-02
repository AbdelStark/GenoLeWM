"""Tests for the credentialed Hub publication helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from geno_lewm.errors import InputError
from tests.unit.test_release_hub_release import PAPER_URL, _write_release_candidate
from tools.release import hub_publish
from tools.release.hub_publish import publish_hub_release


def test_publish_hub_release_runs_uploads_then_writes_final_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_release_candidate(tmp_path)
    commands: list[tuple[str, ...]] = []
    captured_report_args: dict[str, Any] = {}
    plan_output = tmp_path / "hub_release_plan.json"
    candidate_output = tmp_path / "release_candidate_report.json"
    publish_output = tmp_path / "hub_publish_report.json"

    def write_ready_candidate(**kwargs: Any) -> _ReadyCandidate:
        captured_report_args.update(kwargs)
        kwargs["output"].write_text('{"ready": true}\n', encoding="utf-8")
        return _ReadyCandidate()

    monkeypatch.setattr(hub_publish, "write_release_candidate_report", write_ready_candidate)

    report = publish_hub_release(
        model_dir=paths["model_dir"],
        dataset_dir=paths["dataset_dir"],
        demo_dir=paths["demo_dir"],
        paper_path=paths["paper_path"],
        repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        paper_url=PAPER_URL,
        commit_sha="abcdef1234567890",
        plan_output=plan_output,
        candidate_output=candidate_output,
        publish_output=publish_output,
        command_runner=lambda argv: commands.append(tuple(argv)),
        environ={"HF_TOKEN": "hf-token", "GH_TOKEN": "gh-token"},
        generated_at="2026-06-01T12:00:00Z",
    )

    assert report.final_candidate_ready is True
    model_commands = [
        command
        for command in commands
        if command[:3]
        == (
            "huggingface-cli",
            "upload",
            "AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        )
    ]
    dataset_commands = [
        command
        for command in commands
        if command[:3]
        == (
            "huggingface-cli",
            "upload",
            "AbdelStark/geno-lewm-data-v0.1.0-r1",
        )
    ]
    demo_commands = [
        command
        for command in commands
        if command[:4] == ("gh", "release", "upload", "demo-v0.1.0")
        and str(paths["paper_path"]) not in command
    ]
    paper_commands = [
        command
        for command in commands
        if command[:4] == ("gh", "release", "upload", "demo-v0.1.0")
        and str(paths["paper_path"]) in command
    ]
    assert len(model_commands) == len(report.plan.files)
    assert len(dataset_commands) == len(report.plan.dataset_files)
    assert len(demo_commands) == 1
    assert len(paper_commands) == 1
    assert all(command[3] != str(paths["model_dir"]) for command in model_commands)
    assert all(command[4] != "." for command in model_commands)
    assert all(command[3] != str(paths["dataset_dir"]) for command in dataset_commands)
    assert all(command[4] != "." for command in dataset_commands)
    assert {
        (Path(command[3]).name if "/" not in command[4] else command[4])
        for command in model_commands
    } >= {
        "manifest.json",
        "model_card.md",
        "eval/scores.jsonl",
        "eval_metrics.json",
        "SHA256SUMS",
    }
    assert {
        (Path(command[3]).name if "/" not in command[4] else command[4])
        for command in dataset_commands
    } >= {
        "dataset_manifest.json",
        "carbon/windows.jsonl",
        "SHA256SUMS",
    }
    assert demo_commands[0][:4] == ("gh", "release", "upload", "demo-v0.1.0")
    assert "--repo" in demo_commands[0]
    assert "AbdelStark/GenoLeWM" in demo_commands[0]
    assert paper_commands[0] == (
        "gh",
        "release",
        "upload",
        "demo-v0.1.0",
        str(paths["paper_path"]),
        "--repo",
        "AbdelStark/GenoLeWM",
        "--clobber",
    )
    assert all(
        (
            "huggingface-cli",
            "upload",
            str(paths["model_dir"] / file.source),
            file.destination,
        )
        == (command[0], command[1], command[3], command[4])
        for file, command in zip(report.plan.files, model_commands, strict=True)
    )
    assert all(
        (
            "huggingface-cli",
            "upload",
            str(paths["dataset_dir"] / file.source),
            file.destination,
        )
        == (command[0], command[1], command[3], command[4])
        for file, command in zip(report.plan.dataset_files, dataset_commands, strict=True)
    )
    assert captured_report_args["output"] == candidate_output
    assert captured_report_args["paper_url"] == PAPER_URL
    assert captured_report_args["generated_at"] == "2026-06-01T12:00:00Z"
    assert plan_output.is_file()
    assert candidate_output.is_file()
    plan_payload = json.loads(plan_output.read_text(encoding="utf-8"))
    assert plan_payload["schema_version"] == "1.0.0"
    assert plan_payload["generated_by"] == "tools.release.hub_release"
    assert plan_payload["generated_at"] == "2026-06-01T12:00:00Z"
    assert str(tmp_path) not in json.dumps(plan_payload)
    assert plan_payload["paper_file"]["source"] == "paper.md"
    assert plan_payload["paper_file"]["destination"] == "paper.md"
    assert plan_payload["paper_file"]["sha256"].startswith("sha256:")
    payload = json.loads(publish_output.read_text(encoding="utf-8"))
    assert str(tmp_path) not in json.dumps(payload)
    assert payload["generated_at"] == "2026-06-01T12:00:00Z"
    assert payload["plan"]["generated_at"] == "2026-06-01T12:00:00Z"
    assert payload["final_candidate_ready"] is True
    assert [command["name"] for command in payload["commands"]].count("model") == len(
        report.plan.files
    )
    assert [command["name"] for command in payload["commands"]].count("dataset") == len(
        report.plan.dataset_files
    )
    assert payload["commands"][-2]["name"] == "demo"
    assert payload["commands"][-1]["name"] == "paper"
    assert payload["commands"][-1]["argv"] == [
        "gh",
        "release",
        "upload",
        "demo-v0.1.0",
        "paper.md",
        "--repo",
        "AbdelStark/GenoLeWM",
        "--clobber",
    ]


def test_publish_hub_release_requires_hub_credentials(tmp_path: Path) -> None:
    paths = _write_release_candidate(tmp_path)

    with pytest.raises(InputError, match="HF_TOKEN is required"):
        publish_hub_release(
            model_dir=paths["model_dir"],
            dataset_dir=paths["dataset_dir"],
            demo_dir=paths["demo_dir"],
            repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            commit_sha="abcdef1234567890",
            command_runner=lambda _argv: None,
            environ={"GH_TOKEN": "gh-token"},
        )


def test_publish_hub_release_requires_supported_public_targets(tmp_path: Path) -> None:
    paths = _write_release_candidate(tmp_path)

    with pytest.raises(InputError, match="dataset_url must be a Hugging Face dataset URL"):
        publish_hub_release(
            model_dir=paths["model_dir"],
            dataset_dir=paths["dataset_dir"],
            demo_dir=paths["demo_dir"],
            repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            dataset_url="https://example.com/dataset.tar.gz",
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            commit_sha="abcdef1234567890",
            command_runner=lambda _argv: None,
            environ={"HF_TOKEN": "hf-token", "GH_TOKEN": "gh-token"},
        )


def test_publish_hub_release_rejects_unsupported_paper_publication_target(
    tmp_path: Path,
) -> None:
    paths = _write_release_candidate(tmp_path)

    with pytest.raises(InputError, match="paper_url must be a GitHub release download URL"):
        publish_hub_release(
            model_dir=paths["model_dir"],
            dataset_dir=paths["dataset_dir"],
            demo_dir=paths["demo_dir"],
            paper_path=paths["paper_path"],
            repo_id="AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
            dataset_url="https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
            demo_url="https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
            paper_url="https://arxiv.org/abs/2606.00001",
            commit_sha="abcdef1234567890",
            command_runner=lambda _argv: None,
            environ={"HF_TOKEN": "hf-token", "GH_TOKEN": "gh-token"},
        )


class _ReadyCandidate:
    ready = True

    def to_dict(self) -> dict[str, object]:
        return {"ready": True}
