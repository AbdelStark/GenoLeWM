"""Tests for final publication evidence report binding."""

from __future__ import annotations

import json
from pathlib import Path

from geno_lewm.provenance import sha256_file
from tools.release.issue_refs import issue_ref_payload
from tools.release.publication_report import (
    CANDIDATE_ARTIFACT_UPLOADS,
    build_publication_evidence_report,
    main,
    write_publication_evidence_report,
)


def test_publication_evidence_report_binds_publish_and_replay_reports(tmp_path: Path) -> None:
    paths = _write_publication_inputs(tmp_path)
    output = tmp_path / "publication_evidence_report.json"

    report = write_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
        output=output,
        generated_at="2026-06-01T12:00:00Z",
    )

    assert report.ok is True
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert str(tmp_path) not in json.dumps(payload)
    assert {name: identity["path"] for name, identity in payload["source_reports"].items()} == {
        "hub_release_plan": "hub_release_plan.json",
        "release_candidate": "release_candidate_report.json",
        "hub_publish": "hub_publish_report.json",
        "clean_machine_demo": "clean_machine_demo_report.json",
    }
    assert payload["source_reports"]["release_candidate"]["sha256"] == sha256_file(
        paths["candidate"]
    )
    assert payload["release_candidate_artifacts"]["eval_config"]["path"] == (
        "model/eval_config.effective.yaml"
    )
    assert (
        payload["release_candidate_artifacts"]["eval_metrics"]["path"] == "model/eval_metrics.json"
    )
    assert payload["release_candidate_artifacts"]["paper"]["path"] == "paper.md"
    assert payload["release_candidate_artifacts"]["eval_config"]["sha256"] == next(
        artifact["sha256"]
        for artifact in payload["downloaded_artifacts"]
        if artifact["path"].endswith("model/eval_config.effective.yaml")
    )
    readiness = {item["code"]: item for item in payload["release_candidate_readiness"]}
    assert readiness["package_verifier"]["ok"] is True
    assert readiness["public_links"]["issue_refs"] == _issue_refs(163, 166, 167, 101)
    assert readiness["public_artifacts"]["issue_refs"] == _issue_refs(163, 166, 167, 101)
    assert payload["release_candidate_public_links"]["required"] is True
    assert {check["name"] for check in payload["release_candidate_public_links"]["checks"]} == {
        "model",
        "dataset",
        "demo",
        "paper",
    }
    assert all(check["ok"] for check in payload["release_candidate_public_links"]["checks"])
    assert payload["release_candidate_public_artifacts"]["required"] is True
    public_artifacts = {
        check["name"]: check for check in payload["release_candidate_public_artifacts"]["checks"]
    }
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    assert set(public_artifacts) == {"model", "dataset", "demo", "paper"}
    assert public_artifacts["model"]["verified_count"] == len(plan["files"])
    assert public_artifacts["dataset"]["verified_count"] == len(plan["dataset_files"])
    assert public_artifacts["demo"]["verified_count"] == len(plan["demo_files"])
    assert public_artifacts["paper"]["verified_count"] == 1
    assert all(not check["missing"] for check in public_artifacts.values())
    assert all(not check["hash_mismatches"] for check in public_artifacts.values())
    assert all(not check["size_mismatches"] for check in public_artifacts.values())
    assert all(not check["unexpected"] for check in public_artifacts.values())
    assert payload["downloaded_artifact_count"] == sum(
        len(plan[key]) for key in ("files", "dataset_files", "demo_files")
    )
    assert {artifact["group"] for artifact in payload["downloaded_artifacts"]} == {
        "model",
        "dataset",
        "demo",
    }
    assert payload["replay_artifact_count"] == 6
    assert payload["paper_artifact"] == {
        "destination": "paper.md",
        "sha256": sha256_file(paths["paper"]),
        "size_bytes": paths["paper"].stat().st_size,
        "source": "paper.md",
        "url": "https://arxiv.org/abs/2606.00001",
    }
    assert all(
        not Path(artifact["path"]).is_absolute()
        for artifact in (
            *payload["downloaded_artifacts"],
            *payload["replay_artifacts"],
        )
    )
    assert payload["issues"] == []


def test_publication_evidence_report_accepts_clean_machine_relative_paths(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    _make_clean_machine_paths_relative(
        clean_machine,
        root=paths["clean_machine"].parent,
        candidate=paths["candidate"],
    )
    paths["clean_machine"].write_text(
        json.dumps(clean_machine, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is True
    assert report.issues == ()


def test_publication_evidence_report_uses_exact_model_manifest_download(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))

    distractor = _downloaded_artifact(tmp_path / "downloads", "model", "dataset_manifest.json")
    distractor["source_url"] = (
        f"https://huggingface.co/{plan['repo_id']}/resolve/main/dataset_manifest.json"
    )
    plan["files"].insert(
        0,
        {
            "destination": "dataset_manifest.json",
            "sha256": distractor["sha256"],
            "size_bytes": distractor["size_bytes"],
            "source": "model/dataset_manifest.json",
        },
    )
    candidate["hub_plan"] = plan
    candidate["public_artifacts"] = _candidate_public_artifacts(plan)
    publish["plan"] = plan
    publish["final_candidate_report"] = candidate
    clean_machine["downloaded_artifacts"].insert(0, distractor)

    _write_json(paths["plan"], plan)
    _write_json(paths["candidate"], candidate)
    publish["final_candidate_report"] = candidate
    _write_json(paths["publish"], publish)
    clean_machine["release_candidate_report_identity"] = _report_identity(paths["candidate"])
    _write_json(paths["clean_machine"], clean_machine)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is True
    assert report.issues == ()


def test_publication_evidence_report_rejects_mismatched_publish_candidate(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    publish["final_candidate_report"] = {"ready": True, "model_id": "sha256:" + "f" * 64}
    paths["publish"].write_text(json.dumps(publish, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"publish.candidate_mismatch"}
    assert report.issues[0].issue_refs == (167, 101)
    assert report.issues[0].to_dict()["issue_refs"] == _issue_refs(167, 101)


def test_publication_evidence_report_rejects_clean_machine_candidate_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"]["sha256"] = "sha256:" + "0" * 64
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"clean_machine.candidate_hash_mismatch"}
    assert report.issues[0].issue_refs == (163, 166, 101)
    assert report.issues[0].to_dict()["issue_refs"] == _issue_refs(163, 166, 101)


def test_publication_evidence_report_rejects_clean_machine_candidate_identity_path_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"]["path"] = str(
        tmp_path / "other_candidate.json"
    )
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"clean_machine.candidate_identity_path"}


def test_publication_evidence_report_rejects_clean_machine_candidate_size_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"]["size_bytes"] += 1
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"clean_machine.candidate_size_mismatch"}


def test_publication_evidence_report_rejects_unexpected_source_report_generator(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    publish["generated_by"] = "manual-script"
    paths["publish"].write_text(json.dumps(publish, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"publish.generated_by"}


def test_publication_evidence_report_rejects_unexpected_plan_generator(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    plan["generated_by"] = "manual-plan"
    paths["plan"].write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["hub_plan"] = plan
    paths["candidate"].write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    publish["plan"] = plan
    publish["final_candidate_report"] = candidate
    paths["publish"].write_text(json.dumps(publish, indent=2, sort_keys=True) + "\n")
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"] = _report_identity(paths["candidate"])
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"plan.generated_by"}


def test_publication_evidence_report_rejects_plan_candidate_identity_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    plan["model_id"] = "sha256:" + "c" * 64
    paths["plan"].write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["hub_plan"] = plan
    paths["candidate"].write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    publish["plan"] = plan
    publish["final_candidate_report"] = candidate
    paths["publish"].write_text(json.dumps(publish, indent=2, sort_keys=True) + "\n")
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"] = _report_identity(paths["candidate"])
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"plan.model_id_mismatch"}


def test_publication_evidence_report_rejects_candidate_hub_plan_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["hub_plan"]["dataset_files"] = []
    paths["candidate"].write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    publish["final_candidate_report"] = candidate
    paths["publish"].write_text(json.dumps(publish, indent=2, sort_keys=True) + "\n")
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"] = _report_identity(paths["candidate"])
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"candidate.hub_plan_mismatch"}


def test_publication_evidence_report_requires_candidate_readiness(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate.pop("readiness")
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"candidate.readiness_missing"}
    assert report.issues[0].issue_refs == (163, 164, 165, 166, 167, 101)


def test_publication_evidence_report_rejects_nonpassing_candidate_readiness(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    terminal_demo = next(item for item in candidate["readiness"] if item["code"] == "terminal_demo")
    terminal_demo["ok"] = False
    terminal_demo["blockers"] = ["demo.clean_machine_replay_missing"]
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.readiness.terminal_demo.blockers_present",
        "candidate.readiness.terminal_demo.not_ok",
    }
    assert {issue.issue_refs for issue in report.issues} == {(163, 164, 165, 166, 167, 101)}


def test_publication_evidence_report_rejects_candidate_readiness_issue_ref_drift(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    dataset = next(item for item in candidate["readiness"] if item["code"] == "dataset_package")
    dataset["issue_refs"] = _issue_refs(101)
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.readiness.dataset_package.issue_refs"
    }
    assert report.issues[0].issue_refs == (163, 164, 165, 166, 167, 101)


def test_publication_evidence_report_rejects_candidate_blockers_when_ready(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["blockers"] = [
        {
            "code": "paper.url_missing",
            "path": "paper.md",
            "message": "paper URL is missing",
            "issue_refs": _issue_refs(167),
        }
    ]
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"candidate.blockers_not_empty"}
    assert report.issues[0].issue_refs == (163, 164, 165, 166, 167, 101)


def test_publication_evidence_report_requires_candidate_public_links(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate.pop("public_links")
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"candidate.public_links"}
    assert report.issues[0].issue_refs == (163, 166, 167, 101)


def test_publication_evidence_report_rejects_failed_candidate_public_link(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    dataset = next(
        item for item in candidate["public_links"]["checks"] if item["name"] == "dataset"
    )
    dataset["ok"] = False
    dataset["status_code"] = 404
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"candidate.public_links.dataset.not_ok"}
    assert report.issues[0].issue_refs == (163,)


def test_publication_evidence_report_requires_candidate_public_artifact_checks(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["public_artifacts"]["checks"] = [
        item for item in candidate["public_artifacts"]["checks"] if item["name"] != "demo"
    ]
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"candidate.public_artifacts.demo.missing"}
    assert report.issues[0].issue_refs == (166,)


def test_publication_evidence_report_rejects_incomplete_candidate_public_artifact_check(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    model = next(
        item for item in candidate["public_artifacts"]["checks"] if item["name"] == "model"
    )
    model["verified_count"] = model["expected_count"] - 1
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.public_artifacts.model.count_mismatch"
    }
    assert report.issues[0].issue_refs == (101,)


def test_publication_evidence_report_requires_candidate_eval_config_artifact(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["artifacts"].pop("eval_config")
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"candidate.artifacts.eval_config.missing"}
    assert report.issues[0].issue_refs == (165, 167, 101)


def test_publication_evidence_report_requires_candidate_eval_metrics_artifact(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["artifacts"].pop("eval_metrics")
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"candidate.artifacts.eval_metrics.missing"}
    assert report.issues[0].issue_refs == (165, 167, 101)


def test_publication_evidence_report_requires_candidate_paper_artifact_when_paper_url_set(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["artifacts"].pop("paper")
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"candidate.artifacts.paper.missing"}
    assert report.issues[0].issue_refs == (167, 101)


def test_publication_evidence_report_rejects_candidate_private_eval_config_path(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["artifacts"]["eval_config"]["path"] = str(tmp_path / "eval_config.yaml")
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.artifacts.eval_config.missing",
        "candidate.artifacts.eval_config.path",
    }


def test_publication_evidence_report_rejects_candidate_stale_eval_config_hash(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["artifacts"]["eval_config"]["sha256"] = "sha256:" + "b" * 64
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.artifacts.eval_config.download_hash",
        "candidate.artifacts.eval_config.plan_hash",
    }


def test_publication_evidence_report_rejects_candidate_stale_eval_config_size(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["artifacts"]["eval_config"]["size_bytes"] += 1
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.artifacts.eval_config.download_size",
        "candidate.artifacts.eval_config.plan_size",
    }


def test_publication_evidence_report_rejects_candidate_wrong_eval_config_path(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["artifacts"]["eval_config"]["path"] = "model/eval_config-copy.yaml"
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.artifacts.eval_config.path_mismatch"
    }


def test_publication_evidence_report_rejects_candidate_stale_eval_metrics_hash(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["artifacts"]["eval_metrics"]["sha256"] = "sha256:" + "c" * 64
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.artifacts.eval_metrics.download_hash",
        "candidate.artifacts.eval_metrics.plan_hash",
    }


def test_publication_evidence_report_rejects_candidate_wrong_dataset_package_path(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["artifacts"]["dataset_package"]["path"] = "dataset/package/dataset_package.json"
    _write_candidate_bundle(paths, candidate)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.artifacts.dataset_package.path_mismatch"
    }
    assert {issue.issue_refs for issue in report.issues} == {(163,)}


def test_publication_evidence_report_rejects_missing_uploaded_demo_download(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["downloaded_artifacts"] = [
        artifact
        for artifact in clean_machine["downloaded_artifacts"]
        if Path(str(artifact["path"])).name != "terminal-demo-transcript.md"
    ]
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.artifacts.terminal_transcript.download_missing",
        "clean_machine.downloaded_artifact.missing_expected",
    }


def test_publication_evidence_report_rejects_downloaded_artifact_parent_escape(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    outside = _write_artifact_file(tmp_path / "downloads" / "outside.md", "demo:outside\n")
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    transcript = next(
        artifact
        for artifact in clean_machine["downloaded_artifacts"]
        if Path(str(artifact["path"])).name == "terminal-demo-transcript.md"
    )
    transcript["path"] = str(tmp_path / "downloads" / "demo" / ".." / outside.name)
    transcript["sha256"] = sha256_file(outside)
    transcript["size_bytes"] = outside.stat().st_size
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.artifacts.terminal_transcript.download_missing",
        "clean_machine.downloaded_artifact.outside_root",
        "clean_machine.downloaded_artifact.missing_expected",
    }


def test_publication_evidence_report_rejects_replay_artifact_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["replay_artifacts"][0]["sha256"] = "sha256:" + "0" * 64
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.artifact_sha256",
        "clean_machine.replay_artifact.hash_mismatch",
    }


def test_publication_evidence_report_rejects_replay_manifest_model_id_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"]["model_id"] = "sha256:" + "f" * 64
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"clean_machine.replay_manifest.model_id"}


def test_publication_evidence_report_rejects_replay_manifest_model_manifest_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["model_manifest"]["sha256"] = "sha256:" + "e" * 64
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.model_manifest_hash"
    }


def test_publication_evidence_report_requires_replay_manifest_model_manifest_input(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"].pop("model_manifest")
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.model_manifest"
    }


def test_publication_evidence_report_requires_replay_manifest_vcf_input(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"].pop("vcf")
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"clean_machine.replay_manifest.vcf"}


def test_publication_evidence_report_rejects_replay_manifest_vcf_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["vcf"]["sha256"] = "sha256:" + "e" * 64
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"clean_machine.replay_manifest.vcf_sha256"}


def test_publication_evidence_report_rejects_replay_manifest_fasta_path_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"]["fasta"]["path"] = "demo/other.fa"
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.fasta_download_missing"
    }


def test_publication_evidence_report_rejects_replay_manifest_stale_score_receipt_batch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["score_receipt_batch"]["records"] = 2
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.score_receipt_batch.stale"
    }


def test_publication_evidence_report_requires_replay_manifest_runtime_preflight_summary(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["runtime_preflight"]
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.runtime_preflight"
    }


def test_publication_evidence_report_rejects_replay_manifest_stale_runtime_preflight_summary(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_preflight"]["command"]["argv"][-1] = "999"
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.runtime_preflight.stale"
    }


def test_publication_evidence_report_rejects_runtime_preflight_manifest_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    preflight_path = _replay_artifact_path(clean_machine, "runtime preflight report")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["manifest"]["sha256"] = "sha256:" + "e" * 64
    _write_json(preflight_path, preflight)
    _refresh_replay_artifact_identity(clean_machine, "runtime preflight report", preflight_path)
    _refresh_terminal_manifest_artifact_identity(
        clean_machine,
        label="runtime preflight report",
        path=preflight_path,
    )
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.runtime_preflight_report.manifest_sha256"
    }


def test_publication_evidence_report_rejects_runtime_preflight_vcf_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    preflight_path = _replay_artifact_path(clean_machine, "runtime preflight report")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["inputs"]["vcf"]["sha256"] = "sha256:" + "e" * 64
    _write_json(preflight_path, preflight)
    _refresh_replay_artifact_identity(clean_machine, "runtime preflight report", preflight_path)
    _refresh_terminal_manifest_artifact_identity(
        clean_machine,
        label="runtime preflight report",
        path=preflight_path,
    )
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.runtime_preflight_report.vcf_sha256"
    }


def test_publication_evidence_report_rejects_runtime_preflight_fasta_path_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    preflight_path = _replay_artifact_path(clean_machine, "runtime preflight report")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["inputs"]["fasta"]["path"] = "demo/other.fa"
    _write_json(preflight_path, preflight)
    _refresh_replay_artifact_identity(clean_machine, "runtime preflight report", preflight_path)
    _refresh_terminal_manifest_artifact_identity(
        clean_machine,
        label="runtime preflight report",
        path=preflight_path,
    )
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.runtime_preflight_report.fasta_download_missing"
    }


def test_publication_evidence_report_rejects_replay_manifest_stale_artifact_identity(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores = next(artifact for artifact in manifest["artifacts"] if artifact["label"] == "scores")
    scores["sha256"] = "sha256:" + "d" * 64
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.replay_manifest.artifact_sha256"
    }


def test_publication_evidence_report_requires_model_dataset_and_demo_downloads(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["downloaded_artifacts"] = [
        artifact
        for artifact in clean_machine["downloaded_artifacts"]
        if artifact["group"] != "demo"
    ]
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "candidate.artifacts.batch_receipt_report.download_missing",
        "candidate.artifacts.receipts_jsonl.download_missing",
        "candidate.artifacts.runtime_preflight.download_missing",
        "candidate.artifacts.scores_jsonl.download_missing",
        "candidate.artifacts.terminal_demo_manifest.download_missing",
        "candidate.artifacts.terminal_transcript.download_missing",
        "clean_machine.downloaded_artifact.missing_expected",
        "clean_machine.downloaded_artifact.missing_group",
        "clean_machine.replay_manifest.fasta_download_missing",
        "clean_machine.replay_manifest.vcf_download_missing",
        "clean_machine.replay_manifest.runtime_preflight_report.fasta_download_missing",
        "clean_machine.replay_manifest.runtime_preflight_report.vcf_download_missing",
    }
    assert {issue.issue_refs for issue in report.issues} == {
        (163, 166, 101),
        (166, 101),
        (166, 167),
    }


def test_publication_evidence_report_requires_every_planned_download(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    dataset_manifest = next(
        artifact
        for artifact in plan["dataset_files"]
        if artifact["destination"] == "dataset_manifest.json"
    )
    plan["dataset_files"].append(
        {
            "destination": "extra.json",
            "sha256": dataset_manifest["sha256"],
            "size_bytes": dataset_manifest["size_bytes"],
            "source": dataset_manifest["source"],
        }
    )
    paths["plan"].write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["hub_plan"] = plan
    paths["candidate"].write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    publish["plan"] = plan
    publish["final_candidate_report"] = candidate
    paths["publish"].write_text(json.dumps(publish, indent=2, sort_keys=True) + "\n")
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"] = _report_identity(paths["candidate"])
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.downloaded_artifact.missing_expected"
    }


def test_publication_evidence_report_rejects_unplanned_downloads(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    extra = _downloaded_artifact(tmp_path / "downloads", "dataset", "unexpected.parquet")
    clean_machine["downloaded_artifacts"].append(extra)
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.downloaded_artifact.unexpected"
    }


def test_publication_evidence_report_rejects_downloaded_source_url_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["downloaded_artifacts"][0]["source_url"] = "https://example.test/model/file"
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "clean_machine.downloaded_artifact.source_url_mismatch"
    }


def test_publication_evidence_report_rejects_duplicate_demo_asset_names(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    duplicate = dict(plan["demo_files"][0])
    duplicate["destination"] = f"nested/{duplicate['destination']}"
    plan["demo_files"].append(duplicate)
    paths["plan"].write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["hub_plan"] = plan
    paths["candidate"].write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    publish["plan"] = plan
    publish["final_candidate_report"] = candidate
    paths["publish"].write_text(json.dumps(publish, indent=2, sort_keys=True) + "\n")
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"] = _report_identity(paths["candidate"])
    paths["clean_machine"].write_text(json.dumps(clean_machine, indent=2, sort_keys=True) + "\n")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"plan.demo_files.duplicate_asset_name"}


def test_publication_evidence_report_requires_paper_file_when_paper_url_is_set(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    del plan["paper_file"]
    _replace_embedded_plan(paths, plan)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"plan.paper_file"}


def test_publication_evidence_report_rejects_paper_file_without_source(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    plan["paper_file"]["source"] = ""
    _replace_embedded_plan(paths, plan)

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"plan.paper_file.source"}


def test_publication_evidence_report_rejects_missing_paper_file_source(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    paths["paper"].unlink()

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {"plan.paper_file.source_missing"}


def test_publication_evidence_report_rejects_paper_file_hash_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_publication_inputs(tmp_path)
    paths["paper"].write_text("# Tampered Paper\n", encoding="utf-8")

    report = build_publication_evidence_report(
        plan_path=paths["plan"],
        release_candidate_path=paths["candidate"],
        publish_report_path=paths["publish"],
        clean_machine_report_path=paths["clean_machine"],
    )

    assert report.ok is False
    assert {issue.code for issue in report.issues} == {
        "plan.paper_file.hash_mismatch",
        "plan.paper_file.size_mismatch",
    }
    assert {issue.issue_refs for issue in report.issues} == {(167,)}


def test_publication_report_main_uses_ready_exit_code(tmp_path: Path, capsys) -> None:
    paths = _write_publication_inputs(tmp_path)
    output = tmp_path / "publication_evidence_report.json"

    rc = main(
        [
            "--plan",
            str(paths["plan"]),
            "--release-candidate",
            str(paths["candidate"]),
            "--publish-report",
            str(paths["publish"]),
            "--clean-machine-demo-report",
            str(paths["clean_machine"]),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert f"wrote {output}" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def _write_publication_inputs(root: Path) -> dict[str, Path]:
    downloads_dir = root / "downloads"
    replay_dir = root / "replay"
    paper = root / "paper.md"
    paper.write_text("# GenoLeWM First Experiment Report\n", encoding="utf-8")
    candidate_artifacts = _candidate_artifacts(paper=paper)
    artifact_downloads = _candidate_upload_downloads(downloads_dir, candidate_artifacts)
    artifact_downloads["demo_vcf"] = _downloaded_artifact(downloads_dir, "demo", "input.vcf")
    artifact_downloads["demo_fasta"] = _downloaded_artifact(downloads_dir, "demo", "ref.fa")
    _apply_candidate_download_identities(candidate_artifacts, artifact_downloads)
    model_download = artifact_downloads["model_manifest"]
    vcf_download = artifact_downloads["demo_vcf"]
    fasta_download = artifact_downloads["demo_fasta"]
    plan = {
        "schema_version": "1.0.0",
        "generated_by": "tools.release.hub_release",
        "generated_at": "2026-06-01T12:00:00Z",
        "model_id": "sha256:" + "a" * 64,
        "release_id": "geno-lewm-v0.1.0-r1",
        "repo_id": "AbdelStark/geno-lewm-v0.1.0-carbon-500m-r1",
        "commit_sha": "abcdef1234567890",
        "dataset_url": "https://huggingface.co/datasets/AbdelStark/geno-lewm-data-v0.1.0-r1",
        "demo_url": "https://github.com/AbdelStark/GenoLeWM/releases/tag/demo-v0.1.0",
        "paper_url": "https://arxiv.org/abs/2606.00001",
        "paper_file": {
            "destination": "paper.md",
            "sha256": sha256_file(paper),
            "size_bytes": paper.stat().st_size,
            "source": str(paper),
        },
        "ready": True,
        "files": _plan_upload_files(artifact_downloads, candidate_artifacts, group="model"),
        "dataset_files": _plan_upload_files(
            artifact_downloads,
            candidate_artifacts,
            group="dataset",
        ),
        "demo_files": _plan_upload_files(artifact_downloads, candidate_artifacts, group="demo"),
        "commands": [
            "python -m tools.release.paper_package --model-dir model --dataset-dir dataset --demo-dir demo",
        ],
        "requirements": ["A real trained checkpoint, not a fixture manifest."],
    }
    _set_download_source_urls(artifact_downloads, candidate_artifacts, plan)
    _append_demo_input_plan_file(plan, vcf_download, "input.vcf")
    _append_demo_input_plan_file(plan, fasta_download, "ref.fa")
    candidate = {
        "schema_version": "1.0.0",
        "generated_by": "tools.release.release_candidate",
        "generated_at": "2026-06-01T12:00:00Z",
        "ready": True,
        "model_id": plan["model_id"],
        "release_id": plan["release_id"],
        "repo_id": plan["repo_id"],
        "commit_sha": plan["commit_sha"],
        "urls": {
            "model": f"https://huggingface.co/{plan['repo_id']}",
            "dataset": plan["dataset_url"],
            "demo": plan["demo_url"],
            "paper": plan["paper_url"],
        },
        "artifacts": candidate_artifacts,
        "blockers": [],
        "hub_plan": plan,
        "package": {"ok": True, "issues": []},
        "public_links": _candidate_public_links(plan),
        "public_artifacts": _candidate_public_artifacts(plan),
        "readiness": _candidate_readiness(),
    }
    publish = {
        "schema_version": "1.0.0",
        "generated_by": "tools.release.hub_publish",
        "plan": plan,
        "final_candidate_ready": True,
        "final_candidate_report": candidate,
    }
    plan_path = root / "hub_release_plan.json"
    candidate_path = root / "release_candidate_report.json"
    publish_path = root / "hub_publish_report.json"
    _write_json(plan_path, plan)
    _write_json(candidate_path, candidate)
    _write_json(publish_path, publish)
    replay_score_receipt_batch = _replay_score_receipt_batch(plan)
    terminal_transcript = _replay_artifact(
        replay_dir,
        "terminal transcript",
        "terminal-demo-transcript.md",
    )
    scores = _replay_artifact(replay_dir, "scores", "scores.jsonl")
    receipts = _replay_artifact(replay_dir, "receipts", "receipts.jsonl")
    runtime_preflight = _runtime_preflight_report_artifact(
        replay_dir,
        plan=plan,
        model_download=model_download,
        vcf_download=vcf_download,
        fasta_download=fasta_download,
    )
    batch_receipt = _batch_receipt_report_artifact(replay_dir, replay_score_receipt_batch)
    terminal_demo_manifest = _terminal_demo_manifest_artifact(
        replay_dir,
        model_download=model_download,
        vcf_download=vcf_download,
        fasta_download=fasta_download,
        plan=plan,
        score_receipt_batch=replay_score_receipt_batch,
        replay_artifacts=(terminal_transcript, scores, receipts, runtime_preflight, batch_receipt),
    )
    clean_machine = {
        "schema_version": "1.0.0",
        "generated_by": "tools.release.clean_machine_demo",
        "model_dir": str(downloads_dir / "model"),
        "dataset_dir": str(downloads_dir / "dataset"),
        "demo_dir": str(downloads_dir / "demo"),
        "package": {"ok": True, "issues": []},
        "release_candidate_report": str(candidate_path),
        "release_candidate_report_identity": _report_identity(candidate_path),
        "downloaded_artifacts": list(artifact_downloads.values()),
        "replay_artifacts": [
            terminal_transcript,
            terminal_demo_manifest,
            scores,
            receipts,
            runtime_preflight,
            batch_receipt,
        ],
    }
    clean_machine_path = root / "clean_machine_demo_report.json"
    _write_json(clean_machine_path, clean_machine)
    return {
        "plan": plan_path,
        "candidate": candidate_path,
        "publish": publish_path,
        "clean_machine": clean_machine_path,
        "paper": paper,
    }


def _issue_refs(*numbers: int) -> list[dict[str, object]]:
    return issue_ref_payload(numbers)


def _candidate_artifact(path: str, artifact: dict[str, object]) -> dict[str, object]:
    return {
        "path": path,
        "sha256": artifact["sha256"],
        "size_bytes": artifact["size_bytes"],
    }


def _synthetic_candidate_artifact(path: str, seed: str) -> dict[str, object]:
    digest = (seed.encode("utf-8").hex() * 64)[:64]
    return {
        "path": path,
        "sha256": f"sha256:{digest}",
        "size_bytes": max(1, len(seed)),
    }


def _candidate_artifacts(*, paper: Path) -> dict[str, object]:
    artifacts = {
        "model_manifest": _synthetic_candidate_artifact("model/manifest.json", "model_manifest"),
        "model_package": _synthetic_candidate_artifact(
            "model/model_package.json",
            "model_package",
        ),
        "model_card": _synthetic_candidate_artifact("model/model_card.md", "model_card"),
        "model_checksums": _synthetic_candidate_artifact("model/SHA256SUMS", "model_checksums"),
        "predictor": _synthetic_candidate_artifact(
            "model/predictor.safetensors",
            "predictor",
        ),
        "action_encoder": _synthetic_candidate_artifact(
            "model/action_encoder.safetensors",
            "action_encoder",
        ),
        "calibration": _synthetic_candidate_artifact(
            "model/calibration.json",
            "calibration",
        ),
        "training_config": _synthetic_candidate_artifact(
            "model/train_config.yaml",
            "training_config",
        ),
        "training_run_manifest": _synthetic_candidate_artifact(
            "model/training_run_manifest.json",
            "training_run_manifest",
        ),
        "training_run_card": _synthetic_candidate_artifact(
            "model/training_run_card.md",
            "training_run_card",
        ),
        "training_run_checksums": _synthetic_candidate_artifact(
            "model/training_run_SHA256SUMS",
            "training_run_checksums",
        ),
        "training_preflight_report": _synthetic_candidate_artifact(
            "model/training_preflight_report.json",
            "training_preflight_report",
        ),
        "eval_metrics": _synthetic_candidate_artifact(
            "model/eval_metrics.json",
            "eval_metrics",
        ),
        "eval_config": _synthetic_candidate_artifact(
            "model/eval_config.effective.yaml",
            "eval_config",
        ),
        "eval_report": _synthetic_candidate_artifact("model/eval_report.md", "eval_report"),
        "efficiency_report": _synthetic_candidate_artifact(
            "model/efficiency_report.json",
            "efficiency_report",
        ),
        "dataset_manifest": _synthetic_candidate_artifact(
            "dataset/dataset_manifest.json",
            "dataset_manifest",
        ),
        "dataset_package": _synthetic_candidate_artifact(
            "dataset/dataset_package.json",
            "dataset_package",
        ),
        "dataset_snapshot_report": _synthetic_candidate_artifact(
            "dataset/dataset_snapshot_report.json",
            "dataset_snapshot_report",
        ),
        "dataset_input_check_report": _synthetic_candidate_artifact(
            "dataset/dataset_input_check_report.json",
            "dataset_input_check_report",
        ),
        "data_card": _synthetic_candidate_artifact("dataset/data_card.md", "data_card"),
        "dataset_integrity": _synthetic_candidate_artifact(
            "dataset/split_integrity.json",
            "dataset_integrity",
        ),
        "dataset_checksums": _synthetic_candidate_artifact(
            "dataset/SHA256SUMS",
            "dataset_checksums",
        ),
        "terminal_transcript": _synthetic_candidate_artifact(
            "demo/terminal-demo-transcript.md",
            "terminal_transcript",
        ),
        "terminal_demo_manifest": _synthetic_candidate_artifact(
            "demo/terminal_demo_manifest.json",
            "terminal_demo_manifest",
        ),
        "runtime_preflight": _synthetic_candidate_artifact(
            "demo/runtime_preflight_report.json",
            "runtime_preflight",
        ),
        "batch_receipt_report": _synthetic_candidate_artifact(
            "demo/batch_receipt_report.json",
            "batch_receipt_report",
        ),
        "scores_jsonl": _synthetic_candidate_artifact("demo/scores.jsonl", "scores_jsonl"),
        "receipts_jsonl": _synthetic_candidate_artifact(
            "demo/receipts.jsonl",
            "receipts_jsonl",
        ),
        "paper": {
            "path": paper.name,
            "sha256": sha256_file(paper),
            "size_bytes": paper.stat().st_size,
        },
    }
    return artifacts


def _candidate_upload_downloads(
    root: Path,
    artifacts: dict[str, object],
) -> dict[str, dict[str, object]]:
    downloads: dict[str, dict[str, object]] = {}
    for key, (group, fixed_destination) in CANDIDATE_ARTIFACT_UPLOADS.items():
        destination = _candidate_upload_destination(
            artifacts,
            key=key,
            group=group,
            fixed_destination=fixed_destination,
        )
        downloads[key] = _downloaded_artifact(root, group, destination)
    return downloads


def _apply_candidate_download_identities(
    artifacts: dict[str, object],
    downloads: dict[str, dict[str, object]],
) -> None:
    for key, (group, fixed_destination) in CANDIDATE_ARTIFACT_UPLOADS.items():
        destination = _candidate_upload_destination(
            artifacts,
            key=key,
            group=group,
            fixed_destination=fixed_destination,
        )
        artifacts[key] = _candidate_artifact(f"{group}/{destination}", downloads[key])


def _plan_upload_files(
    downloads: dict[str, dict[str, object]],
    artifacts: dict[str, object],
    *,
    group: str,
) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for key, (upload_group, fixed_destination) in CANDIDATE_ARTIFACT_UPLOADS.items():
        if upload_group != group:
            continue
        destination = _candidate_upload_destination(
            artifacts,
            key=key,
            group=group,
            fixed_destination=fixed_destination,
        )
        download = downloads[key]
        files.append(
            {
                "destination": destination,
                "sha256": download["sha256"],
                "size_bytes": download["size_bytes"],
                "source": f"{group}/{destination}",
            }
        )
    return sorted(files, key=lambda item: str(item["destination"]))


def _set_download_source_urls(
    downloads: dict[str, dict[str, object]],
    artifacts: dict[str, object],
    plan: dict[str, object],
) -> None:
    for key, (group, fixed_destination) in CANDIDATE_ARTIFACT_UPLOADS.items():
        destination = _candidate_upload_destination(
            artifacts,
            key=key,
            group=group,
            fixed_destination=fixed_destination,
        )
        download = downloads[key]
        if group == "model":
            download["source_url"] = (
                f"https://huggingface.co/{plan['repo_id']}/resolve/main/{destination}"
            )
        elif group == "dataset":
            download["source_url"] = f"{plan['dataset_url']}/resolve/main/{destination}"
        else:
            download["source_url"] = (
                "https://github.com/AbdelStark/GenoLeWM/releases/download/demo-v0.1.0/"
                f"{Path(destination).name}"
            )


def _append_demo_input_plan_file(
    plan: dict[str, object],
    download: dict[str, object],
    destination: str,
) -> None:
    plan["demo_files"].append(
        {
            "destination": destination,
            "sha256": download["sha256"],
            "size_bytes": download["size_bytes"],
            "source": f"demo/{destination}",
        }
    )
    download["source_url"] = (
        f"https://github.com/AbdelStark/GenoLeWM/releases/download/demo-v0.1.0/{destination}"
    )
    plan["demo_files"] = sorted(plan["demo_files"], key=lambda item: str(item["destination"]))


def _candidate_upload_destination(
    artifacts: dict[str, object],
    *,
    key: str,
    group: str,
    fixed_destination: str | None,
) -> str:
    if fixed_destination is not None:
        return fixed_destination
    artifact = artifacts[key]
    assert isinstance(artifact, dict)
    path = artifact["path"]
    assert isinstance(path, str)
    prefix = f"{group}/"
    assert path.startswith(prefix)
    return path.removeprefix(prefix)


def _candidate_public_links(plan: dict[str, object]) -> dict[str, object]:
    return {
        "required": True,
        "checks": [
            _public_link_check("model", f"https://huggingface.co/{plan['repo_id']}"),
            _public_link_check("dataset", str(plan["dataset_url"])),
            _public_link_check("demo", str(plan["demo_url"])),
            _public_link_check("paper", str(plan["paper_url"])),
        ],
    }


def _public_link_check(name: str, url: str) -> dict[str, object]:
    return {
        "name": name,
        "url": url,
        "ok": True,
        "status_code": 200,
        "error": None,
    }


def _candidate_public_artifacts(plan: dict[str, object]) -> dict[str, object]:
    return {
        "required": True,
        "checks": [
            _public_artifact_check("model", plan["files"]),
            _public_artifact_check("dataset", plan["dataset_files"]),
            _public_artifact_check("demo", plan["demo_files"]),
            _public_artifact_check("paper", [plan["paper_file"]]),
        ],
    }


def _public_artifact_check(name: str, files: object) -> dict[str, object]:
    assert isinstance(files, list)
    return {
        "name": name,
        "url": f"https://example.test/{name}",
        "ok": True,
        "expected_count": len(files),
        "observed_count": len(files),
        "verified_count": len(files),
        "missing": [],
        "hash_mismatches": [],
        "size_mismatches": [],
        "unexpected": [],
        "status_code": 200,
        "error": None,
    }


def _candidate_readiness() -> list[dict[str, object]]:
    return [
        _readiness_item(
            "package_verifier",
            (163, 164, 165, 166, 167, 101),
            evidence=("tools.release.paper_package",),
        ),
        _readiness_item(
            "model_package",
            (164, 165, 101),
            evidence=("release_id=geno-lewm-v0.1.0-r1", "model_files=2"),
        ),
        _readiness_item(
            "dataset_package",
            (163,),
            evidence=("dataset_snapshot_id=geno-lewm-data-v0.1.0-r1", "dataset_files=1"),
        ),
        _readiness_item("terminal_demo", (166,), evidence=("demo_files=1",)),
        _readiness_item(
            "paper_artifact",
            (167,),
            evidence=("paper_path=paper.md", "paper_url=https://arxiv.org/abs/2606.00001"),
        ),
        _readiness_item(
            "public_links",
            (163, 166, 167, 101),
            evidence=("model=ok", "dataset=ok", "demo=ok", "paper=ok"),
        ),
        _readiness_item(
            "public_artifacts",
            (163, 166, 167, 101),
            evidence=("model=ok", "dataset=ok", "demo=ok", "paper=ok"),
        ),
        _readiness_item(
            "hub_publication_plan",
            (167, 101),
            evidence=("model_files=2", "dataset_files=1", "demo_files=1", "commands=1"),
        ),
    ]


def _readiness_item(
    code: str,
    issue_numbers: tuple[int, ...],
    *,
    evidence: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "code": code,
        "ok": True,
        "message": f"{code} passed",
        "evidence": list(evidence),
        "blockers": [],
        "issue_refs": _issue_refs(*issue_numbers),
    }


def _write_candidate_bundle(paths: dict[str, Path], candidate: dict[str, object]) -> None:
    paths["candidate"].write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    publish["final_candidate_report"] = candidate
    paths["publish"].write_text(
        json.dumps(publish, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"] = _report_identity(paths["candidate"])
    paths["clean_machine"].write_text(
        json.dumps(clean_machine, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_clean_machine_paths_relative(
    clean_machine: dict[str, object],
    *,
    root: Path,
    candidate: Path,
) -> None:
    for field in ("model_dir", "dataset_dir", "demo_dir"):
        value = clean_machine[field]
        assert isinstance(value, str)
        clean_machine[field] = Path(value).relative_to(root).as_posix()

    clean_machine["release_candidate_report"] = candidate.name
    identity = clean_machine["release_candidate_report_identity"]
    assert isinstance(identity, dict)
    identity["path"] = candidate.name

    for field in ("downloaded_artifacts", "replay_artifacts"):
        artifacts = clean_machine[field]
        assert isinstance(artifacts, list)
        for artifact in artifacts:
            assert isinstance(artifact, dict)
            value = artifact["path"]
            assert isinstance(value, str)
            artifact["path"] = Path(value).relative_to(root).as_posix()


def _downloaded_artifact(root: Path, group: str, path: str) -> dict[str, object]:
    artifact_path = _write_artifact_file(root / group / path, f"{group}:{path}\n")
    return {
        "group": group,
        "path": str(artifact_path),
        "source_url": f"https://example.test/{group}/{path}",
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }


def _replay_artifact(root: Path, label: str, path: str) -> dict[str, object]:
    artifact_path = _write_artifact_file(root / path, f"{label}:{path}\n")
    return {
        "label": label,
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }


def _terminal_demo_manifest_artifact(
    root: Path,
    *,
    model_download: dict[str, object],
    vcf_download: dict[str, object],
    fasta_download: dict[str, object],
    plan: dict[str, object],
    score_receipt_batch: dict[str, object],
    replay_artifacts: tuple[dict[str, object], ...],
) -> dict[str, object]:
    artifact_path = root / "terminal_demo_manifest.json"
    _write_json(
        artifact_path,
        {
            "generated_by": "tools.demo.terminal_inference",
            "schema_version": "1.0.0",
            "status": "passed",
            "model": {
                "release_id": plan["release_id"],
                "model_id": plan["model_id"],
            },
            "inputs": {
                "model_manifest": {
                    "path": model_download["path"],
                    "sha256": model_download["sha256"],
                    "size_bytes": model_download["size_bytes"],
                },
                "vcf": {
                    "path": vcf_download["path"],
                    "sha256": vcf_download["sha256"],
                    "size_bytes": vcf_download["size_bytes"],
                },
                "fasta": {
                    "path": fasta_download["path"],
                    "sha256": fasta_download["sha256"],
                    "size_bytes": fasta_download["size_bytes"],
                },
            },
            "artifacts": [
                {
                    "label": artifact["label"],
                    "path": artifact["path"],
                    "sha256": artifact["sha256"],
                    "size_bytes": artifact["size_bytes"],
                }
                for artifact in replay_artifacts
            ],
            "score_receipt_batch": score_receipt_batch,
            "runtime_preflight": _runtime_preflight_summary_from_artifact(
                next(
                    artifact
                    for artifact in replay_artifacts
                    if artifact["label"] == "runtime preflight report"
                )
            ),
        },
    )
    return {
        "label": "terminal demo manifest",
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }


def _replay_score_receipt_batch(plan: dict[str, object]) -> dict[str, object]:
    return {
        "records": 1,
        "model_id": plan["model_id"],
        "calibration_hash": "sha256:" + "c" * 64,
        "receipt_schema_version": "1.0.0",
        "receipt_stream": "jsonl_per_scored_alternate_v1",
        "checked_score_fields": ["sigma_raw", "sigma_calibrated"],
        "runtime": {"backend": "cpu", "device": "CPU"},
    }


def _runtime_preflight_report_artifact(
    root: Path,
    *,
    plan: dict[str, object],
    model_download: dict[str, object],
    vcf_download: dict[str, object],
    fasta_download: dict[str, object],
) -> dict[str, object]:
    artifact_path = root / "runtime_preflight_report.json"
    command = [
        "geno-lewm-score",
        "--quiet",
        "--no-banner",
        "--model-dir",
        "model",
        "--backend",
        "cpu",
        "--vcf",
        "demo/input.vcf",
        "--fasta",
        "demo/ref.fa",
        "--output",
        "demo/scores.jsonl",
        "--receipt",
        "demo/receipts.jsonl",
        "--batch-size",
        "64",
        "--no-progress",
    ]
    _write_json(
        artifact_path,
        {
            "schema_version": "1.0.0",
            "generated_by": "tools.release.runtime_preflight",
            "ok": True,
            "model_id": plan["model_id"],
            "release_id": plan["release_id"],
            "requested_backend": "cpu",
            "selected_backend": "cpu",
            "requirements": {
                "native_runtime": True,
                "carbon_cache": False,
                "fixture_manifest_allowed": False,
            },
            "manifest": {
                "path": model_download["path"],
                "sha256": model_download["sha256"],
                "size_bytes": model_download["size_bytes"],
            },
            "inputs": {
                "vcf": {
                    "path": vcf_download["path"],
                    "exists": True,
                    "ok": True,
                    "sha256": vcf_download["sha256"],
                    "size_bytes": vcf_download["size_bytes"],
                },
                "fasta": {
                    "path": fasta_download["path"],
                    "exists": True,
                    "ok": True,
                    "sha256": fasta_download["sha256"],
                    "size_bytes": fasta_download["size_bytes"],
                },
            },
            "command": {
                "argv": command,
                "shell": " ".join(command),
            },
        },
    )
    return {
        "label": "runtime preflight report",
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }


def _runtime_preflight_summary_from_artifact(artifact: dict[str, object]) -> dict[str, object]:
    raw_path = artifact["path"]
    assert isinstance(raw_path, str)
    payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    command = payload["command"]
    assert isinstance(command, dict)
    requirements = payload["requirements"]
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


def _batch_receipt_report_artifact(
    root: Path,
    score_receipt_batch: dict[str, object],
) -> dict[str, object]:
    artifact_path = root / "batch_receipt_report.json"
    _write_json(artifact_path, score_receipt_batch)
    return {
        "label": "batch receipt report",
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
    }


def _replay_artifact_path(clean_machine: dict[str, object], label: str) -> Path:
    artifacts = clean_machine["replay_artifacts"]
    assert isinstance(artifacts, list)
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        if artifact.get("label") == label:
            raw_path = artifact["path"]
            assert isinstance(raw_path, str)
            return Path(raw_path)
    raise AssertionError(f"missing replay artifact: {label}")


def _refresh_replay_artifact_identity(
    clean_machine: dict[str, object],
    label: str,
    path: Path,
) -> None:
    artifacts = clean_machine["replay_artifacts"]
    assert isinstance(artifacts, list)
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        if artifact.get("label") == label:
            artifact["sha256"] = sha256_file(path)
            artifact["size_bytes"] = path.stat().st_size
            return
    raise AssertionError(f"missing replay artifact: {label}")


def _refresh_terminal_manifest_artifact_identity(
    clean_machine: dict[str, object],
    *,
    label: str,
    path: Path,
) -> None:
    manifest_path = _replay_artifact_path(clean_machine, "terminal demo manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        if artifact.get("label") == label:
            artifact["sha256"] = sha256_file(path)
            artifact["size_bytes"] = path.stat().st_size
            break
    else:
        raise AssertionError(f"missing terminal manifest artifact: {label}")
    _write_json(manifest_path, manifest)
    _refresh_replay_artifact_identity(clean_machine, "terminal demo manifest", manifest_path)


def _replace_embedded_plan(paths: dict[str, Path], plan: dict[str, object]) -> None:
    _write_json(paths["plan"], plan)
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    candidate["hub_plan"] = plan
    _write_json(paths["candidate"], candidate)
    publish = json.loads(paths["publish"].read_text(encoding="utf-8"))
    publish["plan"] = plan
    publish["final_candidate_report"] = candidate
    _write_json(paths["publish"], publish)
    clean_machine = json.loads(paths["clean_machine"].read_text(encoding="utf-8"))
    clean_machine["release_candidate_report_identity"] = _report_identity(paths["candidate"])
    _write_json(paths["clean_machine"], clean_machine)


def _report_identity(path: Path) -> dict[str, object]:
    return {
        "label": "release candidate report",
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_artifact_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
