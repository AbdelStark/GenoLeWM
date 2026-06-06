"""Tests for accepted rollout-speed scope reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from tools.release import rollout_speed_scope

ROLLOUT_CLAIM_BOUNDARY = (
    "This benchmark measures local predictor rollout speed only; it is not "
    "model-quality, clinical, privacy, or release-readiness evidence."
)


def test_scope_report_binds_failed_rollout_speed_report(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")

    report = rollout_speed_scope.build_scope_report(
        rollout_speed_report=rollout,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent predictor path reports measured speed below target.",
        replacement_target="Report measured rollout speed while a new target is accepted in #42.",
    )

    assert report["generated_by"] == "tools.release.rollout_speed_scope"
    assert report["ok"] is True
    assert report["decision"] == "rescope_rfc0004_speed_target"
    assert report["issue_refs"] == ["#42", "#197"]
    assert report["rollout_speed_report"]["sha256"].startswith("sha256:")
    summary = report["rollout_speed_summary"]
    assert summary["observed_values"] == {"k5_speedup": 1.8, "k20_speedup": 1.9}
    assert summary["failed_targets"] == [
        {
            "horizon": 5,
            "measured_speedup": 1.8,
            "target_speedup": 2.0,
            "shortfall": pytest.approx(0.2),
        },
        {
            "horizon": 20,
            "measured_speedup": 1.9,
            "target_speedup": 5.0,
            "shortfall": pytest.approx(3.1),
        },
    ]


def test_scope_report_uses_public_safe_paths(tmp_path: Path) -> None:
    rollout = tmp_path / "reports" / "rollout.ar_speed.json"
    rollout.parent.mkdir()
    output = tmp_path / "scope" / "rollout_speed_scope.json"
    payload = _failing_rollout_payload()
    payload["command"] = [
        "python",
        "-m",
        "bench.rollout",
        "--k",
        "5",
        "--k",
        "20",
        "--output-json",
        str(rollout),
        "--out-dir",
        str(tmp_path / "bench"),
        str(tmp_path / "scratch" / "unused.json"),
    ]
    rollout.write_text(json.dumps(payload), encoding="utf-8")

    report = rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=output,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent predictor path reports measured speed below target.",
        replacement_target="Report measured rollout speed while a new target is accepted in #42.",
        command=(
            "python",
            "-m",
            "tools.release.rollout_speed_scope",
            "--rollout-speed-report",
            str(rollout),
            "--output",
            str(output),
        ),
    )

    assert report["rollout_speed_report"]["path"] == "rollout.ar_speed.json"
    assert report["command"] == [
        "python",
        "-m",
        "tools.release.rollout_speed_scope",
        "--rollout-speed-report",
        "rollout.ar_speed.json",
        "--output",
        "rollout_speed_scope.json",
    ]
    assert report["rollout_speed_summary"]["command"] == [
        "python",
        "-m",
        "bench.rollout",
        "--k",
        "5",
        "--k",
        "20",
        "--output-json",
        "rollout.ar_speed.json",
        "--out-dir",
        "bench",
        "unused.json",
    ]
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)


def test_scope_report_rejects_passing_rollout_speed_report(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    payload = _failing_rollout_payload()
    payload["ok"] = True
    for row in payload["rows"]:
        row["measured_speedup"] = row["target_speedup"] + 0.1
        row["target_met"] = True
    rollout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="requires at least one failed target"):
        rollout_speed_scope.build_scope_report(
            rollout_speed_report=rollout,
            accepted_by="maintainer",
            accepted_at="2026-06-06T12:00:00Z",
            decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
            rationale="The target was accepted as met.",
            replacement_target="No re-scope is needed.",
        )


def test_scope_report_requires_k5_and_k20_measurements(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    payload = _failing_rollout_payload()
    payload["rows"] = payload["rows"][:1]
    rollout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="requires K=5 and K=20"):
        rollout_speed_scope.build_scope_report(
            rollout_speed_report=rollout,
            accepted_by="maintainer",
            accepted_at="2026-06-06T12:00:00Z",
            decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
            rationale="Only one target was measured.",
            replacement_target="Report measured rollout speed.",
        )


def test_scope_report_rejects_weakened_rollout_claim_boundary(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    payload = _failing_rollout_payload()
    payload["claim_boundary"] = "This report records rollout speed."
    rollout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="claim_boundary must preserve"):
        rollout_speed_scope.build_scope_report(
            rollout_speed_report=rollout,
            accepted_by="maintainer",
            accepted_at="2026-06-06T12:00:00Z",
            decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
            rationale="The current recurrent predictor path reports measured speed below target.",
            replacement_target="Report measured rollout speed until #42 accepts a new target.",
        )


def test_scope_main_writes_report(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    output = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")

    rc = rollout_speed_scope.main(
        [
            "--rollout-speed-report",
            str(rollout),
            "--output",
            str(output),
            "--accepted-by",
            "maintainer",
            "--accepted-at",
            "2026-06-06T12:00:00Z",
            "--decision-url",
            "https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
            "--rationale",
            "The current measured rollout benchmark missed the RFC-0004 target.",
            "--replacement-target",
            "Report measured rollout speed until #42 accepts a new target.",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["command"][:3] == ["python", "-m", "tools.release.rollout_speed_scope"]


def _failing_rollout_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": "bench.rollout",
        "ok": False,
        "commit": "abcdef1234567890",
        "command": ["python", "-m", "bench.rollout", "--k", "5", "--k", "20"],
        "claim_boundary": ROLLOUT_CLAIM_BOUNDARY,
        "rows": [
            {
                "horizon": 5,
                "measured_speedup": 1.8,
                "target_speedup": 2.0,
                "target_met": False,
            },
            {
                "horizon": 20,
                "measured_speedup": 1.9,
                "target_speedup": 5.0,
                "target_met": False,
            },
        ],
    }
