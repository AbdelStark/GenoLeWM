# SPDX-License-Identifier: Apache-2.0
"""Tests for the GenoLeWM-FX feasibility gate tooling."""

from __future__ import annotations

import json
from pathlib import Path

from tools.research.fx_feasibility import (
    build_feasibility_report,
    build_source_probe_rows,
    main,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "configs" / "fx" / "feasibility_sources.json"


def test_fx_feasibility_manifest_renders_kill_decision() -> None:
    report = build_feasibility_report(
        manifest_path=MANIFEST,
        generated_at="2026-06-11T10:00:00Z",
    )

    assert report["decision"] == "kill"
    assert report["ok_to_continue"] is False
    assert report["epic_issue"] == 257
    assert {blocker["code"] for blocker in report["blockers"]} >= {
        "no_public_teacher_delta_cache",
        "no_bulk_admissible_teacher",
        "source_probe_not_run",
    }
    actions = {action["issue"]: action["action"] for action in report["recommended_issue_actions"]}
    assert actions[258] == "close-completed"
    assert actions[259] == "close-kill"
    assert actions[262] == "close-not-planned"


def test_source_probe_rows_measure_public_label_baselines() -> None:
    source_probe = {
        "dataset": "fixture",
        "config": "complex_traits",
        "split": "test",
        "holdout_chromosomes": ["2"],
        "bootstrap_samples": 20,
        "bootstrap_seed": 257,
        "score_columns": [
            {
                "id": "maf_source_only",
                "column": "maf",
                "direction": "positive",
                "description": "MAF score",
            }
        ],
    }
    records = [
        {"chrom": "1", "label": 0, "maf": 0.1},
        {"chrom": "1", "label": 1, "maf": 0.8},
        {"chrom": "1", "label": 0, "maf": 0.2},
        {"chrom": "1", "label": 1, "maf": 0.9},
        {"chrom": "2", "label": 0, "maf": 0.1},
        {"chrom": "2", "label": 1, "maf": 0.9},
        {"chrom": "2", "label": 0, "maf": 0.2},
        {"chrom": "2", "label": 1, "maf": 0.8},
    ]

    rows = build_source_probe_rows(records=records, source_probe=source_probe)

    assert [row["baseline_id"] for row in rows] == [
        "label_prior_no_teacher",
        "maf_source_only",
    ]
    metrics = {metric["name"]: metric for metric in rows[1]["metrics"]}
    assert metrics["auroc"]["value"] == 1.0
    assert metrics["average_precision"]["value"] == 1.0
    assert metrics["balanced_accuracy"]["value"] == 1.0
    assert rows[1]["target_kind"] == "variant_label_not_teacher_delta"


def test_fx_feasibility_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    exit_code = main(
        [
            "--manifest",
            str(MANIFEST),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--generated-at",
            "2026-06-11T10:00:00Z",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["decision"] == "kill"
    markdown = output_md.read_text(encoding="utf-8")
    assert "Decision: **kill**." in markdown
    assert "no_public_teacher_delta_cache" in markdown


def test_rendered_markdown_explains_no_demo_or_training_path() -> None:
    report = build_feasibility_report(
        manifest_path=MANIFEST,
        generated_at="2026-06-11T10:00:00Z",
    )
    markdown = render_markdown(report)

    assert "The FX pivot is stopped before teacher-cache implementation" in markdown
    assert "No public source-only probe rows were generated." in markdown
    assert "#262" in markdown
    assert "no clinical utility claim" in markdown.lower()
