# SPDX-License-Identifier: Apache-2.0
"""Publish the final GenoLeWM-FX Borzoi rescue outcome."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.fx_borzoi_final"
DEFAULT_OVERLAP_REPORT: Final = Path("docs/research/fx-borzoi-overlap-report.json")
DEFAULT_CACHE_MANIFEST: Final = Path("docs/research/fx-borzoi-cache-manifest.json")
DEFAULT_BASELINE_REPORT: Final = Path("docs/research/fx-borzoi-baseline-report.json")
DEFAULT_RESIDUAL_REPORT: Final = Path("docs/research/fx-borzoi-residual-report.json")
DEFAULT_OUTPUT_JSON: Final = Path("docs/research/fx-borzoi-final-report.json")
DEFAULT_OUTPUT_MD: Final = Path("docs/research/fx-borzoi-final-report.md")

JsonDict = dict[str, Any]


def build_final_report(
    *,
    overlap_report_path: Path = DEFAULT_OVERLAP_REPORT,
    cache_manifest_path: Path = DEFAULT_CACHE_MANIFEST,
    baseline_report_path: Path = DEFAULT_BASELINE_REPORT,
    residual_report_path: Path = DEFAULT_RESIDUAL_REPORT,
    generated_at: str | None = None,
) -> JsonDict:
    """Build the final machine-readable Borzoi rescue outcome."""
    overlap = _load_json(overlap_report_path, label="overlap report")
    cache = _load_json(cache_manifest_path, label="cache manifest")
    baseline = _load_json(baseline_report_path, label="baseline report")
    residual = _load_json(residual_report_path, label="residual report")
    _assert_expected_chain(overlap=overlap, cache=cache, baseline=baseline, residual=residual)
    final_positive = bool(residual["final_positive_claim_supported"])
    outcome = "positive_locked_result" if final_positive else "no_positive_claim_fragile_lift"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "epic_issue": 266,
        "final_issue": 272,
        "outcome": outcome,
        "positive_claim_allowed": final_positive,
        "artifact_receipts": {
            "overlap_report": _receipt(overlap_report_path, overlap),
            "cache_manifest": _receipt(cache_manifest_path, cache),
            "baseline_report": _receipt(baseline_report_path, baseline),
            "residual_report": _receipt(residual_report_path, residual),
            "cache_artifact": cache["cache_artifact"],
            "residual_prediction_artifact": residual["prediction_artifact"],
        },
        "task": {
            "dataset": overlap["traitgym_native_alignment"]["dataset"],
            "config": overlap["traitgym_native_alignment"]["config"],
            "split": overlap["traitgym_native_alignment"]["split"],
            "score_id": cache["score_id"],
            "score_column": cache["score_column"],
            "target_kind": cache["target_kind"],
            "fipip_exact_join_status": cache["fipip_exact_join_status"],
        },
        "gate_summary": {
            "overlap_decision": overlap["decision"],
            "usable_rows": overlap["traitgym_native_alignment"]["usable_rows"],
            "cache_rows": cache["row_count"],
            "baseline_decision": baseline["decision"],
            "residual_decision": residual["decision"],
        },
        "baseline_summary": {
            "strongest_simple_baseline": baseline["strongest_simple_baseline"]["baseline_id"],
            "strongest_simple_baseline_metrics": baseline["strongest_simple_baseline"][
                "holdout_metrics"
            ],
        },
        "residual_summary": {
            "residual_ensemble_metrics": residual["residual_ensemble_metrics"],
            "paired_deltas_vs_strongest_simple_baseline": residual[
                "paired_deltas_vs_strongest_baseline"
            ],
            "seed_variance": residual["seed_variance"],
            "final_positive_claim_supported": residual["final_positive_claim_supported"],
        },
        "negative_findings": [
            "The full fipip table exact join was not run; exact fipip overlap is not claimed.",
            "The residual ensemble's paired AUPRC and AUROC confidence intervals cross zero.",
            "No trained GenoLeWM-FX positive model-quality claim is supported.",
            "No FX demo or paper-positive result should ship from this trajectory.",
        ],
        "recommended_issue_actions": [
            {
                "issue": 272,
                "action": "close-completed",
                "reason": "The final no-positive-claim outcome is source-controlled and claim-bounded.",
            },
            {
                "issue": 266,
                "action": "close-completed",
                "reason": "All child gates completed and the final outcome is published.",
            },
        ],
        "claim_boundary": (
            "This final report is a no-positive-claim FX rescue outcome. It is not a "
            "clinical result, deployment-readiness result, broad VEP superiority claim, "
            "broad Carbon comparison, useful-planning claim, ground-truth biology claim, "
            "or exact fipip overlap claim."
        ),
    }


def write_report(
    *,
    report: Mapping[str, Any],
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
) -> None:
    """Write final JSON and Markdown reports."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the final human-facing report."""
    baseline = _metric_index(
        _require_list(report["baseline_summary"]["strongest_simple_baseline_metrics"])
    )
    residual = _metric_index(_require_list(report["residual_summary"]["residual_ensemble_metrics"]))
    deltas = _metric_index(
        _require_list(report["residual_summary"]["paired_deltas_vs_strongest_simple_baseline"])
    )
    return "\n".join(
        [
            "# GenoLeWM-FX Borzoi final outcome report",
            "",
            f"Generated by `{report['generated_by']}` at `{report['generated_at']}`.",
            "",
            f"Parent epic: #{report['epic_issue']}. Final issue: #{report['final_issue']}.",
            "",
            f"Outcome: **{report['outcome']}**.",
            "",
            f"Positive claim allowed: **{report['positive_claim_allowed']}**.",
            "",
            str(report["claim_boundary"]),
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run python -m tools.research.fx_borzoi_final \\",
            "  --overlap-report docs/research/fx-borzoi-overlap-report.json \\",
            "  --cache-manifest docs/research/fx-borzoi-cache-manifest.json \\",
            "  --baseline-report docs/research/fx-borzoi-baseline-report.json \\",
            "  --residual-report docs/research/fx-borzoi-residual-report.json \\",
            "  --output-json docs/research/fx-borzoi-final-report.json \\",
            "  --output-md docs/research/fx-borzoi-final-report.md",
            "```",
            "",
            "## Final Metrics",
            "",
            "| Model | AUROC | AUPRC | Balanced accuracy |",
            "| --- | ---: | ---: | ---: |",
            f"| Strongest simple baseline | {_metric_value(baseline, 'auroc')} | "
            f"{_metric_value(baseline, 'average_precision')} | "
            f"{_metric_value(baseline, 'balanced_accuracy')} |",
            f"| Residual ensemble | {_metric_value(residual, 'auroc')} | "
            f"{_metric_value(residual, 'average_precision')} | "
            f"{_metric_value(residual, 'balanced_accuracy')} |",
            "",
            "Paired residual deltas versus the strongest simple baseline:",
            "",
            "| Metric | Delta | 95% CI |",
            "| --- | ---: | ---: |",
            f"| AUROC | {_delta_value(deltas, 'auroc')} | {_delta_ci(deltas, 'auroc')} |",
            f"| AUPRC | {_delta_value(deltas, 'average_precision')} | "
            f"{_delta_ci(deltas, 'average_precision')} |",
            "",
            "## Negative Findings",
            "",
            *[f"- {finding}" for finding in _require_list(report["negative_findings"])],
            "",
            "## Final Interpretation",
            "",
            "The rescue path was worth testing because the compact TraitGym-native Borzoi "
            "artifact cleared the overlap/cache/baseline gates. The residual model then "
            "showed a small mean lift, but the paired confidence intervals cross zero. "
            "The correct final outcome is therefore a published no-positive-claim result, "
            "not an FX model-quality, demo, clinical, deployment, broad VEP, broad Carbon, "
            "or useful-planning claim.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-report", type=Path, default=DEFAULT_OVERLAP_REPORT)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--residual-report", type=Path, default=DEFAULT_RESIDUAL_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    try:
        report = build_final_report(
            overlap_report_path=args.overlap_report,
            cache_manifest_path=args.cache_manifest,
            baseline_report_path=args.baseline_report,
            residual_report_path=args.residual_report,
            generated_at=args.generated_at,
        )
        write_report(report=report, output_json=args.output_json, output_md=args.output_md)
    except GenoLeWMError as exc:
        print(exc.to_json(), file=sys.stderr)
        return exit_code_for(exc)
    return 0


def _assert_expected_chain(
    *,
    overlap: Mapping[str, Any],
    cache: Mapping[str, Any],
    baseline: Mapping[str, Any],
    residual: Mapping[str, Any],
) -> None:
    if overlap.get("decision") != "go_traitgym_native_borzoi":
        raise InputError("final report requires a passing overlap report")
    if int(cache.get("row_count", 0)) != int(overlap["traitgym_native_alignment"]["usable_rows"]):
        raise InputError("cache row count does not match overlap usable rows")
    if baseline.get("decision") != "go_residual_model":
        raise InputError("final report requires a completed baseline gate")
    if residual.get("decision") not in {"residual_lift_candidate", "residual_no_lift"}:
        raise InputError("residual report decision is unsupported")


def _receipt(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    return {
        "path": _repo_relative(path),
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "generated_by": payload.get("generated_by"),
    }


def _metric_index(metrics: Sequence[Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(_require_mapping(metric, "metrics[]")["name"]): _require_mapping(metric, "metrics[]")
        for metric in metrics
    }


def _metric_value(metrics: Mapping[str, Mapping[str, Any]], name: str) -> str:
    return f"{float(metrics[name]['value']):.4f}"


def _delta_value(metrics: Mapping[str, Mapping[str, Any]], name: str) -> str:
    return f"{float(metrics[name]['delta']):.4f}"


def _delta_ci(metrics: Mapping[str, Mapping[str, Any]], name: str) -> str:
    low, high = [float(item) for item in _require_list(metrics[name]["ci95"])]
    return f"[{low:.4f}, {high:.4f}]"


def _load_json(path: Path, *, label: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"{label} not found", details={"path": str(path)}) from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object", details={"path": str(path)})
    return cast(JsonDict, payload)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _require_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise InputError("expected a JSON array")
    return value


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
