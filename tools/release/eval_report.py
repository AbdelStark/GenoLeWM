# SPDX-License-Identifier: Apache-2.0
"""Generate a release evaluation report from measured metric JSON.

This helper is intentionally narrower than the future ``geno-lewm-eval-all``
runner. It does not run benchmarks or invent conclusions; it turns a
machine-readable metrics payload into the Markdown artifact required by the
paper/demo release package verifier.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance.hashing import looks_like_sha256

SCHEMA_VERSION: Final = "1.0.0"
EVAL_GENERATED_BY: Final = "geno-lewm-eval"
EVAL_ALL_GENERATED_BY: Final = "geno-lewm-eval-all"
ALLOWED_GENERATORS: Final = frozenset({EVAL_GENERATED_BY, EVAL_ALL_GENERATED_BY})
REQUIRED_ARTIFACTS: Final = (
    "checkpoint",
    "config",
    "dataset_manifest",
    "eval_config",
    "efficiency_report",
)
PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One measured metric row for the release evaluation report."""

    name: str
    value: float
    split: str
    unit: str
    higher_is_better: bool
    baseline: str | None = None
    baseline_value: float | None = None
    delta_vs_baseline: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None
    notes: str | None = None
    evaluated_variant_keys_sha256: str | None = None
    baseline_evaluated_variant_keys_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EvalReportInput:
    """Validated input payload for a generated eval report."""

    schema_version: str
    generated_by: str
    generated_at: str
    model_id: str
    model_release: str
    dataset_snapshot: str
    commit: str
    hardware: str
    metrics: tuple[MetricResult, ...]
    artifacts: tuple[tuple[str, str], ...]
    limitations: tuple[str, ...]
    negative_findings: tuple[str, ...]
    conclusions: tuple[str, ...]


def load_report_input(
    path: Path,
    *,
    allow_placeholders: bool = False,
) -> EvalReportInput:
    """Load and validate a metrics payload from ``path``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError("failed to read metrics JSON", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "metrics JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return parse_report_input(payload, allow_placeholders=allow_placeholders)


def parse_report_input(
    payload: Any,
    *,
    allow_placeholders: bool = False,
) -> EvalReportInput:
    """Validate a decoded metrics payload."""
    if not isinstance(payload, dict):
        raise InputError("metrics payload must be a JSON object")
    schema_version = _required_text(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InputError(
            "unsupported eval-report schema version",
            details={"expected": SCHEMA_VERSION, "observed": schema_version},
        )

    generated_by = _required_text(payload, "generated_by")
    if generated_by not in ALLOWED_GENERATORS:
        raise InputError(
            "eval metrics generated_by is invalid",
            details={"allowed": sorted(ALLOWED_GENERATORS), "observed": generated_by},
        )
    generated_at = _optional_text(payload, "generated_at") or _utc_now()
    model_id = _required_text(payload, "model_id")
    if not looks_like_sha256(model_id):
        raise InputError("model_id must be a sha256:<64hex> identifier")

    metrics = _parse_metrics(payload.get("metrics"))
    artifacts = _parse_artifacts(payload.get("artifacts"))
    _require_baseline_artifacts(metrics, artifacts)
    limitations = _parse_text_list(payload.get("limitations"), field="limitations")
    negative_findings = _parse_text_list(
        payload.get("negative_findings"),
        field="negative_findings",
    )
    conclusions = _parse_text_list(payload.get("conclusions"), field="conclusions")

    text_fields = {
        "generated_by": generated_by,
        "generated_at": generated_at,
        "model_id": model_id,
        "model_release": _required_text(payload, "model_release"),
        "dataset_snapshot": _required_text(payload, "dataset_snapshot"),
        "commit": _required_text(payload, "commit"),
        "hardware": _required_text(payload, "hardware"),
    }
    if not allow_placeholders:
        _reject_placeholders(text_fields)
        _reject_placeholders(dict(artifacts), prefix="artifacts.")
        _reject_placeholders({f"limitations[{i}]": v for i, v in enumerate(limitations)})
        _reject_placeholders(
            {f"negative_findings[{i}]": v for i, v in enumerate(negative_findings)}
        )
        _reject_placeholders({f"conclusions[{i}]": v for i, v in enumerate(conclusions)})
        for index, metric in enumerate(metrics):
            _reject_placeholders(
                {
                    "name": metric.name,
                    "split": metric.split,
                    "unit": metric.unit,
                    "baseline": metric.baseline or "",
                    "notes": metric.notes or "",
                },
                prefix=f"metrics[{index}].",
            )
    _require_metric_conclusions(metrics, conclusions)

    return EvalReportInput(
        schema_version=schema_version,
        generated_by=generated_by,
        generated_at=generated_at,
        model_id=model_id,
        model_release=text_fields["model_release"],
        dataset_snapshot=text_fields["dataset_snapshot"],
        commit=text_fields["commit"],
        hardware=text_fields["hardware"],
        metrics=metrics,
        artifacts=artifacts,
        limitations=limitations,
        negative_findings=negative_findings,
        conclusions=conclusions,
    )


def render_report(report: EvalReportInput) -> str:
    """Render ``report`` as Markdown."""
    lines = [
        "# Evaluation Report",
        "",
        f"Generated by: {report.generated_by}",
        f"Generated: {report.generated_at}",
        "",
        "## Summary",
        "",
        f"- Model release: {report.model_release}",
        f"- Model id: {report.model_id}",
        f"- Dataset snapshot: {report.dataset_snapshot}",
        f"- Commit: {report.commit}",
        f"- Hardware: {report.hardware}",
        "- Result status: measured metrics from the input JSON payload.",
        "- Claim boundary: planned targets are not reported as results.",
        "",
        "## Results",
        "",
        (
            "| Metric | Split | Value | Unit | Baseline | Baseline Value | "
            "Delta vs Baseline | Direction | CI | N | Notes |"
        ),
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: | --- |",
    ]
    lines.extend(_metric_row(metric) for metric in report.metrics)
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Artifact | Path or identifier |",
            "| --- | --- |",
        ]
    )
    for key, value in report.artifacts:
        lines.append(f"| {_md_cell(key)} | {_md_cell(value)} |")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    lines.extend(["", "## Negative Findings", ""])
    lines.extend(f"- {item}" for item in report.negative_findings)
    lines.extend(["", "## Conclusions", ""])
    lines.extend(f"- {item}" for item in report.conclusions)
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report_input = load_report_input(
            args.metrics_json,
            allow_placeholders=args.allow_placeholders,
        )
        rendered = render_report(report_input)
        args.output.write_text(rendered, encoding="utf-8")
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output}\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate eval_report.md from measured metrics JSON.",
    )
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder wording for local drafts. Do not use for releases.",
    )
    return parser


def _parse_metrics(raw: Any) -> tuple[MetricResult, ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError("metrics must be a non-empty list")
    metrics: list[MetricResult] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError("metrics entries must be objects", details={"index": index})
        value = _required_number(item, "value", prefix=f"metrics[{index}].")
        ci_low = _optional_number(item, "ci_low", prefix=f"metrics[{index}].")
        ci_high = _optional_number(item, "ci_high", prefix=f"metrics[{index}].")
        if (ci_low is None) != (ci_high is None):
            raise InputError(
                "ci_low and ci_high must be supplied together",
                details={"field": f"metrics[{index}]"},
            )
        if ci_low is not None and ci_high is not None and ci_low > ci_high:
            raise InputError(
                "ci_low cannot exceed ci_high",
                details={"field": f"metrics[{index}]"},
            )
        n_value = item.get("n")
        if n_value is not None and (
            isinstance(n_value, bool) or not isinstance(n_value, int) or n_value <= 0
        ):
            raise InputError(
                "n must be a positive integer",
                details={"field": f"metrics[{index}].n"},
            )
        higher_is_better = item.get("higher_is_better")
        if not isinstance(higher_is_better, bool):
            raise InputError(
                "higher_is_better must be a boolean",
                details={"field": f"metrics[{index}].higher_is_better"},
            )
        baseline = _optional_text(item, "baseline", prefix=f"metrics[{index}].")
        baseline_value = _optional_number(item, "baseline_value", prefix=f"metrics[{index}].")
        delta_vs_baseline = _optional_number(item, "delta_vs_baseline", prefix=f"metrics[{index}].")
        evaluated_variant_keys_sha256 = _optional_sha256(
            item,
            "evaluated_variant_keys_sha256",
            prefix=f"metrics[{index}].",
        )
        baseline_evaluated_variant_keys_sha256 = _optional_sha256(
            item,
            "baseline_evaluated_variant_keys_sha256",
            prefix=f"metrics[{index}].",
        )
        baseline_fields = (
            baseline is not None,
            baseline_value is not None,
            delta_vs_baseline is not None,
        )
        if any(baseline_fields) and not all(baseline_fields):
            raise InputError(
                "baseline, baseline_value, and delta_vs_baseline must be supplied together",
                details={"field": f"metrics[{index}]"},
            )
        _require_baseline_variant_hashes(
            index=index,
            baseline=baseline,
            baseline_value=baseline_value,
            delta_vs_baseline=delta_vs_baseline,
            evaluated_variant_keys_sha256=evaluated_variant_keys_sha256,
            baseline_evaluated_variant_keys_sha256=baseline_evaluated_variant_keys_sha256,
        )
        metrics.append(
            MetricResult(
                name=_required_text(item, "name", prefix=f"metrics[{index}]."),
                value=value,
                split=_required_text(item, "split", prefix=f"metrics[{index}]."),
                unit=_required_text(item, "unit", prefix=f"metrics[{index}]."),
                higher_is_better=higher_is_better,
                baseline=baseline,
                baseline_value=baseline_value,
                delta_vs_baseline=delta_vs_baseline,
                ci_low=ci_low,
                ci_high=ci_high,
                n=n_value,
                notes=_optional_text(item, "notes", prefix=f"metrics[{index}]."),
                evaluated_variant_keys_sha256=evaluated_variant_keys_sha256,
                baseline_evaluated_variant_keys_sha256=baseline_evaluated_variant_keys_sha256,
            )
        )
    return tuple(metrics)


def _require_baseline_variant_hashes(
    *,
    index: int,
    baseline: str | None,
    baseline_value: float | None,
    delta_vs_baseline: float | None,
    evaluated_variant_keys_sha256: str | None,
    baseline_evaluated_variant_keys_sha256: str | None,
) -> None:
    has_baseline_metrics = (
        baseline is not None and baseline_value is not None and delta_vs_baseline is not None
    )
    if not has_baseline_metrics:
        if baseline_evaluated_variant_keys_sha256 is not None:
            raise InputError(
                "baseline_evaluated_variant_keys_sha256 requires baseline metrics",
                details={"field": f"metrics[{index}].baseline_evaluated_variant_keys_sha256"},
            )
        return
    if evaluated_variant_keys_sha256 is None or baseline_evaluated_variant_keys_sha256 is None:
        raise InputError(
            (
                "baseline metrics require evaluated_variant_keys_sha256 and "
                "baseline_evaluated_variant_keys_sha256"
            ),
            details={"field": f"metrics[{index}]"},
        )
    if evaluated_variant_keys_sha256 != baseline_evaluated_variant_keys_sha256:
        raise InputError(
            "baseline metric variant-key hashes must match",
            details={
                "field": f"metrics[{index}]",
                "evaluated_variant_keys_sha256": evaluated_variant_keys_sha256,
                "baseline_evaluated_variant_keys_sha256": (baseline_evaluated_variant_keys_sha256),
            },
        )


def _parse_artifacts(raw: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, dict) or not raw:
        raise InputError("artifacts must be a non-empty object")
    missing = sorted(set(REQUIRED_ARTIFACTS) - set(raw))
    if missing:
        raise InputError("artifacts are missing required keys", details={"missing": missing})
    artifacts: list[tuple[str, str]] = []
    for key in sorted(raw):
        if not isinstance(key, str) or not key:
            raise InputError("artifact keys must be non-empty strings")
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise InputError("artifact values must be non-empty strings", details={"field": key})
        artifacts.append((key, value.strip()))
    return tuple(artifacts)


def _require_baseline_artifacts(
    metrics: tuple[MetricResult, ...],
    artifacts: tuple[tuple[str, str], ...],
) -> None:
    baseline_names = sorted(
        {
            metric.baseline
            for metric in metrics
            if metric.baseline is not None
            and metric.baseline_value is not None
            and metric.delta_vs_baseline is not None
        }
    )
    if not baseline_names:
        return
    artifact_keys = {key for key, _value in artifacts}
    if not any(_is_baseline_artifact_key(key) for key in artifact_keys):
        raise InputError(
            "baseline metrics require a baseline artifact",
            details={"baselines": baseline_names, "artifact_keys": sorted(artifact_keys)},
        )


def _is_baseline_artifact_key(key: str) -> bool:
    return key in {"baseline_scores", "baseline_rollout_states"} or key.endswith(
        (".baseline_scores", ".baseline_rollout_states")
    )


def _require_metric_conclusions(
    metrics: tuple[MetricResult, ...],
    conclusions: tuple[str, ...],
) -> None:
    raw_text = "\n".join(conclusions)
    normalized_text = _normalize_for_reference(raw_text)
    normalized_text_without_metric_names = _remove_metric_name_references(
        normalized_text,
        metrics,
    )
    missing_metric_names = [
        metric.name
        for metric in metrics
        if _normalize_for_reference(metric.name) not in normalized_text
    ]
    missing_splits = [
        f"{metric.name}:{metric.split}"
        for metric in metrics
        if _normalize_for_reference(metric.split) not in normalized_text_without_metric_names
    ]
    missing_values = [
        f"{metric.name}:{_format_number(metric.value)}"
        for metric in metrics
        if not _number_is_referenced(metric.value, raw_text)
    ]
    missing_baselines = [
        f"{metric.name}:{metric.baseline}"
        for metric in metrics
        if metric.baseline is not None
        and _normalize_for_reference(metric.baseline) not in normalized_text
    ]
    missing_baseline_deltas = [
        f"{metric.name}:{_format_number(metric.delta_vs_baseline)}"
        for metric in metrics
        if metric.delta_vs_baseline is not None
        and not _number_is_referenced(metric.delta_vs_baseline, raw_text)
    ]
    if (
        missing_metric_names
        or missing_splits
        or missing_values
        or missing_baselines
        or missing_baseline_deltas
    ):
        raise InputError(
            "conclusions must reference each measured metric name, split, value, and baseline delta",
            details={
                "missing_metric_names": sorted(set(missing_metric_names)),
                "missing_splits": sorted(set(missing_splits)),
                "missing_values": sorted(set(missing_values)),
                "missing_baselines": sorted(set(missing_baselines)),
                "missing_baseline_deltas": sorted(set(missing_baseline_deltas)),
            },
        )


def _parse_text_list(raw: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError(f"{field} must be a non-empty list")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise InputError(
                f"{field} entries must be non-empty strings",
                details={"field": f"{field}[{index}]"},
            )
        values.append(item.strip())
    return tuple(values)


def _required_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{prefix}{key} must be a non-empty string")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{prefix}{key} must be a non-empty string when supplied")
    return value.strip()


def _optional_sha256(payload: dict[str, Any], key: str, *, prefix: str = "") -> str | None:
    value = _optional_text(payload, key, prefix=prefix)
    if value is None:
        return None
    if not looks_like_sha256(value):
        raise InputError(f"{prefix}{key} must be a sha256:<64hex> identifier")
    return value


def _required_number(payload: dict[str, Any], key: str, *, prefix: str = "") -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise InputError(f"{prefix}{key} must be a finite number")
    return float(value)


def _optional_number(payload: dict[str, Any], key: str, *, prefix: str = "") -> float | None:
    if key not in payload:
        return None
    return _required_number(payload, key, prefix=prefix)


def _reject_placeholders(values: dict[str, str], *, prefix: str = "") -> None:
    for key, value in values.items():
        if PLACEHOLDER_RE.search(value):
            raise InputError(
                "placeholder text is not allowed in release eval reports",
                details={"field": f"{prefix}{key}"},
            )


def _normalize_for_reference(value: str) -> str:
    return " ".join(re.split(r"[^a-z0-9]+", value.lower())).strip()


def _remove_metric_name_references(
    normalized_text: str,
    metrics: tuple[MetricResult, ...],
) -> str:
    text = f" {normalized_text} "
    normalized_names = sorted(
        {_normalize_for_reference(metric.name) for metric in metrics},
        key=len,
        reverse=True,
    )
    for name in normalized_names:
        text = re.sub(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", " ", text)
    return " ".join(text.split())


def _metric_row(metric: MetricResult) -> str:
    direction = "higher" if metric.higher_is_better else "lower"
    ci = "-"
    if metric.ci_low is not None and metric.ci_high is not None:
        ci = f"{_format_number(metric.ci_low)} to {_format_number(metric.ci_high)}"
    n = "-" if metric.n is None else str(metric.n)
    baseline_value = "-" if metric.baseline_value is None else _format_number(metric.baseline_value)
    delta = "-" if metric.delta_vs_baseline is None else _format_number(metric.delta_vs_baseline)
    return (
        f"| {_md_cell(metric.name)} "
        f"| {_md_cell(metric.split)} "
        f"| {_format_number(metric.value)} "
        f"| {_md_cell(metric.unit)} "
        f"| {_md_cell(metric.baseline or 'not reported')} "
        f"| {baseline_value} "
        f"| {delta} "
        f"| {direction} "
        f"| {_md_cell(ci)} "
        f"| {n} "
        f"| {_md_cell(metric.notes or '')} |"
    )


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _number_is_referenced(value: float, text: str) -> bool:
    return any(
        re.search(rf"(?<![0-9A-Za-z_]){re.escape(form)}(?![0-9A-Za-z_])", text)
        for form in _number_reference_forms(value)
    )


def _number_reference_forms(value: float) -> tuple[str, ...]:
    forms = {
        _format_number(value),
        str(value),
        f"{value:.6f}".rstrip("0").rstrip("."),
    }
    if float(value).is_integer():
        forms.add(str(int(value)))
        forms.add(f"{value:.1f}")
    return tuple(sorted(form for form in forms if form))


def _md_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", r"\|").strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
