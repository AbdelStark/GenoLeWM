# SPDX-License-Identifier: Apache-2.0
"""Generate a first-experiment paper draft from release artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import Manifest, load_manifest
from tools.demo.terminal_inference import DEMO_MANIFEST_NAME, summarize_vcf_input
from tools.release.batch_receipt_report import REPORT_NAME as BATCH_RECEIPT_REPORT_NAME
from tools.release.dataset_integrity import DEFAULT_REPORT_NAME as DATASET_INTEGRITY_NAME
from tools.release.dataset_snapshot import (
    INPUT_CHECK_REPORT_NAME as DATASET_INPUT_CHECK_REPORT_NAME,
    REPORT_NAME as DATASET_SNAPSHOT_REPORT_NAME,
)
from tools.release.efficiency_report import (
    REPORT_NAME as EFFICIENCY_REPORT_NAME,
    load_efficiency_report,
)
from tools.release.eval_report import load_report_input, render_report
from tools.release.model_package import EVAL_METRICS_NAME, MODEL_PACKAGE_NAME, load_model_package
from tools.release.runtime_preflight import REPORT_NAME as RUNTIME_PREFLIGHT_REPORT_NAME

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.paper_draft"
PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum|go here)\b",
    re.IGNORECASE,
)
UTC_TIMESTAMP_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True, slots=True)
class PaperDraftReport:
    """Files and identities covered by the generated paper draft."""

    path: Path
    model_id: str
    model_release: str
    model_package: Path
    dataset_snapshot: str
    dataset_input_check_report: Path
    dataset_snapshot_report: Path
    efficiency_report: Path
    demo_transcript: Path
    demo_manifest: Path
    runtime_preflight_report: Path
    batch_receipt_report: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path.name,
            "model_id": self.model_id,
            "model_release": self.model_release,
            "model_package": f"model/{self.model_package.name}",
            "dataset_snapshot": self.dataset_snapshot,
            "dataset_input_check_report": f"dataset/{self.dataset_input_check_report.name}",
            "dataset_snapshot_report": f"dataset/{self.dataset_snapshot_report.name}",
            "efficiency_report": f"model/{self.efficiency_report.name}",
            "demo_transcript": f"demo/{self.demo_transcript.name}",
            "demo_manifest": f"demo/{self.demo_manifest.name}",
            "runtime_preflight_report": f"demo/{self.runtime_preflight_report.name}",
            "batch_receipt_report": f"demo/{self.batch_receipt_report.name}",
        }


@dataclass(frozen=True, slots=True)
class _PaperDraftArtifacts:
    model_root: Path
    dataset_root: Path
    demo_root: Path
    manifest: Manifest
    model_package_path: Path
    dataset_manifest: dict[str, Any]
    dataset_snapshot: str
    dataset_input_check_report_path: Path
    dataset_snapshot_report_path: Path
    eval_report_path: Path
    eval_report: str
    eval_config_path: Path
    efficiency_report_path: Path
    efficiency_report: dict[str, Any]
    batch_report_path: Path
    batch_report: dict[str, Any]
    runtime_preflight_path: Path
    runtime_preflight: dict[str, Any]
    transcript_path: Path
    transcript: str
    demo_manifest_path: Path
    demo_manifest: dict[str, Any]
    training_manifest: dict[str, Any]


def build_paper_draft(
    *,
    model_dir: str | Path,
    dataset_dir: str | Path,
    demo_dir: str | Path,
    output: str | Path,
    title: str = "GenoLeWM First Experiment Report",
    generated_at: str | None = None,
    allow_placeholders: bool = False,
) -> PaperDraftReport:
    """Generate a Markdown paper/report draft from verified release artifacts."""
    artifacts = _load_paper_draft_artifacts(
        model_dir=Path(model_dir),
        dataset_dir=Path(dataset_dir),
        demo_dir=Path(demo_dir),
    )
    text = _render_paper_draft(
        title=title,
        generated_at=generated_at or _utc_now(),
        artifacts=artifacts,
    )
    if not allow_placeholders:
        _reject_placeholders(text)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return PaperDraftReport(
        path=output_path,
        model_id=artifacts.manifest.model_id(),
        model_release=artifacts.manifest.release_id,
        model_package=artifacts.model_package_path,
        dataset_snapshot=artifacts.dataset_snapshot,
        dataset_input_check_report=artifacts.dataset_input_check_report_path,
        dataset_snapshot_report=artifacts.dataset_snapshot_report_path,
        efficiency_report=artifacts.efficiency_report_path,
        demo_transcript=artifacts.transcript_path,
        demo_manifest=artifacts.demo_manifest_path,
        runtime_preflight_report=artifacts.runtime_preflight_path,
        batch_receipt_report=artifacts.batch_report_path,
    )


def render_paper_draft(
    *,
    model_dir: str | Path,
    dataset_dir: str | Path,
    demo_dir: str | Path,
    title: str = "GenoLeWM First Experiment Report",
    generated_at: str,
) -> str:
    """Render the expected paper draft text from the current release artifacts."""
    artifacts = _load_paper_draft_artifacts(
        model_dir=Path(model_dir),
        dataset_dir=Path(dataset_dir),
        demo_dir=Path(demo_dir),
    )
    return _render_paper_draft(
        title=title,
        generated_at=generated_at,
        artifacts=artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = build_paper_draft(
            model_dir=args.model_dir,
            dataset_dir=args.dataset_dir,
            demo_dir=args.demo_dir,
            output=args.output,
            title=args.title,
            allow_placeholders=args.allow_placeholders,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a first-experiment paper draft from release artifacts.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="GenoLeWM First Experiment Report")
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder wording for local drafts only.",
    )
    return parser


def _render_paper_draft(
    *,
    title: str,
    generated_at: str,
    artifacts: _PaperDraftArtifacts,
) -> str:
    _require_utc_timestamp(generated_at, field="generated_at")
    manifest = artifacts.manifest
    model_id = manifest.model_id()
    results = _extract_markdown_section(artifacts.eval_report, "Results")
    conclusions = _extract_markdown_section(artifacts.eval_report, "Conclusions")
    limitations = _extract_markdown_section(artifacts.eval_report, "Limitations")
    negative_findings = _extract_markdown_section(artifacts.eval_report, "Negative Findings")
    commit = _optional_text_from_mapping(artifacts.training_manifest, "commit") or "not recorded"
    command = _optional_text_from_mapping(artifacts.training_manifest, "command") or "not recorded"
    record_count = artifacts.batch_report.get("records")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count <= 0:
        raise InputError("batch receipt report records must be a positive integer")
    if artifacts.runtime_preflight.get("ok") is not True:
        raise InputError("runtime preflight report must have ok=true")
    if artifacts.demo_manifest.get("status") != "passed":
        raise InputError("terminal demo manifest must have status=passed")
    demo_vcf_summary = _verified_demo_vcf_summary(artifacts.demo_manifest, artifacts.demo_root)
    measurements = artifacts.efficiency_report.get("measurements")
    if not isinstance(measurements, dict) or not measurements:
        raise InputError("efficiency report measurements must be a non-empty object")
    lines = [
        f"# {title}",
        "",
        f"Generated by: {GENERATED_BY}",
        f"Generated: {generated_at}",
        "",
        "## Abstract",
        "",
        (
            "This draft is generated from the GenoLeWM release artifact set. "
            "Its results and conclusions are sourced from the generated evaluation report; "
            "the draft does not add claims beyond those measured artifacts."
        ),
        "",
        "## Citation Metadata",
        "",
        f"- Title: {title}",
        f"- Report generator: `{GENERATED_BY}`",
        f"- Generated at: {generated_at}",
        f"- Model release: {manifest.release_id}",
        f"- Model version: {manifest.model_version}",
        f"- Model id: `{model_id}`",
        f"- Dataset snapshot: `{artifacts.dataset_snapshot}`",
        f"- Source metrics: `{EVAL_METRICS_NAME}`",
        f"- Eval config: `{_rel(artifacts.eval_config_path, artifacts.model_root)}`",
        f"- Evaluation report: `{_rel(artifacts.eval_report_path, artifacts.model_root)}`",
        f"- Efficiency report: `{_rel(artifacts.efficiency_report_path, artifacts.model_root)}`",
        f"- Terminal demo manifest: `{_rel(artifacts.demo_manifest_path, artifacts.demo_root)}`",
        "",
        "## Experiment",
        "",
        f"- Model release: {manifest.release_id}",
        f"- Model version: {manifest.model_version}",
        f"- Model id: {model_id}",
        f"- Dataset snapshot: {artifacts.dataset_snapshot}",
        f"- Encoder: {manifest.encoder.id}",
        f"- Encoder revision: {manifest.encoder.revision}",
        f"- Training commit: {commit}",
        f"- Training command: `{command}`",
        "",
        "## Methods",
        "",
        "### Model",
        "",
        (
            "The checkpoint package is defined by `manifest.json`, "
            f"`{MODEL_PACKAGE_NAME}`, `model_card.md`, "
            "`training_run_manifest.json`, and `SHA256SUMS`."
        ),
        "",
        "### Data",
        "",
        (
            f"The dataset package is `{artifacts.dataset_snapshot}` and is defined by "
            "`dataset_package.json`, `dataset_manifest.json`, "
            f"`{DATASET_INPUT_CHECK_REPORT_NAME}`, "
            f"`{DATASET_SNAPSHOT_REPORT_NAME}`, `data_card.md`, and "
            f"`{DATASET_INTEGRITY_NAME}`."
        ),
        "",
    ]
    lines.extend(_dataset_split_lines(artifacts.dataset_manifest))
    lines.extend(
        [
            "",
            "### Evaluation",
            "",
            f"The source metrics artifact is `{EVAL_METRICS_NAME}`.",
            (
                "The effective evaluation config is "
                f"`{_rel(artifacts.eval_config_path, artifacts.model_root)}`."
            ),
            (
                "The generated evaluation report is "
                f"`{_rel(artifacts.eval_report_path, artifacts.model_root)}`."
            ),
            (
                "The generated efficiency report is "
                f"`{_rel(artifacts.efficiency_report_path, artifacts.model_root)}`."
            ),
            "The Results, Conclusions, and Limitations sections below are copied from that report.",
            "",
            "### Demo Evidence",
            "",
            f"The terminal transcript is `{_rel(artifacts.transcript_path, artifacts.demo_root)}`.",
            (
                "The terminal demo manifest is "
                f"`{_rel(artifacts.demo_manifest_path, artifacts.demo_root)}`."
            ),
            (
                "The runtime preflight report is "
                f"`{_rel(artifacts.runtime_preflight_path, artifacts.demo_root)}`."
            ),
            (
                "The batch receipt report is "
                f"`{_rel(artifacts.batch_report_path, artifacts.demo_root)}`."
            ),
            (
                "The packaged demo VCF contains "
                f"{demo_vcf_summary['variant_records']} variant record(s), "
                f"{demo_vcf_summary['alternate_alleles']} alternate allele(s), and contig(s) "
                f"{_field_list(_string_sequence(demo_vcf_summary['contigs']))}."
            ),
            f"First demo variant(s): {_variant_summary(demo_vcf_summary['first_variants'])}.",
            f"The demo batch contains {record_count} scored JSONL row(s).",
            f"The transcript status is {_transcript_status(artifacts.transcript)}.",
            "",
            "## Results",
            "",
            results,
            "",
            "## Conclusions",
            "",
            conclusions,
            "",
            "## Negative Findings",
            "",
            negative_findings,
            "",
            "## Limitations",
            "",
            limitations,
            "",
            "## Artifact Availability",
            "",
            "- Model package root: model release package",
            "- Model manifest: `manifest.json`",
            f"- Model package metadata: `{MODEL_PACKAGE_NAME}`",
            "- Model card: `model_card.md`",
            f"- Model id: `{model_id}`",
            "- Dataset package root: dataset release package",
            "- Dataset package metadata: `dataset_package.json`",
            "- Dataset manifest: `dataset_manifest.json`",
            f"- Dataset input-check report: `{DATASET_INPUT_CHECK_REPORT_NAME}`",
            f"- Dataset snapshot report: `{DATASET_SNAPSHOT_REPORT_NAME}`",
            f"- Dataset snapshot id: `{artifacts.dataset_snapshot}`",
            f"- Dataset integrity report: `{DATASET_INTEGRITY_NAME}`",
            f"- Source metrics: `{EVAL_METRICS_NAME}`",
            f"- Eval config: `{_rel(artifacts.eval_config_path, artifacts.model_root)}`",
            f"- Evaluation report: `{_rel(artifacts.eval_report_path, artifacts.model_root)}`",
            f"- Efficiency report: `{_rel(artifacts.efficiency_report_path, artifacts.model_root)}`",
            "- Demo package root: terminal demo package",
            f"- Terminal transcript: `{_rel(artifacts.transcript_path, artifacts.demo_root)}`",
            (
                "- Terminal demo manifest: "
                f"`{_rel(artifacts.demo_manifest_path, artifacts.demo_root)}`"
            ),
            (
                "- Runtime preflight report: "
                f"`{_rel(artifacts.runtime_preflight_path, artifacts.demo_root)}`"
            ),
            f"- Batch receipt report: `{_rel(artifacts.batch_report_path, artifacts.demo_root)}`",
            "",
            "## Reproducibility",
            "",
            "```console",
            "$ python -m tools.release.paper_package --model-dir MODEL_DIR --dataset-dir DATASET_DIR --demo-dir DEMO_DIR --paper-path PAPER.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _load_paper_draft_artifacts(
    *,
    model_dir: Path,
    dataset_dir: Path,
    demo_dir: Path,
) -> _PaperDraftArtifacts:
    manifest = load_manifest(model_dir / "manifest.json")
    model_package_path = model_dir / MODEL_PACKAGE_NAME
    load_model_package(model_package_path)
    dataset_manifest = _load_json_object(dataset_dir / "dataset_manifest.json", "dataset manifest")
    dataset_snapshot = _required_text(dataset_manifest, "snapshot_id")
    dataset_input_check_report_path = dataset_dir / DATASET_INPUT_CHECK_REPORT_NAME
    if not dataset_input_check_report_path.is_file():
        raise InputError(
            "dataset input-check report is missing",
            details={"path": str(dataset_input_check_report_path)},
        )
    dataset_snapshot_report_path = dataset_dir / DATASET_SNAPSHOT_REPORT_NAME
    if not dataset_snapshot_report_path.is_file():
        raise InputError(
            "dataset snapshot report is missing",
            details={"path": str(dataset_snapshot_report_path)},
        )
    eval_report_path = model_dir / manifest.eval.file
    eval_report = _read_text(eval_report_path, "eval report")
    eval_input = load_report_input(model_dir / EVAL_METRICS_NAME)
    expected_eval_report = render_report(eval_input)
    if eval_report != expected_eval_report:
        raise InputError(
            f"{manifest.eval.file} does not match render of {EVAL_METRICS_NAME}",
            details={
                "report_path": str(eval_report_path),
                "metrics_path": str(model_dir / EVAL_METRICS_NAME),
            },
        )
    eval_config_path = _eval_config_artifact_path(eval_input.artifacts, model_dir)
    efficiency_report_path = model_dir / EFFICIENCY_REPORT_NAME
    efficiency_report = load_efficiency_report(efficiency_report_path).to_dict()
    batch_report_path = demo_dir / BATCH_RECEIPT_REPORT_NAME
    batch_report = _load_json_object(batch_report_path, "batch receipt report")
    runtime_preflight_path = demo_dir / RUNTIME_PREFLIGHT_REPORT_NAME
    runtime_preflight = _load_json_object(runtime_preflight_path, "runtime preflight report")
    transcript_path = demo_dir / "terminal-demo-transcript.md"
    transcript = _read_text(transcript_path, "terminal transcript")
    demo_manifest_path = demo_dir / DEMO_MANIFEST_NAME
    demo_manifest = _load_json_object(demo_manifest_path, "terminal demo manifest")
    _verified_demo_vcf_summary(demo_manifest, demo_dir)
    training_manifest = _load_json_object(
        model_dir / "training_run_manifest.json",
        "training run manifest",
    )
    return _PaperDraftArtifacts(
        model_root=model_dir,
        dataset_root=dataset_dir,
        demo_root=demo_dir,
        manifest=manifest,
        model_package_path=model_package_path,
        dataset_manifest=dataset_manifest,
        dataset_snapshot=dataset_snapshot,
        dataset_input_check_report_path=dataset_input_check_report_path,
        dataset_snapshot_report_path=dataset_snapshot_report_path,
        eval_report_path=eval_report_path,
        eval_report=eval_report,
        eval_config_path=eval_config_path,
        efficiency_report_path=efficiency_report_path,
        efficiency_report=efficiency_report,
        batch_report_path=batch_report_path,
        batch_report=batch_report,
        runtime_preflight_path=runtime_preflight_path,
        runtime_preflight=runtime_preflight,
        transcript_path=transcript_path,
        transcript=transcript,
        demo_manifest_path=demo_manifest_path,
        demo_manifest=demo_manifest,
        training_manifest=training_manifest,
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"failed to read {label}", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{label} is invalid JSON",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object", details={"path": str(path)})
    return payload


def _eval_config_artifact_path(artifacts: tuple[tuple[str, str], ...], model_dir: Path) -> Path:
    by_key = dict(artifacts)
    raw_path = by_key.get("eval_config")
    if raw_path is None:
        raise InputError("eval metrics artifacts must include eval_config")
    if "://" in raw_path:
        raise InputError("eval_config artifact path must be package-relative, not a URL")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise InputError("eval_config artifact path must be relative and stay inside the package")
    if relative.parts[0] == "dataset":
        raise InputError("eval_config artifact must be model-package local")
    if relative.parts[0] == "model":
        if len(relative.parts) == 1:
            raise InputError("eval_config artifact path must name a file")
        relative = Path(*relative.parts[1:])
    path = model_dir / relative
    if not path.is_file():
        raise InputError("eval_config artifact is missing", details={"path": str(path)})
    return path


def _read_text(path: Path, label: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"failed to read {label}", details={"path": str(path)}) from exc
    if not text.strip():
        raise InputError(f"{label} must be non-empty", details={"path": str(path)})
    return text


def _extract_markdown_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"(?ms)^## {re.escape(heading)}\s*\n(?P<body>.*?)(?=^## |\Z)",
    )
    match = pattern.search(text)
    if match is None:
        raise InputError(f"eval report is missing ## {heading}")
    body = match.group("body").strip()
    if not body:
        raise InputError(f"eval report ## {heading} section is empty")
    return body


def _dataset_split_lines(dataset_manifest: dict[str, Any]) -> list[str]:
    splits = dataset_manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise InputError("dataset manifest splits must be a non-empty object")
    lines = ["| Split | Records | Description |", "| --- | ---: | --- |"]
    for name, raw in sorted(splits.items()):
        if not isinstance(name, str) or not name:
            raise InputError("dataset split names must be non-empty strings")
        if not isinstance(raw, dict):
            raise InputError("dataset split entries must be objects", details={"split": name})
        records = raw.get("records", "-")
        if isinstance(records, bool) or not isinstance(records, int | str):
            raise InputError(
                "dataset split records must be an integer or string", details={"split": name}
            )
        description = raw.get("description", "")
        if not isinstance(description, str):
            raise InputError("dataset split description must be a string", details={"split": name})
        lines.append(f"| {_md_cell(name)} | {_md_cell(str(records))} | {_md_cell(description)} |")
    return lines


def _verified_demo_vcf_summary(
    demo_manifest: dict[str, Any],
    demo_root: Path,
) -> dict[str, object]:
    inputs = demo_manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise InputError("terminal demo manifest inputs must be an object")
    recorded = inputs.get("vcf_summary")
    if not isinstance(recorded, dict):
        raise InputError("terminal demo manifest must include inputs.vcf_summary")
    vcf_path = _demo_manifest_input_path(inputs.get("vcf"), demo_root, label="vcf")
    expected = summarize_vcf_input(vcf_path)
    if recorded != expected:
        raise InputError(
            "terminal demo manifest vcf_summary does not match the packaged demo VCF",
            details={"vcf": str(vcf_path)},
        )
    return expected


def _demo_manifest_input_path(raw: object, demo_root: Path, *, label: str) -> Path:
    if not isinstance(raw, dict):
        raise InputError(f"terminal demo manifest inputs.{label} identity is required")
    raw_path = raw.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise InputError(f"terminal demo manifest inputs.{label}.path must be non-empty")
    path = _resolve_demo_path(raw_path, demo_root)
    if not _path_is_within(path, demo_root):
        raise InputError(
            f"terminal demo manifest inputs.{label}.path must stay inside the demo package",
            details={"path": str(path)},
        )
    if not path.is_file():
        raise InputError(
            f"terminal demo manifest inputs.{label}.path is missing",
            details={"path": str(path)},
        )
    return path


def _resolve_demo_path(raw_path: str, demo_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    candidates = (demo_root / candidate, demo_root.parent / candidate)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _variant_summary(value: object) -> str:
    if not isinstance(value, list):
        return "-"
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        chrom = item.get("chrom")
        pos = item.get("pos")
        ref = item.get("ref")
        alts = item.get("alts")
        if not isinstance(chrom, str) or not isinstance(pos, int) or not isinstance(ref, str):
            continue
        alt_text = "/".join(_string_sequence(alts))
        if alt_text:
            parts.append(f"{chrom}:{pos}:{ref}>{alt_text}")
    return ", ".join(parts) if parts else "-"


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _field_list(fields: tuple[str, ...]) -> str:
    return ", ".join(fields) if fields else "-"


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_text_from_mapping(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _transcript_status(text: str) -> str:
    match = re.search(r"(?m)^- Status: (?P<status>\w+)$", text)
    if match is None:
        raise InputError("terminal transcript is missing status marker")
    return match.group("status")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _reject_placeholders(text: str) -> None:
    if PLACEHOLDER_RE.search(text):
        raise InputError("placeholder text is not allowed in generated paper drafts")


def _require_utc_timestamp(value: str, *, field: str) -> None:
    if UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise InputError(f"{field} must be a UTC ISO-8601 timestamp ending in Z")


def _md_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", r"\|").strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
