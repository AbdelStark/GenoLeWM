# SPDX-License-Identifier: Apache-2.0
"""Build release model-card and checksum artifacts from a checkpoint manifest."""

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
from geno_lewm.provenance import Manifest, load_manifest, sha256_file
from geno_lewm.training.preflight import REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME
from tools.release.efficiency_report import (
    REPORT_NAME as EFFICIENCY_REPORT_NAME,
    EfficiencyReport,
    load_efficiency_report,
)
from tools.release.eval_report import EvalReportInput, load_report_input, render_report
from tools.release.training_run import (
    CARD_NAME as TRAINING_RUN_CARD_NAME,
    CHECKSUMS_NAME as TRAINING_RUN_CHECKSUMS_NAME,
    MANIFEST_NAME as TRAINING_RUN_MANIFEST_NAME,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.model_package"
MODEL_PACKAGE_NAME: Final = "model_package.json"
EVAL_METRICS_NAME: Final = "eval_metrics.json"
EVAL_CONFIG_NAME: Final = "eval_config.effective.yaml"
CALIBRATION_REPORT_NAME: Final = "calibration_report.json"
CALIBRATION_REPORT_GENERATED_BY: Final = "tools.release.build_calibration"
PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum)\b",
    re.IGNORECASE,
)
GENERATED_FILES: Final = frozenset({MODEL_PACKAGE_NAME, "model_card.md", "SHA256SUMS"})
REQUIRED_RELEASE_EVIDENCE_FILES: Final = (
    TRAINING_PREFLIGHT_REPORT_NAME,
    TRAINING_RUN_MANIFEST_NAME,
    TRAINING_RUN_CARD_NAME,
    TRAINING_RUN_CHECKSUMS_NAME,
)


@dataclass(frozen=True, slots=True)
class ModelPackage:
    """Validated release metadata for a checkpoint package."""

    schema_version: str
    generated_by: str
    generated_at: str
    summary: str
    data: tuple[str, ...]
    hardware: tuple[str, ...]
    license: str
    intended_use: str
    limitations: tuple[str, ...]
    training: tuple[str, ...]
    evaluation: tuple[str, ...]
    runtime: tuple[str, ...]
    release_notes: tuple[str, ...]
    extra_files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "data": list(self.data),
            "hardware": list(self.hardware),
            "license": self.license,
            "intended_use": self.intended_use,
            "limitations": list(self.limitations),
            "training": list(self.training),
            "evaluation": list(self.evaluation),
            "runtime": list(self.runtime),
            "release_notes": list(self.release_notes),
            "extra_files": list(self.extra_files),
        }


@dataclass(frozen=True, slots=True)
class ModelPackageReport:
    """Files written by :func:`build_model_package`."""

    model_id: str
    metadata_path: Path
    model_card_path: Path
    checksums_path: Path
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_by": GENERATED_BY,
            "model_id": self.model_id,
            "metadata_path": self.metadata_path.name,
            "model_card_path": self.model_card_path.name,
            "checksums_path": self.checksums_path.name,
            "files": list(self.files),
        }


def build_model_package(
    model_dir: Path,
    metadata_path: Path,
    *,
    allow_fixture_manifest: bool = False,
    allow_placeholders: bool = False,
) -> ModelPackageReport:
    """Generate ``model_card.md`` and ``SHA256SUMS`` for ``model_dir``."""
    manifest_path = model_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    if not allow_fixture_manifest and _looks_like_fixture_manifest(manifest):
        raise InputError("fixture/test manifests cannot back a model release package")
    package = load_model_package(metadata_path, allow_placeholders=allow_placeholders)
    _verify_manifest_artifacts(model_dir, manifest)
    eval_input = _verify_eval_report_source(model_dir, manifest.eval.file)
    efficiency_report = _verify_efficiency_report_source(model_dir)
    _verify_release_evidence_identity(manifest, eval_input, efficiency_report)
    _verify_extra_files(model_dir, package.extra_files)
    eval_artifact_files = _eval_artifact_checksum_files(model_dir, eval_input)

    metadata_output_path = model_dir / MODEL_PACKAGE_NAME
    model_card_path = model_dir / "model_card.md"
    checksums_path = model_dir / "SHA256SUMS"
    metadata_output_path.write_text(
        json.dumps(package.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_card_path.write_text(render_model_card(manifest, package), encoding="utf-8")
    checksum_files = _checksum_files(manifest, package.extra_files, eval_artifact_files)
    _write_sha256sums(model_dir, checksums_path, checksum_files)
    return ModelPackageReport(
        model_id=manifest.model_id(),
        metadata_path=metadata_output_path,
        model_card_path=model_card_path,
        checksums_path=checksums_path,
        files=checksum_files,
    )


def load_model_package(
    metadata_path: Path,
    *,
    allow_placeholders: bool = False,
) -> ModelPackage:
    """Load and validate model release metadata."""
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "failed to read model metadata", details={"path": str(metadata_path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "model metadata JSON is invalid",
            details={"path": str(metadata_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return parse_model_package(payload, allow_placeholders=allow_placeholders)


def parse_model_package(
    payload: Any,
    *,
    allow_placeholders: bool = False,
) -> ModelPackage:
    """Validate decoded model release metadata."""
    if not isinstance(payload, dict):
        raise InputError("model metadata must be a JSON object")
    schema_version = _required_text(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InputError(
            "unsupported model-package schema version",
            details={"expected": SCHEMA_VERSION, "observed": schema_version},
        )
    generated_by = _required_text(payload, "generated_by")
    if generated_by != GENERATED_BY:
        raise InputError(
            "model-package generated_by is invalid",
            details={"expected": GENERATED_BY, "observed": generated_by},
        )
    package = ModelPackage(
        schema_version=schema_version,
        generated_by=generated_by,
        generated_at=_optional_text(payload, "generated_at") or _utc_now(),
        summary=_required_text(payload, "summary"),
        data=_parse_text_list(payload.get("data"), field="data"),
        hardware=_parse_text_list(payload.get("hardware"), field="hardware"),
        license=_required_text(payload, "license"),
        intended_use=_required_text(payload, "intended_use"),
        limitations=_parse_text_list(payload.get("limitations"), field="limitations"),
        training=_parse_text_list(payload.get("training"), field="training"),
        evaluation=_parse_text_list(payload.get("evaluation"), field="evaluation"),
        runtime=_parse_text_list(payload.get("runtime"), field="runtime"),
        release_notes=_parse_text_list(payload.get("release_notes"), field="release_notes"),
        extra_files=_parse_extra_files(payload.get("extra_files", [])),
    )
    _require_release_evidence_extra_files(package.extra_files)
    if not allow_placeholders:
        _reject_placeholders(_text_fields(package))
    return package


def render_model_card(manifest: Manifest, package: ModelPackage) -> str:
    """Render a Markdown model card from manifest and release metadata."""
    artifacts = _manifest_artifact_rows(manifest)
    lines = [
        f"# Model Card: {manifest.release_id}",
        "",
        f"Generated by: {package.generated_by}",
        f"Generated: {package.generated_at}",
        "",
        "## Summary",
        "",
        package.summary,
        "",
        "## Model Identity",
        "",
        f"- Model name: {manifest.model_name}",
        f"- Model version: {manifest.model_version}",
        f"- Release id: {manifest.release_id}",
        f"- Model id: {manifest.model_id()}",
        f"- Manifest schema: {manifest.schema_version}",
        f"- Encoder: {manifest.encoder.id}",
        f"- Encoder revision: {manifest.encoder.revision}",
        f"- Encoder hash: {manifest.encoder.hash}",
        "",
        "## Data",
        "",
    ]
    lines.extend(f"- {item}" for item in package.data)
    if manifest.training.data_snapshot:
        lines.append("- Manifest data snapshot:")
        lines.extend(
            f"  - {key}: {value}" for key, value in sorted(manifest.training.data_snapshot.items())
        )
    lines.extend(["", "## Hardware", ""])
    lines.extend(f"- {item}" for item in package.hardware)
    lines.extend(["", "## Training", ""])
    lines.extend(f"- {item}" for item in package.training)
    lines.extend(["", "## Evaluation", ""])
    lines.extend(f"- {item}" for item in package.evaluation)
    lines.extend(["", "## Artifacts", "", "| Artifact | Path | SHA-256 | Detail |"])
    lines.append("| --- | --- | --- | --- |")
    lines.extend(
        f"| {_md_cell(name)} | {_md_cell(path)} | {_md_cell(digest)} | {_md_cell(detail)} |"
        for name, path, digest, detail in artifacts
    )
    lines.extend(["", "## Runtime", ""])
    lines.extend(f"- {item}" for item in package.runtime)
    lines.extend(["", "## License", "", package.license, "", "## Intended Use", ""])
    lines.append(package.intended_use)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in package.limitations)
    lines.extend(["", "## Release Notes", ""])
    lines.extend(f"- {item}" for item in package.release_notes)
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = build_model_package(
            args.model_dir,
            args.metadata_json,
            allow_fixture_manifest=args.allow_fixture_manifest,
            allow_placeholders=args.allow_placeholders,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a release model card and SHA256SUMS from manifest metadata.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument(
        "--allow-fixture-manifest",
        action="store_true",
        help="Allow fixture/test manifests for local verifier tests only.",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder wording for local drafts. Do not use for releases.",
    )
    return parser


def _manifest_artifact_rows(manifest: Manifest) -> tuple[tuple[str, str, str, str], ...]:
    return (
        ("manifest", "manifest.json", manifest.model_id(), "canonical manifest model_id"),
        (
            "predictor",
            manifest.predictor.file,
            manifest.predictor.hash,
            _artifact_detail(dtype=manifest.predictor.dtype, version=manifest.predictor.version),
        ),
        (
            "action_encoder",
            manifest.action_encoder.file,
            manifest.action_encoder.hash,
            _artifact_detail(
                dtype=manifest.action_encoder.dtype,
                version=manifest.action_encoder.version,
            ),
        ),
        (
            "calibration",
            manifest.calibration.file,
            manifest.calibration.hash,
            _artifact_detail(
                dtype=manifest.calibration.dtype,
                version=manifest.calibration.version,
            ),
        ),
        (
            "training_config",
            manifest.training.config_file,
            manifest.training.hash,
            "training configuration",
        ),
        (
            "evaluation_report",
            manifest.eval.file,
            manifest.eval.hash,
            _artifact_detail(dtype=manifest.eval.dtype, version=manifest.eval.version),
        ),
    )


def _verify_manifest_artifacts(model_dir: Path, manifest: Manifest) -> None:
    expected = {
        manifest.predictor.file: manifest.predictor.hash,
        manifest.action_encoder.file: manifest.action_encoder.hash,
        manifest.calibration.file: manifest.calibration.hash,
        manifest.training.config_file: manifest.training.hash,
        manifest.eval.file: manifest.eval.hash,
    }
    for relative, expected_hash in expected.items():
        path = _safe_relative(model_dir, relative)
        if not path.is_file():
            raise InputError("manifest artifact is missing", details={"path": str(path)})
        observed = sha256_file(path)
        if observed != expected_hash:
            raise InputError(
                "manifest artifact hash mismatch",
                details={"path": str(path), "expected": expected_hash, "observed": observed},
            )


def _verify_eval_report_source(model_dir: Path, eval_report_file: str) -> EvalReportInput:
    metrics_path = model_dir / EVAL_METRICS_NAME
    if not metrics_path.is_file():
        raise InputError(
            f"{EVAL_METRICS_NAME} is required for model package generation",
            details={"path": str(metrics_path)},
        )
    report_path = _safe_relative(model_dir, eval_report_file)
    report_input = load_report_input(metrics_path)
    expected = render_report(report_input)
    observed = report_path.read_text(encoding="utf-8")
    if observed != expected:
        raise InputError(
            f"{eval_report_file} does not match render of {EVAL_METRICS_NAME}",
            details={"report_path": str(report_path), "metrics_path": str(metrics_path)},
        )
    return report_input


def _verify_efficiency_report_source(model_dir: Path) -> EfficiencyReport:
    report_path = model_dir / EFFICIENCY_REPORT_NAME
    if not report_path.is_file():
        raise InputError(
            f"{EFFICIENCY_REPORT_NAME} is required for model package generation",
            details={"path": str(report_path)},
        )
    return load_efficiency_report(report_path)


def _verify_release_evidence_identity(
    manifest: Manifest,
    eval_input: EvalReportInput,
    efficiency_report: EfficiencyReport,
) -> None:
    if eval_input.model_release != manifest.release_id:
        raise InputError(
            "eval_metrics.json model_release must match manifest release_id",
            details={
                "expected": manifest.release_id,
                "observed": eval_input.model_release,
            },
        )
    if efficiency_report.model_release != manifest.release_id:
        raise InputError(
            "efficiency_report.json model_release must match manifest release_id",
            details={
                "expected": manifest.release_id,
                "observed": efficiency_report.model_release,
            },
        )
    expected_snapshot = _manifest_dataset_snapshot(manifest)
    if expected_snapshot is not None and eval_input.dataset_snapshot != expected_snapshot:
        raise InputError(
            "eval_metrics.json dataset_snapshot must match manifest training data snapshot",
            details={
                "expected": expected_snapshot,
                "observed": eval_input.dataset_snapshot,
            },
        )
    if expected_snapshot is not None and efficiency_report.dataset_snapshot != expected_snapshot:
        raise InputError(
            "efficiency_report.json dataset_snapshot must match manifest training data snapshot",
            details={
                "expected": expected_snapshot,
                "observed": efficiency_report.dataset_snapshot,
            },
        )
    if eval_input.dataset_snapshot != efficiency_report.dataset_snapshot:
        raise InputError(
            "eval_metrics.json and efficiency_report.json dataset snapshots must match",
            details={
                "eval_metrics": eval_input.dataset_snapshot,
                "efficiency_report": efficiency_report.dataset_snapshot,
            },
        )
    if eval_input.model_id != efficiency_report.model_id:
        raise InputError(
            "eval_metrics.json and efficiency_report.json model ids must match",
            details={
                "eval_metrics": eval_input.model_id,
                "efficiency_report": efficiency_report.model_id,
            },
        )
    if eval_input.commit.lower() != efficiency_report.commit:
        raise InputError(
            "eval_metrics.json and efficiency_report.json commits must match",
            details={
                "eval_metrics": eval_input.commit,
                "efficiency_report": efficiency_report.commit,
            },
        )


def _manifest_dataset_snapshot(manifest: Manifest) -> str | None:
    for key in ("snapshot", "dataset_snapshot", "snapshot_id"):
        value = manifest.training.data_snapshot.get(key)
        if value:
            return value
    if len(manifest.training.data_snapshot) == 1:
        return next(iter(manifest.training.data_snapshot.values()))
    return None


def _verify_extra_files(model_dir: Path, files: tuple[str, ...]) -> None:
    for relative in files:
        path = _safe_relative(model_dir, relative)
        if not path.is_file():
            raise InputError("extra model package file is missing", details={"path": str(path)})
    if CALIBRATION_REPORT_NAME in files:
        _verify_calibration_report(model_dir, _load_manifest_for_verification(model_dir))


def _eval_artifact_checksum_files(
    model_dir: Path,
    eval_input: EvalReportInput,
) -> tuple[str, ...]:
    files: list[str] = []
    for key, raw_path in eval_input.artifacts:
        relative = _model_relative_eval_artifact(raw_path, key=key)
        if relative is None:
            continue
        path = _safe_relative(model_dir, relative)
        if not path.is_file():
            raise InputError(
                "eval metrics artifact is missing from model package",
                details={"artifact": key, "path": str(path)},
            )
        files.append(relative)
    return tuple(dict.fromkeys(files))


def _load_manifest_for_verification(model_dir: Path) -> Manifest:
    return load_manifest(model_dir / "manifest.json")


def _verify_calibration_report(model_dir: Path, manifest: Manifest) -> None:
    report_path = model_dir / CALIBRATION_REPORT_NAME
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "failed to read calibration_report.json",
            details={"path": str(report_path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "calibration_report.json is invalid JSON",
            details={"path": str(report_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("calibration_report.json must be a JSON object")
    if payload.get("generated_by") != CALIBRATION_REPORT_GENERATED_BY:
        raise InputError(
            "calibration_report.json generated_by is invalid",
            details={
                "expected": CALIBRATION_REPORT_GENERATED_BY,
                "observed": payload.get("generated_by"),
            },
        )
    if payload.get("model_id") != manifest.model_id():
        raise InputError(
            "calibration_report.json model_id must match manifest",
            details={"expected": manifest.model_id(), "observed": payload.get("model_id")},
        )
    if payload.get("model_release") != manifest.release_id:
        raise InputError(
            "calibration_report.json model_release must match manifest",
            details={"expected": manifest.release_id, "observed": payload.get("model_release")},
        )
    _verify_identity(
        payload.get("model_manifest"),
        expected_path="model/manifest.json",
        actual_path=model_dir / "manifest.json",
        field="calibration_report.json model_manifest",
    )
    _verify_identity(
        payload.get("calibration_artifact"),
        expected_path=manifest.calibration.file,
        actual_path=_safe_relative(model_dir, manifest.calibration.file),
        field="calibration_report.json calibration_artifact",
    )
    _verify_input_identity(payload.get("inputs"), key="vcf")
    _verify_input_identity(payload.get("inputs"), key="fasta")


def _verify_input_identity(raw_inputs: object, *, key: str) -> None:
    if not isinstance(raw_inputs, dict):
        raise InputError("calibration_report.json inputs must be an object")
    raw_identity = raw_inputs.get(key)
    if not isinstance(raw_identity, dict):
        raise InputError(
            "calibration_report.json input identity is missing",
            details={"input": key},
        )
    _required_text(raw_identity, "path", prefix=f"inputs.{key}.")
    _required_sha256(raw_identity.get("sha256"), field=f"inputs.{key}.sha256")
    _required_positive_int(raw_identity.get("size_bytes"), field=f"inputs.{key}.size_bytes")


def _verify_identity(
    raw_identity: object,
    *,
    expected_path: str,
    actual_path: Path,
    field: str,
) -> None:
    if not isinstance(raw_identity, dict):
        raise InputError(f"{field} identity must be an object")
    observed_path = _required_text(raw_identity, "path", prefix=f"{field}.")
    if observed_path != expected_path:
        raise InputError(
            f"{field} path must match expected artifact",
            details={"expected": expected_path, "observed": observed_path},
        )
    observed_hash = _required_sha256(raw_identity.get("sha256"), field=f"{field}.sha256")
    expected_hash = sha256_file(actual_path)
    if observed_hash != expected_hash:
        raise InputError(
            f"{field} hash must match packaged artifact",
            details={"expected": expected_hash, "observed": observed_hash},
        )
    observed_size = _required_positive_int(
        raw_identity.get("size_bytes"),
        field=f"{field}.size_bytes",
    )
    expected_size = actual_path.stat().st_size
    if observed_size != expected_size:
        raise InputError(
            f"{field} size must match packaged artifact",
            details={"expected": expected_size, "observed": observed_size},
        )


def _model_relative_eval_artifact(raw_path: str, *, key: str) -> str | None:
    if "://" in raw_path:
        raise InputError(
            "eval metrics artifact paths must be package-relative, not URLs",
            details={"artifact": key, "path": raw_path},
        )
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise InputError(
            "eval metrics artifact paths must be relative and stay inside the release package",
            details={"artifact": key, "path": raw_path},
        )
    parts = relative.parts
    if parts[0] == "dataset":
        return None
    if parts[0] == "model":
        if len(parts) == 1:
            raise InputError(
                "eval metrics artifact path must name a file",
                details={"artifact": key, "path": raw_path},
            )
        return Path(*parts[1:]).as_posix()
    return relative.as_posix()


def _checksum_files(
    manifest: Manifest,
    extra_files: tuple[str, ...],
    eval_artifact_files: tuple[str, ...],
) -> tuple[str, ...]:
    files = (
        "manifest.json",
        MODEL_PACKAGE_NAME,
        "model_card.md",
        manifest.predictor.file,
        manifest.action_encoder.file,
        manifest.calibration.file,
        manifest.training.config_file,
        manifest.eval.file,
        EVAL_METRICS_NAME,
        EFFICIENCY_REPORT_NAME,
        *eval_artifact_files,
        *extra_files,
    )
    return tuple(dict.fromkeys(files))


def _write_sha256sums(model_dir: Path, path: Path, files: tuple[str, ...]) -> None:
    lines = []
    for relative in files:
        artifact_path = _safe_relative(model_dir, relative)
        digest = sha256_file(artifact_path).removeprefix("sha256:")
        lines.append(f"{digest}  {relative}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_extra_files(raw: Any) -> tuple[str, ...]:
    if raw == []:
        return ()
    if not isinstance(raw, list):
        raise InputError("extra_files must be a list")
    files: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise InputError(
                "extra_files entries must be non-empty strings",
                details={"field": f"extra_files[{index}]"},
            )
        relative = item.strip()
        if relative in GENERATED_FILES:
            raise InputError(
                "generated model package files cannot be listed as extra files",
                details={"path": relative},
            )
        if relative in seen:
            raise InputError("extra_files contains duplicate paths", details={"path": relative})
        _validate_relative(relative)
        seen.add(relative)
        files.append(relative)
    return tuple(files)


def _require_release_evidence_extra_files(files: tuple[str, ...]) -> None:
    missing = sorted(set(REQUIRED_RELEASE_EVIDENCE_FILES) - set(files))
    if missing:
        raise InputError(
            "extra_files must include release training-run evidence",
            details={"missing": missing},
        )


def _safe_relative(root: Path, relative: str) -> Path:
    _validate_relative(relative)
    return root / Path(relative)


def _validate_relative(relative: str) -> None:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise InputError(
            "model package paths must be relative and stay inside model_dir",
            details={"path": relative},
        )


def _looks_like_fixture_manifest(manifest: Manifest) -> bool:
    parts = [
        manifest.release_id,
        manifest.model_version,
        *manifest.training.data_snapshot.keys(),
        *manifest.training.data_snapshot.values(),
    ]
    text = " ".join(parts).lower()
    return any(token in text for token in ("fixture", "dummy", "test"))


def _required_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{prefix}{key} must be a non-empty string")
    return value.strip()


def _required_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise InputError(f"{field} must be a sha256:<hex> digest")
    return value


def _required_positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(f"{field} must be a positive integer")
    return value


def _optional_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{prefix}{key} must be a non-empty string when supplied")
    return value.strip()


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


def _text_fields(package: ModelPackage) -> dict[str, str]:
    fields = {
        "generated_by": package.generated_by,
        "generated_at": package.generated_at,
        "summary": package.summary,
        "license": package.license,
        "intended_use": package.intended_use,
    }
    for group in (
        "data",
        "hardware",
        "limitations",
        "training",
        "evaluation",
        "runtime",
        "release_notes",
        "extra_files",
    ):
        values = getattr(package, group)
        for index, value in enumerate(values):
            fields[f"{group}[{index}]"] = value
    return fields


def _reject_placeholders(values: dict[str, str]) -> None:
    for key, value in values.items():
        if PLACEHOLDER_RE.search(value):
            raise InputError(
                "placeholder text is not allowed in release model packages",
                details={"field": key},
            )


def _artifact_detail(*, dtype: str | None, version: str | None) -> str:
    parts = []
    if dtype is not None:
        parts.append(f"dtype={dtype}")
    if version is not None:
        parts.append(f"version={version}")
    return ", ".join(parts)


def _md_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", r"\|").strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
