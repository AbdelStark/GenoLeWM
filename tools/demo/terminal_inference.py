# SPDX-License-Identifier: Apache-2.0
"""Generate a terminal-demo transcript from a real ``geno-lewm-score`` run."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from geno_lewm.data._vcf import iter_vcf_rows
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import Manifest, load_manifest, sha256_bytes, sha256_file
from tools.release.batch_receipt_report import write_batch_receipt_report
from tools.release.runtime_preflight import (
    GENERATED_BY as RUNTIME_PREFLIGHT_GENERATED_BY,
    REPORT_NAME as RUNTIME_PREFLIGHT_REPORT_NAME,
    SCHEMA_VERSION as RUNTIME_PREFLIGHT_SCHEMA_VERSION,
    RuntimePreflightRequest,
    write_runtime_preflight_report,
)

_DEFAULT_COMMAND = "geno-lewm-score"
DEMO_MANIFEST_NAME = "terminal_demo_manifest.json"
DEMO_MANIFEST_SCHEMA_VERSION = "1.0.0"
GENERATED_BY = "tools.demo.terminal_inference"
_MAX_SUMMARY_VARIANTS = 5


@dataclass(frozen=True, slots=True)
class DemoRequest:
    """Inputs needed to run and record the terminal inference demo."""

    model_dir: Path
    vcf: Path
    fasta: Path
    output_dir: Path
    backend: str = "auto"
    batch_size: int = 64
    command: str = _DEFAULT_COMMAND
    allow_fixture_manifest: bool = False
    require_native_runtime: bool = True
    carbon_cache_dir: Path | None = None
    require_carbon_cache: bool = False

    @property
    def scores_path(self) -> Path:
        return self.output_dir / "scores.jsonl"

    @property
    def receipts_path(self) -> Path:
        return self.output_dir / "receipts.jsonl"

    @property
    def transcript_path(self) -> Path:
        return self.output_dir / "terminal-demo-transcript.md"

    @property
    def batch_receipt_report_path(self) -> Path:
        return self.output_dir / "batch_receipt_report.json"

    @property
    def runtime_preflight_report_path(self) -> Path:
        return self.output_dir / RUNTIME_PREFLIGHT_REPORT_NAME

    @property
    def demo_manifest_path(self) -> Path:
        return self.output_dir / DEMO_MANIFEST_NAME


@dataclass(frozen=True, slots=True)
class DemoArtifact:
    """One output artifact produced by the terminal demo command."""

    label: str
    path: Path
    sha256: str
    size_bytes: int
    jsonl_records: int | None = None
    jsonl_fields: tuple[str, ...] = ()

    def to_dict(self, *, roots: tuple[Path, ...] = ()) -> dict[str, object]:
        payload: dict[str, object] = {
            "label": self.label,
            "path": _portable_path(self.path, roots),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if self.jsonl_records is not None:
            payload["jsonl_records"] = self.jsonl_records
        if self.jsonl_fields:
            payload["jsonl_fields"] = list(self.jsonl_fields)
        return payload


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def build_score_command(request: DemoRequest) -> tuple[str, ...]:
    """Build the exact score command recorded in the transcript."""
    _require_positive_batch_size(request.batch_size)
    return (
        request.command,
        "--quiet",
        "--no-banner",
        "--model-dir",
        str(request.model_dir),
        "--backend",
        request.backend,
        "--vcf",
        str(request.vcf),
        "--fasta",
        str(request.fasta),
        "--output",
        str(request.scores_path),
        "--receipt",
        str(request.receipts_path),
        "--batch-size",
        str(request.batch_size),
        "--no-progress",
    )


def run_demo_transcript(
    request: DemoRequest,
    *,
    runner: Runner | None = None,
    now: datetime | None = None,
) -> Path:
    """Run the score command and write a Markdown transcript.

    The transcript is evidence only if the command exits successfully.
    Fixture/test manifests are rejected by default so release demos do
    not accidentally publish scaffold output as model behavior.
    """
    manifest = _load_manifest_for_demo(
        request.model_dir,
        allow_fixture_manifest=request.allow_fixture_manifest,
    )
    request.output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = now or _utc_now()
    preflight = write_runtime_preflight_report(
        RuntimePreflightRequest(
            model_dir=request.model_dir,
            vcf=request.vcf,
            fasta=request.fasta,
            output_dir=request.output_dir,
            backend=request.backend,
            batch_size=request.batch_size,
            command=request.command,
            allow_fixture_manifest=request.allow_fixture_manifest,
            require_native_runtime=request.require_native_runtime,
            carbon_cache_dir=request.carbon_cache_dir,
            require_carbon_cache=request.require_carbon_cache,
        ),
        request.runtime_preflight_report_path,
        generated_at=generated_at.isoformat().replace("+00:00", "Z"),
    )
    if not preflight.ok:
        raise InputError(
            "terminal demo runtime preflight failed",
            details={
                "report": str(request.runtime_preflight_report_path),
                "issues": [issue.to_dict() for issue in preflight.issues],
            },
            remediation="fix the model artifacts, demo inputs, or runtime environment first",
        )
    vcf_summary = summarize_vcf_input(request.vcf)
    command = build_score_command(request)
    _remove_stale_demo_outputs(request)
    if runner is None:
        runner = _run_score_command
    completed = runner(command)
    artifact_error: GenoLeWMError | None = None
    artifacts: tuple[DemoArtifact, ...] = ()
    score_receipt_summary: dict[str, object] | None = None
    runtime_preflight_summary: dict[str, object] | None = None
    if completed.returncode == 0:
        try:
            write_batch_receipt_report(
                request.scores_path,
                request.receipts_path,
                request.batch_receipt_report_path,
                generated_at=generated_at.isoformat().replace("+00:00", "Z"),
            )
            runtime_preflight_summary = _load_runtime_preflight_summary(
                request.runtime_preflight_report_path,
                request=request,
                model_manifest=manifest,
                command=command,
            )
            artifacts = _collect_demo_artifacts(request)
            score_receipt_summary = _load_batch_receipt_summary(request.batch_receipt_report_path)
        except GenoLeWMError as exc:
            artifact_error = exc
    transcript = _render_transcript(
        request=request,
        manifest=manifest,
        command=command,
        completed=completed,
        artifacts=artifacts,
        score_receipt_summary=score_receipt_summary,
        artifact_error=artifact_error,
        vcf_summary=vcf_summary,
        generated_at=generated_at,
    )
    request.transcript_path.write_text(transcript, encoding="utf-8")
    if completed.returncode != 0:
        raise InputError(
            "terminal demo command failed",
            details={
                "returncode": completed.returncode,
                "transcript": str(request.transcript_path),
            },
            remediation="fix the model artifacts or score command before publishing the transcript",
        )
    if artifact_error is not None:
        raise InputError(
            "terminal demo artifact verification failed",
            details={
                "error": artifact_error.message or str(artifact_error),
                "transcript": str(request.transcript_path),
            },
        ) from artifact_error
    write_demo_manifest(
        request=request,
        model_manifest=manifest,
        command=command,
        completed=completed,
        artifacts=artifacts,
        score_receipt_summary=score_receipt_summary or {},
        runtime_preflight_summary=runtime_preflight_summary or {},
        vcf_summary=vcf_summary,
        generated_at=generated_at,
    )
    return request.transcript_path


def summarize_vcf_input(path: str | Path) -> dict[str, object]:
    """Return a compact, deterministic summary of the scored VCF input."""
    records = 0
    alternate_alleles = 0
    contigs: list[str] = []
    seen_contigs: set[str] = set()
    first_variants: list[dict[str, object]] = []
    for row in iter_vcf_rows(path):
        records += 1
        alternate_alleles += len(row.alts)
        if row.chrom not in seen_contigs:
            contigs.append(row.chrom)
            seen_contigs.add(row.chrom)
        if len(first_variants) < _MAX_SUMMARY_VARIANTS:
            first_variants.append(
                {
                    "chrom": row.chrom,
                    "pos": row.pos,
                    "ref": row.ref,
                    "alts": list(row.alts),
                }
            )
    if records <= 0:
        raise InputError(
            "terminal demo VCF input contains no scoreable variant records",
            details={"path": str(path)},
        )
    return {
        "format": "vcf",
        "variant_records": records,
        "alternate_alleles": alternate_alleles,
        "contigs": contigs,
        "first_variants": first_variants,
    }


def write_demo_manifest(
    *,
    request: DemoRequest,
    model_manifest: Manifest,
    command: Sequence[str],
    completed: subprocess.CompletedProcess[str],
    artifacts: tuple[DemoArtifact, ...],
    score_receipt_summary: Mapping[str, object] | None = None,
    runtime_preflight_summary: Mapping[str, object] | None = None,
    vcf_summary: Mapping[str, object] | None = None,
    generated_at: datetime,
) -> Path:
    """Write machine-readable evidence for a successful terminal demo run."""
    artifact_roots = (request.output_dir.parent,)
    transcript_artifact = _inspect_file_artifact("terminal transcript", request.transcript_path)
    manifest_command = _portable_score_command(command, request)
    payload = {
        "schema_version": DEMO_MANIFEST_SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "status": "passed",
        "command": {
            "argv": list(manifest_command),
            "shell": shlex.join(manifest_command),
            "returncode": completed.returncode,
            "stdout": _stream_identity(completed.stdout),
            "stderr": _stream_identity(completed.stderr),
        },
        "model": {
            "release_id": model_manifest.release_id,
            "model_version": model_manifest.model_version,
            "model_id": model_manifest.model_id(),
            "encoder_id": model_manifest.encoder.id,
            "encoder_revision": model_manifest.encoder.revision,
            "encoder_hash": model_manifest.encoder.hash,
        },
        "inputs": {
            "model_manifest": _file_identity(
                request.model_dir / "manifest.json",
                roots=(request.model_dir.parent,),
            ),
            "vcf": _file_identity(request.vcf, roots=artifact_roots),
            "vcf_summary": dict(vcf_summary or summarize_vcf_input(request.vcf)),
            "fasta": _file_identity(request.fasta, roots=artifact_roots),
        },
        "artifacts": [
            artifact.to_dict(roots=artifact_roots) for artifact in (*artifacts, transcript_artifact)
        ],
        "runtime_preflight": dict(runtime_preflight_summary or {}),
        "score_receipt_batch": dict(score_receipt_summary or {}),
    }
    request.demo_manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return request.demo_manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    request = DemoRequest(
        model_dir=args.model_dir,
        vcf=args.vcf,
        fasta=args.fasta,
        output_dir=args.output_dir,
        backend=args.backend,
        batch_size=args.batch_size,
        command=args.command,
        allow_fixture_manifest=args.allow_fixture_manifest,
        require_native_runtime=not args.no_require_native_runtime,
        carbon_cache_dir=args.carbon_cache_dir,
        require_carbon_cache=args.require_carbon_cache,
    )
    try:
        transcript = run_demo_transcript(request)
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"{transcript}\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run geno-lewm-score and write a terminal-demo transcript.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vcf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--command", default=_DEFAULT_COMMAND)
    parser.add_argument(
        "--allow-fixture-manifest",
        action="store_true",
        help="Allow fixture/test manifests for local tool tests only.",
    )
    parser.add_argument(
        "--no-require-native-runtime",
        action="store_true",
        help="Skip optional native-runtime dependency checks for local tool tests only.",
    )
    parser.add_argument("--carbon-cache-dir", type=Path)
    parser.add_argument(
        "--require-carbon-cache",
        action="store_true",
        help="Require --carbon-cache-dir to point at a non-empty local Carbon cache marker.",
    )
    return parser


def _run_score_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
    )


def _load_manifest_for_demo(
    model_dir: Path,
    *,
    allow_fixture_manifest: bool,
) -> Manifest:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise InputError(
            "terminal demo requires a manifest-backed model directory",
            details={"manifest": str(manifest_path)},
            remediation="pass --model-dir pointing at a released checkpoint directory",
        )
    manifest = load_manifest(manifest_path)
    if not allow_fixture_manifest and _looks_like_fixture_manifest(manifest):
        raise InputError(
            "terminal demo refuses fixture/test manifests by default",
            details={"release_id": manifest.release_id, "model_version": manifest.model_version},
            remediation=(
                "use released model artifacts, or pass --allow-fixture-manifest only for "
                "local tool tests"
            ),
        )
    return manifest


def _looks_like_fixture_manifest(manifest: Manifest) -> bool:
    parts = [
        manifest.release_id,
        manifest.model_version,
        *manifest.training.data_snapshot.keys(),
        *manifest.training.data_snapshot.values(),
    ]
    text = " ".join(parts).lower()
    return any(token in text for token in ("fixture", "dummy", "test"))


def _render_transcript(
    *,
    request: DemoRequest,
    manifest: Manifest,
    command: Sequence[str],
    completed: subprocess.CompletedProcess[str],
    artifacts: tuple[DemoArtifact, ...],
    score_receipt_summary: Mapping[str, object] | None,
    artifact_error: GenoLeWMError | None,
    vcf_summary: Mapping[str, object],
    generated_at: datetime,
) -> str:
    status = "passed" if completed.returncode == 0 and artifact_error is None else "failed"
    artifact_roots = (request.output_dir.parent,)
    display_command = _portable_score_command(command, request)
    lines = [
        "# GenoLeWM Terminal Inference Transcript",
        "",
        f"- Generated: {generated_at.isoformat().replace('+00:00', 'Z')}",
        f"- Status: {status}",
        f"- Exit code: {completed.returncode}",
        f"- Model release: {manifest.release_id}",
        f"- Model version: {manifest.model_version}",
        f"- Model id: {manifest.model_id()}",
        f"- Input VCF records: {vcf_summary.get('variant_records')}",
        f"- Input alternate alleles: {vcf_summary.get('alternate_alleles')}",
        f"- Input contigs: {_field_list(_string_sequence(vcf_summary.get('contigs')))}",
        f"- First input variants: {_variant_summary(vcf_summary.get('first_variants'))}",
        f"- Scores: {_portable_path(request.scores_path, artifact_roots)}",
        f"- Receipts: {_portable_path(request.receipts_path, artifact_roots)}",
        (
            "- Runtime preflight report: "
            f"{_portable_path(request.runtime_preflight_report_path, artifact_roots)}"
        ),
        (
            "- Batch receipt report: "
            f"{_portable_path(request.batch_receipt_report_path, artifact_roots)}"
        ),
        f"- Demo manifest: {_portable_path(request.demo_manifest_path, artifact_roots)}",
        "",
        "## Command",
        "",
        "```console",
        "$ " + " ".join(shlex.quote(part) for part in display_command),
    ]
    if completed.stdout:
        lines.extend(completed.stdout.rstrip().splitlines())
    lines.append("```")
    if completed.stderr:
        lines.extend(
            [
                "",
                "## Stderr",
                "",
                "```text",
                *completed.stderr.rstrip().splitlines(),
                "```",
            ]
        )
    if artifact_error is not None:
        lines.extend(
            [
                "",
                "## Artifact Verification",
                "",
                "```text",
                artifact_error.message or str(artifact_error),
                "```",
            ]
        )
    if artifacts:
        lines.extend(
            [
                "",
                "## Output Artifacts",
                "",
                "| Artifact | Path | SHA-256 | Bytes | JSONL rows |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        lines.extend(
            (
                f"| {artifact.label} | {_portable_path(artifact.path, artifact_roots)} | {artifact.sha256} | "
                f"{artifact.size_bytes} | {_row_count_cell(artifact.jsonl_records)} |"
            )
            for artifact in artifacts
        )
        for artifact in artifacts:
            title = artifact.label.title()
            lines.extend(
                [
                    f"- {title} SHA-256: {artifact.sha256}",
                ]
            )
            if artifact.jsonl_records is not None:
                lines.append(f"- {title} JSONL rows: {artifact.jsonl_records}")
            if artifact.jsonl_fields:
                lines.append(f"- {title} JSONL fields: {_field_list(artifact.jsonl_fields)}")
    if artifacts and score_receipt_summary:
        score_artifact = _artifact_by_label(artifacts, "scores")
        receipt_artifact = _artifact_by_label(artifacts, "receipts")
        runtime = score_receipt_summary.get("runtime")
        runtime_backend = (
            _string_dict_value(runtime, "backend") if isinstance(runtime, Mapping) else None
        )
        runtime_device = (
            _string_dict_value(runtime, "device") if isinstance(runtime, Mapping) else None
        )
        lines.extend(
            [
                "",
                "## Score And Receipt Summary",
                "",
                f"- Records: {score_receipt_summary.get('records')}",
                f"- Score fields: {_field_list(() if score_artifact is None else score_artifact.jsonl_fields)}",
                f"- Receipt fields: {_field_list(() if receipt_artifact is None else receipt_artifact.jsonl_fields)}",
                "- Checked score fields: "
                + _field_list(_string_sequence(score_receipt_summary.get("checked_score_fields"))),
                f"- Receipt stream: {score_receipt_summary.get('receipt_stream')}",
                f"- Receipt schema: {score_receipt_summary.get('receipt_schema_version')}",
                f"- Receipt model id: {score_receipt_summary.get('model_id')}",
                f"- Calibration hash: {score_receipt_summary.get('calibration_hash')}",
            ]
        )
        if runtime_backend is not None:
            lines.append(f"- Runtime backend: {runtime_backend}")
        if runtime_device is not None:
            lines.append(f"- Runtime device: {runtime_device}")
    lines.extend(
        [
            "",
            "## Artifact Inputs",
            "",
            f"- Model directory: {_portable_path(request.model_dir, artifact_roots)}",
            f"- Manifest: {_portable_path(request.model_dir / 'manifest.json', artifact_roots)}",
            f"- VCF: {_portable_path(request.vcf, artifact_roots)}",
            f"- FASTA: {_portable_path(request.fasta, artifact_roots)}",
            "",
            "This transcript records command behavior only. Model-quality claims require the "
            "published evaluation report linked from the release.",
            "",
        ]
    )
    return "\n".join(lines)


def _collect_demo_artifacts(request: DemoRequest) -> tuple[DemoArtifact, ...]:
    return (
        _inspect_jsonl_artifact("scores", request.scores_path),
        _inspect_jsonl_artifact("receipts", request.receipts_path),
        _inspect_file_artifact("runtime preflight report", request.runtime_preflight_report_path),
        _inspect_file_artifact("batch receipt report", request.batch_receipt_report_path),
    )


def _remove_stale_demo_outputs(request: DemoRequest) -> None:
    for path in (
        request.scores_path,
        request.receipts_path,
        request.batch_receipt_report_path,
        request.demo_manifest_path,
    ):
        if not path.exists():
            continue
        if not path.is_file():
            raise InputError(
                "terminal demo output path must be a file",
                details={"path": str(path)},
            )
        try:
            path.unlink()
        except OSError as exc:
            raise InputError(
                "failed to remove stale terminal demo output",
                details={"path": str(path)},
            ) from exc


def _inspect_jsonl_artifact(label: str, path: Path) -> DemoArtifact:
    if not path.is_file():
        raise InputError(f"terminal demo {label} artifact is missing", details={"path": str(path)})
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise InputError(f"terminal demo {label} artifact is empty", details={"path": str(path)})
    records = 0
    fields: list[str] = []
    seen_fields: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputError(
                        f"terminal demo {label} artifact must be JSONL",
                        details={"path": str(path), "line": line_no},
                    ) from exc
                if not isinstance(payload, dict):
                    raise InputError(
                        f"terminal demo {label} artifact rows must be JSON objects",
                        details={"path": str(path), "line": line_no},
                    )
                for key in payload:
                    if key not in seen_fields:
                        fields.append(key)
                        seen_fields.add(key)
                records += 1
    except OSError as exc:
        raise InputError(
            f"failed to read terminal demo {label} artifact", details={"path": str(path)}
        ) from exc
    if records <= 0:
        raise InputError(
            f"terminal demo {label} artifact must contain at least one JSONL row",
            details={"path": str(path)},
        )
    return DemoArtifact(
        label=label,
        path=path,
        sha256=sha256_file(path),
        size_bytes=size_bytes,
        jsonl_records=records,
        jsonl_fields=tuple(fields),
    )


def _inspect_file_artifact(label: str, path: Path) -> DemoArtifact:
    if not path.is_file():
        raise InputError(f"terminal demo {label} artifact is missing", details={"path": str(path)})
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise InputError(f"terminal demo {label} artifact is empty", details={"path": str(path)})
    return DemoArtifact(
        label=label,
        path=path,
        sha256=sha256_file(path),
        size_bytes=size_bytes,
    )


def _file_identity(path: Path, *, roots: tuple[Path, ...] = ()) -> dict[str, object]:
    if not path.is_file():
        raise InputError("terminal demo manifest input is missing", details={"path": str(path)})
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise InputError("terminal demo manifest input is empty", details={"path": str(path)})
    return {
        "path": _portable_path(path, roots),
        "sha256": sha256_file(path),
        "size_bytes": size_bytes,
    }


def _portable_score_command(command: Sequence[str], request: DemoRequest) -> tuple[str, ...]:
    values = list(command)
    replacements = {
        "--model-dir": _portable_path(request.model_dir, (request.model_dir.parent,)),
        "--vcf": _portable_path(request.vcf, (request.output_dir.parent,)),
        "--fasta": _portable_path(request.fasta, (request.output_dir.parent,)),
        "--output": _portable_path(request.scores_path, (request.output_dir.parent,)),
        "--receipt": _portable_path(request.receipts_path, (request.output_dir.parent,)),
    }
    for flag, value in replacements.items():
        try:
            index = values.index(flag)
        except ValueError:
            continue
        value_index = index + 1
        if value_index < len(values):
            values[value_index] = value
    return tuple(values)


def _portable_path(path: Path, roots: tuple[Path, ...]) -> str:
    for root in roots:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, RuntimeError, ValueError):
            continue
    return str(path)


def _stream_identity(text: str) -> dict[str, object]:
    data = text.encode("utf-8")
    return {
        "sha256": sha256_bytes(data),
        "size_bytes": len(data),
        "line_count": 0 if not text else len(text.splitlines()),
    }


def _load_batch_receipt_summary(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(
            "terminal demo batch receipt report must be valid JSON",
            details={"path": str(path)},
        ) from exc
    except OSError as exc:
        raise InputError(
            "failed to read terminal demo batch receipt report",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(
            "terminal demo batch receipt report must be a JSON object",
            details={"path": str(path)},
        )
    required = (
        "records",
        "model_id",
        "calibration_hash",
        "receipt_schema_version",
        "receipt_stream",
        "checked_score_fields",
        "runtime",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise InputError(
            "terminal demo batch receipt report is missing summary fields",
            details={"path": str(path), "missing": missing},
        )
    return {field: payload[field] for field in required}


def _load_runtime_preflight_summary(
    path: Path,
    *,
    request: DemoRequest,
    model_manifest: Manifest,
    command: Sequence[str],
) -> dict[str, object]:
    payload = _load_json_object(path, label="runtime preflight report")
    problems: list[str] = []
    _expect_value(
        payload,
        "schema_version",
        RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        problems,
    )
    _expect_value(payload, "generated_by", RUNTIME_PREFLIGHT_GENERATED_BY, problems)
    _expect_value(payload, "ok", True, problems)
    _expect_value(payload, "model_id", model_manifest.model_id(), problems)
    _expect_value(payload, "release_id", model_manifest.release_id, problems)
    _expect_value(payload, "requested_backend", request.backend, problems)
    requirements = _mapping_field(payload, "requirements", problems)
    if requirements is not None:
        _expect_value(
            requirements,
            "native_runtime",
            request.require_native_runtime,
            problems,
            path="requirements.native_runtime",
        )
        _expect_value(
            requirements,
            "carbon_cache",
            request.require_carbon_cache,
            problems,
            path="requirements.carbon_cache",
        )
        _expect_value(
            requirements,
            "fixture_manifest_allowed",
            request.allow_fixture_manifest,
            problems,
            path="requirements.fixture_manifest_allowed",
        )
    command_payload = _mapping_field(payload, "command", problems)
    expected_command = list(_portable_score_command(command, request))
    if command_payload is not None:
        _expect_value(command_payload, "argv", expected_command, problems, path="command.argv")
    manifest_identity = _mapping_field(payload, "manifest", problems)
    if manifest_identity is not None:
        _expect_file_identity(
            manifest_identity,
            _file_identity(request.model_dir / "manifest.json", roots=(request.model_dir.parent,)),
            problems,
            label="manifest",
        )
    inputs = _mapping_field(payload, "inputs", problems)
    if inputs is not None:
        vcf_identity = _mapping_field(inputs, "vcf", problems, path="inputs.vcf")
        if vcf_identity is not None:
            _expect_file_identity(
                vcf_identity,
                _file_identity(request.vcf, roots=(request.output_dir.parent,)),
                problems,
                label="inputs.vcf",
                require_ok=True,
            )
        fasta_identity = _mapping_field(inputs, "fasta", problems, path="inputs.fasta")
        if fasta_identity is not None:
            _expect_file_identity(
                fasta_identity,
                _file_identity(request.fasta, roots=(request.output_dir.parent,)),
                problems,
                label="inputs.fasta",
                require_ok=True,
            )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        problems.append("artifacts must be a non-empty list")
    else:
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                problems.append(f"artifacts[{index}] must be a JSON object")
                continue
            if artifact.get("ok") is not True:
                problems.append(f"artifacts[{index}].ok must be true")
    if problems:
        raise InputError(
            "terminal demo runtime preflight report is inconsistent",
            details={"path": str(path), "problems": problems},
        )
    return {
        "schema_version": payload["schema_version"],
        "generated_by": payload["generated_by"],
        "ok": payload["ok"],
        "model_id": payload["model_id"],
        "release_id": payload["release_id"],
        "requested_backend": payload["requested_backend"],
        "selected_backend": payload.get("selected_backend"),
        "requirements": dict(requirements or {}),
        "command": {"argv": expected_command, "shell": shlex.join(expected_command)},
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(
            f"terminal demo {label} must be valid JSON",
            details={"path": str(path)},
        ) from exc
    except OSError as exc:
        raise InputError(
            f"failed to read terminal demo {label}", details={"path": str(path)}
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(
            f"terminal demo {label} must be a JSON object",
            details={"path": str(path)},
        )
    return payload


def _mapping_field(
    payload: Mapping[str, object],
    key: str,
    problems: list[str],
    *,
    path: str | None = None,
) -> Mapping[str, object] | None:
    value = payload.get(key)
    field_path = path or key
    if not isinstance(value, Mapping):
        problems.append(f"{field_path} must be a JSON object")
        return None
    return value


def _expect_value(
    payload: Mapping[str, object],
    key: str,
    expected: object,
    problems: list[str],
    *,
    path: str | None = None,
) -> None:
    observed = payload.get(key)
    if observed != expected:
        problems.append(f"{path or key} must be {expected!r}, observed {observed!r}")


def _expect_file_identity(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    problems: list[str],
    *,
    label: str,
    require_ok: bool = False,
) -> None:
    problems.extend(
        (f"{label}.{key} must be {expected.get(key)!r}, observed {observed.get(key)!r}")
        for key in ("path", "sha256", "size_bytes")
        if observed.get(key) != expected.get(key)
    )
    if require_ok and observed.get("ok") is not True:
        problems.append(f"{label}.ok must be true")


def _artifact_by_label(
    artifacts: tuple[DemoArtifact, ...],
    label: str,
) -> DemoArtifact | None:
    return next((artifact for artifact in artifacts if artifact.label == label), None)


def _field_list(fields: Sequence[str]) -> str:
    return ", ".join(fields) if fields else "-"


def _variant_summary(value: object) -> str:
    if not isinstance(value, list):
        return "-"
    parts: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        chrom = item.get("chrom")
        pos = item.get("pos")
        ref = item.get("ref")
        alts = item.get("alts")
        if not isinstance(chrom, str) or not isinstance(pos, int) or not isinstance(ref, str):
            continue
        alt_text = "/".join(_string_sequence(alts))
        if not alt_text:
            continue
        parts.append(f"{chrom}:{pos}:{ref}>{alt_text}")
    return ", ".join(parts) if parts else "-"


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _string_dict_value(mapping: Mapping[object, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None


def _row_count_cell(value: int | None) -> str:
    return "-" if value is None else str(value)


def _require_positive_batch_size(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            "batch_size must be a positive integer",
            details={"batch_size": value, "type": type(value).__name__},
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
