# SPDX-License-Identifier: Apache-2.0
"""Build clean-machine runtime evidence for the terminal demo."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import shlex
import socket
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

from geno_lewm.deploy.runtime import fail_closed_network_guard, probe_backends, select_backend
from geno_lewm.errors import GenoLeWMError, InputError, NetworkCallProhibitedError, exit_code_for
from geno_lewm.provenance import Manifest, load_manifest, sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256

REPORT_NAME: Final = "runtime_preflight_report.json"
SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.runtime_preflight"
DEFAULT_COMMAND: Final = "geno-lewm-score"
REQUIRED_NATIVE_MODULES: Final = (
    "torch",
    "safetensors.torch",
    "transformers",
    "pyarrow",
    "pyarrow.parquet",
)

Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class RuntimePreflightRequest:
    """Inputs needed to verify a release terminal-demo runtime envelope."""

    model_dir: Path
    vcf: Path
    fasta: Path
    output_dir: Path
    backend: str = "auto"
    batch_size: int = 64
    command: str = DEFAULT_COMMAND
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


@dataclass(frozen=True, slots=True)
class RuntimePreflightIssue:
    """One preflight issue."""

    severity: Severity
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class DependencyProbe:
    """Importability probe for one optional runtime dependency."""

    import_name: str
    package: str
    required: bool
    available: bool
    version: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "import_name": self.import_name,
            "package": self.package,
            "required": self.required,
            "available": self.available,
            "version": self.version,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RuntimePreflightReport:
    """Machine-readable clean-machine readiness report for a demo run."""

    schema_version: str
    generated_by: str
    generated_at: str
    ok: bool
    model_id: str | None
    release_id: str | None
    requested_backend: str
    selected_backend: str | None
    command: tuple[str, ...]
    requirements: dict[str, bool]
    manifest: dict[str, object]
    artifacts: tuple[dict[str, object], ...]
    inputs: dict[str, dict[str, object]]
    dependencies: tuple[DependencyProbe, ...]
    backend_probes: tuple[dict[str, object], ...]
    network_guard: dict[str, object]
    carbon: dict[str, object]
    issues: tuple[RuntimePreflightIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "ok": self.ok,
            "model_id": self.model_id,
            "release_id": self.release_id,
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "command": {
                "argv": list(self.command),
                "shell": shlex.join(self.command),
            },
            "requirements": self.requirements,
            "manifest": self.manifest,
            "artifacts": list(self.artifacts),
            "inputs": self.inputs,
            "dependencies": [probe.to_dict() for probe in self.dependencies],
            "backend_probes": list(self.backend_probes),
            "network_guard": self.network_guard,
            "carbon": self.carbon,
            "issues": [issue.to_dict() for issue in self.issues],
        }


DependencyProbeFn = Callable[[str, bool], DependencyProbe]


def build_runtime_preflight_report(
    request: RuntimePreflightRequest,
    *,
    generated_at: str | None = None,
    dependency_probe: DependencyProbeFn | None = None,
) -> RuntimePreflightReport:
    """Verify the local runtime envelope needed before publishing a terminal demo."""
    _require_positive_batch_size(request.batch_size)
    issues: list[RuntimePreflightIssue] = []
    dependency_probe = dependency_probe or _probe_dependency
    command = _portable_terminal_demo_command(request)

    manifest = _load_manifest(request.model_dir, issues)
    if (
        manifest is not None
        and not request.allow_fixture_manifest
        and _looks_like_fixture_manifest(manifest)
    ):
        _issue(
            issues,
            "error",
            "model.fixture_manifest",
            request.model_dir / "manifest.json",
            "fixture/test manifests cannot back a release terminal demo",
        )
    manifest_identity = _manifest_identity(
        request.model_dir,
        manifest,
        roots=(request.model_dir.parent,),
    )
    artifacts = _artifact_checks(
        request.model_dir,
        manifest,
        issues,
        roots=(request.model_dir.parent,),
    )
    inputs = {
        "vcf": _required_file_identity(
            request.vcf,
            issues,
            code_prefix="input.vcf",
            roots=(request.output_dir.parent,),
        ),
        "fasta": _required_file_identity(
            request.fasta,
            issues,
            code_prefix="input.fasta",
            roots=(request.output_dir.parent,),
        ),
    }
    dependencies = tuple(
        dependency_probe(name, request.require_native_runtime) for name in REQUIRED_NATIVE_MODULES
    )
    _record_dependency_issues(
        dependencies,
        issues,
        required=request.require_native_runtime,
    )
    backend_probes, selected_backend = _runtime_backend_report(request, issues)
    network_guard = _network_guard_report(issues)
    carbon = _carbon_report(request, manifest, issues)

    ok = not any(issue.severity == "error" for issue in issues)
    return RuntimePreflightReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        ok=ok,
        model_id=None if manifest is None else manifest.model_id(),
        release_id=None if manifest is None else manifest.release_id,
        requested_backend=request.backend,
        selected_backend=selected_backend,
        command=command,
        requirements={
            "native_runtime": request.require_native_runtime,
            "carbon_cache": request.require_carbon_cache,
            "fixture_manifest_allowed": request.allow_fixture_manifest,
        },
        manifest=manifest_identity,
        artifacts=tuple(artifacts),
        inputs=inputs,
        dependencies=dependencies,
        backend_probes=tuple(backend_probes),
        network_guard=network_guard,
        carbon=carbon,
        issues=tuple(issues),
    )


def write_runtime_preflight_report(
    request: RuntimePreflightRequest,
    output: str | Path,
    *,
    generated_at: str | None = None,
    dependency_probe: DependencyProbeFn | None = None,
) -> RuntimePreflightReport:
    """Build and write ``runtime_preflight_report.json``."""
    report = build_runtime_preflight_report(
        request,
        generated_at=generated_at,
        dependency_probe=dependency_probe,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_terminal_demo_command(request: RuntimePreflightRequest) -> tuple[str, ...]:
    """Return the exact terminal scoring command covered by the preflight."""
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


def _portable_terminal_demo_command(request: RuntimePreflightRequest) -> tuple[str, ...]:
    command = list(build_terminal_demo_command(request))
    replacements = {
        "--model-dir": _portable_path(request.model_dir, (request.model_dir.parent,)),
        "--vcf": _portable_path(request.vcf, (request.output_dir.parent,)),
        "--fasta": _portable_path(request.fasta, (request.output_dir.parent,)),
        "--output": _portable_path(request.scores_path, (request.output_dir.parent,)),
        "--receipt": _portable_path(request.receipts_path, (request.output_dir.parent,)),
    }
    for flag, value in replacements.items():
        try:
            index = command.index(flag)
        except ValueError:
            continue
        value_index = index + 1
        if value_index < len(command):
            command[value_index] = value
    return tuple(command)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    request = RuntimePreflightRequest(
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
    output = args.output or args.output_dir / REPORT_NAME
    try:
        report = write_runtime_preflight_report(request, output)
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {output}\n")
    return 0 if report.ok else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build runtime_preflight_report.json before a terminal demo release.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vcf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--command", default=DEFAULT_COMMAND)
    parser.add_argument(
        "--allow-fixture-manifest",
        action="store_true",
        help="Allow fixture/test manifests for local tool tests only.",
    )
    parser.add_argument(
        "--no-require-native-runtime",
        action="store_true",
        help="Do not fail the report when optional native runtime dependencies are absent.",
    )
    parser.add_argument("--carbon-cache-dir", type=Path)
    parser.add_argument(
        "--require-carbon-cache",
        action="store_true",
        help="Require --carbon-cache-dir to point at a non-empty local Carbon cache marker.",
    )
    return parser


def _load_manifest(model_dir: Path, issues: list[RuntimePreflightIssue]) -> Manifest | None:
    path = model_dir / "manifest.json"
    if not path.is_file():
        _issue(issues, "error", "model.manifest_missing", path, "manifest.json is required")
        return None
    try:
        return load_manifest(path)
    except GenoLeWMError as exc:
        _issue(
            issues,
            "error",
            "model.manifest_invalid",
            path,
            exc.message or str(exc),
        )
        return None


def _manifest_identity(
    model_dir: Path,
    manifest: Manifest | None,
    *,
    roots: tuple[Path, ...] = (),
) -> dict[str, object]:
    path = model_dir / "manifest.json"
    payload: dict[str, object] = {
        "path": _portable_path(path, roots),
        "exists": path.is_file(),
    }
    if path.is_file():
        payload["sha256"] = sha256_file(path)
        payload["size_bytes"] = path.stat().st_size
    if manifest is not None:
        payload.update(
            {
                "schema_version": manifest.schema_version,
                "model_name": manifest.model_name,
                "model_version": manifest.model_version,
                "release_id": manifest.release_id,
                "model_id": manifest.model_id(),
                "encoder_id": manifest.encoder.id,
                "encoder_revision": manifest.encoder.revision,
                "encoder_hash": manifest.encoder.hash,
            }
        )
    return payload


def _artifact_checks(
    model_dir: Path,
    manifest: Manifest | None,
    issues: list[RuntimePreflightIssue],
    *,
    roots: tuple[Path, ...] = (),
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if manifest is not None:
        expected = (
            ("predictor", manifest.predictor.file, manifest.predictor.hash),
            ("action_encoder", manifest.action_encoder.file, manifest.action_encoder.hash),
            ("calibration", manifest.calibration.file, manifest.calibration.hash),
            ("training_config", manifest.training.config_file, manifest.training.hash),
            ("eval_report", manifest.eval.file, manifest.eval.hash),
        )
        for label, relative, expected_hash in expected:
            checks.append(
                _required_file_identity(
                    model_dir / relative,
                    issues,
                    code_prefix=f"model.{label}",
                    expected_sha256=expected_hash,
                    roots=roots,
                )
            )
    checks.append(
        _required_file_identity(
            model_dir / "model_card.md",
            issues,
            code_prefix="model.model_card",
            roots=roots,
        )
    )
    return checks


def _required_file_identity(
    path: Path,
    issues: list[RuntimePreflightIssue],
    *,
    code_prefix: str,
    expected_sha256: str | None = None,
    roots: tuple[Path, ...] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": _portable_path(path, roots),
        "exists": path.is_file(),
        "expected_sha256": expected_sha256,
        "ok": False,
    }
    if not path.is_file():
        _issue(issues, "error", f"{code_prefix}.missing", path, "required file is missing")
        return payload
    size_bytes = path.stat().st_size
    payload["size_bytes"] = size_bytes
    if size_bytes <= 0:
        _issue(issues, "error", f"{code_prefix}.empty", path, "required file is empty")
        return payload
    observed = sha256_file(path)
    payload["sha256"] = observed
    if expected_sha256 is not None:
        if not looks_like_sha256(expected_sha256):
            _issue(
                issues,
                "error",
                f"{code_prefix}.expected_hash_invalid",
                path,
                "expected hash is not a sha256:<64hex> string",
            )
            return payload
        if observed != expected_sha256:
            _issue(
                issues,
                "error",
                f"{code_prefix}.hash_mismatch",
                path,
                f"expected {expected_sha256}, observed {observed}",
            )
            return payload
    payload["ok"] = True
    return payload


def _portable_path(path: Path, roots: tuple[Path, ...]) -> str:
    for root in roots:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, RuntimeError, ValueError):
            continue
    return str(path)


def _record_dependency_issues(
    dependencies: tuple[DependencyProbe, ...],
    issues: list[RuntimePreflightIssue],
    *,
    required: bool,
) -> None:
    for probe in dependencies:
        if probe.available:
            continue
        severity: Severity = "error" if required and probe.required else "warning"
        _issue(
            issues,
            severity,
            "runtime.dependency_unavailable",
            probe.import_name,
            probe.reason,
        )


def _runtime_backend_report(
    request: RuntimePreflightRequest,
    issues: list[RuntimePreflightIssue],
) -> tuple[list[dict[str, object]], str | None]:
    try:
        probes = probe_backends(request.model_dir)
        selected = select_backend(request.backend, probes=probes)
    except GenoLeWMError as exc:
        _issue(
            issues,
            "error",
            "runtime.backend_unavailable",
            request.model_dir,
            exc.message or str(exc),
        )
        return [], None
    return (
        [
            {
                "backend": probe.backend,
                "available": probe.available,
                "reason": probe.reason,
            }
            for probe in probes
        ],
        selected,
    )


def _network_guard_report(issues: list[RuntimePreflightIssue]) -> dict[str, object]:
    try:
        with fail_closed_network_guard():
            socket.create_connection(("127.0.0.1", 9), timeout=0.001)
    except NetworkCallProhibitedError:
        return {
            "ok": True,
            "probe": "socket.create_connection",
            "reason": "network calls are blocked inside runtime guard",
        }
    except Exception as exc:
        _issue(
            issues,
            "error",
            "runtime.network_guard_failed",
            "socket.create_connection",
            f"network guard did not raise NetworkCallProhibitedError: {exc}",
        )
        return {
            "ok": False,
            "probe": "socket.create_connection",
            "reason": str(exc),
        }
    _issue(
        issues,
        "error",
        "runtime.network_guard_failed",
        "socket.create_connection",
        "network guard allowed a connection attempt",
    )
    return {
        "ok": False,
        "probe": "socket.create_connection",
        "reason": "network guard allowed a connection attempt",
    }


def _carbon_report(
    request: RuntimePreflightRequest,
    manifest: Manifest | None,
    issues: list[RuntimePreflightIssue],
) -> dict[str, object]:
    cache_dir = request.carbon_cache_dir
    cache_exists = False
    cache_nonempty = False
    if cache_dir is not None:
        cache_exists = cache_dir.is_dir()
        cache_nonempty = cache_exists and any(cache_dir.iterdir())
    if request.require_carbon_cache and not cache_nonempty:
        _issue(
            issues,
            "error",
            "carbon.cache_missing",
            "" if cache_dir is None else cache_dir,
            "--require-carbon-cache needs a non-empty --carbon-cache-dir",
        )
    elif cache_dir is None:
        _issue(
            issues,
            "warning",
            "carbon.cache_unverified",
            "",
            "no Carbon cache directory was provided; terminal command remains the final check",
        )
    return {
        "encoder_id": None if manifest is None else manifest.encoder.id,
        "encoder_revision": None if manifest is None else manifest.encoder.revision,
        "encoder_hash": None if manifest is None else manifest.encoder.hash,
        "local_files_only": True,
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "cache_exists": cache_exists,
        "cache_nonempty": cache_nonempty,
        "cache_required": request.require_carbon_cache,
    }


def _probe_dependency(import_name: str, required: bool) -> DependencyProbe:
    package = import_name.split(".", 1)[0]
    try:
        spec = importlib.util.find_spec(import_name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return DependencyProbe(
            import_name=import_name,
            package=package,
            required=required,
            available=False,
            version=None,
            reason=f"{import_name} import probe failed: {exc}",
        )
    if spec is None:
        return DependencyProbe(
            import_name=import_name,
            package=package,
            required=required,
            available=False,
            version=None,
            reason=f"{import_name} is not installed",
        )
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return DependencyProbe(
        import_name=import_name,
        package=package,
        required=required,
        available=True,
        version=version,
        reason="importable",
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


def _issue(
    issues: list[RuntimePreflightIssue],
    severity: Severity,
    code: str,
    path: str | Path,
    message: str,
) -> None:
    issues.append(
        RuntimePreflightIssue(
            severity=severity,
            code=code,
            path=str(path),
            message=message,
        )
    )


def _require_positive_batch_size(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            "batch_size must be a positive integer",
            details={"batch_size": value, "type": type(value).__name__},
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
