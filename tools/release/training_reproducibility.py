# SPDX-License-Identifier: Apache-2.0
"""Compare real training-run archives for deterministic reproducibility evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from tools.release.training_run import TrainingRunManifest, verify_training_run_manifest

REPORT_NAME: Final = "training_reproducibility_report.json"
SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.training_reproducibility"
_DETERMINISTIC_PAIR_FIELDS: Final = (
    "run_id",
    "commit_sha",
    "package_version",
    "dataset_snapshot_id",
    "seeds",
    "determinism",
)
_THROUGHPUT_IDENTITY_FIELDS: Final = (
    "commit_sha",
    "package_version",
    "dataset_snapshot_id",
    "seeds",
)
_REQUIRED_PAIR_ARTIFACT_KINDS: Final = frozenset(
    {"dataset_manifest", "training_config", "checkpoint"}
)
_THROUGHPUT_KEYS: Final = (
    "samples_per_second",
    "samples_per_s",
    "throughput_samples_per_s",
    "train_samples_per_second",
)
_ELAPSED_KEYS: Final = ("elapsed_seconds", "wall_time_seconds", "duration_seconds")


@dataclass(frozen=True, slots=True)
class ReproducibilityBlocker:
    """One failed reproducibility-evidence requirement."""

    code: str
    message: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Stable identity for one training-run artifact."""

    key: str
    kind: str
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class TrainingRunEvidence:
    """Validated evidence loaded from one training-run archive."""

    label: str
    run_dir: str
    run_id: str
    commit_sha: str
    package_version: str
    dataset_snapshot_id: str
    seeds: dict[str, int]
    determinism: str
    deterministic_enabled: bool
    sample_count: int | None
    samples_per_second: float | None
    artifacts: tuple[ArtifactIdentity, ...]

    def artifact_map(self) -> dict[str, ArtifactIdentity]:
        return {artifact.key: artifact for artifact in self.artifacts}

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "commit_sha": self.commit_sha,
            "package_version": self.package_version,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "seeds": self.seeds,
            "determinism": self.determinism,
            "deterministic_enabled": self.deterministic_enabled,
            "sample_count": self.sample_count,
            "samples_per_second": self.samples_per_second,
            "artifacts": {artifact.key: artifact.to_dict() for artifact in self.artifacts},
        }


@dataclass(frozen=True, slots=True)
class DeterministicPairComparison:
    """Bit-exact deterministic rerun comparison."""

    ok: bool
    matched_fields: tuple[str, ...]
    matched_artifacts: tuple[str, ...]
    blockers: tuple[ReproducibilityBlocker, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "matched_fields": list(self.matched_fields),
            "matched_artifacts": list(self.matched_artifacts),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


@dataclass(frozen=True, slots=True)
class ThroughputComparison:
    """Deterministic-throughput comparison against a non-deterministic baseline."""

    ok: bool
    max_drop_fraction: float
    baseline_samples_per_second: float | None
    deterministic_samples_per_second: dict[str, float]
    deterministic_min_samples_per_second: float | None
    drop_fraction: float | None
    blockers: tuple[ReproducibilityBlocker, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "max_drop_fraction": self.max_drop_fraction,
            "baseline_samples_per_second": self.baseline_samples_per_second,
            "deterministic_samples_per_second": self.deterministic_samples_per_second,
            "deterministic_min_samples_per_second": self.deterministic_min_samples_per_second,
            "drop_fraction": self.drop_fraction,
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


@dataclass(frozen=True, slots=True)
class TrainingReproducibilityReport:
    """Machine-readable deterministic training evidence."""

    schema_version: str
    generated_by: str
    generated_at: str
    ok: bool
    max_throughput_drop_fraction: float
    runs: tuple[TrainingRunEvidence, ...]
    deterministic_pair: DeterministicPairComparison
    throughput: ThroughputComparison

    @property
    def blockers(self) -> tuple[ReproducibilityBlocker, ...]:
        return (*self.deterministic_pair.blockers, *self.throughput.blockers)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "ok": self.ok,
            "max_throughput_drop_fraction": self.max_throughput_drop_fraction,
            "runs": [run.to_dict() for run in self.runs],
            "deterministic_pair": self.deterministic_pair.to_dict(),
            "throughput": self.throughput.to_dict(),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


def build_training_reproducibility_report(
    *,
    deterministic_run_a: Path,
    deterministic_run_b: Path,
    baseline_run_dir: Path | None = None,
    max_throughput_drop_fraction: float = 0.15,
    require_preflight: bool = False,
    generated_at: str | None = None,
) -> TrainingReproducibilityReport:
    """Build a report for issue #47's remaining real-run evidence gates."""
    _validate_drop_threshold(max_throughput_drop_fraction)
    run_a = load_training_run_evidence(
        deterministic_run_a,
        label="deterministic_a",
        require_preflight=require_preflight,
    )
    run_b = load_training_run_evidence(
        deterministic_run_b,
        label="deterministic_b",
        require_preflight=require_preflight,
    )
    baseline = (
        None
        if baseline_run_dir is None
        else load_training_run_evidence(
            baseline_run_dir,
            label="baseline",
            require_preflight=require_preflight,
        )
    )
    deterministic_pair = compare_deterministic_pair(run_a, run_b)
    throughput = compare_throughput(
        baseline,
        run_a,
        run_b,
        max_drop_fraction=max_throughput_drop_fraction,
    )
    runs = (run_a, run_b) if baseline is None else (baseline, run_a, run_b)
    ok = deterministic_pair.ok and throughput.ok
    return TrainingReproducibilityReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        ok=ok,
        max_throughput_drop_fraction=max_throughput_drop_fraction,
        runs=runs,
        deterministic_pair=deterministic_pair,
        throughput=throughput,
    )


def load_training_run_evidence(
    run_dir: Path,
    *,
    label: str,
    require_preflight: bool = False,
) -> TrainingRunEvidence:
    """Load one validated training-run archive into comparison-ready evidence."""
    manifest = verify_training_run_manifest(run_dir, require_preflight=require_preflight)
    metrics = _load_metrics_payload(run_dir, manifest)
    return TrainingRunEvidence(
        label=label,
        run_dir=_public_path(run_dir),
        run_id=manifest.run_id,
        commit_sha=manifest.commit_sha,
        package_version=manifest.package_version,
        dataset_snapshot_id=manifest.dataset_snapshot_id,
        seeds=dict(manifest.seeds),
        determinism=manifest.determinism,
        deterministic_enabled=_deterministic_enabled(manifest.determinism),
        sample_count=_extract_sample_count(metrics),
        samples_per_second=_extract_samples_per_second(metrics),
        artifacts=_artifact_identities(manifest),
    )


def compare_deterministic_pair(
    run_a: TrainingRunEvidence,
    run_b: TrainingRunEvidence,
) -> DeterministicPairComparison:
    """Compare two deterministic runs for bit-exact checkpoint evidence."""
    blockers: list[ReproducibilityBlocker] = []
    matched_fields: list[str] = []
    for field in _DETERMINISTIC_PAIR_FIELDS:
        left = getattr(run_a, field)
        right = getattr(run_b, field)
        if left == right:
            matched_fields.append(field)
        else:
            blockers.append(
                _blocker(
                    "deterministic_pair.field_mismatch",
                    "deterministic run identity fields must match",
                    field=field,
                    run_a=left,
                    run_b=right,
                )
            )
    if not run_a.deterministic_enabled:
        blockers.append(
            _blocker(
                "deterministic_pair.run_a_not_deterministic",
                "first deterministic run does not record deterministic torch controls",
                run=run_a.run_dir,
            )
        )
    if not run_b.deterministic_enabled:
        blockers.append(
            _blocker(
                "deterministic_pair.run_b_not_deterministic",
                "second deterministic run does not record deterministic torch controls",
                run=run_b.run_dir,
            )
        )

    matched_artifacts = _compare_pair_artifacts(run_a, run_b, blockers)
    return DeterministicPairComparison(
        ok=not blockers,
        matched_fields=tuple(matched_fields),
        matched_artifacts=tuple(matched_artifacts),
        blockers=tuple(blockers),
    )


def compare_throughput(
    baseline: TrainingRunEvidence | None,
    run_a: TrainingRunEvidence,
    run_b: TrainingRunEvidence,
    *,
    max_drop_fraction: float,
) -> ThroughputComparison:
    """Compare deterministic throughput to a non-deterministic baseline run."""
    _validate_drop_threshold(max_drop_fraction)
    blockers: list[ReproducibilityBlocker] = []
    if baseline is None:
        blockers.append(
            _blocker(
                "throughput.missing_baseline_run",
                "deterministic throughput evidence requires a baseline run directory",
            )
        )
        return ThroughputComparison(
            ok=False,
            max_drop_fraction=max_drop_fraction,
            baseline_samples_per_second=None,
            deterministic_samples_per_second={},
            deterministic_min_samples_per_second=None,
            drop_fraction=None,
            blockers=tuple(blockers),
        )

    if baseline.deterministic_enabled:
        blockers.append(
            _blocker(
                "throughput.baseline_is_deterministic",
                "baseline run must not enable deterministic torch controls",
                run=baseline.run_dir,
            )
        )
    blockers.extend(
        _blocker(
            "throughput.deterministic_run_not_deterministic",
            "deterministic throughput run must record deterministic torch controls",
            run=run.run_dir,
        )
        for run in (run_a, run_b)
        if not run.deterministic_enabled
    )
    for field in _THROUGHPUT_IDENTITY_FIELDS:
        expected = getattr(run_a, field)
        for observed in (baseline, run_b):
            value = getattr(observed, field)
            if value != expected:
                blockers.append(
                    _blocker(
                        "throughput.identity_mismatch",
                        "throughput runs must use the same commit, package, dataset, and seeds",
                        field=field,
                        expected=expected,
                        observed=value,
                        run=observed.run_dir,
                    )
                )

    if baseline.sample_count is None:
        blockers.append(
            _blocker(
                "throughput.missing_sample_count",
                "baseline metrics must record a positive sample_count",
                run=baseline.run_dir,
            )
        )
    for run in (run_a, run_b):
        if run.sample_count is None:
            blockers.append(
                _blocker(
                    "throughput.missing_sample_count",
                    "deterministic metrics must record a positive sample_count",
                    run=run.run_dir,
                )
            )
        elif baseline.sample_count is not None and run.sample_count != baseline.sample_count:
            blockers.append(
                _blocker(
                    "throughput.sample_count_mismatch",
                    "throughput runs must measure the same sample count",
                    baseline=baseline.sample_count,
                    deterministic=run.sample_count,
                    run=run.run_dir,
                )
            )

    baseline_rate = baseline.samples_per_second
    deterministic_rates = {
        run.label: run.samples_per_second
        for run in (run_a, run_b)
        if run.samples_per_second is not None
    }
    if baseline_rate is None:
        blockers.append(
            _blocker(
                "throughput.missing_baseline_rate",
                "baseline metrics must record samples_per_second or elapsed_seconds",
                run=baseline.run_dir,
            )
        )
    blockers.extend(
        _blocker(
            "throughput.missing_deterministic_rate",
            "deterministic metrics must record samples_per_second or elapsed_seconds",
            run=run.run_dir,
        )
        for run in (run_a, run_b)
        if run.samples_per_second is None
    )

    deterministic_min = min(deterministic_rates.values()) if len(deterministic_rates) == 2 else None
    drop_fraction = (
        None
        if baseline_rate is None or deterministic_min is None
        else max(0.0, (baseline_rate - deterministic_min) / baseline_rate)
    )
    if drop_fraction is not None and drop_fraction > max_drop_fraction:
        blockers.append(
            _blocker(
                "throughput.drop_exceeds_threshold",
                "deterministic throughput drop exceeds the configured threshold",
                baseline_samples_per_second=baseline_rate,
                deterministic_min_samples_per_second=deterministic_min,
                drop_fraction=drop_fraction,
                max_drop_fraction=max_drop_fraction,
            )
        )

    return ThroughputComparison(
        ok=not blockers,
        max_drop_fraction=max_drop_fraction,
        baseline_samples_per_second=baseline_rate,
        deterministic_samples_per_second=dict(deterministic_rates),
        deterministic_min_samples_per_second=deterministic_min,
        drop_fraction=drop_fraction,
        blockers=tuple(blockers),
    )


def write_training_reproducibility_report(
    *,
    deterministic_run_a: Path,
    deterministic_run_b: Path,
    output: Path,
    baseline_run_dir: Path | None = None,
    max_throughput_drop_fraction: float = 0.15,
    require_preflight: bool = False,
) -> TrainingReproducibilityReport:
    """Build and write a normalized training reproducibility report."""
    report = build_training_reproducibility_report(
        deterministic_run_a=deterministic_run_a,
        deterministic_run_b=deterministic_run_b,
        baseline_run_dir=baseline_run_dir,
        max_throughput_drop_fraction=max_throughput_drop_fraction,
        require_preflight=require_preflight,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = write_training_reproducibility_report(
            deterministic_run_a=args.deterministic_run_a,
            deterministic_run_b=args.deterministic_run_b,
            baseline_run_dir=args.baseline_run_dir,
            output=args.output,
            max_throughput_drop_fraction=args.max_throughput_drop,
            require_preflight=args.require_preflight,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0 if report.ok else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare completed training-run archives for deterministic rerun and "
            "deterministic-throughput evidence."
        ),
    )
    parser.add_argument("--deterministic-run-a", type=Path, required=True)
    parser.add_argument("--deterministic-run-b", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-throughput-drop", type=float, default=0.15)
    parser.add_argument(
        "--require-preflight",
        action="store_true",
        help="Require a validated training_preflight_report.json artifact in each run.",
    )
    return parser


def _compare_pair_artifacts(
    run_a: TrainingRunEvidence,
    run_b: TrainingRunEvidence,
    blockers: list[ReproducibilityBlocker],
) -> list[str]:
    left = {
        key: artifact
        for key, artifact in run_a.artifact_map().items()
        if artifact.kind in _REQUIRED_PAIR_ARTIFACT_KINDS
    }
    right = {
        key: artifact
        for key, artifact in run_b.artifact_map().items()
        if artifact.kind in _REQUIRED_PAIR_ARTIFACT_KINDS
    }
    matched: list[str] = []
    missing_left = sorted(set(right) - set(left))
    missing_right = sorted(set(left) - set(right))
    if missing_left or missing_right:
        blockers.append(
            _blocker(
                "deterministic_pair.artifact_set_mismatch",
                "deterministic runs must expose matching dataset, config, and checkpoint artifacts",
                missing_from_run_a=missing_left,
                missing_from_run_b=missing_right,
            )
        )
    observed_kinds = {artifact.kind for artifact in (*left.values(), *right.values())}
    missing_kinds = sorted(_REQUIRED_PAIR_ARTIFACT_KINDS - observed_kinds)
    if missing_kinds:
        blockers.append(
            _blocker(
                "deterministic_pair.missing_required_artifact_kind",
                "deterministic comparison requires dataset, config, and checkpoint artifacts",
                missing_kinds=missing_kinds,
            )
        )
    for key in sorted(set(left) & set(right)):
        left_artifact = left[key]
        right_artifact = right[key]
        if (
            left_artifact.sha256 == right_artifact.sha256
            and left_artifact.size_bytes == right_artifact.size_bytes
        ):
            matched.append(key)
            continue
        blockers.append(
            _blocker(
                "deterministic_pair.artifact_mismatch",
                "deterministic reruns must produce bit-exact config, dataset, and checkpoint artifacts",
                artifact=key,
                run_a_sha256=left_artifact.sha256,
                run_b_sha256=right_artifact.sha256,
                run_a_size_bytes=left_artifact.size_bytes,
                run_b_size_bytes=right_artifact.size_bytes,
            )
        )
    return matched


def _load_metrics_payload(run_dir: Path, manifest: TrainingRunManifest) -> dict[str, Any]:
    metrics_artifacts = tuple(
        artifact for artifact in manifest.artifacts if artifact.kind == "metrics"
    )
    if len(metrics_artifacts) != 1:
        raise InputError(
            "training-run manifest must contain exactly one metrics artifact",
            details={"observed": len(metrics_artifacts)},
        )
    path = run_dir / metrics_artifacts[0].path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError("failed to read training metrics", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "training metrics JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("training metrics must be a JSON object", details={"path": str(path)})
    return payload


def _artifact_identities(manifest: TrainingRunManifest) -> tuple[ArtifactIdentity, ...]:
    artifacts = [
        ArtifactIdentity(
            key=f"{artifact.kind}:{artifact.path}",
            kind=artifact.kind,
            path=artifact.path,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
        )
        for artifact in manifest.artifacts
    ]
    return tuple(sorted(artifacts, key=lambda artifact: artifact.key))


def _deterministic_enabled(determinism: str) -> bool:
    try:
        payload = json.loads(determinism)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("deterministic") is True
        and payload.get("torch_deterministic_algorithms") is True
    )


def _extract_samples_per_second(payload: dict[str, Any]) -> float | None:
    for key in _THROUGHPUT_KEYS:
        value = _positive_float(_metric_value(payload, key))
        if value is not None:
            return value
    sample_count = _extract_sample_count(payload)
    if sample_count is None:
        return None
    for key in _ELAPSED_KEYS:
        elapsed = _positive_float(_metric_value(payload, key))
        if elapsed is not None:
            return sample_count / elapsed
    return None


def _extract_sample_count(payload: dict[str, Any]) -> int | None:
    value = _metric_value(payload, "sample_count")
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _metric_value(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(key)
    return None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        return None
    return converted


def _validate_drop_threshold(value: float) -> None:
    if not math.isfinite(value) or value < 0 or value >= 1:
        raise InputError(
            "max throughput drop fraction must be in [0, 1)",
            details={"max_throughput_drop_fraction": value},
        )


def _blocker(code: str, message: str, **details: object) -> ReproducibilityBlocker:
    return ReproducibilityBlocker(code=code, message=message, details=dict(details))


def _public_path(path: Path) -> str:
    if path.is_absolute():
        return path.name
    if ".." in path.parts or not path.parts:
        return path.name
    return path.as_posix()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
