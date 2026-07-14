# SPDX-License-Identifier: Apache-2.0
"""Validate the immutable issue #47 H200 N-D-D-N launch contract."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml

from geno_lewm.config import load_config
from geno_lewm.errors import GenoLeWMError
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "2.0.0"
GENERATED_BY: Final = "tools.research.training_reproducibility_preflight"

EXPECTED_DETERMINISTIC_CONFIG_PATH: Final = Path(
    "configs/reproducibility/train-carbon-500m-snv-deterministic-500.yaml"
)
EXPECTED_BASELINE_CONFIG_PATH: Final = Path(
    "configs/reproducibility/train-carbon-500m-snv-baseline-500.yaml"
)
EXPECTED_DATASET_REFERENCE_PATH: Final = Path("configs/reproducibility/dataset-reference-r2.json")
EXPECTED_CONTAINER_IMAGE: Final = (
    "ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
)
EXPECTED_DATASET_REPO: Final = "abdelstark/geno-lewm-runs"
EXPECTED_UPLOAD_REPO: Final = "abdelstark/geno-lewm-runs"
EXPECTED_DATASET_REVISION: Final = "1200467a6b940cb5b1230d9a7db0be74e51bd50d"
EXPECTED_DATASET_PATH: Final = "geno-lewm-l2-p1-smoke-304128e4d4f3-50-r2/dataset"
EXPECTED_DATASET_SNAPSHOT_ID: Final = "geno-lewm-data-correction-control-l2-p1-proof-v1"
EXPECTED_DATASET_MANIFEST_SHA256: Final = (
    "sha256:8d60360f365185451ebac80cb8c37f8aa4324bb915e16243ac9ce661d6748621"
)
EXPECTED_RUN_ID: Final = "training-reproducibility-h200-nddn-500-v2"
EXPECTED_STEPS: Final = 500
EXPECTED_SAMPLE_COUNT: Final = 4_000
EXPECTED_SEED: Final = 104729
EXPECTED_CARBON_MODEL_DIR: Final = "/carbon"
EXPECTED_CARBON_REVISION: Final = "5d31d59b3c845b288a13aedb1358934196852eec"
EXPECTED_CARBON_RUNTIME_HASH: Final = (
    "sha256:a1fd1dd20756c7248b7f9ca95c59c821f0329530fd49c6fea253a8df9a6a6311"
)
EXPECTED_MIN_CUDA_VRAM_GB: Final = 120.0
EXPECTED_MAX_THROUGHPUT_DROP: Final = 0.15
EXPECTED_MAX_REPEAT_SPREAD: Final = 0.05
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}")
_MISSING: Final = object()


@dataclass(frozen=True, slots=True)
class TrainingReproducibilityPreflightRequest:
    """Inputs defining one immutable H200 N-D-D-N launch."""

    repo_root: Path
    deterministic_config: Path
    baseline_config: Path
    dataset_reference: Path
    expected_commit_sha: str
    run_name: str
    run_attempt: int
    steps: int
    expected_sample_count: int
    dataset_repo: str
    dataset_revision: str
    dataset_path: str
    carbon_model_dir: str
    expected_carbon_runtime_hash: str
    upload_repo: str
    container_image: str
    min_cuda_vram_gb: float
    max_throughput_drop: float
    max_repeat_spread: float


@dataclass(frozen=True, slots=True)
class TrainingReproducibilityPreflightIssue:
    """One mismatch against the immutable launch contract."""

    code: str
    path: str
    message: str
    expected: object | None
    observed: object | None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "expected": _json_value(self.expected),
            "observed": _json_value(self.observed),
        }


@dataclass(frozen=True, slots=True)
class TrainingReproducibilityPreflightReport:
    """Machine-readable static launch validation."""

    schema_version: str
    generated_by: str
    generated_at: str
    ok: bool
    repository: dict[str, object]
    job: dict[str, object]
    configs: dict[str, object]
    dataset: dict[str, object]
    issues: tuple[TrainingReproducibilityPreflightIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "ok": self.ok,
            "repository": self.repository,
            "job": self.job,
            "configs": self.configs,
            "dataset": self.dataset,
            "issues": [issue.to_dict() for issue in self.issues],
            "claim_boundary": (
                "This preflight validates a bounded four-arm engineering benchmark launch. "
                "It does not establish model quality, scientific utility, or clinical validity."
            ),
        }


def build_training_reproducibility_preflight_report(
    request: TrainingReproducibilityPreflightRequest,
    *,
    generated_at: str | None = None,
) -> TrainingReproducibilityPreflightReport:
    """Validate the N-D-D-N request without accelerator, network, or output writes."""
    issues: list[TrainingReproducibilityPreflightIssue] = []
    root = request.repo_root.resolve()
    deterministic_path = _repo_path(root, request.deterministic_config)
    baseline_path = _repo_path(root, request.baseline_config)
    dataset_path = _repo_path(root, request.dataset_reference)

    observed_root, observed_head, git_error = _git_identity(root)
    dirty_paths, status_error = _git_dirty_paths(root) if git_error is None else ((), None)
    if git_error is not None:
        _issue(
            issues,
            "repository.git_unavailable",
            "repo_root",
            "repository identity could not be resolved",
            "a Git worktree",
            git_error,
        )
    else:
        _expect(
            issues,
            "repository.root_mismatch",
            "repo_root",
            None if observed_root is None else str(observed_root),
            str(root),
        )
        _expect(
            issues,
            "repository.head_mismatch",
            "expected_commit_sha",
            observed_head,
            request.expected_commit_sha,
        )
        if status_error is not None:
            _issue(
                issues,
                "repository.status_unavailable",
                "repo_root",
                "repository status could not be resolved",
                "a readable clean worktree",
                status_error,
            )
        elif dirty_paths:
            _issue(
                issues,
                "repository.worktree_dirty",
                "repo_root",
                "reproducibility execution requires a clean exact-SHA worktree",
                [],
                list(dirty_paths),
            )

    if _COMMIT_RE.fullmatch(request.expected_commit_sha) is None:
        _issue(
            issues,
            "request.commit_sha_format",
            "expected_commit_sha",
            "expected commit must be a full lowercase Git SHA",
            "40 lowercase hexadecimal characters",
            request.expected_commit_sha,
        )
    attempt_valid = (
        isinstance(request.run_attempt, int)
        and not isinstance(request.run_attempt, bool)
        and request.run_attempt > 0
    )
    expected_run_name = (
        f"geno-lewm-repro-h200-{request.expected_commit_sha[:12]}-500-r{request.run_attempt}"
        if attempt_valid and _COMMIT_RE.fullmatch(request.expected_commit_sha)
        else None
    )
    _expect(
        issues,
        "request.run_name_mismatch",
        "run_name",
        request.run_name,
        expected_run_name,
    )
    _validate_request_values(request, issues)
    _expect_path(
        issues,
        "request.deterministic_config_path_mismatch",
        "deterministic_config",
        deterministic_path,
        (root / EXPECTED_DETERMINISTIC_CONFIG_PATH).resolve(),
    )
    _expect_path(
        issues,
        "request.baseline_config_path_mismatch",
        "baseline_config",
        baseline_path,
        (root / EXPECTED_BASELINE_CONFIG_PATH).resolve(),
    )
    _expect_path(
        issues,
        "request.dataset_reference_path_mismatch",
        "dataset_reference",
        dataset_path,
        (root / EXPECTED_DATASET_REFERENCE_PATH).resolve(),
    )

    deterministic = _load_yaml(deterministic_path, "deterministic_config", issues)
    baseline = _load_yaml(baseline_path, "baseline_config", issues)
    if deterministic is not None:
        _validate_training_config(deterministic, True, deterministic_path, issues)
    if baseline is not None:
        _validate_training_config(baseline, False, baseline_path, issues)
    if deterministic is not None and baseline is not None:
        expected_baseline = dict(deterministic)
        expected_baseline["deterministic"] = False
        _expect(
            issues,
            "config.baseline_diff",
            "baseline_config",
            baseline,
            expected_baseline,
        )

    dataset_reference = _load_json(dataset_path, "dataset_reference", issues)
    if dataset_reference is not None:
        _validate_dataset_reference(dataset_reference, dataset_path, issues)

    repository: dict[str, object] = {
        "root": ".",
        "expected_commit_sha": request.expected_commit_sha,
        "observed_commit_sha": observed_head,
        "worktree_clean": status_error is None and not dirty_paths,
        "dirty_paths": list(dirty_paths),
    }
    job: dict[str, object] = {
        "run_name": request.run_name,
        "run_attempt": request.run_attempt,
        "execution_order": [
            "baseline_a",
            "deterministic_a",
            "deterministic_b",
            "baseline_b",
        ],
        "steps": request.steps,
        "expected_sample_count": request.expected_sample_count,
        "carbon_model_dir": request.carbon_model_dir,
        "expected_carbon_runtime_hash": request.expected_carbon_runtime_hash,
        "upload_repo": request.upload_repo,
        "container_image": request.container_image,
        "min_cuda_vram_gb": request.min_cuda_vram_gb,
        "max_throughput_drop": request.max_throughput_drop,
        "max_repeat_spread": request.max_repeat_spread,
    }
    configs: dict[str, object] = {
        "deterministic": _file_summary(root, deterministic_path, deterministic),
        "baseline": _file_summary(root, baseline_path, baseline),
    }
    dataset: dict[str, object] = {
        "reference": _file_summary(root, dataset_path, dataset_reference),
        "source": {
            "repo_id": request.dataset_repo,
            "repo_type": "model",
            "revision": request.dataset_revision,
            "path": request.dataset_path,
        },
    }
    return TrainingReproducibilityPreflightReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        ok=not issues,
        repository=repository,
        job=job,
        configs=configs,
        dataset=dataset,
        issues=tuple(issues),
    )


def write_training_reproducibility_preflight_report(
    request: TrainingReproducibilityPreflightRequest,
    output: Path,
) -> TrainingReproducibilityPreflightReport:
    """Validate and write the launch report."""
    report = build_training_reproducibility_preflight_report(request)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the immutable launch preflight CLI."""
    args = _parser().parse_args(argv)
    request = TrainingReproducibilityPreflightRequest(
        repo_root=args.repo_root,
        deterministic_config=args.deterministic_config,
        baseline_config=args.baseline_config,
        dataset_reference=args.dataset_reference,
        expected_commit_sha=args.expected_commit_sha,
        run_name=args.run_name,
        run_attempt=args.run_attempt,
        steps=args.steps,
        expected_sample_count=args.expected_sample_count,
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
        dataset_path=args.dataset_path,
        carbon_model_dir=args.carbon_model_dir,
        expected_carbon_runtime_hash=args.expected_carbon_runtime_hash,
        upload_repo=args.upload_repo,
        container_image=args.container_image,
        min_cuda_vram_gb=args.min_cuda_vram_gb,
        max_throughput_drop=args.max_throughput_drop,
        max_repeat_spread=args.max_repeat_spread,
    )
    report = (
        build_training_reproducibility_preflight_report(request)
        if args.output is None
        else write_training_reproducibility_preflight_report(request, args.output)
    )
    sys.stdout.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
    return 0 if report.ok else 2


def _validate_request_values(
    request: TrainingReproducibilityPreflightRequest,
    issues: list[TrainingReproducibilityPreflightIssue],
) -> None:
    expected = (
        ("steps", request.steps, EXPECTED_STEPS),
        ("expected_sample_count", request.expected_sample_count, EXPECTED_SAMPLE_COUNT),
        ("dataset_repo", request.dataset_repo, EXPECTED_DATASET_REPO),
        ("dataset_revision", request.dataset_revision, EXPECTED_DATASET_REVISION),
        ("dataset_path", request.dataset_path, EXPECTED_DATASET_PATH),
        ("carbon_model_dir", request.carbon_model_dir, EXPECTED_CARBON_MODEL_DIR),
        (
            "expected_carbon_runtime_hash",
            request.expected_carbon_runtime_hash,
            EXPECTED_CARBON_RUNTIME_HASH,
        ),
        ("upload_repo", request.upload_repo, EXPECTED_UPLOAD_REPO),
        ("container_image", request.container_image, EXPECTED_CONTAINER_IMAGE),
        ("min_cuda_vram_gb", request.min_cuda_vram_gb, EXPECTED_MIN_CUDA_VRAM_GB),
        (
            "max_throughput_drop",
            request.max_throughput_drop,
            EXPECTED_MAX_THROUGHPUT_DROP,
        ),
        (
            "max_repeat_spread",
            request.max_repeat_spread,
            EXPECTED_MAX_REPEAT_SPREAD,
        ),
    )
    for name, observed, wanted in expected:
        _expect(issues, f"request.{name}_mismatch", name, observed, wanted)


def _validate_training_config(
    payload: dict[str, Any],
    deterministic: bool,
    path: Path,
    issues: list[TrainingReproducibilityPreflightIssue],
) -> None:
    try:
        load_config(payload)
    except GenoLeWMError as exc:
        _issue(
            issues,
            "config.schema_invalid",
            str(path),
            "training config failed the typed GenoLeWM schema",
            "a valid schema-1.1 config",
            exc.to_dict(),
        )
    expected = (
        (("run_id",), EXPECTED_RUN_ID),
        (("seed",), EXPECTED_SEED),
        (("phase",), "phase1"),
        (("deterministic",), deterministic),
        (("schema_version",), "1.1.0"),
        (("encoder", "model_id"), EXPECTED_CARBON_MODEL_DIR),
        (("encoder", "revision"), EXPECTED_CARBON_REVISION),
        (("encoder", "normalize"), True),
        (("encoder", "state_contract_version"), "l2_normalized_v2"),
        (("encoder", "trust_remote_code"), False),
        (("predictor", "d_state"), 1024),
        (("training", "max_steps"), EXPECTED_STEPS),
        (("optimizer", "lr"), 3.0e-5),
        (("data", "batch_size"), 8),
        (("data", "num_workers"), 0),
        (("data", "shuffle_buffer"), 0),
        (("runtime", "backend"), "torch"),
        (("runtime", "device"), "cuda"),
    )
    for keys, wanted in expected:
        _expect_nested(issues, "config", payload, keys, wanted)


def _validate_dataset_reference(
    payload: dict[str, Any],
    path: Path,
    issues: list[TrainingReproducibilityPreflightIssue],
) -> None:
    expected = {
        "schema_version": "1.0.0",
        "repo_id": EXPECTED_DATASET_REPO,
        "repo_type": "model",
        "revision": EXPECTED_DATASET_REVISION,
        "path": EXPECTED_DATASET_PATH,
        "snapshot_id": EXPECTED_DATASET_SNAPSHOT_ID,
        "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
    }
    for key, wanted in expected.items():
        _expect(
            issues,
            f"dataset_reference.{key}_mismatch",
            f"dataset_reference.{key}",
            payload.get(key, _MISSING),
            wanted,
        )
    files = payload.get("expected_files")
    if not isinstance(files, dict) or files.get("dataset_manifest.json") != (
        EXPECTED_DATASET_MANIFEST_SHA256
    ):
        _issue(
            issues,
            "dataset_reference.expected_files_invalid",
            str(path),
            "dataset reference must pin the downloaded manifest in expected_files",
            {"dataset_manifest.json": EXPECTED_DATASET_MANIFEST_SHA256},
            files,
        )


def _load_yaml(
    path: Path,
    label: str,
    issues: list[TrainingReproducibilityPreflightIssue],
) -> dict[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        _issue(issues, f"{label}.invalid", label, "YAML could not be loaded", "object", str(exc))
        return None
    if not isinstance(payload, dict):
        _issue(issues, f"{label}.invalid", label, "YAML root must be an object", "object", payload)
        return None
    return payload


def _load_json(
    path: Path,
    label: str,
    issues: list[TrainingReproducibilityPreflightIssue],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _issue(issues, f"{label}.invalid", label, "JSON could not be loaded", "object", str(exc))
        return None
    if not isinstance(payload, dict):
        _issue(issues, f"{label}.invalid", label, "JSON root must be an object", "object", payload)
        return None
    return payload


def _git_identity(root: Path) -> tuple[Path | None, str | None, str | None]:
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, None, str(exc)
    return Path(top).resolve(), head, None


def _git_dirty_paths(root: Path) -> tuple[tuple[str, ...], str | None]:
    try:
        output = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return (), str(exc)
    return tuple(line[3:] for line in output.splitlines() if len(line) >= 4), None


def _repo_path(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _expect_nested(
    issues: list[TrainingReproducibilityPreflightIssue],
    prefix: str,
    payload: dict[str, Any],
    keys: tuple[str, ...],
    expected: object,
) -> None:
    observed: object = payload
    for key in keys:
        if not isinstance(observed, dict) or key not in observed:
            observed = _MISSING
            break
        observed = observed[key]
    label = ".".join((prefix, *keys))
    _expect(issues, f"{label}_mismatch", label, observed, expected)


def _expect_path(
    issues: list[TrainingReproducibilityPreflightIssue],
    code: str,
    label: str,
    observed: Path,
    expected: Path,
) -> None:
    _expect(issues, code, label, str(observed), str(expected))


def _expect(
    issues: list[TrainingReproducibilityPreflightIssue],
    code: str,
    path: str,
    observed: object,
    expected: object,
) -> None:
    if observed != expected or type(observed) is not type(expected):
        _issue(
            issues, code, path, f"{path} does not match the immutable contract", expected, observed
        )


def _issue(
    issues: list[TrainingReproducibilityPreflightIssue],
    code: str,
    path: str,
    message: str,
    expected: object | None,
    observed: object | None,
) -> None:
    issues.append(
        TrainingReproducibilityPreflightIssue(
            code=code,
            path=path,
            message=message,
            expected=expected,
            observed=observed,
        )
    )


def _file_summary(root: Path, path: Path, payload: dict[str, Any] | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": _display_path(root, path),
        "exists": path.is_file(),
    }
    if path.is_file():
        summary.update({"sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    if payload is not None:
        for key in ("run_id", "schema_version", "snapshot_id"):
            if isinstance(payload.get(key), str):
                summary[key] = payload[key]
    return summary


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _json_value(value: object) -> object:
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--deterministic-config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--dataset-reference", type=Path, required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--expected-sample-count", type=int, required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--carbon-model-dir", required=True)
    parser.add_argument("--expected-carbon-runtime-hash", required=True)
    parser.add_argument("--upload-repo", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--min-cuda-vram-gb", type=float, required=True)
    parser.add_argument("--max-throughput-drop", type=float, required=True)
    parser.add_argument("--max-repeat-spread", type=float, required=True)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
