# SPDX-License-Identifier: Apache-2.0
"""Run or plan the v0.2 benchmark suite from release-shaped inputs.

This runner composes existing benchmark-producing commands. It does not
generate private-data fixtures and a plan-only report is not benchmark
evidence; ``ok`` is true only after ``--execute`` clears each step's
declared output files, runs every planned step successfully, and observes
the declared output files again.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, Protocol

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.v02_benchmark_suite"
MAX_CAPTURE_CHARS: Final = 2000


class CommandRunner(Protocol):
    """Callable shape used to execute one planned command."""

    def __call__(
        self,
        args: list[str],
        *,
        cwd: Path,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True, slots=True)
class SuiteStep:
    """One executable benchmark-suite step."""

    step_id: str
    kind: str
    command: tuple[str, ...]
    outputs: tuple[str, ...]
    issue_refs: tuple[str, ...]

    def to_report_dict(self) -> dict[str, object]:
        return {
            "id": self.step_id,
            "kind": self.kind,
            "command": list(self.command),
            "outputs": list(self.outputs),
            "issue_refs": list(self.issue_refs),
        }


def build_suite_steps(manifest_path: Path) -> tuple[SuiteStep, ...]:
    """Load ``manifest_path`` and return the planned benchmark commands."""
    manifest = _load_manifest(manifest_path)
    root = _path_text(manifest.get("artifact_root", "."), field="artifact_root", allow_dot=True)
    identity = _required_mapping(manifest, "identity")
    artifacts = _required_mapping(manifest, "artifacts")
    benchmarks = _required_list(manifest, "benchmarks")
    aggregate = _required_mapping(manifest, "aggregate")
    readiness = _optional_mapping(manifest, "readiness")

    shared = {
        "artifact_root": root,
        "model_id": _required_text(identity, "model_id"),
        "model_release": _required_text(identity, "model_release"),
        "dataset_snapshot": _required_text(identity, "dataset_snapshot"),
        "commit": _required_text(identity, "commit"),
        "hardware": _required_text(identity, "hardware"),
        "model_dir": _path_text(artifacts.get("model_dir"), field="artifacts.model_dir"),
        "checkpoint": _path_text(artifacts.get("checkpoint"), field="artifacts.checkpoint"),
        "config": _path_text(artifacts.get("config"), field="artifacts.config"),
        "dataset_manifest": _path_text(
            artifacts.get("dataset_manifest"),
            field="artifacts.dataset_manifest",
        ),
        "efficiency_report": _path_text(
            artifacts.get("efficiency_report"),
            field="artifacts.efficiency_report",
        ),
    }

    steps: list[SuiteStep] = []
    metric_inputs: list[str] = []
    seen_ids: set[str] = set()
    for index, raw_benchmark in enumerate(benchmarks, start=1):
        benchmark = _mapping_from_list_item(raw_benchmark, field=f"benchmarks[{index - 1}]")
        benchmark_id = _required_text(benchmark, "id")
        if benchmark_id in seen_ids:
            raise InputError("benchmark ids must be unique", details={"id": benchmark_id})
        seen_ids.add(benchmark_id)
        kind = _required_text(benchmark, "kind")
        if kind == "vep":
            metric_inputs.extend(_append_vep_steps(steps, benchmark, shared))
        elif kind == "rollout":
            metric_inputs.extend(_append_rollout_steps(steps, benchmark, shared))
        else:
            raise InputError(
                "unsupported benchmark kind",
                details={"benchmark": benchmark_id, "kind": kind},
            )

    if not metric_inputs:
        raise InputError("benchmark suite manifest produced no metrics outputs")
    aggregate_metrics = _path_text(
        aggregate.get("metrics_json"),
        field="aggregate.metrics_json",
    )
    aggregate_report = _path_text(
        aggregate.get("report_md"),
        field="aggregate.report_md",
    )
    aggregate_command = [
        "geno-lewm-eval-all",
        "--quiet",
        "--no-banner",
    ]
    for metric_path in metric_inputs:
        aggregate_command.extend(("--metrics-json", metric_path))
    aggregate_command.extend(("--output-metrics", aggregate_metrics))
    aggregate_command.extend(("--output-report", aggregate_report))
    if _optional_bool(aggregate, "require_v02_vep_metrics", default=False):
        aggregate_command.append("--require-v02-vep-metrics")
    if _optional_bool(aggregate, "require_v02_rollout_metrics", default=False):
        aggregate_command.append("--require-v02-rollout-metrics")
    steps.append(
        SuiteStep(
            step_id="aggregate.eval_all",
            kind="aggregate_eval",
            command=tuple(aggregate_command),
            outputs=(aggregate_metrics, aggregate_report),
            issue_refs=("#56", "#197"),
        )
    )

    if readiness is not None:
        steps.append(_readiness_step(readiness, aggregate_metrics, shared["efficiency_report"]))
    _validate_unique_step_outputs(steps)
    return tuple(steps)


def write_suite_report(
    *,
    manifest_path: Path,
    output_report: Path,
    execute: bool = False,
    runner: CommandRunner = subprocess.run,
) -> dict[str, object]:
    """Write a benchmark-suite plan or execution report."""
    steps = build_suite_steps(manifest_path)
    status = "planned"
    step_reports: list[dict[str, object]] = []
    failed = False
    for step in steps:
        report = step.to_report_dict()
        if not execute:
            report["status"] = "planned"
        elif failed:
            report["status"] = "not_run"
        else:
            output_findings = _clear_declared_outputs(step, root=manifest_path.parent)
            if output_findings:
                report["status"] = "failed"
                report["output_findings"] = output_findings
                failed = True
            else:
                completed = runner(
                    list(step.command),
                    cwd=manifest_path.parent,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                report["exit_code"] = completed.returncode
                report["stdout_tail"] = _tail_text(completed.stdout)
                report["stderr_tail"] = _tail_text(completed.stderr)
                if completed.returncode == 0:
                    output_findings = _missing_output_findings(step, root=manifest_path.parent)
                    if output_findings:
                        report["status"] = "failed"
                        report["output_findings"] = output_findings
                        failed = True
                    else:
                        report["status"] = "pass"
                        report["output_identities"] = _output_identities(
                            step,
                            root=manifest_path.parent,
                        )
                else:
                    report["status"] = "failed"
                    failed = True
        step_reports.append(report)
    if execute:
        status = "failed" if failed else "pass"
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": _utc_now(),
        "ok": execute and not failed,
        "status": status,
        "manifest_path": manifest_path.name,
        "manifest": _manifest_identity(manifest_path),
        "execute": execute,
        "steps": step_reports,
        "negative_findings": _negative_findings(execute=execute, failed=failed),
        "claim_boundary": (
            "This report is benchmark-suite orchestration evidence only; measured model-quality "
            "claims require the generated metrics, efficiency, rollout-speed, and readiness "
            "artifacts to validate separately."
        ),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = write_suite_report(
            manifest_path=args.manifest,
            output_report=args.output_report,
            execute=args.execute,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output_report}\n")
    if args.execute and not bool(report["ok"]):
        for line in _failed_step_diagnostics(report):
            sys.stderr.write(line + "\n")
        return 1
    return 0


def _append_vep_steps(
    steps: list[SuiteStep],
    benchmark: dict[str, object],
    shared: dict[str, str],
) -> list[str]:
    benchmark_id = _required_text(benchmark, "id")
    split = _required_text(benchmark, "split")
    vcf = _path_text(benchmark.get("vcf"), field=f"{benchmark_id}.vcf")
    fasta = _path_text(benchmark.get("fasta"), field=f"{benchmark_id}.fasta")
    labels = _path_text(benchmark.get("labels_jsonl"), field=f"{benchmark_id}.labels_jsonl")
    scores = _path_text(benchmark.get("scores_jsonl"), field=f"{benchmark_id}.scores_jsonl")
    metrics = _path_text(benchmark.get("metrics_json"), field=f"{benchmark_id}.metrics_json")
    score_cfg = _optional_mapping(benchmark, "score")
    baseline_cfg = _optional_mapping(benchmark, "carbon_baseline")
    baseline_scores: str | None = None

    if score_cfg is not None and _optional_bool(score_cfg, "enabled", default=False):
        command = [
            "geno-lewm-score",
            "--quiet",
            "--no-banner",
            "--model-dir",
            shared["model_dir"],
            "--backend",
            _optional_text(score_cfg, "backend", default="auto"),
            "--vcf",
            vcf,
            "--fasta",
            fasta,
            "--output",
            scores,
            "--batch-size",
            str(_optional_int(score_cfg, "batch_size", default=64)),
            "--no-progress",
        ]
        steps.append(
            SuiteStep(
                step_id=f"{benchmark_id}.score",
                kind="vep_score",
                command=tuple(command),
                outputs=(scores,),
                issue_refs=("#53", "#56", "#197"),
            )
        )

    if baseline_cfg is not None and _optional_bool(baseline_cfg, "enabled", default=False):
        baseline_scores = _path_text(
            baseline_cfg.get("scores_jsonl"),
            field=f"{benchmark_id}.carbon_baseline.scores_jsonl",
        )
        command = [
            "geno-lewm-carbon-baseline",
            "--quiet",
            "--no-banner",
            "--artifact-root",
            shared["artifact_root"],
            "--vcf",
            vcf,
            "--fasta",
            fasta,
            "--carbon-model-dir",
            _path_text(
                baseline_cfg.get("carbon_model_dir"),
                field=f"{benchmark_id}.carbon_baseline.carbon_model_dir",
            ),
            "--output-scores",
            baseline_scores,
            "--carbon-revision",
            _optional_text(baseline_cfg, "carbon_revision", default="main"),
            "--dtype",
            _optional_text(baseline_cfg, "dtype", default="bf16"),
            "--window-bp",
            str(_optional_int(baseline_cfg, "window_bp", default=12288)),
        ]
        device = _optional_text(baseline_cfg, "device", default="")
        if device:
            command.extend(("--device", device))
        metadata = _optional_path_text(
            baseline_cfg,
            "metadata_json",
            field=f"{benchmark_id}.carbon_baseline.metadata_json",
        )
        baseline_outputs = [baseline_scores]
        if metadata is not None:
            command.extend(("--metadata-output", metadata))
            baseline_outputs.append(metadata)
        cache = _optional_path_text(
            baseline_cfg,
            "logp_cache_jsonl",
            field=f"{benchmark_id}.carbon_baseline.logp_cache_jsonl",
        )
        if cache is not None:
            command.extend(("--logp-cache-jsonl", cache))
        if _optional_bool(baseline_cfg, "trust_remote_code", default=False):
            command.append("--trust-remote-code")
        if _optional_bool(baseline_cfg, "allow_network_download", default=False):
            command.append("--allow-network-download")
        steps.append(
            SuiteStep(
                step_id=f"{benchmark_id}.carbon_baseline",
                kind="carbon_baseline",
                command=tuple(command),
                outputs=tuple(baseline_outputs),
                issue_refs=("#55", "#56", "#197"),
            )
        )

    eval_command = [
        "geno-lewm-eval",
        "--quiet",
        "--no-banner",
        "--scores-jsonl",
        scores,
        "--labels-jsonl",
        labels,
        "--output-metrics",
        metrics,
        "--artifact-root",
        shared["artifact_root"],
        "--split",
        split,
        "--model-id",
        shared["model_id"],
        "--model-release",
        shared["model_release"],
        "--dataset-snapshot",
        shared["dataset_snapshot"],
        "--commit",
        shared["commit"],
        "--hardware",
        shared["hardware"],
        "--checkpoint",
        shared["checkpoint"],
        "--config-artifact",
        shared["config"],
        "--dataset-manifest",
        shared["dataset_manifest"],
        "--efficiency-report",
        shared["efficiency_report"],
    ]
    metric_mode = _optional_text(benchmark, "metric_mode", default="binary")
    if metric_mode not in {"binary", "spearman"}:
        raise InputError(
            "unsupported VEP metric_mode",
            details={"benchmark": benchmark_id, "metric_mode": metric_mode},
        )
    if metric_mode != "binary":
        eval_command.extend(("--metric-mode", metric_mode))
    score_field = _optional_text(benchmark, "score_field", default="")
    if score_field:
        eval_command.extend(("--score-field", score_field))
    label_field = _optional_text(benchmark, "label_field", default="")
    if label_field:
        eval_command.extend(("--label-field", label_field))
    baseline_score_field = _optional_text(
        baseline_cfg or {},
        "score_field",
        default="",
    )
    if baseline_scores is not None:
        eval_command.extend(
            (
                "--baseline-scores-jsonl",
                baseline_scores,
                "--baseline-score-field",
                baseline_score_field or "carbon_zero_shot_score",
                "--baseline-name",
                "carbon_zero_shot",
            )
        )
    bootstrap_resamples = _optional_int(benchmark, "bootstrap_resamples", default=0)
    if bootstrap_resamples:
        eval_command.extend(("--bootstrap-resamples", str(bootstrap_resamples)))
    bootstrap_seed = _optional_int(benchmark, "bootstrap_seed", default=0)
    if bootstrap_seed:
        eval_command.extend(("--bootstrap-seed", str(bootstrap_seed)))
    steps.append(
        SuiteStep(
            step_id=f"{benchmark_id}.eval",
            kind="vep_eval",
            command=tuple(eval_command),
            outputs=(metrics,),
            issue_refs=("#53", "#55", "#56", "#197"),
        )
    )
    return [metrics]


def _append_rollout_steps(
    steps: list[SuiteStep],
    benchmark: dict[str, object],
    shared: dict[str, str],
) -> list[str]:
    benchmark_id = _required_text(benchmark, "id")
    _required_text(benchmark, "split")
    states = _path_text(benchmark.get("states_jsonl"), field=f"{benchmark_id}.states_jsonl")
    metrics = _path_text(benchmark.get("metrics_json"), field=f"{benchmark_id}.metrics_json")
    state_generation = _optional_mapping(benchmark, "state_generation")
    if state_generation is not None:
        examples = _path_text(
            state_generation.get("examples_jsonl"),
            field=f"{benchmark_id}.state_generation.examples_jsonl",
        )
        spec_jsonl = _optional_path_text(
            state_generation,
            "spec_jsonl",
            field=f"{benchmark_id}.state_generation.spec_jsonl",
        )
        if spec_jsonl is not None:
            examples_report = _path_text(
                state_generation.get("examples_report_json"),
                field=f"{benchmark_id}.state_generation.examples_report_json",
            )
            steps.append(
                SuiteStep(
                    step_id=f"{benchmark_id}.rollout_examples",
                    kind="rollout_state_examples",
                    command=(
                        "python",
                        "-m",
                        "tools.release.rollout_state_examples",
                        "--spec-jsonl",
                        spec_jsonl,
                        "--cache-dir",
                        _path_text(
                            state_generation.get("cache_dir"),
                            field=f"{benchmark_id}.state_generation.cache_dir",
                        ),
                        "--artifact-root",
                        shared["artifact_root"],
                        "--output-jsonl",
                        examples,
                        "--output-report",
                        examples_report,
                    ),
                    outputs=(examples, examples_report),
                    issue_refs=("#57", "#197"),
                )
            )
        report = _path_text(
            state_generation.get("report_json"),
            field=f"{benchmark_id}.state_generation.report_json",
        )
        steps.append(
            SuiteStep(
                step_id=f"{benchmark_id}.rollout_states",
                kind="rollout_state_generation",
                command=(
                    "python",
                    "-m",
                    "tools.release.rollout_state_rows",
                    "--examples-jsonl",
                    examples,
                    "--model-dir",
                    shared["model_dir"],
                    "--artifact-root",
                    shared["artifact_root"],
                    "--output-jsonl",
                    states,
                    "--output-report",
                    report,
                ),
                outputs=(states, report),
                issue_refs=("#57", "#197"),
            )
        )
    command = [
        "geno-lewm-rollout",
        "--quiet",
        "--no-banner",
        "--states-jsonl",
        states,
        "--output-metrics",
        metrics,
        "--artifact-root",
        shared["artifact_root"],
        "--recall-k",
        str(_optional_int(benchmark, "recall_k", default=10)),
        "--model-id",
        shared["model_id"],
        "--model-release",
        shared["model_release"],
        "--dataset-snapshot",
        shared["dataset_snapshot"],
        "--commit",
        shared["commit"],
        "--hardware",
        shared["hardware"],
        "--checkpoint",
        shared["checkpoint"],
        "--config-artifact",
        shared["config"],
        "--dataset-manifest",
        shared["dataset_manifest"],
        "--efficiency-report",
        shared["efficiency_report"],
    ]
    if state_generation is not None:
        command.extend(
            (
                "--rollout-state-rows-report",
                _path_text(
                    state_generation.get("report_json"),
                    field=f"{benchmark_id}.state_generation.report_json",
                ),
            )
        )
        if state_generation.get("examples_report_json") is not None:
            command.extend(
                (
                    "--rollout-state-examples-report",
                    _path_text(
                        state_generation.get("examples_report_json"),
                        field=f"{benchmark_id}.state_generation.examples_report_json",
                    ),
                )
            )
    steps.append(
        SuiteStep(
            step_id=f"{benchmark_id}.rollout",
            kind="rollout_eval",
            command=tuple(command),
            outputs=(metrics,),
            issue_refs=("#42", "#57", "#197"),
        )
    )
    return [metrics]


def _readiness_step(
    readiness: dict[str, object],
    aggregate_metrics: str,
    efficiency_report: str,
) -> SuiteStep:
    output = _path_text(readiness.get("output_json"), field="readiness.output_json")
    command = [
        "python",
        "-m",
        "tools.release.v02_benchmark_readiness",
        "--metrics-json",
        aggregate_metrics,
        "--rollout-speed-report",
        _path_text(
            readiness.get("rollout_speed_report"),
            field="readiness.rollout_speed_report",
        ),
        "--efficiency-report",
        efficiency_report,
    ]
    suite_report = readiness.get("suite_report")
    if suite_report is not None:
        command.extend(
            (
                "--suite-report",
                _path_text(suite_report, field="readiness.suite_report"),
            )
        )
    command.extend(("--output", output))
    rollout_speed_scope_report = readiness.get("rollout_speed_scope_report")
    if rollout_speed_scope_report is not None:
        command.extend(
            (
                "--rollout-speed-scope-report",
                _path_text(
                    rollout_speed_scope_report,
                    field="readiness.rollout_speed_scope_report",
                ),
            )
        )
    if _optional_bool(readiness, "require_ok", default=False):
        command.append("--require-ok")
    if _optional_bool(readiness, "require_release_inputs", default=False):
        command.append("--require-release-inputs")
    return SuiteStep(
        step_id="readiness.v02",
        kind="v02_readiness",
        command=tuple(command),
        outputs=(output,),
        issue_refs=("#42", "#53", "#55", "#56", "#57", "#197"),
    )


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "failed to read benchmark suite manifest", details={"path": str(path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "benchmark suite manifest JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("benchmark suite manifest must be a JSON object")
    schema_version = _required_text(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InputError(
            "unsupported benchmark suite schema version",
            details={"expected": SCHEMA_VERSION, "observed": schema_version},
        )
    return payload


def _required_mapping(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise InputError(f"benchmark suite manifest requires object {key}")
    return value


def _optional_mapping(parent: dict[str, object], key: str) -> dict[str, object] | None:
    value = parent.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InputError(f"benchmark suite manifest field {key} must be an object")
    return value


def _mapping_from_list_item(raw: object, *, field: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise InputError(f"benchmark suite manifest field {field} must be an object")
    return raw


def _required_list(parent: dict[str, object], key: str) -> list[object]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise InputError(f"benchmark suite manifest requires non-empty list {key}")
    return value


def _required_text(parent: dict[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"benchmark suite manifest requires text field {key}")
    return value.strip()


def _optional_text(parent: dict[str, object], key: str, *, default: str) -> str:
    value = parent.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise InputError(f"benchmark suite manifest field {key} must be text")
    return value.strip()


def _optional_int(parent: dict[str, object], key: str, *, default: int) -> int:
    value = parent.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"benchmark suite manifest field {key} must be an integer")
    if value <= 0:
        raise InputError(f"benchmark suite manifest field {key} must be positive")
    return value


def _optional_bool(parent: dict[str, object], key: str, *, default: bool) -> bool:
    value = parent.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise InputError(f"benchmark suite manifest field {key} must be a boolean")
    return value


def _optional_path_text(
    parent: dict[str, object],
    key: str,
    *,
    field: str,
) -> str | None:
    value = parent.get(key)
    if value is None:
        return None
    return _path_text(value, field=field)


def _validate_unique_step_outputs(steps: list[SuiteStep]) -> None:
    owners: dict[str, str] = {}
    for step in steps:
        for output in step.outputs:
            owner = owners.get(output)
            if owner is not None:
                raise InputError(
                    "benchmark suite step outputs must be unique",
                    details={
                        "output": output,
                        "first_step": owner,
                        "duplicate_step": step.step_id,
                    },
                )
            owners[output] = step.step_id


def _path_text(raw: object, *, field: str, allow_dot: bool = False) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise InputError(f"benchmark suite manifest requires path field {field}")
    value = raw.strip()
    if allow_dot and value == ".":
        return value
    if "://" in value:
        raise InputError(f"{field} must be package-relative, not a URL")
    if "\\" in value or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise InputError(f"{field} must be package-relative")
    candidate = Path(value)
    if ".." in candidate.parts or not candidate.parts:
        raise InputError(f"{field} must be package-relative")
    return value


def _tail_text(value: str | None) -> str:
    if not value:
        return ""
    return value[-MAX_CAPTURE_CHARS:]


def _manifest_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise InputError("benchmark suite manifest does not exist", details={"path": str(path)})
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _clear_declared_outputs(step: SuiteStep, *, root: Path) -> list[str]:
    findings: list[str] = []
    for output in step.outputs:
        path = root / output
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.exists():
            findings.append(f"declared output {output} exists but is not a file")
    return findings


def _missing_output_findings(step: SuiteStep, *, root: Path) -> list[str]:
    return [
        f"missing declared output {output}"
        for output in step.outputs
        if not (root / output).is_file()
    ]


def _output_identities(step: SuiteStep, *, root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": output,
            "sha256": sha256_file(root / output),
            "size_bytes": (root / output).stat().st_size,
        }
        for output in step.outputs
    ]


def _negative_findings(*, execute: bool, failed: bool) -> list[str]:
    if not execute:
        return [
            "The benchmark suite was planned but not executed; this is not measured evidence.",
        ]
    if failed:
        return [
            "At least one benchmark-suite command failed or omitted a declared output before the suite completed."
        ]
    return [
        "No suite-runner failures were observed; measured claims still depend on downstream artifact validators.",
    ]


def _failed_step_diagnostics(report: dict[str, object]) -> tuple[str, ...]:
    """Return compact stderr lines for failed suite executions."""
    raw_steps = report.get("steps")
    if not isinstance(raw_steps, list):
        return ()
    lines: list[str] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict) or raw_step.get("status") != "failed":
            continue
        step_id = raw_step.get("id", "<unknown>")
        kind = raw_step.get("kind", "<unknown>")
        exit_code = raw_step.get("exit_code")
        if exit_code is None:
            lines.append(f"failed step: {step_id} ({kind})")
        else:
            lines.append(f"failed step: {step_id} ({kind}) exit_code={exit_code}")
        output_findings = raw_step.get("output_findings")
        if isinstance(output_findings, list) and output_findings:
            lines.extend(f"output finding: {finding}" for finding in output_findings[:5])
        stderr_tail = raw_step.get("stderr_tail")
        if isinstance(stderr_tail, str) and stderr_tail.strip():
            lines.append("stderr tail:")
            lines.append(stderr_tail.strip())
        stdout_tail = raw_step.get("stdout_tail")
        if isinstance(stdout_tail, str) and stdout_tail.strip():
            lines.append("stdout tail:")
            lines.append(stdout_tail.strip())
        break
    return tuple(lines)


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the v0.2 benchmark suite from a JSON manifest.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute planned benchmark commands. Omit to write a non-evidence command plan.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
