# SPDX-License-Identifier: Apache-2.0
"""Generate and verify the v0.2 serious-completion paper artifact.

This is the paper path for the post-v0.2 evidence package tracked by #205.
It consumes the broader benchmark suite/readiness artifacts and the released
planning demo. It does not reuse the v0.1 terminal-demo package shape.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256
from tools.release.efficiency_report import (
    EfficiencyReport,
    load_efficiency_report,
)
from tools.release.eval_report import EvalReportInput, load_report_input, render_report
from tools.release.rollout_speed_scope import (
    GENERATED_BY as ROLLOUT_SPEED_SCOPE_GENERATED_BY,
)
from tools.release.v02_benchmark_readiness import (
    GENERATED_BY as READINESS_GENERATED_BY,
)
from tools.release.v02_benchmark_suite import (
    GENERATED_BY as SUITE_GENERATED_BY,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.paper_draft"
PAPER_KIND: Final = "serious_completion_v0.2"
DEFAULT_TITLE: Final = (
    "GenoLeWM: Evidence-Bound Genomic Edit World Models, Benchmarks, and Negative Results"
)

EVAL_METRICS_NAME: Final = "eval_metrics.v02.json"
EVAL_REPORT_NAME: Final = "eval_report.v02.md"
EFFICIENCY_REPORT_NAME: Final = "efficiency_report.v02.json"
READINESS_REPORT_NAME: Final = "v0.2_benchmark_readiness_report.json"
SUITE_REPORT_NAME: Final = "v0.2_benchmark_suite_report.json"
ROLLOUT_SPEED_NAME: Final = "rollout.ar_speed.json"
ROLLOUT_SPEED_SCOPE_NAME: Final = "rollout_speed_scope.json"
PLANNING_MANIFEST_NAME: Final = "planning_demo_manifest.json"
PLAN_NAME: Final = "plan.json"
PLAN_STDOUT_NAME: Final = "plan.stdout.json"
PLANNING_TRANSCRIPT_NAME: Final = "planning-demo-transcript.md"
TARGET_EDITS_NAME: Final = "target_edits.json"
SOURCE_WINDOW_NAME: Final = "source_window.fa"
TARGET_WINDOW_NAME: Final = "target_window.fa"

REQUIRED_BENCHMARKS: Final = (
    "clinvar_coding",
    "clinvar_noncoding",
    "brca2_saturation",
    "traitgym_mendelian",
    "rollout_phased_haplotypes",
    "rollout_synthetic_edit_chains",
    "inference_efficiency",
    "ar_rollout_speed",
    "release_inputs",
)
PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum|go here)\b",
    re.IGNORECASE,
)
UTC_TIMESTAMP_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class SeriousCompletionPaperPaths:
    """Paths for the v0.2 serious-completion paper package."""

    suite_dir: Path
    planning_demo_dir: Path
    paper_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SeriousCompletionPaperIssue:
    """One serious-completion paper verification issue."""

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
class SeriousCompletionPaperReport:
    """Verification result for the serious-completion paper package."""

    ok: bool
    model_id: str | None
    issues: tuple[SeriousCompletionPaperIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "model_id": self.model_id,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class SeriousCompletionPaperDraftReport:
    """Files and identities covered by a generated serious-completion paper."""

    path: Path
    model_id: str
    model_release: str
    dataset_snapshot: str
    commit: str
    readiness_report: Path
    suite_report: Path
    eval_metrics: Path
    eval_report: Path
    efficiency_report: Path
    planning_manifest: Path
    plan: Path
    transcript: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path.name,
            "model_id": self.model_id,
            "model_release": self.model_release,
            "dataset_snapshot": self.dataset_snapshot,
            "commit": self.commit,
            "readiness_report": f"suite/model/{self.readiness_report.name}",
            "suite_report": f"suite/model/{self.suite_report.name}",
            "eval_metrics": f"suite/model/{self.eval_metrics.name}",
            "eval_report": f"suite/model/{self.eval_report.name}",
            "efficiency_report": f"suite/model/{self.efficiency_report.name}",
            "planning_manifest": f"planning-demo/{self.planning_manifest.name}",
            "plan": f"planning-demo/{self.plan.name}",
            "transcript": f"planning-demo/{self.transcript.name}",
        }


@dataclass(frozen=True, slots=True)
class _SeriousCompletionArtifacts:
    suite_dir: Path
    planning_demo_dir: Path
    eval_metrics_path: Path
    eval_report_path: Path
    efficiency_report_path: Path
    readiness_report_path: Path
    suite_report_path: Path
    rollout_speed_path: Path
    rollout_speed_scope_path: Path
    planning_manifest_path: Path
    plan_path: Path
    plan_stdout_path: Path
    transcript_path: Path
    source_window_path: Path
    target_window_path: Path
    target_edits_path: Path
    eval_input: EvalReportInput
    eval_report_text: str
    efficiency: EfficiencyReport
    readiness: dict[str, Any]
    suite_report: dict[str, Any]
    rollout_speed: dict[str, Any]
    rollout_speed_scope: dict[str, Any]
    planning_manifest: dict[str, Any]
    plan: dict[str, Any]
    plan_stdout: dict[str, Any]
    transcript: str
    target_edits: tuple[dict[str, Any], ...]


def build_serious_completion_paper(
    *,
    suite_dir: str | Path,
    planning_demo_dir: str | Path,
    output: str | Path,
    title: str = DEFAULT_TITLE,
    generated_at: str | None = None,
) -> SeriousCompletionPaperDraftReport:
    """Generate a serious-completion Markdown paper from measured artifacts."""

    artifacts = _load_serious_completion_artifacts(
        suite_dir=Path(suite_dir),
        planning_demo_dir=Path(planning_demo_dir),
    )
    text = _render_serious_completion_paper(
        title=title,
        generated_at=generated_at or _utc_now(),
        artifacts=artifacts,
    )
    if PLACEHOLDER_RE.search(text):
        raise InputError("generated paper contains placeholder text")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return SeriousCompletionPaperDraftReport(
        path=output_path,
        model_id=artifacts.eval_input.model_id,
        model_release=artifacts.eval_input.model_release,
        dataset_snapshot=artifacts.eval_input.dataset_snapshot,
        commit=artifacts.eval_input.commit,
        readiness_report=artifacts.readiness_report_path,
        suite_report=artifacts.suite_report_path,
        eval_metrics=artifacts.eval_metrics_path,
        eval_report=artifacts.eval_report_path,
        efficiency_report=artifacts.efficiency_report_path,
        planning_manifest=artifacts.planning_manifest_path,
        plan=artifacts.plan_path,
        transcript=artifacts.transcript_path,
    )


def render_serious_completion_paper(
    *,
    suite_dir: str | Path,
    planning_demo_dir: str | Path,
    title: str = DEFAULT_TITLE,
    generated_at: str,
) -> str:
    """Render the expected serious-completion paper text."""

    artifacts = _load_serious_completion_artifacts(
        suite_dir=Path(suite_dir),
        planning_demo_dir=Path(planning_demo_dir),
    )
    return _render_serious_completion_paper(
        title=title,
        generated_at=generated_at,
        artifacts=artifacts,
    )


def verify_serious_completion_paper(
    paths: SeriousCompletionPaperPaths,
) -> SeriousCompletionPaperReport:
    """Verify a serious-completion paper against the benchmark/demo artifacts."""

    issues: list[SeriousCompletionPaperIssue] = []
    artifacts: _SeriousCompletionArtifacts | None = None
    try:
        artifacts = _load_serious_completion_artifacts(
            suite_dir=paths.suite_dir,
            planning_demo_dir=paths.planning_demo_dir,
        )
    except GenoLeWMError as exc:
        _error(
            issues,
            "serious_paper.artifacts_invalid",
            paths.suite_dir,
            exc.message or str(exc),
        )

    if paths.paper_path is None:
        _error(
            issues,
            "serious_paper.path_missing",
            paths.suite_dir,
            "paper_path is required",
        )
    elif not paths.paper_path.is_file():
        _error(
            issues,
            "serious_paper.missing",
            paths.paper_path,
            "generated serious-completion paper is required",
        )
    else:
        _verify_paper_path(paths.paper_path, artifacts, issues)

    model_id = None if artifacts is None else artifacts.eval_input.model_id
    return SeriousCompletionPaperReport(
        ok=not any(issue.severity == "error" for issue in issues),
        model_id=model_id,
        issues=tuple(issues),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    paths = SeriousCompletionPaperPaths(
        suite_dir=args.suite_dir,
        planning_demo_dir=args.planning_demo_dir,
        paper_path=args.paper_path,
    )
    try:
        if args.verify:
            report = verify_serious_completion_paper(paths)
            sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
            return 0 if report.ok else 2
        if args.paper_path is None:
            raise InputError("--paper-path is required unless --verify is set")
        draft = build_serious_completion_paper(
            suite_dir=args.suite_dir,
            planning_demo_dir=args.planning_demo_dir,
            output=args.paper_path,
            title=args.title,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(draft.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify the v0.2 serious-completion paper artifact.",
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--planning-demo-dir", type=Path, required=True)
    parser.add_argument("--paper-path", type=Path)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--verify", action="store_true")
    return parser


def _load_serious_completion_artifacts(
    *,
    suite_dir: Path,
    planning_demo_dir: Path,
) -> _SeriousCompletionArtifacts:
    model_dir = suite_dir / "model"
    bench_dir = suite_dir / "bench"
    eval_metrics_path = model_dir / EVAL_METRICS_NAME
    eval_report_path = model_dir / EVAL_REPORT_NAME
    efficiency_report_path = model_dir / EFFICIENCY_REPORT_NAME
    readiness_report_path = model_dir / READINESS_REPORT_NAME
    suite_report_path = model_dir / SUITE_REPORT_NAME
    rollout_speed_path = bench_dir / ROLLOUT_SPEED_NAME
    rollout_speed_scope_path = bench_dir / ROLLOUT_SPEED_SCOPE_NAME
    planning_manifest_path = planning_demo_dir / PLANNING_MANIFEST_NAME
    plan_path = planning_demo_dir / PLAN_NAME
    plan_stdout_path = planning_demo_dir / PLAN_STDOUT_NAME
    transcript_path = planning_demo_dir / PLANNING_TRANSCRIPT_NAME
    source_window_path = planning_demo_dir / SOURCE_WINDOW_NAME
    target_window_path = planning_demo_dir / TARGET_WINDOW_NAME
    target_edits_path = planning_demo_dir / TARGET_EDITS_NAME

    eval_input = load_report_input(eval_metrics_path)
    eval_report_text = _read_text(eval_report_path, "eval report")
    expected_eval_report = render_report(eval_input)
    if eval_report_text != expected_eval_report:
        raise InputError(f"{EVAL_REPORT_NAME} does not match render of {EVAL_METRICS_NAME}")
    efficiency = load_efficiency_report(efficiency_report_path)
    _require_shared_identity(eval_input, efficiency)

    readiness = _read_json_object(readiness_report_path, "readiness report")
    suite_report = _read_json_object(suite_report_path, "suite report")
    rollout_speed = _read_json_object(rollout_speed_path, "rollout speed report")
    rollout_speed_scope = _read_json_object(
        rollout_speed_scope_path,
        "rollout speed scope report",
    )
    planning_manifest = _read_json_object(planning_manifest_path, "planning demo manifest")
    plan = _read_json_object(plan_path, "planning plan")
    plan_stdout = _read_json_object(plan_stdout_path, "planning stdout")
    transcript = _read_text(transcript_path, "planning transcript")
    target_edits = _read_target_edits(target_edits_path)

    artifacts = _SeriousCompletionArtifacts(
        suite_dir=suite_dir,
        planning_demo_dir=planning_demo_dir,
        eval_metrics_path=eval_metrics_path,
        eval_report_path=eval_report_path,
        efficiency_report_path=efficiency_report_path,
        readiness_report_path=readiness_report_path,
        suite_report_path=suite_report_path,
        rollout_speed_path=rollout_speed_path,
        rollout_speed_scope_path=rollout_speed_scope_path,
        planning_manifest_path=planning_manifest_path,
        plan_path=plan_path,
        plan_stdout_path=plan_stdout_path,
        transcript_path=transcript_path,
        source_window_path=source_window_path,
        target_window_path=target_window_path,
        target_edits_path=target_edits_path,
        eval_input=eval_input,
        eval_report_text=eval_report_text,
        efficiency=efficiency,
        readiness=readiness,
        suite_report=suite_report,
        rollout_speed=rollout_speed,
        rollout_speed_scope=rollout_speed_scope,
        planning_manifest=planning_manifest,
        plan=plan,
        plan_stdout=plan_stdout,
        transcript=transcript,
        target_edits=target_edits,
    )
    _validate_artifacts(artifacts)
    return artifacts


def _render_serious_completion_paper(
    *,
    title: str,
    generated_at: str,
    artifacts: _SeriousCompletionArtifacts,
) -> str:
    _require_utc_timestamp(generated_at, field="generated_at")
    eval_input = artifacts.eval_input
    planning_summary = _required_mapping(artifacts.planning_manifest, "plan_summary")
    lines = [
        f"# {title}",
        "",
        f"Generated by: {GENERATED_BY}",
        f"Generated: {generated_at}",
        f"Schema: {SCHEMA_VERSION}",
        f"Paper kind: {PAPER_KIND}",
        "",
        "## Abstract",
        "",
        (
            "We study GenoLeWM, an action-conditioned latent world model for genomic "
            "edits that predicts changes in frozen Carbon-500M sequence representations "
            "rather than directly reconstructing DNA. The paper reports the v0.2 "
            "serious-completion experiments: ClinVar coding and non-coding SNV "
            "classification, BRCA2 saturation and TraitGym Mendelian continuous-label "
            "benchmarks, rollout-fidelity measurements, autoregressive rollout-speed "
            "measurements, and a released-artifact multi-edit planning demo. The main "
            "finding is negative: although the artifact pipeline is reproducible and "
            "the broader benchmark package is complete, GenoLeWM trails Carbon zero-shot "
            "on most variant-effect rows, rollout fidelity is weak versus source-state "
            "baselines, the K20 rollout-speed target remains open under #42, and the "
            "planning demo records execution rather than useful planning behavior."
        ),
        "",
        "## Introduction",
        "",
        (
            "Genomic foundation models can score and represent DNA sequence context, "
            "but edit-conditioned reasoning asks a different question: given a local "
            "reference state and an explicit genomic edit, can a learned model predict "
            "the edited state cheaply enough to support scoring, rollout, and planning? "
            "GenoLeWM tests this question by freezing Carbon-500M as the state encoder "
            "and training a smaller predictor over edit actions and latent states."
        ),
        "",
        (
            "The contribution of this paper is deliberately bounded. It is not a claim "
            "of clinical utility, deployment readiness, privacy assurance, runtime "
            "assurance, or broad superiority over Carbon. It is a negative-results and "
            "systems paper: an evidence-bound account of what worked operationally, "
            "what failed empirically, and which benchmark artifacts support those "
            "conclusions."
        ),
        "",
        "The paper makes three concrete contributions:",
        "",
        (
            "1. A reproducible artifact chain for training, evaluating, benchmarking, "
            "and demoing an action-conditioned genomic latent predictor."
        ),
        (
            "2. A broader v0.2 benchmark package showing mixed and mostly negative "
            "variant-effect performance against Carbon zero-shot baselines."
        ),
        (
            "3. A planning and rollout audit showing that released-artifact execution "
            "is possible, but useful multi-edit planning behavior is not established."
        ),
        "",
        "## Related Work",
        "",
        (
            "GenoLeWM sits between DNA foundation modeling, variant-effect prediction, "
            "and latent predictive world models. DNABERT adapted BERT-style masked "
            "language modeling to genomic DNA; HyenaDNA studied long-context genome "
            "modeling at single-nucleotide resolution; and Nucleotide Transformer "
            "evaluated large transformer representations for human genomics. Carbon-500M "
            "is the frozen DNA language model used here as a state encoder."
        ),
        "",
        (
            "The evaluation setup is also shaped by variant interpretation resources. "
            "ClinVar provides submitted clinical variant interpretations, gnomAD provides "
            "large-scale population variation context, TraitGym targets causal regulatory "
            "variant prediction, and AlphaMissense is an example of a high-performing "
            "protein-centric variant-effect model. GenoLeWM does not compete with those "
            "systems as a clinical predictor; it asks whether action-conditioned latent "
            "prediction adds useful evidence over Carbon-style sequence scoring."
        ),
        "",
        (
            "The modeling objective follows the broader joint-embedding predictive "
            "architecture idea: predict representation-space targets rather than "
            "high-entropy observations. In this project the input and target spaces are "
            "Carbon-encoded DNA windows, and the conditioning variable is an explicit "
            "genomic edit action."
        ),
        "",
        "## Method",
        "",
        (
            "For each training tuple, GenoLeWM encodes a source reference window into a "
            "state `s_t`, encodes a canonical edit action, and trains a cross-attention "
            "predictor to approximate the edited-window target state `s_{t+1}` produced "
            "by the frozen Carbon encoder. The predictor and action encoder are the "
            "trainable artifacts; Carbon-500M remains frozen and is required at runtime."
        ),
        "",
        (
            "Variant scoring uses surprise-style comparisons derived from model-predicted "
            "state changes and calibration artifacts. Rollout experiments repeatedly apply "
            "the predictor over edit chains. Planning uses the released manifest-backed "
            "runtime to search over candidate SNV edits against a target latent state."
        ),
        "",
        "## Experiments",
        "",
        (
            "The v0.2 experiment package evaluates the same released model identity across "
            "binary ClinVar coding and non-coding SNV slices, continuous-label BRCA2 and "
            "TraitGym Mendelian slices, two rollout-fidelity slices, inference-efficiency "
            "measurements, autoregressive rollout-speed measurements at K5 and K20, and "
            "one synthetic multi-SNV planning demo. Every result below is loaded from the "
            "artifact files listed in the reproducibility sections rather than copied from "
            "hand-authored tables."
        ),
        "",
        "## Citation Metadata",
        "",
        f"- Model release: `{eval_input.model_release}`",
        f"- Model id: `{eval_input.model_id}`",
        f"- Dataset snapshot: `{eval_input.dataset_snapshot}`",
        f"- Commit: `{eval_input.commit}`",
        f"- Hardware: `{eval_input.hardware}`",
        f"- Benchmark readiness: `{_package_path(artifacts.readiness_report_path)}`",
        f"- Benchmark suite: `{_package_path(artifacts.suite_report_path)}`",
        f"- Planning demo manifest: `{_package_path(artifacts.planning_manifest_path)}`",
        "",
        "## Artifact Inputs",
        "",
        "| Artifact | Package path | SHA-256 | Bytes |",
        "| --- | --- | --- | ---: |",
    ]
    lines.extend(_artifact_identity_rows(artifacts))
    lines.extend(
        [
            "",
            "### Model And Data Artifact References",
            "",
            "| Reference | Path or identifier |",
            "| --- | --- |",
        ]
    )
    for key, value in artifacts.eval_input.artifacts:
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "### Planning Model Artifacts",
            "",
            "| Artifact | Path | SHA-256 | Bytes |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for key in ("model_manifest", "predictor", "action_encoder"):
        identity = _required_mapping(
            _required_mapping(artifacts.planning_manifest, "artifacts"), key
        )
        lines.append(
            "| "
            f"{key} | `{_required_text(identity, 'path')}` | "
            f"`{_required_text(identity, 'sha256')}` | "
            f"{_required_int(identity, 'size_bytes')} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "### Benchmark Readiness Summary",
            "",
            "| Benchmark | Status | Observed values | Baseline deltas | Issue refs |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(_benchmark_table_row(row) for row in _benchmark_rows(artifacts.readiness))
    lines.extend(
        [
            "",
            "### Measured Metric Rows",
            "",
            "| Metric | Split | Value | Baseline | Baseline value | Delta vs baseline | Direction | N |",
            "| --- | --- | ---: | --- | ---: | ---: | --- | ---: |",
        ]
    )
    lines.extend(_metric_table_row(metric) for metric in eval_input.metrics)
    lines.extend(
        [
            "",
            "### Efficiency And Rollout Speed",
            "",
            (
                f"- Single-variant latency: "
                f"`{_format_value(artifacts.efficiency.measurements.single_variant_latency_ms)}` ms."
            ),
            (
                f"- Batched throughput: "
                f"`{_format_value(artifacts.efficiency.measurements.batched_throughput_variants_per_s)}` "
                "variants/s."
            ),
            f"- Peak memory: `{artifacts.efficiency.measurements.peak_memory_bytes}` bytes.",
            (
                "- AR rollout speed: "
                f"K5 `{_format_value(_readiness_observed(artifacts, 'ar_rollout_speed', 'k5_speedup'))}`x; "
                f"K20 `{_format_value(_readiness_observed(artifacts, 'ar_rollout_speed', 'k20_speedup'))}`x. "
                "The K20 miss is recorded as an accepted v0.2 scope decision, not closure of #42."
            ),
            "",
            "## Planning Demo Evidence",
            "",
            f"- Planning status: `{_required_text(artifacts.planning_manifest, 'status')}`.",
            f"- Evaluation mode: `{_required_text(planning_summary, 'evaluation_mode')}`.",
            f"- Model id: `{_required_text(planning_summary, 'model_id')}`.",
            f"- Best distance: `{_format_value(_required_number(planning_summary, 'best_distance'))}`.",
            f"- Best objective: `{_format_value(_required_number(planning_summary, 'best_objective'))}`.",
            f"- Evaluations: `{_required_int(planning_summary, 'n_evaluations')}`.",
            f"- Elapsed seconds: `{_format_value(_required_number(planning_summary, 'elapsed_seconds'))}`.",
            f"- Stopped reason: `{_required_text(planning_summary, 'stopped_reason')}`.",
            "",
            "### Synthetic Target Edits",
            "",
            "| Chrom | Pos | Rel pos | Ref | Alt |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(
        (
            "| "
            f"{_required_text(edit, 'chrom')} | "
            f"{_required_int(edit, 'pos')} | "
            f"{_required_int(edit, 'rel_pos')} | "
            f"{_required_text(edit, 'ref')} | "
            f"{_required_text(edit, 'alt')} |"
        )
        for edit in artifacts.target_edits
    )
    lines.extend(
        [
            "",
            "### Best Planned Edits",
            "",
            "| Rel pos | Ref | Alt | Type |",
            "| ---: | --- | --- | --- |",
        ]
    )
    planned_edit_rows = (
        _mapping_item(raw_edit, field="plan_summary.best_edits")
        for raw_edit in _required_list(planning_summary, "best_edits")
    )
    lines.extend(
        (
            "| "
            f"{_required_int(row, 'rel_pos')} | "
            f"{_required_text(row, 'ref_bases')} | "
            f"{_required_text(row, 'alt_bases')} | "
            f"{_required_text(row, 'edit_type')} |"
        )
        for row in planned_edit_rows
    )
    lines.extend(
        [
            "",
            "## Discussion and Learnings",
            "",
            (
                "The central lesson is that an artifact-complete genomic world-model "
                "pipeline is easier to achieve than a useful learned edit model. The "
                "training, packaging, scoring, evaluation, rollout, planning, and release "
                "contracts now bind their inputs and outputs, but those contracts mainly "
                "make the negative findings auditable."
            ),
            "",
            (
                "The most informative failure is the gap between narrow classification "
                "successes and latent rollout behavior. Coding ClinVar balanced accuracy "
                "improves slightly over Carbon on this tiny slice, but AUROC and average "
                "precision trail Carbon, non-coding ClinVar is worse on every reported "
                "metric, BRCA2 saturation correlation trails Carbon, and rollout-state "
                "cosine similarity is far below the source-state baseline. This suggests "
                "that the current predictor has not learned an edit-transition model that "
                "transfers cleanly across evaluation modes."
            ),
            "",
            (
                "The planning demo is therefore best interpreted as integration evidence. "
                "It proves that a released checkpoint can be loaded through the planning "
                "runtime and can emit a candidate edit sequence, but the measured non-zero "
                "best distance and patience stop show that the run should not be presented "
                "as useful genomic planning behavior."
            ),
            "",
            "## Negative Findings",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in _negative_findings(artifacts))
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This paper is not clinical, privacy-assurance, runtime-assurance, deployment-readiness, or broad model-quality evidence.",
            "- The benchmark suite is broader than v0.1 but remains a bounded v0.2 evidence package.",
            "- The K20 autoregressive rollout-speed target remains open in #42 despite the accepted v0.2 scope decision.",
            "- The planning demo uses one deterministic synthetic multi-SNV target window and does not prove useful planning behavior.",
            "",
            "## Conclusions",
            "",
            "- The v0.2 evidence supports a systems and reproducibility contribution with negative model-quality findings.",
            "- Artifact generation, benchmark aggregation, readiness checks, and released-artifact planning execution are now bound by public-safe identities.",
            "- The measured result does not justify claims of broad GenoLeWM improvement over Carbon or useful multi-edit planning.",
            "",
            "## Reproducibility",
            "",
            (
                "The manuscript is generated from machine-readable benchmark, readiness, "
                "efficiency, rollout-speed, and planning-demo artifacts. Verification "
                "re-renders the paper from those artifacts and rejects stale text, missing "
                "negative findings, missing K20 scope markers, mutated planning evidence, "
                "and private absolute workstation paths."
            ),
            "",
            (
                "The artifact chain records model release, model id, dataset snapshot, "
                "commit, hardware string, package-relative artifact paths, SHA-256 hashes, "
                "and file sizes. These commitments support reproducible inspection of the "
                "reported experiments; they do not establish runtime assurance."
            ),
            "",
            "## Artifact Availability",
            "",
            "- Suite package root: `suite/`.",
            "- Planning demo package root: `planning-demo/`.",
            f"- Eval metrics: `{_package_path(artifacts.eval_metrics_path)}`.",
            f"- Eval report: `{_package_path(artifacts.eval_report_path)}`.",
            f"- Efficiency report: `{_package_path(artifacts.efficiency_report_path)}`.",
            f"- Benchmark readiness report: `{_package_path(artifacts.readiness_report_path)}`.",
            f"- Benchmark suite report: `{_package_path(artifacts.suite_report_path)}`.",
            f"- Rollout speed report: `{_package_path(artifacts.rollout_speed_path)}`.",
            f"- Rollout speed scope report: `{_package_path(artifacts.rollout_speed_scope_path)}`.",
            f"- Planning manifest: `{_package_path(artifacts.planning_manifest_path)}`.",
            f"- Planning plan: `{_package_path(artifacts.plan_path)}`.",
            f"- Planning transcript: `{_package_path(artifacts.transcript_path)}`.",
            "",
            "## References",
            "",
            (
                "1. HuggingFaceBio. Carbon-500M model card. "
                "https://huggingface.co/HuggingFaceBio/Carbon-500M"
            ),
            (
                "2. Assran et al. Self-Supervised Learning from Images with a "
                "Joint-Embedding Predictive Architecture. arXiv:2301.08243, 2023. "
                "https://arxiv.org/abs/2301.08243"
            ),
            (
                "3. Ji et al. DNABERT: pre-trained Bidirectional Encoder Representations "
                "from Transformers model for DNA-language in genome. Bioinformatics, "
                "2021. https://doi.org/10.1093/bioinformatics/btab083"
            ),
            (
                "4. Nguyen et al. HyenaDNA: Long-Range Genomic Sequence Modeling at "
                "Single Nucleotide Resolution. arXiv:2306.15794, 2023. "
                "https://arxiv.org/abs/2306.15794"
            ),
            (
                "5. Dalla-Torre et al. Nucleotide Transformer: building and evaluating "
                "robust foundation models for human genomics. Nature Methods, 2025. "
                "https://www.nature.com/articles/s41592-024-02523-z"
            ),
            (
                "6. Landrum et al. ClinVar: public archive of relationships among "
                "sequence variation and human phenotype. Nucleic Acids Research, 2014. "
                "https://academic.oup.com/nar/article/42/D1/D980/1051029"
            ),
            (
                "7. Karczewski et al. The mutational constraint spectrum quantified "
                "from variation in 141,456 humans. Nature, 2020. "
                "https://www.nature.com/articles/s41586-020-2308-7"
            ),
            (
                "8. Benegas et al. Benchmarking DNA Sequence Models for Causal Regulatory "
                "Variant Prediction in Human Genetics. bioRxiv, 2025. "
                "https://doi.org/10.1101/2025.02.11.637758"
            ),
            (
                "9. Cheng et al. Accurate proteome-wide missense variant effect prediction "
                "with AlphaMissense. Science, 2023. "
                "https://doi.org/10.1126/science.adg7492"
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    if PLACEHOLDER_RE.search(text):
        raise InputError("generated paper contains placeholder text")
    return text


def _verify_paper_path(
    path: Path,
    artifacts: _SeriousCompletionArtifacts | None,
    issues: list[SeriousCompletionPaperIssue],
) -> None:
    text = path.read_text(encoding="utf-8")
    _require_sections(
        path,
        text,
        issues,
        sections=(
            "Abstract",
            "Introduction",
            "Related Work",
            "Method",
            "Experiments",
            "Citation Metadata",
            "Artifact Inputs",
            "Results",
            "Planning Demo Evidence",
            "Discussion and Learnings",
            "Negative Findings",
            "Limitations",
            "Reproducibility",
            "Conclusions",
            "Artifact Availability",
            "References",
        ),
    )
    required_patterns = {
        "serious_paper.generated_by": rf"(?m)^Generated by: {re.escape(GENERATED_BY)}$",
        "serious_paper.generated_at": r"(?m)^Generated: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
        "serious_paper.kind": rf"(?m)^Paper kind: {re.escape(PAPER_KIND)}$",
        "serious_paper.carbon": r"Carbon-500M",
        "serious_paper.jepa": r"Joint-Embedding Predictive Architecture",
        "serious_paper.genomic_foundation_models": r"DNABERT.*HyenaDNA.*Nucleotide Transformer",
        "serious_paper.eval_metrics": re.escape(EVAL_METRICS_NAME),
        "serious_paper.eval_report": re.escape(EVAL_REPORT_NAME),
        "serious_paper.efficiency_report": re.escape(EFFICIENCY_REPORT_NAME),
        "serious_paper.readiness": re.escape(READINESS_REPORT_NAME),
        "serious_paper.suite": re.escape(SUITE_REPORT_NAME),
        "serious_paper.rollout_speed": re.escape(ROLLOUT_SPEED_NAME),
        "serious_paper.rollout_speed_scope": re.escape(ROLLOUT_SPEED_SCOPE_NAME),
        "serious_paper.planning_manifest": re.escape(PLANNING_MANIFEST_NAME),
        "serious_paper.plan": re.escape(PLAN_NAME),
        "serious_paper.transcript": re.escape(PLANNING_TRANSCRIPT_NAME),
        "serious_paper.k20_open": r"K20.*#42",
        "serious_paper.weak_planning": r"does not prove useful planning behavior",
        "serious_paper.negative_results": r"negative-results and systems",
        "serious_paper.references": r"(?m)^## References$",
    }
    for code, pattern in required_patterns.items():
        if re.search(pattern, text) is None:
            _error(issues, code, path, f"missing paper marker: {code}")
    if PLACEHOLDER_RE.search(text):
        _error(
            issues, "serious_paper.placeholder", path, "paper cannot contain placeholder wording"
        )
    if artifacts is None:
        return
    if artifacts.eval_input.model_id not in text:
        _error(issues, "serious_paper.model_id", path, "paper does not name the model id")
    if artifacts.eval_input.dataset_snapshot not in text:
        _error(
            issues,
            "serious_paper.dataset_snapshot",
            path,
            "paper does not name the dataset snapshot id",
        )
    title = _paper_title(text)
    generated_at = _paper_generated_at(text)
    if title is None:
        _error(issues, "serious_paper.title", path, "generated paper must start with an H1 title")
        return
    if generated_at is None:
        return
    try:
        expected = _render_serious_completion_paper(
            title=title,
            generated_at=generated_at,
            artifacts=artifacts,
        )
    except GenoLeWMError as exc:
        _error(issues, "serious_paper.render_failed", path, exc.message or str(exc))
        return
    if text != expected:
        _error(
            issues,
            "serious_paper.stale",
            path,
            "paper does not match render of current serious-completion artifacts",
        )


def _validate_artifacts(artifacts: _SeriousCompletionArtifacts) -> None:
    _validate_suite_report(artifacts.suite_report)
    _validate_readiness_report(artifacts.readiness, artifacts.eval_input)
    _validate_rollout_speed(artifacts.rollout_speed, expected_commit=artifacts.eval_input.commit)
    _validate_rollout_speed_scope(artifacts.rollout_speed_scope)
    _validate_planning_demo(artifacts)


def _validate_suite_report(payload: dict[str, Any]) -> None:
    _require_value(payload, "schema_version", SCHEMA_VERSION, label="suite report")
    _require_value(payload, "generated_by", SUITE_GENERATED_BY, label="suite report")
    if payload.get("ok") is not True or payload.get("status") != "pass":
        raise InputError("suite report must have ok=true and status=pass")
    _require_claim_boundary(payload, "suite report")
    steps = _required_list(payload, "steps")
    for item in steps:
        step = _mapping_item(item, field="suite steps")
        if step.get("status") != "pass":
            raise InputError("suite report contains a non-passing step")


def _validate_readiness_report(payload: dict[str, Any], eval_input: EvalReportInput) -> None:
    _require_value(payload, "schema_version", SCHEMA_VERSION, label="readiness report")
    _require_value(payload, "generated_by", READINESS_GENERATED_BY, label="readiness report")
    if payload.get("ok") is not True:
        raise InputError("readiness report must have ok=true")
    _require_identity_mapping(payload, eval_input, label="readiness report")
    _require_claim_boundary(payload, "readiness report")
    if payload.get("release_inputs_required") is not True:
        raise InputError("readiness report must require release inputs")
    if payload.get("missing_or_failed_benchmarks") not in ([], ()):
        raise InputError("readiness report must not have missing or failed benchmarks")
    if not _text_list(payload.get("negative_findings"), "readiness negative_findings"):
        raise InputError("readiness report must carry negative findings")
    rows = {str(row.get("benchmark_id")): row for row in _benchmark_rows(payload)}
    missing = [benchmark for benchmark in REQUIRED_BENCHMARKS if benchmark not in rows]
    if missing:
        raise InputError(
            "readiness report is missing required benchmark rows", details={"missing": missing}
        )
    for benchmark in REQUIRED_BENCHMARKS:
        status = rows[benchmark].get("status")
        expected = "rescoped" if benchmark == "ar_rollout_speed" else "pass"
        if status != expected:
            raise InputError(
                "readiness benchmark row has unexpected status",
                details={"benchmark": benchmark, "expected": expected, "observed": status},
            )
    _require_negative_delta(rows["clinvar_noncoding"], "auroc")
    _require_negative_delta(rows["brca2_saturation"], "spearman_rho")
    _require_rollout_weakness(rows["rollout_phased_haplotypes"])
    _require_rollout_weakness(rows["rollout_synthetic_edit_chains"])
    ar_values = _required_mapping(rows["ar_rollout_speed"], "observed_values")
    if _required_number(ar_values, "k20_speedup") >= 5.0:
        raise InputError("serious-completion paper expects K20 to remain a recorded #42 miss")
    scope_decision = _required_mapping(rows["ar_rollout_speed"], "scope_decision")
    if scope_decision.get("status") != "accepted":
        raise InputError("AR rollout-speed scope decision must be accepted")


def _validate_rollout_speed(payload: dict[str, Any], *, expected_commit: str) -> None:
    _require_value(payload, "schema_version", SCHEMA_VERSION, label="rollout speed report")
    _require_value(payload, "generated_by", "bench.rollout", label="rollout speed report")
    if payload.get("ok") is not False:
        raise InputError("rollout speed report must preserve the failed K20 target with ok=false")
    if _required_text(payload, "commit") != expected_commit:
        raise InputError("rollout speed report commit does not match eval identity")
    _require_claim_boundary(payload, "rollout speed report")
    rows = {
        _required_int(_mapping_item(row, field="rollout speed rows"), "horizon"): row
        for row in _required_list(payload, "rows")
    }
    for horizon in (5, 20):
        if horizon not in rows:
            raise InputError(
                "rollout speed report is missing horizon", details={"horizon": horizon}
            )
    k20 = _mapping_item(rows[20], field="rollout speed k20")
    if k20.get("target_met") is not False or _required_number(
        k20, "measured_speedup"
    ) >= _required_number(k20, "target_speedup"):
        raise InputError("rollout speed K20 row must preserve the target miss")


def _validate_rollout_speed_scope(payload: dict[str, Any]) -> None:
    _require_value(payload, "schema_version", SCHEMA_VERSION, label="rollout speed scope")
    _require_value(
        payload,
        "generated_by",
        ROLLOUT_SPEED_SCOPE_GENERATED_BY,
        label="rollout speed scope",
    )
    if payload.get("ok") is not True or payload.get("status") != "accepted":
        raise InputError("rollout speed scope report must have ok=true and status=accepted")
    _require_claim_boundary(payload, "rollout speed scope")
    if "#42" not in _text_list(payload.get("issue_refs"), "rollout speed scope issue_refs"):
        raise InputError("rollout speed scope report must retain #42 issue reference")
    if not _text_list(payload.get("negative_findings"), "rollout speed scope negative_findings"):
        raise InputError("rollout speed scope report must carry negative findings")


def _validate_planning_demo(artifacts: _SeriousCompletionArtifacts) -> None:
    manifest = artifacts.planning_manifest
    plan = artifacts.plan
    _require_value(manifest, "schema_version", SCHEMA_VERSION, label="planning manifest")
    _require_value(
        manifest, "generated_by", "tools.jobs.planning_demo_run", label="planning manifest"
    )
    _require_value(manifest, "status", "passed", label="planning manifest")
    _require_claim_boundary(manifest, "planning manifest")
    boundary = _required_text(manifest, "claim_boundary").lower()
    if "useful planning behavior" not in boundary:
        raise InputError("planning manifest must not claim useful planning behavior")
    if not _text_list(manifest.get("negative_findings"), "planning negative_findings"):
        raise InputError("planning manifest must carry negative findings")

    _require_value(plan, "generated_by", "geno-lewm-plan", label="planning plan")
    _require_value(plan, "evaluation_mode", "manifest_runtime", label="planning plan")
    _require_claim_boundary(plan, "planning plan")
    runtime = _required_mapping(plan, "runtime")
    summary = _required_mapping(manifest, "plan_summary")
    result = _required_mapping(plan, "result")
    if _required_text(runtime, "model_id") != artifacts.eval_input.model_id:
        raise InputError("planning plan runtime model_id does not match benchmark model_id")
    if _required_text(summary, "model_id") != artifacts.eval_input.model_id:
        raise InputError("planning manifest model_id does not match benchmark model_id")
    for key in ("best_distance", "best_objective", "elapsed_seconds"):
        if _required_number(summary, key) != _required_number(result, key):
            raise InputError(
                "planning manifest summary does not match plan result", details={"field": key}
            )
    for key in ("n_evaluations", "stopped_reason"):
        if summary.get(key) != result.get(key):
            raise InputError(
                "planning manifest summary does not match plan result", details={"field": key}
            )
    if _required_number(summary, "best_distance") <= 0.0:
        raise InputError("planning best_distance must preserve weak non-zero behavior")
    if _required_text(summary, "stopped_reason") != "patience":
        raise InputError("planning demo must preserve patience-stop evidence")
    if (
        "# GenoLeWM Planning Demo" not in artifacts.transcript
        or "best_distance" not in artifacts.transcript
    ):
        raise InputError("planning transcript must include result evidence")

    manifest_artifacts = _required_mapping(manifest, "artifacts")
    _require_model_artifact_identity(manifest_artifacts, artifacts.eval_input.model_id)
    local_files = {
        "plan": artifacts.plan_path,
        "plan_stdout": artifacts.plan_stdout_path,
        "source_window": artifacts.source_window_path,
        "target_window": artifacts.target_window_path,
        "target_edits": artifacts.target_edits_path,
        "transcript": artifacts.transcript_path,
    }
    for label, path in local_files.items():
        _require_manifest_file_identity(manifest_artifacts, label, path)


def _require_shared_identity(eval_input: EvalReportInput, efficiency: EfficiencyReport) -> None:
    mismatches = {
        "model_id": (eval_input.model_id, efficiency.model_id),
        "model_release": (eval_input.model_release, efficiency.model_release),
        "dataset_snapshot": (eval_input.dataset_snapshot, efficiency.dataset_snapshot),
        "commit": (eval_input.commit, efficiency.commit),
        "hardware": (eval_input.hardware, efficiency.hardware),
    }
    observed = {
        key: {"eval": expected, "efficiency": actual}
        for key, (expected, actual) in mismatches.items()
        if expected != actual
    }
    if observed:
        raise InputError("efficiency report identity does not match eval metrics", details=observed)


def _require_identity_mapping(
    payload: dict[str, Any], eval_input: EvalReportInput, *, label: str
) -> None:
    expected = {
        "model_id": eval_input.model_id,
        "model_release": eval_input.model_release,
        "dataset_snapshot": eval_input.dataset_snapshot,
        "commit": eval_input.commit,
        "hardware": eval_input.hardware,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise InputError(f"{label} identity does not match eval metrics", details=mismatches)


def _require_negative_delta(row: dict[str, Any], metric: str) -> None:
    deltas = _required_mapping(row, "delta_vs_baseline")
    if _required_number(deltas, metric) >= 0.0:
        raise InputError("expected negative baseline delta is missing", details={"metric": metric})


def _require_rollout_weakness(row: dict[str, Any]) -> None:
    deltas = _required_mapping(row, "delta_vs_baseline")
    if _required_number(deltas, "cosine_similarity_mean") >= 0.0:
        raise InputError("rollout cosine delta must preserve weak source-baseline result")
    if _required_number(deltas, "l2_distance_mean") <= 0.0:
        raise InputError("rollout L2 delta must preserve weak source-baseline result")


def _require_model_artifact_identity(
    manifest_artifacts: dict[str, Any],
    model_id: str,
) -> None:
    model_manifest = _required_mapping(manifest_artifacts, "model_manifest")
    if _required_text(model_manifest, "sha256") != model_id:
        raise InputError("planning model_manifest identity does not match benchmark model_id")
    for label in ("predictor", "action_encoder"):
        identity = _required_mapping(manifest_artifacts, label)
        if not looks_like_sha256(_required_text(identity, "sha256")):
            raise InputError("planning model artifact has invalid sha256", details={"label": label})
        _required_int(identity, "size_bytes")
        _required_text(identity, "path")


def _require_manifest_file_identity(
    manifest_artifacts: dict[str, Any],
    label: str,
    path: Path,
) -> None:
    identity = _required_mapping(manifest_artifacts, label)
    expected_path = "demo/" + path.name
    observed_path = _required_text(identity, "path")
    if observed_path != expected_path:
        raise InputError(
            "planning manifest artifact path is not canonical",
            details={"label": label, "expected": expected_path, "observed": observed_path},
        )
    observed_hash = _required_text(identity, "sha256")
    observed_size = _required_int(identity, "size_bytes")
    expected_hash = sha256_file(path)
    expected_size = path.stat().st_size
    if observed_hash != expected_hash or observed_size != expected_size:
        raise InputError(
            "planning manifest artifact identity is stale",
            details={
                "label": label,
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                "expected_size_bytes": expected_size,
                "observed_size_bytes": observed_size,
            },
        )


def _artifact_identity_rows(artifacts: _SeriousCompletionArtifacts) -> list[str]:
    paths = (
        artifacts.eval_metrics_path,
        artifacts.eval_report_path,
        artifacts.efficiency_report_path,
        artifacts.readiness_report_path,
        artifacts.suite_report_path,
        artifacts.rollout_speed_path,
        artifacts.rollout_speed_scope_path,
        artifacts.planning_manifest_path,
        artifacts.plan_path,
        artifacts.plan_stdout_path,
        artifacts.transcript_path,
        artifacts.source_window_path,
        artifacts.target_window_path,
        artifacts.target_edits_path,
    )
    return [
        f"| {_artifact_label(path)} | `{_package_path(path)}` | `{sha256_file(path)}` | {path.stat().st_size} |"
        for path in paths
    ]


def _benchmark_table_row(row: dict[str, Any]) -> str:
    return (
        "| "
        f"{_required_text(row, 'benchmark_id')} | "
        f"{_required_text(row, 'status')} | "
        f"{_mapping_summary(row.get('observed_values'))} | "
        f"{_mapping_summary(row.get('delta_vs_baseline'))} | "
        f"{', '.join(_text_list(row.get('issue_refs'), 'issue_refs'))} |"
    )


def _metric_table_row(metric: Any) -> str:
    baseline = metric.baseline or "not reported"
    baseline_value = "-" if metric.baseline_value is None else _format_value(metric.baseline_value)
    delta = "-" if metric.delta_vs_baseline is None else _format_value(metric.delta_vs_baseline)
    n_value = "-" if metric.n is None else str(metric.n)
    direction = "higher" if metric.higher_is_better else "lower"
    return (
        "| "
        f"{metric.name} | "
        f"{metric.split} | "
        f"{_format_value(metric.value)} | "
        f"{baseline} | "
        f"{baseline_value} | "
        f"{delta} | "
        f"{direction} | "
        f"{n_value} |"
    )


def _negative_findings(artifacts: _SeriousCompletionArtifacts) -> list[str]:
    rows = {str(row.get("benchmark_id")): row for row in _benchmark_rows(artifacts.readiness)}
    findings = [
        (
            "GenoLeWM trails Carbon on most VEP rows: ClinVar non-coding AUROC delta "
            f"`{_format_value(_readiness_delta(rows, 'clinvar_noncoding', 'auroc'))}` and "
            "BRCA2 Spearman delta "
            f"`{_format_value(_readiness_delta(rows, 'brca2_saturation', 'spearman_rho'))}`."
        ),
        (
            "Only narrow coding ClinVar balanced-accuracy and accuracy deltas are positive "
            f"(`{_format_value(_readiness_delta(rows, 'clinvar_coding', 'balanced_accuracy'))}` each); "
            "this is not a broad model-quality win."
        ),
        (
            "Rollout fidelity is weak versus source-state baselines: phased-haplotype cosine delta "
            f"`{_format_value(_readiness_delta(rows, 'rollout_phased_haplotypes', 'cosine_similarity_mean'))}` "
            "and synthetic-chain cosine delta "
            f"`{_format_value(_readiness_delta(rows, 'rollout_synthetic_edit_chains', 'cosine_similarity_mean'))}`."
        ),
        (
            "K20 rollout speed remains below the RFC-0004 target: measured "
            f"`{_format_value(_readiness_observed(artifacts, 'ar_rollout_speed', 'k20_speedup'))}`x "
            "against a 5.0x target, with #42 still open."
        ),
        (
            "The planning demo stopped by patience with best_distance "
            f"`{_format_value(_required_number(_required_mapping(artifacts.planning_manifest, 'plan_summary'), 'best_distance'))}`; "
            "it records execution from released artifacts, not useful planning behavior."
        ),
    ]
    for payload in (
        artifacts.eval_input.negative_findings,
        _text_list(artifacts.readiness.get("negative_findings"), "readiness negative_findings"),
        _text_list(
            artifacts.rollout_speed_scope.get("negative_findings"),
            "rollout speed scope negative_findings",
        ),
        _text_list(
            artifacts.planning_manifest.get("negative_findings"),
            "planning negative_findings",
        ),
    ):
        for item in payload:
            if item not in findings:
                findings.append(item)
    return findings


def _readiness_delta(rows: dict[str, dict[str, Any]], benchmark: str, metric: str) -> float:
    return _required_number(_required_mapping(rows[benchmark], "delta_vs_baseline"), metric)


def _readiness_observed(
    artifacts: _SeriousCompletionArtifacts,
    benchmark: str,
    metric: str,
) -> float:
    rows = {str(row.get("benchmark_id")): row for row in _benchmark_rows(artifacts.readiness)}
    return _required_number(_required_mapping(rows[benchmark], "observed_values"), metric)


def _mapping_summary(raw: object) -> str:
    if raw is None:
        return "not reported"
    if not isinstance(raw, dict) or not raw:
        return "not reported"
    return "; ".join(f"{key}={_format_value(value)}" for key, value in sorted(raw.items()))


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        raise InputError("boolean values cannot be formatted as metrics")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, str):
        return value
    raise InputError("unsupported metric value type", details={"type": type(value).__name__})


def _artifact_label(path: Path) -> str:
    if path.name == EVAL_METRICS_NAME:
        return "v0.2 eval metrics"
    if path.name == EVAL_REPORT_NAME:
        return "v0.2 eval report"
    if path.name == EFFICIENCY_REPORT_NAME:
        return "v0.2 efficiency report"
    if path.name == READINESS_REPORT_NAME:
        return "benchmark readiness report"
    if path.name == SUITE_REPORT_NAME:
        return "benchmark suite report"
    if path.name == ROLLOUT_SPEED_NAME:
        return "AR rollout speed report"
    if path.name == ROLLOUT_SPEED_SCOPE_NAME:
        return "AR rollout speed scope report"
    if path.name == PLANNING_MANIFEST_NAME:
        return "planning demo manifest"
    if path.name == PLAN_NAME:
        return "planning plan"
    if path.name == PLAN_STDOUT_NAME:
        return "planning stdout"
    if path.name == PLANNING_TRANSCRIPT_NAME:
        return "planning transcript"
    if path.name == SOURCE_WINDOW_NAME:
        return "planning source FASTA"
    if path.name == TARGET_WINDOW_NAME:
        return "planning target FASTA"
    if path.name == TARGET_EDITS_NAME:
        return "planning target edits"
    return path.name


def _package_path(path: Path) -> str:
    parts = path.parts
    if "suite" in parts:
        index = parts.index("suite")
        return "/".join(parts[index:])
    if "planning-demo" in parts:
        index = parts.index("planning-demo")
        return "/".join(parts[index:])
    if path.parent.name == "model":
        return f"suite/model/{path.name}"
    if path.parent.name == "bench":
        return f"suite/bench/{path.name}"
    return f"planning-demo/{path.name}"


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
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


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"failed to read {label}", details={"path": str(path)}) from exc


def _read_target_edits(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError("failed to read target edits", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "target edits JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise InputError("target edits must be a non-empty JSON list")
    return tuple(_mapping_item(item, field="target_edits") for item in payload)


def _benchmark_rows(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = _required_list(payload, "benchmark_rows")
    return tuple(_mapping_item(row, field="benchmark_rows") for row in rows)


def _require_sections(
    path: Path,
    text: str,
    issues: list[SeriousCompletionPaperIssue],
    *,
    sections: tuple[str, ...],
) -> None:
    for section in sections:
        if re.search(rf"(?m)^## {re.escape(section)}$", text) is None:
            _error(
                issues,
                "serious_paper.section_missing",
                path,
                f"missing required Markdown section: {section}",
            )


def _paper_title(text: str) -> str | None:
    match = re.search(r"(?m)^# (?P<title>.+)$", text)
    if match is None:
        return None
    title = match.group("title").strip()
    return title or None


def _paper_generated_at(text: str) -> str | None:
    match = re.search(r"(?m)^Generated: (?P<generated_at>.+)$", text)
    if match is None:
        return None
    generated_at = match.group("generated_at").strip()
    return generated_at or None


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise InputError(f"{key} must be an object")
    return value


def _mapping_item(raw: object, *, field: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InputError(f"{field} entries must be objects")
    return raw


def _required_list(raw: dict[str, Any], key: str) -> tuple[object, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise InputError(f"{key} must be a non-empty list")
    return tuple(value)


def _required_text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise InputError(f"{key} must be a non-empty string")
    return value


def _required_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{key} must be an integer")
    return value


def _required_number(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(f"{key} must be numeric")
    return float(value)


def _text_list(raw: object, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise InputError(f"{field} must be a list")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item:
            raise InputError(f"{field}[{index}] must be a non-empty string")
        values.append(item)
    return tuple(values)


def _require_value(
    payload: dict[str, Any],
    key: str,
    expected: object,
    *,
    label: str,
) -> None:
    observed = payload.get(key)
    if observed != expected:
        raise InputError(
            f"{label} {key} is invalid",
            details={"expected": expected, "observed": observed},
        )


def _require_claim_boundary(payload: dict[str, Any], label: str) -> None:
    boundary = _required_text(payload, "claim_boundary").lower()
    required_any = (
        "clinical",
        "privacy",
        "runtime",
        "model-quality",
        "release-readiness",
        "deployment",
        "evidence only",
    )
    if not any(phrase in boundary for phrase in required_any):
        raise InputError(f"{label} claim_boundary is incomplete")


def _require_utc_timestamp(value: str, *, field: str) -> None:
    if UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise InputError(f"{field} must be a UTC ISO-8601 timestamp ending in Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _error(
    issues: list[SeriousCompletionPaperIssue],
    code: str,
    path: Path,
    message: str,
) -> None:
    issues.append(
        SeriousCompletionPaperIssue(
            severity="error",
            code=code,
            path=str(path),
            message=message,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
