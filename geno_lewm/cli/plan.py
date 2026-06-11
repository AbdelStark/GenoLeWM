# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-plan`` - CEM-based latent planning."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from geno_lewm.action import EditType, RelEdit, apply_edits
from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.deploy import GenoLeWMRuntime
from geno_lewm.encoder.windowing import canonicalize_dna
from geno_lewm.errors import InputError, RuntimeSetupError, SchemaCompatError
from geno_lewm.planning.cem import (
    CandidateEvaluation,
    PlanningConfig,
    PlanningResult,
    cosine_distance,
    l2_distance,
    plan,
)
from geno_lewm.planning.costs import bp_cost, count_cost, weighted_type_cost
from geno_lewm.planning.sampling import ActionSampler
from geno_lewm.provenance import load_manifest, sha256_file
from geno_lewm.training import EditTypeWeight

__all__ = ["app", "cli_main"]

GENERATED_BY = "geno-lewm-plan"
SCHEMA_VERSION = "0.1"
SEQUENCE_PROXY_MODE = "sequence_proxy"
MANIFEST_RUNTIME_MODE = "manifest_runtime"
_DISTANCES = ("l2", "cosine")
_COSTS = ("count", "bp", "weighted_type")
_DEFAULT_EDIT_TYPES = "snv"
_BASE_TO_STATE: dict[str, tuple[float, float, float, float]] = {
    "A": (1.0, 0.0, 0.0, 0.0),
    "C": (0.0, 1.0, 0.0, 0.0),
    "G": (0.0, 0.0, 1.0, 0.0),
    "T": (0.0, 0.0, 0.0, 1.0),
    "N": (0.0, 0.0, 0.0, 0.0),
}

app = typer.Typer(
    name="geno-lewm-plan",
    help=(
        "Plan edit sequences with the CEM solver. Manifest-backed mode uses "
        "local model artifacts; sequence-proxy mode is an explicit development "
        "smoke path, not learned-model evidence."
    ),
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@app.callback(invoke_without_command=True)
def main(
    window_fasta: Annotated[
        Path | None,
        typer.Option("--window-fasta", help="Reference FASTA window for the initial sequence."),
    ] = None,
    target_fasta: Annotated[
        Path | None,
        typer.Option("--target-fasta", help="Target FASTA window to encode or proxy-match."),
    ] = None,
    initial_state: Annotated[
        Path | None,
        typer.Option(
            "--initial-state",
            help="Precomputed initial latent state as JSON list or .npy array.",
        ),
    ] = None,
    target_state: Annotated[
        Path | None,
        typer.Option(
            "--target-state",
            help="Precomputed target latent state as JSON list or .npy array.",
        ),
    ] = None,
    model_dir: Annotated[
        Path | None,
        typer.Option("--model-dir", help="Local model directory for manifest-backed planning."),
    ] = None,
    backend: Annotated[
        str,
        typer.Option("--backend", help="Runtime backend: auto, cpu, cuda, onnx, or coreml."),
    ] = "auto",
    output: Annotated[
        Path,
        typer.Option("--output", help="Destination plan.json path."),
    ] = Path("plan.json"),
    horizon: Annotated[int, typer.Option("--horizon", help="Number of edits to plan.")] = 5,
    iterations: Annotated[
        int,
        typer.Option("--iterations", help="CEM iterations."),
    ] = 5,
    samples: Annotated[
        int,
        typer.Option("--samples", help="Candidate sequences per CEM iteration."),
    ] = 1024,
    elite: Annotated[
        int,
        typer.Option("--elite", help="Elite candidate count per CEM iteration."),
    ] = 64,
    stopping_eps: Annotated[
        float,
        typer.Option("--stopping-eps", help="Stop once best distance is below this threshold."),
    ] = 0.05,
    patience: Annotated[
        int,
        typer.Option("--patience", help="Stop after this many stale iterations."),
    ] = 2,
    smoothing: Annotated[
        float,
        typer.Option("--smoothing", help="CEM proposal smoothing in [0, 1]."),
    ] = 0.1,
    cost_weight: Annotated[
        float,
        typer.Option("--cost-weight", help="Weight for edit-sequence cost in the objective."),
    ] = 0.0,
    cost: Annotated[
        str,
        typer.Option("--cost", help="Cost function: count, bp, or weighted_type."),
    ] = "count",
    distance: Annotated[
        str,
        typer.Option("--distance", help="Distance function: l2 or cosine."),
    ] = "l2",
    edit_types: Annotated[
        str,
        typer.Option(
            "--edit-types",
            help="Comma-separated edit types to sample: snv,ins,del,mnv,indel.",
        ),
    ] = _DEFAULT_EDIT_TYPES,
    edge_margin: Annotated[
        int,
        typer.Option("--edge-margin", help="Sampler edge margin in bp."),
    ] = 64,
    position_bin_bp: Annotated[
        int,
        typer.Option("--position-bin-bp", help="Sampler position-bin width in bp."),
    ] = 8,
    allow_sequence_proxy: Annotated[
        bool,
        typer.Option(
            "--allow-sequence-proxy",
            help=(
                "Allow FASTA-to-FASTA planning without model artifacts using an explicit "
                "sequence-state proxy. This is not learned predictor evidence."
            ),
        ),
    ] = False,
    config: Annotated[str | None, _S["config"]] = None,
    set_overrides: Annotated[list[str] | None, _S["set_overrides"]] = None,
    seed: Annotated[int | None, _S["seed"]] = None,
    deterministic: Annotated[bool, _S["deterministic"]] = False,
    log_level: Annotated[str, _S["log_level"]] = "info",
    log_dir: Annotated[str | None, _S["log_dir"]] = None,
    run_id: Annotated[str | None, _S["run_id"]] = None,
    wandb_project: Annotated[str | None, _S["wandb_project"]] = None,
    no_receipt: Annotated[bool, _S["no_receipt"]] = False,
    print_config: Annotated[bool, _S["print_config"]] = False,
    print_config_tree: Annotated[bool, _S["print_config_tree"]] = False,
    explain: Annotated[str | None, _S["explain"]] = None,
    quiet: Annotated[bool, _S["quiet"]] = False,
    no_banner: Annotated[bool, _S["no_banner"]] = False,
    version: Annotated[bool, _S["version"]] = False,
) -> None:
    opts: SharedOptions | None = finalize_shared(
        config=config,
        set_overrides=set_overrides,
        seed=seed,
        deterministic=deterministic,
        log_level=log_level,
        log_dir=log_dir,
        run_id=run_id,
        wandb_project=wandb_project,
        no_receipt=no_receipt,
        print_config=print_config,
        print_config_tree=print_config_tree,
        explain=explain,
        quiet=quiet,
        no_banner=no_banner,
        version=version,
        default_config_name="plan",
    )
    if opts is None:
        return

    window_path = _required_path("window-fasta", window_fasta)
    window = _load_fasta_sequence(window_path)
    config_obj = PlanningConfig(
        horizon=horizon,
        n_iterations=iterations,
        n_samples=samples,
        n_elite=elite,
        cost_weight=cost_weight,
        stopping_eps=stopping_eps,
        patience=patience,
        seed=opts.seed,
        smoothing=smoothing,
    )
    sampler = ActionSampler(
        window,
        seed=opts.seed,
        edge_margin=edge_margin,
        type_weights=_parse_edit_type_weights(edit_types),
        position_bin_bp=position_bin_bp,
    )
    distance_name = _normalize_choice("distance", distance, _DISTANCES)
    cost_name = _normalize_choice("cost", cost, _COSTS)
    runtime = _runtime(model_dir=model_dir, backend=backend)
    evaluator, mode, target_vector, input_artifacts, negative_findings = _build_evaluator(
        window=window,
        window_path=window_path,
        target_fasta=target_fasta,
        initial_state=initial_state,
        target_state=target_state,
        runtime=runtime,
        distance_name=distance_name,
        allow_sequence_proxy=allow_sequence_proxy,
    )
    result = plan(
        evaluator,
        sampler,
        config=config_obj,
        cost_fn=_cost_function(cost_name),
    )
    payload = _plan_payload(
        result,
        mode=mode,
        target_vector=target_vector,
        config=config_obj,
        distance=distance_name,
        cost=cost_name,
        edit_types=edit_types,
        edge_margin=edge_margin,
        position_bin_bp=position_bin_bp,
        backend=backend,
        model_dir=model_dir,
        input_artifacts=input_artifacts,
        negative_findings=negative_findings,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "best_distance": result.best_distance,
                "evaluation_mode": mode,
                "n_evaluations": result.n_evaluations,
                "output_path": str(output),
                "stopped_reason": result.stopped_reason,
            },
            sort_keys=True,
        )
    )


def _runtime(*, model_dir: Path | None, backend: str) -> GenoLeWMRuntime | None:
    if model_dir is None:
        return None
    return GenoLeWMRuntime(model_dir, backend=backend)


def _build_evaluator(
    *,
    window: str,
    window_path: Path,
    target_fasta: Path | None,
    initial_state: Path | None,
    target_state: Path | None,
    runtime: GenoLeWMRuntime | None,
    distance_name: str,
    allow_sequence_proxy: bool,
) -> tuple[
    Any,
    str,
    tuple[float, ...],
    dict[str, dict[str, object]],
    list[str],
]:
    if target_fasta is not None and target_state is not None:
        raise InputError("choose either --target-fasta or --target-state, not both")
    if target_fasta is None and target_state is None:
        raise InputError("provide --target-fasta or --target-state")

    artifacts: dict[str, dict[str, object]] = {"window_fasta": _path_identity(window_path)}
    distance_fn = _distance_function(distance_name)

    if target_state is not None:
        if runtime is None:
            raise InputError("--target-state planning requires --model-dir")
        if initial_state is None:
            raise InputError("--target-state planning requires --initial-state")
        initial_vector = _load_numeric_state(initial_state)
        target_vector = _load_numeric_state(target_state)
        artifacts["initial_state"] = _path_identity(initial_state)
        artifacts["target_state"] = _path_identity(target_state)

        def evaluate_runtime_state(edits: Sequence[RelEdit]) -> CandidateEvaluation:
            predicted = _final_state_vector(runtime.predict(initial_vector, edits), target_vector)
            return CandidateEvaluation(
                distance=distance_fn(predicted, target_vector),
                predicted_state=predicted,
            )

        return evaluate_runtime_state, MANIFEST_RUNTIME_MODE, target_vector, artifacts, []

    target_path = _required_path("target-fasta", target_fasta)
    target_window = _load_fasta_sequence(target_path)
    artifacts["target_fasta"] = _path_identity(target_path)

    if runtime is not None:
        initial_vector = _final_state_vector(runtime.encode_window(window), ())
        target_vector = _final_state_vector(runtime.encode_window(target_window), ())

        def evaluate_runtime_fasta(edits: Sequence[RelEdit]) -> CandidateEvaluation:
            predicted = _final_state_vector(runtime.predict(initial_vector, edits), target_vector)
            return CandidateEvaluation(
                distance=distance_fn(predicted, target_vector),
                predicted_state=predicted,
            )

        return evaluate_runtime_fasta, MANIFEST_RUNTIME_MODE, target_vector, artifacts, []

    if not allow_sequence_proxy:
        raise RuntimeSetupError(
            "FASTA planning without --model-dir requires --allow-sequence-proxy",
            remediation=(
                "pass --model-dir for learned predictor planning, or pass "
                "--allow-sequence-proxy for a non-model smoke run"
            ),
        )
    if len(window) != len(target_window):
        raise InputError(
            "sequence-proxy planning requires equal-length window and target FASTA",
            details={"window_len": len(window), "target_len": len(target_window)},
        )
    target_vector = _sequence_state(target_window)

    def evaluate_sequence_proxy(edits: Sequence[RelEdit]) -> CandidateEvaluation:
        edited_window = apply_edits(window, edits, preserve_length=True)
        predicted = _sequence_state(edited_window)
        return CandidateEvaluation(
            distance=distance_fn(predicted, target_vector),
            predicted_state=predicted,
        )

    return (
        evaluate_sequence_proxy,
        SEQUENCE_PROXY_MODE,
        target_vector,
        artifacts,
        [
            (
                "Sequence-proxy planning is not learned predictor evidence; "
                "use manifest_runtime mode for model evidence."
            )
        ],
    )


def _plan_payload(
    result: PlanningResult,
    *,
    mode: str,
    target_vector: tuple[float, ...],
    config: PlanningConfig,
    distance: str,
    cost: str,
    edit_types: str,
    edge_margin: int,
    position_bin_bp: int,
    backend: str,
    model_dir: Path | None,
    input_artifacts: dict[str, dict[str, object]],
    negative_findings: list[str],
) -> dict[str, object]:
    return {
        "generated_by": GENERATED_BY,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluation_mode": mode,
        "claim_boundary": (
            "Planning output is research evidence only. It is not clinical, deployment, "
            "privacy-assurance, or runtime-assurance evidence."
        ),
        "negative_findings": negative_findings,
        "inputs": {
            **input_artifacts,
            **({} if model_dir is None else {"model_dir": {"path": _public_path_label(model_dir)}}),
        },
        "runtime": {
            "backend": backend,
            "model_id": None if model_dir is None else _runtime_model_id(model_dir),
        },
        "config": {
            "horizon": config.horizon,
            "n_iterations": config.n_iterations,
            "n_samples": config.n_samples,
            "n_elite": config.n_elite,
            "cost_weight": config.cost_weight,
            "stopping_eps": config.stopping_eps,
            "patience": config.patience,
            "seed": config.seed,
            "smoothing": config.smoothing,
            "distance": distance,
            "cost": cost,
            "edit_types": edit_types,
            "edge_margin": edge_margin,
            "position_bin_bp": position_bin_bp,
        },
        "target_state_summary": _state_summary(target_vector),
        "result": {
            "best_edits": [_edit_payload(edit) for edit in result.best_edits],
            "best_distance": result.best_distance,
            "best_cost": result.best_cost,
            "best_objective": result.best_objective,
            "best_predicted_state": _json_state(result.best_predicted_state),
            "n_evaluations": result.n_evaluations,
            "n_predictor_calls": result.n_predictor_calls,
            "reproducibility_sha256": result.reproducibility_sha256,
            "elapsed_seconds": result.elapsed_seconds,
            "stopped_reason": result.stopped_reason,
            "iterations": [
                {
                    "iteration": item.iteration,
                    "best_distance": item.best_distance,
                    "best_cost": item.best_cost,
                    "best_objective": item.best_objective,
                    "elite_mean_distance": item.elite_mean_distance,
                    "elite_mean_objective": item.elite_mean_objective,
                    "n_candidates": item.n_candidates,
                }
                for item in result.iterations
            ],
        },
    }


def _load_fasta_sequence(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError("failed to read FASTA", details={"path": str(path)}) from exc
    sequence = "".join(line.strip() for line in lines if line.strip() and not line.startswith(">"))
    if not sequence:
        raise InputError("FASTA contains no sequence", details={"path": str(path)})
    return canonicalize_dna(sequence)


def _load_numeric_state(path: Path) -> tuple[float, ...]:
    if path.suffix == ".npy":
        try:
            import numpy as np  # type: ignore[import-not-found,unused-ignore]
        except ImportError as exc:
            raise RuntimeSetupError(
                ".npy target states require numpy",
                remediation="install geno-lewm[train] or provide a JSON state file",
            ) from exc
        payload = np.load(path, allow_pickle=False).tolist()
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputError("failed to read numeric state", details={"path": str(path)}) from exc
        if isinstance(payload, dict) and "state" in payload:
            payload = payload["state"]
    return _numeric_vector(payload, name=str(path))


def _sequence_state(sequence: str) -> tuple[float, ...]:
    values: list[float] = []
    for base in sequence:
        try:
            values.extend(_BASE_TO_STATE[base])
        except KeyError as exc:  # pragma: no cover - canonicalize_dna protects this.
            raise InputError("unsupported base in sequence-proxy state") from exc
    return tuple(values)


def _final_state_vector(value: object, target_vector: Sequence[float]) -> tuple[float, ...]:
    payload = _materialize_numeric(value)
    selected = _select_final_state(payload)
    vector = _numeric_vector(selected, name="predicted state")
    if (
        target_vector
        and len(vector) % len(target_vector) == 0
        and len(vector) != len(target_vector)
    ):
        return vector[-len(target_vector) :]
    return vector


def _select_final_state(value: object) -> object:
    current = value
    while isinstance(current, Sequence) and not isinstance(current, str | bytes):
        if not current:
            raise InputError("state payload must not be empty")
        items = cast(Sequence[object], current)
        if all(_is_number(item) for item in items):
            return current
        if len(items) == 1:
            current = items[0]
        else:
            current = items[-1]
    return current


def _materialize_numeric(value: object) -> object:
    out = value
    for attr in ("detach", "cpu"):
        method = getattr(out, attr, None)
        if callable(method):
            out = method()
    tolist = getattr(out, "tolist", None)
    if callable(tolist):
        out = tolist()
    return out


def _numeric_vector(value: object, *, name: str) -> tuple[float, ...]:
    values: list[float] = []
    _collect_numbers(value, values, name=name)
    if not values:
        raise InputError(f"{name} must contain at least one numeric value")
    return tuple(values)


def _collect_numbers(value: object, values: list[float], *, name: str) -> None:
    item = getattr(value, "item", None)
    if callable(item):
        _collect_numbers(item(), values, name=name)
        return
    if isinstance(value, bool):
        raise InputError(f"{name} must contain numeric values, not bools")
    if isinstance(value, int | float):
        number = float(value)
        if not math.isfinite(number):
            raise InputError(f"{name} must contain finite numeric values")
        values.append(number)
        return
    if isinstance(value, str | bytes):
        raise InputError(f"{name} must contain numeric values")
    if isinstance(value, Sequence):
        for item_value in cast(Sequence[object], value):
            _collect_numbers(item_value, values, name=name)
        return
    raise InputError(
        f"{name} must be a numeric scalar or sequence",
        details={"type": type(value).__name__},
    )


def _distance_function(name: str) -> Any:
    if name == "l2":
        return l2_distance
    if name == "cosine":
        return cosine_distance
    raise InputError("unsupported distance", details={"distance": name})


def _cost_function(name: str) -> Any:
    if name == "count":
        return count_cost
    if name == "bp":
        return bp_cost
    if name == "weighted_type":
        return weighted_type_cost
    raise InputError("unsupported cost", details={"cost": name})


def _parse_edit_type_weights(raw: str) -> tuple[EditTypeWeight, ...]:
    names = [name.strip().upper() for name in raw.split(",") if name.strip()]
    if not names:
        raise InputError("--edit-types must include at least one edit type")
    entries: list[EditTypeWeight] = []
    seen: set[EditType] = set()
    for name in names:
        try:
            edit_type = EditType[name]
        except KeyError as exc:
            raise InputError(
                "unsupported edit type in --edit-types",
                details={"edit_type": name, "supported": ["SNV", "INS", "DEL", "MNV", "INDEL"]},
            ) from exc
        if edit_type is EditType.SV:
            raise InputError("SV edits are outside the v1 planner")
        if edit_type in seen:
            raise InputError("duplicate edit type in --edit-types", details={"edit_type": name})
        seen.add(edit_type)
        entries.append(EditTypeWeight(edit_type, 1.0))
    return tuple(entries)


def _normalize_choice(name: str, value: str, allowed: Iterable[str]) -> str:
    normalized = value.lower()
    allowed_values = tuple(allowed)
    if normalized not in allowed_values:
        raise InputError(
            f"--{name} is unsupported",
            details={name: value, "supported": list(allowed_values)},
        )
    return normalized


def _path_identity(path: Path) -> dict[str, object]:
    return {
        "path": _public_path_label(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _public_path_label(path: Path) -> str:
    return path.name if path.is_absolute() else path.as_posix()


def _runtime_model_id(model_dir: Path) -> str | None:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, InputError, SchemaCompatError):
        return None
    return manifest.model_id()


def _edit_payload(edit: RelEdit) -> dict[str, object]:
    return {
        "rel_pos": edit.rel_pos,
        "edit_type": edit.edit_type.name,
        "edit_type_id": int(edit.edit_type),
        "ref_bases": edit.ref_bases,
        "alt_bases": edit.alt_bases,
    }


def _state_summary(vector: Sequence[float]) -> dict[str, object]:
    return {
        "length": len(vector),
        "l2_norm": math.sqrt(sum(value * value for value in vector)),
    }


def _json_state(value: object) -> list[float] | None:
    if value is None:
        return None
    return list(_final_state_vector(value, ()))


def _required_path(name: str, value: Path | None) -> Path:
    if value is None:
        raise InputError(f"requires --{name}")
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def cli_main() -> int:
    return run_app(app)
