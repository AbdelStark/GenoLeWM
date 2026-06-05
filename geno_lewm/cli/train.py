# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-train`` — train or smoke-test the predictor path."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from geno_lewm import __version__
from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.config import (
    GenoLeWMConfig,
    config_to_dict,
    load_config,
    load_default,
    write_resolved_config,
)
from geno_lewm.errors import InputError
from geno_lewm.training import run_carbon_training, run_fixture_training
from geno_lewm.training.preflight import (
    MIN_CUDA_VRAM_GB,
    REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME,
    TrainingPreflightReport,
    TrainingPreflightRequest,
    write_training_preflight_report,
)

__all__ = ["app", "cli_main"]

app = typer.Typer(
    name="geno-lewm-train",
    help=(
        "Train the predictor path. Supports fixture smoke, Carbon preflight, "
        "and explicit Carbon-backed training launches."
    ),
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()
_EFFECTIVE_TRAINING_CONFIG_NAME = "training_config.effective.yaml"


@app.callback(invoke_without_command=True)
def main(
    fixture_smoke: Annotated[
        bool,
        typer.Option(
            "--fixture-smoke",
            help="Run deterministic fixture-tier training; not Carbon-backed model training.",
        ),
    ] = False,
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Directory where training artifacts are written."),
    ] = None,
    steps: Annotated[
        int | None,
        typer.Option(
            "--steps",
            help=(
                "Override total training steps. Defaults to 50 for fixture smoke and "
                "training.max_steps for Carbon training."
            ),
        ),
    ] = None,
    resume_from: Annotated[
        Path | None,
        typer.Option(
            "--resume-from",
            help="Resume fixture smoke or Carbon training from a compatible checkpoint.",
        ),
    ] = None,
    carbon_preflight: Annotated[
        bool,
        typer.Option(
            "--carbon-preflight",
            help="Write real Carbon-backed training readiness evidence without training.",
        ),
    ] = False,
    carbon_train: Annotated[
        bool,
        typer.Option(
            "--carbon-train",
            help="Run the single-process Carbon-backed trainer after preflight succeeds.",
        ),
    ] = False,
    dataset_dir: Annotated[
        Path | None,
        typer.Option("--dataset-dir", help="Packaged dataset directory for Carbon training."),
    ] = None,
    carbon_model_dir: Annotated[
        Path | None,
        typer.Option("--carbon-model-dir", help="Local Carbon Transformers model directory."),
    ] = None,
    training_config: Annotated[
        Path | None,
        typer.Option("--training-config", help="Committed YAML config for the real training run."),
    ] = None,
    preflight_output: Annotated[
        Path | None,
        typer.Option("--preflight-output", help="Output path for training_preflight_report.json."),
    ] = None,
    allow_fixture_dataset: Annotated[
        bool,
        typer.Option(
            "--allow-fixture-dataset", help="Allow fixture datasets for local tests only."
        ),
    ] = False,
    require_native_runtime: Annotated[
        bool,
        typer.Option("--require-native-runtime/--no-require-native-runtime"),
    ] = True,
    require_accelerator: Annotated[
        bool,
        typer.Option(
            "--require-accelerator/--no-require-accelerator",
            help="Require CUDA accelerator readiness in Carbon training preflight.",
        ),
    ] = True,
    min_cuda_vram_gb: Annotated[
        float,
        typer.Option(
            "--min-cuda-vram-gb",
            help="Minimum CUDA device memory required by Carbon training preflight.",
        ),
    ] = MIN_CUDA_VRAM_GB,
    package_release_run: Annotated[
        bool,
        typer.Option(
            "--package-release-run",
            help=(
                "After a successful --carbon-train run, build training-run release "
                "manifest/card/checksums in the run directory."
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
        default_config_name="train",
    )
    if opts is None:
        return
    selected_modes = sum(bool(value) for value in (fixture_smoke, carbon_preflight, carbon_train))
    if selected_modes > 1:
        raise InputError("choose exactly one training mode")
    if package_release_run and not carbon_train:
        raise InputError("--package-release-run requires --carbon-train")
    if resume_from is not None and carbon_preflight and not carbon_train:
        raise InputError("--resume-from requires --fixture-smoke or --carbon-train")
    if carbon_preflight or carbon_train:
        if run_dir is None:
            mode = "--carbon-train" if carbon_train else "--carbon-preflight"
            raise InputError(f"geno-lewm-train {mode} requires --run-dir")
        if dataset_dir is None:
            mode = "--carbon-train" if carbon_train else "--carbon-preflight"
            raise InputError(f"geno-lewm-train {mode} requires --dataset-dir")
        if carbon_model_dir is None:
            mode = "--carbon-train" if carbon_train else "--carbon-preflight"
            raise InputError(f"geno-lewm-train {mode} requires --carbon-model-dir")
        if training_config is None:
            mode = "--carbon-train" if carbon_train else "--carbon-preflight"
            raise InputError(f"geno-lewm-train {mode} requires --training-config")
        resolved = _resolve_training_config(
            config_path=str(training_config),
            set_overrides=opts.set_overrides,
            seed=opts.seed,
            deterministic=opts.deterministic,
            run_id=opts.run_id,
        )
        carbon_steps = _resolve_carbon_steps(resolved, steps)
        effective_training_config = write_resolved_config(
            resolved,
            run_dir / _EFFECTIVE_TRAINING_CONFIG_NAME,
        )
        preflight_report = write_training_preflight_report(
            TrainingPreflightRequest(
                dataset_dir=dataset_dir,
                carbon_model_dir=carbon_model_dir,
                training_config=effective_training_config,
                run_dir=run_dir,
                allow_fixture_dataset=allow_fixture_dataset,
                require_native_runtime=require_native_runtime,
                require_accelerator=require_accelerator,
                min_cuda_vram_gb=min_cuda_vram_gb,
            ),
            preflight_output,
        )
        if not preflight_report.ok:
            typer.echo(json.dumps(preflight_report.to_dict(), sort_keys=True))
            raise InputError(
                "Carbon training preflight failed",
                details={"issues": len(preflight_report.issues)},
                remediation="fix training_preflight_report.json before launching --carbon-train",
            )
        if not carbon_train:
            typer.echo(json.dumps(preflight_report.to_dict(), sort_keys=True))
            return
        _write_preflight_copy_in_run_dir(preflight_report, run_dir)
        carbon_report = run_carbon_training(
            config=resolved,
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_model_dir,
            run_dir=run_dir,
            steps=carbon_steps,
            command=_carbon_train_command_string(
                run_dir=run_dir,
                dataset_dir=dataset_dir,
                carbon_model_dir=carbon_model_dir,
                training_config=training_config,
                steps=carbon_steps,
                steps_override=steps is not None,
                set_overrides=opts.set_overrides,
                seed=opts.seed,
                deterministic=opts.deterministic,
                run_id=opts.run_id,
                resume_from=resume_from,
                allow_fixture_dataset=allow_fixture_dataset,
                require_native_runtime=require_native_runtime,
                require_accelerator=require_accelerator,
                min_cuda_vram_gb=min_cuda_vram_gb,
                package_release_run=package_release_run,
            ),
            commit_sha=_current_commit_sha(Path.cwd()),
            package_version=__version__,
            preflight_report=preflight_report,
            resume_from=resume_from,
        )
        payload = carbon_report.to_dict()
        if package_release_run:
            package_report = _build_release_training_run_package(
                run_dir,
                carbon_report.training_metadata_path,
            )
            payload["training_run_package"] = package_report.to_dict()
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    if not fixture_smoke:
        raise InputError(
            "geno-lewm-train currently requires --fixture-smoke, --carbon-preflight, or --carbon-train",
            remediation=(
                "use --carbon-preflight to validate real training inputs before launching "
                "--carbon-train"
            ),
        )
    if run_dir is None:
        raise InputError("geno-lewm-train --fixture-smoke requires --run-dir")

    fixture_steps = _resolve_fixture_steps(steps)
    resolved = _resolve_training_config(
        config_path=config,
        set_overrides=opts.set_overrides,
        seed=opts.seed,
        deterministic=opts.deterministic,
        run_id=opts.run_id,
    )
    fixture_report = run_fixture_training(
        config=resolved,
        run_dir=run_dir,
        steps=fixture_steps,
        resume_from=resume_from,
        command=_command_string(
            run_dir=run_dir,
            steps=fixture_steps,
            resume_from=resume_from,
            config_path=config,
            set_overrides=opts.set_overrides,
            seed=opts.seed,
            deterministic=opts.deterministic,
            run_id=opts.run_id,
        ),
        commit_sha=_current_commit_sha(Path.cwd()),
        package_version=__version__,
    )
    typer.echo(json.dumps(fixture_report.to_dict(), sort_keys=True))


def cli_main() -> int:
    return run_app(app)


def _resolve_training_config(
    *,
    config_path: str | None,
    set_overrides: tuple[str, ...],
    seed: int | None,
    deterministic: bool,
    run_id: str | None,
) -> GenoLeWMConfig:
    cfg = load_config(Path(config_path)) if config_path is not None else load_default("train")
    payload = config_to_dict(cfg)
    for raw in set_overrides:
        _apply_set_override(payload, raw)
    if seed is not None:
        payload["seed"] = seed
    if deterministic:
        payload["deterministic"] = True
    if run_id is not None:
        payload["run_id"] = run_id
    resolved = load_config(payload)
    if deterministic and not resolved.deterministic:
        resolved = dataclasses.replace(resolved, deterministic=True)
    return resolved


def _resolve_fixture_steps(steps: int | None) -> int:
    return 50 if steps is None else steps


def _resolve_carbon_steps(config: GenoLeWMConfig, steps_override: int | None) -> int:
    steps = config.training.max_steps if steps_override is None else steps_override
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise InputError(
            "Carbon training steps must be a positive integer",
            details={
                "source": "training.max_steps" if steps_override is None else "--steps",
                "value": steps,
            },
        )
    if config.optimizer.schedule == "wsd" and steps <= config.optimizer.warmup_steps:
        raise InputError(
            "Carbon training steps must exceed optimizer.warmup_steps for the WSD schedule",
            details={
                "source": "training.max_steps" if steps_override is None else "--steps",
                "steps": steps,
                "warmup_steps": config.optimizer.warmup_steps,
            },
            remediation=(
                "increase training.max_steps in the committed config or pass an explicit "
                "--steps override greater than optimizer.warmup_steps"
            ),
        )
    return steps


def _apply_set_override(payload: dict[str, Any], raw: str) -> None:
    if "=" not in raw:
        raise InputError("--set override must have the form key=value", details={"override": raw})
    key, value_text = raw.split("=", maxsplit=1)
    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise InputError("--set override key must be a non-empty dotted path", details={"key": key})
    try:
        value = yaml.safe_load(value_text)
    except yaml.YAMLError as exc:
        raise InputError(
            "--set override value is not valid YAML",
            details={"override": raw, "error": str(exc)},
        ) from exc
    target: dict[str, Any] = payload
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise InputError("--set override path does not resolve to a config block")
        target = child
    target[parts[-1]] = value


def _write_preflight_copy_in_run_dir(
    report: TrainingPreflightReport,
    run_dir: Path,
) -> Path:
    """Keep Carbon train metadata packageable even with a custom preflight output path."""
    path = run_dir / TRAINING_PREFLIGHT_REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_release_training_run_package(run_dir: Path, metadata_path: Path) -> Any:
    from tools.release.training_run import build_training_run_package

    return build_training_run_package(run_dir, metadata_path)


def _command_string(
    *,
    run_dir: Path,
    steps: int,
    resume_from: Path | None,
    config_path: str | None,
    set_overrides: tuple[str, ...],
    seed: int | None,
    deterministic: bool,
    run_id: str | None,
) -> str:
    parts = ["geno-lewm-train", "--fixture-smoke", "--run-dir", str(run_dir), "--steps", str(steps)]
    if resume_from is not None:
        parts.extend(["--resume-from", str(resume_from)])
    if config_path is not None:
        parts.extend(["--config", config_path])
    for override in set_overrides:
        parts.extend(["--set", override])
    if seed is not None:
        parts.extend(["--seed", str(seed)])
    if deterministic:
        parts.append("--deterministic")
    if run_id is not None:
        parts.extend(["--run-id", run_id])
    return " ".join(parts)


def _carbon_train_command_string(
    *,
    run_dir: Path,
    dataset_dir: Path,
    carbon_model_dir: Path,
    training_config: Path,
    steps: int,
    steps_override: bool,
    set_overrides: tuple[str, ...],
    seed: int | None,
    deterministic: bool,
    run_id: str | None,
    resume_from: Path | None,
    allow_fixture_dataset: bool,
    require_native_runtime: bool,
    require_accelerator: bool,
    min_cuda_vram_gb: float,
    package_release_run: bool,
) -> str:
    parts = [
        "geno-lewm-train",
        "--carbon-train",
        "--run-dir",
        str(run_dir),
        "--dataset-dir",
        str(dataset_dir),
        "--carbon-model-dir",
        str(carbon_model_dir),
        "--training-config",
        str(training_config),
    ]
    if steps_override:
        parts.extend(["--steps", str(steps)])
    for override in set_overrides:
        parts.extend(["--set", override])
    if seed is not None:
        parts.extend(["--seed", str(seed)])
    if deterministic:
        parts.append("--deterministic")
    if run_id is not None:
        parts.extend(["--run-id", run_id])
    if resume_from is not None:
        parts.extend(["--resume-from", str(resume_from)])
    if allow_fixture_dataset:
        parts.append("--allow-fixture-dataset")
    if not require_native_runtime:
        parts.append("--no-require-native-runtime")
    if not require_accelerator:
        parts.append("--no-require-accelerator")
    if min_cuda_vram_gb != MIN_CUDA_VRAM_GB:
        parts.extend(["--min-cuda-vram-gb", str(min_cuda_vram_gb)])
    if package_release_run:
        parts.append("--package-release-run")
    return " ".join(parts)


def _current_commit_sha(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        candidate = result.stdout.strip().lower()
        if candidate:
            return candidate
    return "0000000"
