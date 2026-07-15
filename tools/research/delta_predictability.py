# SPDX-License-Identifier: Apache-2.0
"""Is the edit response *predictable* from the pooled state and the action (study R3)?

This is the world-model / efficiency thesis stated correctly. The original
project trained ``g(s_ref, action) -> s_alt``, which is degenerate: a point edit
barely moves the pooled state (``cos(s_ref, s_alt) ~ 0.999``), so "predict no
change" already wins and the task measures nothing. Removing that copy leaves the
displacement

    Delta = s_alt - s_ref

and the honest question: can ``Delta`` be predicted from the *reference* state
plus the action alone, **without encoding the edited window**? If it can, the
edit response is cheap. If it cannot, there is no shortcut through the pooled
latent and the encoder must actually be run on the edited sequence.

Consuming :mod:`tools.research.edit_response_spectroscopy` embeddings (the same
Parquet :mod:`tools.research.edit_response_analysis` reads), this tool restricts
to SNVs at one pooling radius -- where every variant sits at the window centre,
so the action reduces to the 12 ``REF>ALT`` substitution classes -- and compares,
under K-fold cross-validation:

``global_mean``
    The mean training ``Delta``. Knows nothing about the variant; the floor.
``class_mean``
    The mean ``Delta`` of the variant's substitution class: a lookup table with
    no model in it at all. **This is the baseline that matters.** If a learned
    predictor cannot beat it, the "world model" is a lookup table and there is
    nothing being learned.
``ridge_action_only``
    Ridge on the one-hot action alone. One-hot ridge *is* the class means, so
    this must reproduce ``class_mean``; it is the harness's self-check.
``ridge_sref_action``
    Ridge on ``[s_ref, action]``, **swept over lambda**. An under-regularised
    failure would prove nothing, so the sweep is mandatory rather than optional
    and the reported number is the best lambda's.
``mlp_mse``
    A nonlinear predictor, giving the thesis its best shot.

Each predictor is scored by ``cos(Dhat, Delta)``, mean relative error, and the
AUROC of ``||Dhat||`` for ClinVar pathogenic-vs-benign, against the AUROC of the
*true* ``Delta`` as the ceiling any predictor is chasing.

Two method points are load-bearing:

1. **A cosine-trained MLP's magnitude is meaningless.** Cosine loss is
   scale-invariant, so it leaves ``||Dhat||`` unconstrained; an AUROC of that
   magnitude would be an artefact of the loss, not a finding. The tool therefore
   refuses to emit one (``mlp_objective="cosine"`` reports ``cos`` only). The
   MSE-trained model, whose objective constrains magnitude, is the only fair
   magnitude test.
2. **Ridge sweeps lambda**, as above.

``fit_intercept`` records a deviation from the validated R3 prototype. The
prototype standardised a constant column, whose zero variance turned it into a
column of zeros, so it fit *no* intercept and could not represent ``E[Delta]``,
while its action-only ridge (left unstandardised) kept one -- making the two
ridges non-comparable. The default here fits the intercept properly; setting
``fit_intercept=False`` reproduces the prototype exactly. The choice moves the
real-data number only slightly and changes no conclusion.

NumPy, PyArrow and PyTorch are imported lazily, mirroring
:mod:`tools.research.edit_response_analysis`, so the module imports and
type-checks without the ``train`` / ``eval`` extras installed.

Usage::

    python -m tools.research.delta_predictability \
        --embeddings emb.parquet --radius 8 --out-report r3.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.evaluation import _auroc as _mann_whitney_auroc
from tools.research.edit_response_analysis import (
    VariantGeometry,
    _stack_vectors as _stack_state_vectors,
    load_embeddings,
)

__all__ = [
    "GENERATED_BY",
    "MLP_OBJECTIVES",
    "SCHEMA_VERSION",
    "PredictabilityConfig",
    "evaluate_predictability",
    "main",
    "run_predictability",
]

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.delta_predictability"

_CLINVAR_PATH: Final = "clinvar_path"
_CLINVAR_BENIGN: Final = "clinvar_benign"

#: Training objectives available to the MLP. ``cosine`` optimises direction only.
MLP_OBJECTIVES: Final[tuple[str, ...]] = ("mse", "cosine")

_COSINE_MAGNITUDE_NOTE: Final = (
    "cosine loss is scale-invariant, so it leaves ||Dhat|| unconstrained; an AUROC of "
    "this model's magnitude would measure the loss, not the edit response. Use the "
    "MSE-trained model for any magnitude claim."
)


@dataclass(frozen=True, slots=True)
class PredictabilityConfig:
    """Deterministic knobs for the delta-predictability study.

    Every random draw (fold assignment, weight initialisation, batch shuffling,
    dropout) is seeded so the report is reproducible.
    """

    radius: int = 8
    kfold_splits: int = 5
    fold_seed: int = 0
    ridge_lambdas: tuple[float, ...] = (1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
    ridge_action_only_lambda: float = 1.0
    fit_intercept: bool = True
    min_class_count: int = 5
    min_clinvar_total: int = 20
    enable_mlp: bool = True
    mlp_objective: str = "mse"
    mlp_hidden: int = 512
    mlp_epochs: int = 60
    mlp_batch_size: int = 256
    mlp_lr: float = 1e-3
    mlp_weight_decay: float = 1e-2
    mlp_dropout: float = 0.1
    mlp_seed: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int("radius", self.radius)
        _require_positive_int("kfold_splits", self.kfold_splits)
        _require_positive_int("mlp_hidden", self.mlp_hidden)
        _require_positive_int("mlp_epochs", self.mlp_epochs)
        _require_positive_int("mlp_batch_size", self.mlp_batch_size)
        _require_non_negative_int("min_class_count", self.min_class_count)
        _require_non_negative_int("min_clinvar_total", self.min_clinvar_total)
        if self.kfold_splits < 2:
            raise InputError(
                "kfold_splits must be at least 2 for cross-validation",
                details={"kfold_splits": self.kfold_splits},
            )
        if not self.ridge_lambdas:
            raise InputError("ridge_lambdas must not be empty; lambda must be swept")
        if any(value <= 0.0 for value in self.ridge_lambdas):
            raise InputError(
                "ridge_lambdas must all be positive",
                details={"ridge_lambdas": list(self.ridge_lambdas)},
            )
        if self.ridge_action_only_lambda <= 0.0:
            raise InputError(
                "ridge_action_only_lambda must be positive",
                details={"ridge_action_only_lambda": self.ridge_action_only_lambda},
            )
        if self.mlp_objective not in MLP_OBJECTIVES:
            raise InputError(
                "mlp_objective must be one of the supported objectives",
                details={"mlp_objective": self.mlp_objective, "supported": list(MLP_OBJECTIVES)},
            )
        if not 0.0 <= self.mlp_dropout < 1.0:
            raise InputError(
                "mlp_dropout must lie in [0, 1)", details={"mlp_dropout": self.mlp_dropout}
            )


# ---------------------------------------------------------------------------
# Lazy optional dependencies.


def _numpy() -> Any:
    return cast(Any, importlib.import_module("numpy"))


def _torch() -> Any:
    """Import PyTorch, which is an optional extra, failing with a typed error."""
    try:
        return cast(Any, importlib.import_module("torch"))
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise InputError(
            "the MLP predictor requires PyTorch; install the train extra or pass --no-mlp",
            details={"error": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Metric helpers.


def _cos_rows(predicted: Any, actual: Any) -> Any:
    """Row-wise cosine between predicted and actual displacements."""
    np = _numpy()
    norms = np.linalg.norm(predicted, axis=1) * np.linalg.norm(actual, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (predicted * actual).sum(axis=1) / norms


def _relative_error(predicted: Any, actual: Any) -> float:
    np = _numpy()
    denominator = np.maximum(np.linalg.norm(actual, axis=1), 1e-9)
    return float((np.linalg.norm(predicted - actual, axis=1) / denominator).mean())


def _magnitude_auroc(predicted: Any, clinvar: _ClinVarLabels | None) -> float | None:
    """AUROC of ``||Dhat||`` for path-vs-benign, or ``None`` when unlabelled."""
    if clinvar is None:
        return None
    np = _numpy()
    magnitudes = np.linalg.norm(predicted[clinvar.mask], axis=1)
    return _mann_whitney_auroc(
        clinvar.labels,
        [float(value) for value in magnitudes.tolist()],
    )


@dataclass(frozen=True, slots=True)
class _ClinVarLabels:
    """The ClinVar path/benign subset of the analysed rows."""

    mask: Any
    labels: list[bool]
    n_path: int
    n_benign: int


def _clinvar_labels(
    rows: Sequence[VariantGeometry], config: PredictabilityConfig
) -> _ClinVarLabels | None:
    np = _numpy()
    groups = np.asarray([row.label_group or "" for row in rows])
    is_path = groups == _CLINVAR_PATH
    is_benign = groups == _CLINVAR_BENIGN
    mask = is_path | is_benign
    n_path = int(is_path.sum())
    n_benign = int(is_benign.sum())
    if (
        int(mask.sum()) < config.min_clinvar_total
        or n_path < config.min_class_count
        or n_benign < config.min_class_count
    ):
        return None
    return _ClinVarLabels(
        mask=mask,
        labels=[bool(value) for value in is_path[mask].tolist()],
        n_path=n_path,
        n_benign=n_benign,
    )


def _score(
    predicted: Any,
    actual: Any,
    clinvar: _ClinVarLabels | None,
) -> dict[str, Any]:
    return {
        "cos": float(_cos_rows(predicted, actual).mean()),
        "rel_err": _relative_error(predicted, actual),
        "auroc_magnitude": _magnitude_auroc(predicted, clinvar),
    }


# ---------------------------------------------------------------------------
# Feature construction and folds.


def _one_hot_actions(rows: Sequence[VariantGeometry]) -> tuple[Any, Any, list[str]]:
    """One-hot encode the ``REF>ALT`` substitution class of every SNV row."""
    np = _numpy()
    substitutions = np.asarray([row.substitution for row in rows])
    classes = sorted({str(value) for value in substitutions.tolist()})
    index_of = {name: index for index, name in enumerate(classes)}
    actions = np.zeros((len(rows), len(classes)), dtype=np.float64)
    actions[np.arange(len(rows)), [index_of[str(value)] for value in substitutions.tolist()]] = 1.0
    return actions, substitutions, classes


def _fold_assignments(n_rows: int, config: PredictabilityConfig) -> Any:
    np = _numpy()
    rng = np.random.default_rng(config.fold_seed)
    return rng.integers(0, config.kfold_splits, n_rows)


# ---------------------------------------------------------------------------
# Predictors. Each returns the cross-validated held-out prediction matrix.


def _cv_global_mean(delta: Any, folds: Any, config: PredictabilityConfig) -> Any:
    np = _numpy()
    predicted = np.zeros_like(delta)
    for fold in range(config.kfold_splits):
        test = folds == fold
        train = ~test
        if not bool(test.any()) or not bool(train.any()):
            continue
        predicted[test] = delta[train].mean(axis=0)
    return predicted


def _cv_class_mean(
    delta: Any,
    substitutions: Any,
    folds: Any,
    config: PredictabilityConfig,
) -> Any:
    """Per-substitution-class mean Delta, falling back to the global mean.

    A class with too few training rows would give a mean dominated by noise, so
    it defers to the global mean rather than pretending to know something.
    """
    np = _numpy()
    predicted = np.zeros_like(delta)
    for fold in range(config.kfold_splits):
        test = folds == fold
        train = ~test
        if not bool(test.any()) or not bool(train.any()):
            continue
        fallback = delta[train].mean(axis=0)
        for name in np.unique(substitutions):
            in_test = test & (substitutions == name)
            if not bool(in_test.any()):
                continue
            in_train = train & (substitutions == name)
            predicted[in_test] = (
                delta[in_train].mean(axis=0)
                if int(in_train.sum()) >= config.min_class_count
                else fallback
            )
    return predicted


def _solve_ridge(features: Any, targets: Any, lam: float) -> Any:
    np = _numpy()
    gram = features.T @ features + lam * np.eye(features.shape[1])
    return np.linalg.solve(gram, features.T @ targets)


def _cv_ridge(
    features: Any,
    delta: Any,
    folds: Any,
    config: PredictabilityConfig,
    *,
    lam: float,
    standardize: bool,
    fit_intercept: bool,
) -> Any:
    """Cross-validated ridge, standardising and centring on the training fold only.

    ``fit_intercept`` adds back the training-fold mean of ``Delta``. Without it a
    model on centred features cannot represent ``E[Delta]`` at all.
    """
    np = _numpy()
    predicted = np.zeros_like(delta)
    for fold in range(config.kfold_splits):
        test = folds == fold
        train = ~test
        if not bool(test.any()) or not bool(train.any()):
            continue
        train_features = features[train]
        test_features = features[test]
        if standardize:
            mean = train_features.mean(axis=0)
            std = train_features.std(axis=0)
            std = np.where(std == 0.0, 1.0, std)
            train_features = (train_features - mean) / std
            test_features = (test_features - mean) / std
        offset = delta[train].mean(axis=0) if fit_intercept else np.zeros(delta.shape[1])
        weights = _solve_ridge(train_features, delta[train] - offset, lam)
        predicted[test] = test_features @ weights + offset
    return predicted


def _ridge_sref_action(
    s_ref: Any,
    actions: Any,
    delta: Any,
    folds: Any,
    clinvar: _ClinVarLabels | None,
    config: PredictabilityConfig,
) -> dict[str, Any]:
    """Ridge on ``[s_ref, action]`` with a mandatory lambda sweep."""
    np = _numpy()
    features = np.hstack([s_ref, actions])
    # ``ridge_lambdas`` is validated non-empty, so the sweep always has a winner.
    scored_by_lambda = [
        (
            lam,
            _score(
                _cv_ridge(
                    features,
                    delta,
                    folds,
                    config,
                    lam=lam,
                    standardize=True,
                    fit_intercept=config.fit_intercept,
                ),
                delta,
                clinvar,
            ),
        )
        for lam in config.ridge_lambdas
    ]
    best_lambda, best = max(scored_by_lambda, key=lambda item: item[1]["cos"])
    return {
        **best,
        "best_lambda": float(best_lambda),
        "fit_intercept": config.fit_intercept,
        "lambda_sweep": [
            {"lambda": float(lam), "cos": scored["cos"], "rel_err": scored["rel_err"]}
            for lam, scored in scored_by_lambda
        ],
    }


# ---------------------------------------------------------------------------
# The nonlinear predictor: the thesis's best shot.


def _build_mlp(torch: Any, n_features: int, n_outputs: int, config: PredictabilityConfig) -> Any:
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(n_features, config.mlp_hidden),
        nn.GELU(),
        nn.Dropout(config.mlp_dropout),
        nn.Linear(config.mlp_hidden, config.mlp_hidden),
        nn.GELU(),
        nn.Linear(config.mlp_hidden, n_outputs),
    )


def _mlp_loss(torch: Any, predicted: Any, target: Any, objective: str) -> Any:
    if objective == "cosine":
        return (1.0 - torch.nn.functional.cosine_similarity(predicted, target, dim=1)).mean()
    return torch.nn.functional.mse_loss(predicted, target)


def _cv_mlp(
    s_ref: Any,
    actions: Any,
    delta: Any,
    folds: Any,
    config: PredictabilityConfig,
) -> Any:
    """Cross-validated MLP on ``[standardised s_ref, action] -> Delta``.

    The one-hot action is left unscaled; only ``s_ref`` is standardised, and only
    against the training fold.
    """
    np = _numpy()
    torch = _torch()
    torch.manual_seed(config.mlp_seed)
    state = s_ref.astype(np.float32)
    action = actions.astype(np.float32)
    target = delta.astype(np.float32)
    predicted = np.zeros_like(target)

    for fold in range(config.kfold_splits):
        test = folds == fold
        train = ~test
        if not bool(test.any()) or not bool(train.any()):
            continue
        mean = state[train].mean(axis=0)
        std = state[train].std(axis=0) + 1e-6
        train_x = torch.tensor(np.hstack([(state[train] - mean) / std, action[train]]))
        train_y = torch.tensor(target[train])
        test_x = torch.tensor(np.hstack([(state[test] - mean) / std, action[test]]))

        net = _build_mlp(torch, int(train_x.shape[1]), int(target.shape[1]), config)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=config.mlp_lr, weight_decay=config.mlp_weight_decay
        )
        net.train()
        for _epoch in range(config.mlp_epochs):
            order = torch.randperm(len(train_x))
            for start in range(0, len(train_x), config.mlp_batch_size):
                batch = order[start : start + config.mlp_batch_size]
                optimizer.zero_grad()
                loss = _mlp_loss(torch, net(train_x[batch]), train_y[batch], config.mlp_objective)
                loss.backward()
                optimizer.step()
        net.eval()
        with torch.no_grad():
            predicted[test] = net(test_x).numpy()
    return predicted.astype(np.float64)


def _mlp_block(
    s_ref: Any,
    actions: Any,
    delta: Any,
    folds: Any,
    clinvar: _ClinVarLabels | None,
    config: PredictabilityConfig,
) -> dict[str, dict[str, Any]]:
    """Score the MLP, withholding the magnitude AUROC for a cosine objective."""
    if not config.enable_mlp:
        return {"mlp": {"skipped": "MLP disabled"}}
    predicted = _cv_mlp(s_ref, actions, delta, folds, config)
    scored = _score(predicted, delta, clinvar)
    scored["objective"] = config.mlp_objective
    if config.mlp_objective == "cosine":
        # Refuse to publish a magnitude the loss never constrained.
        scored["auroc_magnitude"] = None
        scored["auroc_magnitude_omitted_because"] = _COSINE_MAGNITUDE_NOTE
    return {f"mlp_{config.mlp_objective}": scored}


# ---------------------------------------------------------------------------
# Top-level analysis.


def evaluate_predictability(
    rows: Sequence[VariantGeometry],
    *,
    config: PredictabilityConfig | None = None,
) -> dict[str, Any]:
    """Compare every Delta predictor against the baselines for one radius."""
    np = _numpy()
    cfg = config if config is not None else PredictabilityConfig()
    if not rows:
        raise InputError("evaluate_predictability requires at least one variant row")

    snv_rows = [row for row in rows if row.is_snv]
    n_excluded = len(rows) - len(snv_rows)
    if not snv_rows:
        raise InputError(
            "no SNV rows to analyse; the action encoding is defined for substitutions only",
            details={"n_rows": len(rows), "n_non_snv_excluded": n_excluded},
        )

    s_ref = _stack_state_vectors(snv_rows, "s_ref")
    delta = _stack_state_vectors(snv_rows, "s_alt") - s_ref
    actions, substitutions, classes = _one_hot_actions(snv_rows)
    folds = _fold_assignments(len(snv_rows), cfg)
    clinvar = _clinvar_labels(snv_rows, cfg)

    predictors: dict[str, Any] = {
        "global_mean": _score(_cv_global_mean(delta, folds, cfg), delta, clinvar),
        "class_mean": _score(_cv_class_mean(delta, substitutions, folds, cfg), delta, clinvar),
        "ridge_action_only": _score(
            _cv_ridge(
                np.hstack([actions, np.ones((len(snv_rows), 1))]),
                delta,
                folds,
                cfg,
                lam=cfg.ridge_action_only_lambda,
                standardize=False,
                fit_intercept=False,
            ),
            delta,
            clinvar,
        ),
        "ridge_sref_action": _ridge_sref_action(s_ref, actions, delta, folds, clinvar, cfg),
    }
    predictors.update(_mlp_block(s_ref, actions, delta, folds, clinvar, cfg))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "config": _config_payload(cfg),
        "radius": cfg.radius,
        "n": len(snv_rows),
        "n_non_snv_excluded": n_excluded,
        "state_dim": int(delta.shape[1]),
        "substitution_classes": classes,
        "clinvar": (
            {"n_path": clinvar.n_path, "n_benign": clinvar.n_benign}
            if clinvar is not None
            else {"skipped": "insufficient ClinVar path/benign SNVs"}
        ),
        "predictors": predictors,
        # The ceiling every predictor is chasing: what the true displacement scores.
        "ceiling": {"true_delta_auroc": _magnitude_auroc(delta, clinvar)},
    }


def _config_payload(config: PredictabilityConfig) -> dict[str, Any]:
    return {
        "radius": config.radius,
        "kfold_splits": config.kfold_splits,
        "fold_seed": config.fold_seed,
        "ridge_lambdas": [float(value) for value in config.ridge_lambdas],
        "ridge_action_only_lambda": config.ridge_action_only_lambda,
        "fit_intercept": config.fit_intercept,
        "min_class_count": config.min_class_count,
        "min_clinvar_total": config.min_clinvar_total,
        "enable_mlp": config.enable_mlp,
        "mlp_objective": config.mlp_objective,
        "mlp_hidden": config.mlp_hidden,
        "mlp_epochs": config.mlp_epochs,
        "mlp_batch_size": config.mlp_batch_size,
        "mlp_lr": config.mlp_lr,
        "mlp_weight_decay": config.mlp_weight_decay,
        "mlp_dropout": config.mlp_dropout,
        "mlp_seed": config.mlp_seed,
    }


def run_predictability(
    path: str | Path,
    *,
    config: PredictabilityConfig | None = None,
) -> dict[str, Any]:
    """Load the embeddings Parquet and run the R3 study at one radius."""
    cfg = config if config is not None else PredictabilityConfig()
    rows_by_radius = load_embeddings(path)
    if cfg.radius not in rows_by_radius:
        raise InputError(
            "requested pooling radius is absent from the embeddings Parquet",
            details={
                "radius": cfg.radius,
                "available": sorted(rows_by_radius),
                "path": str(Path(path)),
            },
        )
    report = evaluate_predictability(rows_by_radius[cfg.radius], config=cfg)
    report["source"] = str(Path(path))
    report["provenance"] = {"git_commit": _git_commit()}
    return report


# ---------------------------------------------------------------------------
# Validation helpers.


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{name} must be a positive integer", details={name: value})
    return value


def _require_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(f"{name} must be a non-negative integer", details={name: value})
    return value


# ---------------------------------------------------------------------------
# CLI.


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _parser() -> argparse.ArgumentParser:
    defaults = PredictabilityConfig()
    parser = argparse.ArgumentParser(
        description="Is the edit response predictable from (s_ref, action)? (R3)."
    )
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=defaults.radius)
    parser.add_argument("--kfold-splits", type=int, default=defaults.kfold_splits)
    parser.add_argument("--fold-seed", type=int, default=defaults.fold_seed)
    parser.add_argument(
        "--ridge-lambdas",
        type=float,
        nargs="+",
        default=list(defaults.ridge_lambdas),
        help="lambda sweep for ridge; an under-regularised failure proves nothing",
    )
    parser.add_argument(
        "--no-fit-intercept",
        action="store_true",
        help="reproduce the R3 prototype, which fit no intercept on standardised features",
    )
    parser.add_argument("--no-mlp", action="store_true", help="skip the PyTorch predictor")
    parser.add_argument(
        "--mlp-objective",
        choices=MLP_OBJECTIVES,
        default=defaults.mlp_objective,
        help="cosine optimises direction only; its magnitude AUROC is never reported",
    )
    parser.add_argument("--mlp-epochs", type=int, default=defaults.mlp_epochs)
    parser.add_argument("--mlp-seed", type=int, default=defaults.mlp_seed)
    return parser


def _config_from_args(args: argparse.Namespace) -> PredictabilityConfig:
    return replace(
        PredictabilityConfig(),
        radius=args.radius,
        kfold_splits=args.kfold_splits,
        fold_seed=args.fold_seed,
        ridge_lambdas=tuple(float(value) for value in args.ridge_lambdas),
        fit_intercept=not args.no_fit_intercept,
        enable_mlp=not args.no_mlp,
        mlp_objective=args.mlp_objective,
        mlp_epochs=args.mlp_epochs,
        mlp_seed=args.mlp_seed,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the R3 study and write the JSON report."""
    args = _parser().parse_args(argv)
    try:
        report = run_predictability(args.embeddings, config=_config_from_args(args))
        _write_report(args.out_report, report)
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        if exc.details:
            sys.stderr.write(json.dumps(exc.details, sort_keys=True) + "\n")
        return exit_code_for(exc)
    sys.stdout.write(
        json.dumps(
            {
                "out_report": str(args.out_report),
                "radius": report["radius"],
                "n": report["n"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
