# SPDX-License-Identifier: Apache-2.0
"""Pure-Python CEM solver core for planning contract planning.

The predictor-backed planner and CLI still need runtime integration and
benchmark evidence. This module provides the deterministic search loop
over existing ``RelEdit`` samplers so downstream integrations can bind a
real rollout/evaluation function without importing optional ML runtimes.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from geno_lewm.action import EditType, RelEdit
from geno_lewm.errors import InputError
from geno_lewm.planning.costs import count_cost
from geno_lewm.planning.sampling import ActionSampler
from geno_lewm.training import EditTypeWeight

__all__ = [
    "CEMIterationLog",
    "CandidateEvaluation",
    "PlanningConfig",
    "PlanningResult",
    "cosine_distance",
    "l2_distance",
    "plan",
    "projection_distance",
    "region_distance",
]

_ALL_V1_EDIT_TYPES: tuple[EditType, ...] = (
    EditType.SNV,
    EditType.INS,
    EditType.DEL,
    EditType.MNV,
    EditType.INDEL,
)
_DEFAULT_SMOOTHING = 0.1


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    """CEM search configuration for a fixed-horizon edit sequence."""

    horizon: int = 5
    n_iterations: int = 5
    n_samples: int = 1024
    n_elite: int = 64
    cost_weight: float = 0.0
    stopping_eps: float = 0.05
    patience: int = 2
    seed: int | None = None
    smoothing: float = _DEFAULT_SMOOTHING

    def __post_init__(self) -> None:
        _require_nonnegative_int("horizon", self.horizon)
        _require_positive_int("n_iterations", self.n_iterations)
        _require_positive_int("n_samples", self.n_samples)
        _require_positive_int("n_elite", self.n_elite)
        if self.n_elite > self.n_samples:
            raise InputError(
                "n_elite must be <= n_samples",
                details={"n_elite": self.n_elite, "n_samples": self.n_samples},
            )
        _validate_nonnegative_finite("cost_weight", self.cost_weight)
        _validate_nonnegative_finite("stopping_eps", self.stopping_eps)
        _require_positive_int("patience", self.patience)
        _validate_seed(self.seed)
        if (
            isinstance(self.smoothing, bool)
            or not isinstance(self.smoothing, int | float)
            or not math.isfinite(float(self.smoothing))
            or not 0.0 <= float(self.smoothing) <= 1.0
        ):
            raise InputError(
                "smoothing must be a finite number in [0, 1]",
                details={"value": self.smoothing, "type": type(self.smoothing).__name__},
            )
        object.__setattr__(self, "cost_weight", float(self.cost_weight))
        object.__setattr__(self, "stopping_eps", float(self.stopping_eps))
        object.__setattr__(self, "smoothing", float(self.smoothing))


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Evaluation returned by a caller-provided rollout/scoring function."""

    distance: float
    predicted_state: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "distance", _validate_nonnegative_finite("distance", self.distance)
        )


@dataclass(frozen=True, slots=True)
class CEMIterationLog:
    """Summary statistics for one CEM iteration."""

    iteration: int
    best_distance: float
    best_cost: float
    best_objective: float
    elite_mean_distance: float
    elite_mean_objective: float
    n_candidates: int

    def __post_init__(self) -> None:
        _require_positive_int("iteration", self.iteration)
        _require_positive_int("n_candidates", self.n_candidates)
        object.__setattr__(
            self, "best_distance", _validate_nonnegative_finite("best_distance", self.best_distance)
        )
        object.__setattr__(
            self, "best_cost", _validate_nonnegative_finite("best_cost", self.best_cost)
        )
        object.__setattr__(
            self,
            "best_objective",
            _validate_nonnegative_finite("best_objective", self.best_objective),
        )
        object.__setattr__(
            self,
            "elite_mean_distance",
            _validate_nonnegative_finite("elite_mean_distance", self.elite_mean_distance),
        )
        object.__setattr__(
            self,
            "elite_mean_objective",
            _validate_nonnegative_finite("elite_mean_objective", self.elite_mean_objective),
        )


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """Best edit sequence and reproducibility trace from a CEM run."""

    best_edits: tuple[RelEdit, ...]
    best_distance: float
    best_cost: float
    best_objective: float
    best_predicted_state: Any | None
    n_evaluations: int
    iterations: tuple[CEMIterationLog, ...]
    elapsed_seconds: float
    stopped_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "best_edits", tuple(self.best_edits))
        object.__setattr__(
            self, "best_distance", _validate_nonnegative_finite("best_distance", self.best_distance)
        )
        object.__setattr__(
            self, "best_cost", _validate_nonnegative_finite("best_cost", self.best_cost)
        )
        object.__setattr__(
            self,
            "best_objective",
            _validate_nonnegative_finite("best_objective", self.best_objective),
        )
        _require_nonnegative_int("n_evaluations", self.n_evaluations)
        object.__setattr__(self, "iterations", tuple(self.iterations))
        object.__setattr__(
            self,
            "elapsed_seconds",
            _validate_nonnegative_finite("elapsed_seconds", self.elapsed_seconds),
        )
        if self.stopped_reason not in {"max_iterations", "distance_threshold", "patience"}:
            raise InputError(
                "stopped_reason must be one of max_iterations, distance_threshold, patience",
                details={"stopped_reason": self.stopped_reason},
            )

    @property
    def n_predictor_calls(self) -> int:
        """Compatibility alias for the final predictor-backed API shape."""
        return self.n_evaluations


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    edits: tuple[RelEdit, ...]
    evaluation: CandidateEvaluation
    cost: float
    objective: float
    index: int


def plan(
    evaluate: Callable[[Sequence[RelEdit]], float | CandidateEvaluation],
    sampler: ActionSampler,
    *,
    config: PlanningConfig | None = None,
    cost_fn: Callable[[Sequence[RelEdit]], float] | None = None,
) -> PlanningResult:
    """Run CEM over valid ``RelEdit`` sequences from ``sampler``.

    ``evaluate`` is the integration boundary: it receives a candidate edit
    sequence and returns either a non-negative distance or a
    :class:`CandidateEvaluation` with optional predicted-state payload.
    The objective minimized by CEM is ``distance + cost_weight * cost``.
    """

    if not callable(evaluate):
        raise InputError("evaluate must be callable")
    if not isinstance(sampler, ActionSampler):
        raise InputError(
            "sampler must be an ActionSampler",
            details={"type": type(sampler).__name__},
        )
    resolved_cost_fn = count_cost if cost_fn is None else cost_fn
    if not callable(resolved_cost_fn):
        raise InputError("cost_fn must be callable")

    resolved_config = config if config is not None else PlanningConfig()
    rng = random.Random(resolved_config.seed)
    current_sampler = _clone_sampler(sampler, rng=rng)
    best: _ScoredCandidate | None = None
    logs: list[CEMIterationLog] = []
    n_evaluations = 0
    stale_iterations = 0
    stopped_reason = "max_iterations"
    started = time.perf_counter()

    for iteration in range(1, resolved_config.n_iterations + 1):
        candidates = current_sampler.sample_sequences(
            resolved_config.n_samples,
            resolved_config.horizon,
        )
        scored = tuple(
            _score_candidate(
                edits,
                index=index,
                evaluate=evaluate,
                cost_fn=resolved_cost_fn,
                cost_weight=resolved_config.cost_weight,
            )
            for index, edits in enumerate(candidates)
        )
        n_evaluations += len(scored)
        ranked = sorted(scored, key=_candidate_sort_key)
        iteration_best = ranked[0]
        elites = tuple(candidate.edits for candidate in ranked[: resolved_config.n_elite])

        if best is None or _candidate_value_key(iteration_best) < _candidate_value_key(best):
            best = iteration_best
            stale_iterations = 0
        else:
            stale_iterations += 1

        logs.append(
            _iteration_log(
                iteration,
                ranked[: resolved_config.n_elite],
                iteration_best,
                n_candidates=len(scored),
            )
        )

        if best.evaluation.distance < resolved_config.stopping_eps:
            stopped_reason = "distance_threshold"
            break
        if stale_iterations >= resolved_config.patience:
            stopped_reason = "patience"
            break

        current_sampler = _refit_sampler(
            current_sampler,
            elites=elites,
            rng=rng,
            smoothing=resolved_config.smoothing,
        )

    if best is None:  # pragma: no cover - PlanningConfig validation prevents this.
        raise InputError("CEM produced no candidates")

    return PlanningResult(
        best_edits=best.edits,
        best_distance=best.evaluation.distance,
        best_cost=best.cost,
        best_objective=best.objective,
        best_predicted_state=best.evaluation.predicted_state,
        n_evaluations=n_evaluations,
        iterations=tuple(logs),
        elapsed_seconds=time.perf_counter() - started,
        stopped_reason=stopped_reason,
    )


def l2_distance(predicted: Iterable[float], target: Iterable[float]) -> float:
    """Return Euclidean distance between two finite numeric vectors."""

    left, right = _paired_vectors(predicted, target)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def cosine_distance(predicted: Iterable[float], target: Iterable[float]) -> float:
    """Return ``1 - cosine_similarity`` for two non-zero vectors."""

    left, right = _paired_vectors(predicted, target)
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise InputError("cosine distance requires non-zero vectors")
    cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(0.0, 1.0 - max(-1.0, min(1.0, cosine)))


def region_distance(
    predicted: Iterable[float],
    target: Iterable[float],
    indices: Iterable[int],
) -> float:
    """Return L2 distance restricted to explicit vector indices."""

    left, right = _paired_vectors(predicted, target)
    selected = _normalize_indices(indices, len(left))
    return l2_distance((left[idx] for idx in selected), (right[idx] for idx in selected))


def projection_distance(
    predicted: Iterable[float],
    target: Iterable[float],
    projection: Iterable[Iterable[float]],
) -> float:
    """Return L2 distance after applying a row-major linear projection."""

    left, right = _paired_vectors(predicted, target)
    rows = tuple(_normalize_projection_row(row, width=len(left)) for row in projection)
    if not rows:
        raise InputError("projection must contain at least one row")
    projected_left = tuple(
        sum(weight * value for weight, value in zip(row, left, strict=True)) for row in rows
    )
    projected_right = tuple(
        sum(weight * value for weight, value in zip(row, right, strict=True)) for row in rows
    )
    return l2_distance(projected_left, projected_right)


def _score_candidate(
    edits: Sequence[RelEdit],
    *,
    index: int,
    evaluate: Callable[[Sequence[RelEdit]], float | CandidateEvaluation],
    cost_fn: Callable[[Sequence[RelEdit]], float],
    cost_weight: float,
) -> _ScoredCandidate:
    normalized_edits = tuple(edits)
    evaluation = _normalize_evaluation(evaluate(normalized_edits))
    cost = _validate_nonnegative_finite("candidate cost", cost_fn(normalized_edits))
    objective = evaluation.distance + cost_weight * cost
    return _ScoredCandidate(
        edits=normalized_edits,
        evaluation=evaluation,
        cost=cost,
        objective=_validate_nonnegative_finite("candidate objective", objective),
        index=index,
    )


def _normalize_evaluation(value: float | CandidateEvaluation) -> CandidateEvaluation:
    if isinstance(value, CandidateEvaluation):
        return value
    return CandidateEvaluation(distance=value)


def _candidate_sort_key(candidate: _ScoredCandidate) -> tuple[float, float, float, int]:
    return (
        candidate.objective,
        candidate.evaluation.distance,
        candidate.cost,
        candidate.index,
    )


def _candidate_value_key(candidate: _ScoredCandidate) -> tuple[float, float, float]:
    return (
        candidate.objective,
        candidate.evaluation.distance,
        candidate.cost,
    )


def _iteration_log(
    iteration: int,
    elites: Sequence[_ScoredCandidate],
    iteration_best: _ScoredCandidate,
    *,
    n_candidates: int,
) -> CEMIterationLog:
    elite_count = float(len(elites))
    return CEMIterationLog(
        iteration=iteration,
        best_distance=iteration_best.evaluation.distance,
        best_cost=iteration_best.cost,
        best_objective=iteration_best.objective,
        elite_mean_distance=sum(candidate.evaluation.distance for candidate in elites)
        / elite_count,
        elite_mean_objective=sum(candidate.objective for candidate in elites) / elite_count,
        n_candidates=n_candidates,
    )


def _refit_sampler(
    sampler: ActionSampler,
    *,
    elites: Sequence[Sequence[RelEdit]],
    rng: random.Random,
    smoothing: float,
) -> ActionSampler:
    type_weights = _refit_type_weights(_sampler_type_weights(sampler), elites, smoothing=smoothing)
    position_weights = _refit_position_weights(sampler, elites, smoothing=smoothing)
    return _clone_sampler(
        sampler,
        rng=rng,
        type_weights=type_weights,
        position_weights=dict(position_weights),
    )


def _refit_type_weights(
    previous: Sequence[EditTypeWeight],
    elites: Sequence[Sequence[RelEdit]],
    *,
    smoothing: float,
) -> tuple[EditTypeWeight, ...]:
    prior = {entry.edit_type: entry.weight for entry in previous}
    total_prior = sum(prior.values())
    counts = dict.fromkeys(prior, 0)
    total = 0
    for sequence in elites:
        for edit in sequence:
            if edit.edit_type in counts:
                counts[edit.edit_type] += 1
                total += 1
    if total == 0:
        return tuple(previous)
    entries: list[EditTypeWeight] = []
    for edit_type in sorted(prior, key=int):
        old_prob = prior[edit_type] / total_prior
        mle = counts[edit_type] / total
        entries.append(EditTypeWeight(edit_type, smoothing * old_prob + (1.0 - smoothing) * mle))
    return tuple(entries)


def _refit_position_weights(
    sampler: ActionSampler,
    elites: Sequence[Sequence[RelEdit]],
    *,
    smoothing: float,
) -> tuple[tuple[int, float], ...]:
    previous = _sampler_position_weights(sampler)
    counts = {bin_index: 0 for bin_index, _weight in previous}
    total = 0
    for sequence in elites:
        for edit in sequence:
            bin_index = max(0, (edit.rel_pos - sampler.edge_margin) // sampler.position_bin_bp)
            if bin_index in counts:
                counts[bin_index] += 1
                total += 1
    if total == 0:
        return previous
    total_prior = sum(weight for _bin_index, weight in previous)
    return tuple(
        (
            bin_index,
            smoothing * (weight / total_prior) + (1.0 - smoothing) * (counts[bin_index] / total),
        )
        for bin_index, weight in previous
    )


def _clone_sampler(
    sampler: ActionSampler,
    *,
    rng: random.Random,
    type_weights: Sequence[EditTypeWeight] | None = None,
    position_weights: Mapping[int, float] | Sequence[float] | None = None,
) -> ActionSampler:
    return ActionSampler(
        sampler.window,
        rng=rng,
        edge_margin=sampler.edge_margin,
        type_weights=tuple(type_weights)
        if type_weights is not None
        else _sampler_type_weights(sampler),
        length_dist=_sampler_length_dist(sampler),
        position_bin_bp=sampler.position_bin_bp,
        position_weights=position_weights
        if position_weights is not None
        else _sampler_raw_position_weights(sampler),
        max_attempts=sampler.max_attempts,
    )


def _sampler_type_weights(sampler: ActionSampler) -> tuple[EditTypeWeight, ...]:
    weights = getattr(sampler, "_type_weights", None)
    if weights is None:
        return tuple(EditTypeWeight(edit_type, 1.0) for edit_type in _ALL_V1_EDIT_TYPES)
    return tuple(weights)


def _sampler_length_dist(sampler: ActionSampler) -> Mapping[int, float] | Sequence[float] | None:
    value = getattr(sampler, "_length_dist", None)
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return None
    return tuple(value)


def _sampler_raw_position_weights(
    sampler: ActionSampler,
) -> Mapping[int, float] | None:
    value = getattr(sampler, "_position_weights", None)
    if value is None:
        return None
    return dict(value)


def _sampler_position_weights(sampler: ActionSampler) -> tuple[tuple[int, float], ...]:
    configured = _sampler_raw_position_weights(sampler)
    if configured is not None:
        return tuple(sorted(configured.items()))

    interior = max(1, len(sampler.window) - 2 * sampler.edge_margin)
    n_bins = max(1, math.ceil(interior / sampler.position_bin_bp))
    return tuple((bin_index, 1.0) for bin_index in range(n_bins))


def _paired_vectors(
    predicted: Iterable[float],
    target: Iterable[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    left = _numeric_vector("predicted", predicted)
    right = _numeric_vector("target", target)
    if len(left) != len(right):
        raise InputError(
            "distance vectors must have the same length",
            details={"predicted_len": len(left), "target_len": len(right)},
        )
    return left, right


def _numeric_vector(name: str, values: Iterable[float]) -> tuple[float, ...]:
    if isinstance(values, str | bytes):
        raise InputError(
            f"{name} must be an iterable of finite numbers",
            details={"type": type(values).__name__},
        )
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise InputError(
            f"{name} must be an iterable of finite numbers",
            details={"type": type(values).__name__},
        ) from exc
    if not raw_values:
        raise InputError(f"{name} must contain at least one value")
    return tuple(
        _validate_finite_number(f"{name}[{idx}]", value) for idx, value in enumerate(raw_values)
    )


def _normalize_indices(indices: Iterable[int], width: int) -> tuple[int, ...]:
    try:
        raw_indices = tuple(indices)
    except TypeError as exc:
        raise InputError("indices must be an iterable of integers") from exc
    if not raw_indices:
        raise InputError("indices must contain at least one entry")
    normalized: list[int] = []
    for idx, value in enumerate(raw_indices):
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < width:
            raise InputError(
                "indices must be valid non-negative vector positions",
                details={"index": idx, "value": value, "width": width},
            )
        normalized.append(value)
    return tuple(normalized)


def _normalize_projection_row(row: Iterable[float], *, width: int) -> tuple[float, ...]:
    normalized = _numeric_vector("projection row", row)
    if len(normalized) != width:
        raise InputError(
            "projection row width must match vector length",
            details={"row_width": len(normalized), "vector_width": width},
        )
    return normalized


def _validate_seed(seed: int | None) -> None:
    if seed is None:
        return
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise InputError(
            "seed must be an integer or None",
            details={"seed": seed, "type": type(seed).__name__},
        )


def _validate_finite_number(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise InputError(
            f"{name} must be a finite number",
            details={"value": value, "type": type(value).__name__},
        )
    return float(value)


def _validate_nonnegative_finite(name: str, value: float) -> float:
    normalized = _validate_finite_number(name, value)
    if normalized < 0.0:
        raise InputError(
            f"{name} must be non-negative",
            details={"value": value},
        )
    return normalized


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(
            f"{name} must be a non-negative integer",
            details={"value": value, "type": type(value).__name__},
        )


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InputError(
            f"{name} must be a positive integer",
            details={"value": value, "type": type(value).__name__},
        )
