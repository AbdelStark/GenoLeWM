# SPDX-License-Identifier: Apache-2.0
"""RFC-0005 collapse monitoring for training batches.

The monitor accepts plain Python nested sequences and common tensor-like
objects that expose ``detach()``, ``cpu()``, and/or ``tolist()``. Core
math stays dependency-free so the diagnostics remain available before a
full ML runtime is installed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from geno_lewm.errors import InputError
from geno_lewm.metrics import get_counter, get_gauge
from geno_lewm.observability import GenoLeWMLogger

__all__ = [
    "CollapseAlert",
    "CollapseCheck",
    "CollapseMetrics",
    "CollapseMonitor",
    "CollapseThresholds",
    "compute_collapse_metrics",
    "detect_collapse",
    "record_collapse_metrics",
]

Matrix: TypeAlias = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class CollapseMetrics:
    """Scalar RFC-0005 §3.6 collapse diagnostics for one batch."""

    pred_cos_mean: float
    pred_l2_mean: float
    target_var_per_dim: float
    pred_var_per_dim: float
    pred_target_corr: float
    pairwise_pred_dist_mean: float
    kl_reg: float


@dataclass(frozen=True, slots=True)
class CollapseThresholds:
    """Alert thresholds from RFC-0005 §3.6."""

    pred_var_to_target_var: float = 0.5
    pairwise_to_initial: float = 0.5
    kl_reg_max: float = 10.0

    def __post_init__(self) -> None:
        _require_positive_finite("pred_var_to_target_var", self.pred_var_to_target_var)
        _require_positive_finite("pairwise_to_initial", self.pairwise_to_initial)
        _require_positive_finite("kl_reg_max", self.kl_reg_max)


@dataclass(frozen=True, slots=True)
class CollapseAlert:
    """One tripped collapse criterion."""

    criterion: str
    value: float
    threshold: float


@dataclass(frozen=True, slots=True)
class CollapseCheck:
    """Metrics and alerts produced by one collapse-monitor observation."""

    metrics: CollapseMetrics
    alerts: tuple[CollapseAlert, ...]

    @property
    def tripped(self) -> bool:
        """Return whether any collapse alert tripped."""
        return bool(self.alerts)


@dataclass(slots=True)
class CollapseMonitor:
    """Compute, register, and optionally log collapse diagnostics.

    ``observe`` returns ``None`` on non-logging steps, otherwise a
    :class:`CollapseCheck`. The first logged batch establishes the
    pairwise-distance baseline unless the caller supplies one.
    """

    log_every_steps: int = 500
    thresholds: CollapseThresholds = field(default_factory=CollapseThresholds)
    initial_pairwise_pred_dist_mean: float | None = None

    def __post_init__(self) -> None:
        _require_positive_int("log_every_steps", self.log_every_steps)
        if self.initial_pairwise_pred_dist_mean is not None:
            _require_nonnegative_finite(
                "initial_pairwise_pred_dist_mean",
                self.initial_pairwise_pred_dist_mean,
            )

    def should_log(self, step: int) -> bool:
        """Return whether ``step`` is a scheduled collapse-monitor step."""
        _require_nonnegative_int("step", step)
        return step > 0 and step % self.log_every_steps == 0

    def observe(
        self,
        prediction: object,
        target: object,
        *,
        kl_reg: float,
        step: int,
        logger: GenoLeWMLogger | None = None,
        force: bool = False,
    ) -> CollapseCheck | None:
        """Observe a validation batch at ``step``.

        Metrics are computed, written to the registry, and logged only
        when ``step`` is a scheduled monitoring step unless ``force`` is
        true.
        """
        _require_nonnegative_int("step", step)
        if not force and not self.should_log(step):
            return None

        metrics = compute_collapse_metrics(prediction, target, kl_reg=kl_reg)
        if self.initial_pairwise_pred_dist_mean is None:
            self.initial_pairwise_pred_dist_mean = metrics.pairwise_pred_dist_mean
        alerts = detect_collapse(
            metrics,
            thresholds=self.thresholds,
            initial_pairwise_pred_dist_mean=self.initial_pairwise_pred_dist_mean,
        )
        record_collapse_metrics(metrics, alerts=alerts, logger=logger, step=step)
        return CollapseCheck(metrics=metrics, alerts=alerts)


def compute_collapse_metrics(
    prediction: object,
    target: object,
    *,
    kl_reg: float,
) -> CollapseMetrics:
    """Compute RFC-0005 §3.6 collapse metrics for one ``[N, D]`` batch."""
    pred_rows = _as_rows(prediction, "prediction")
    target_rows = _as_rows(target, "target")
    if len(pred_rows) != len(target_rows):
        raise InputError(
            "prediction and target must have the same batch size",
            details={"prediction_rows": len(pred_rows), "target_rows": len(target_rows)},
        )
    dim = len(pred_rows[0])
    if dim != len(target_rows[0]):
        raise InputError(
            "prediction and target must have the same latent dimension",
            details={"prediction_dim": dim, "target_dim": len(target_rows[0])},
        )

    kl_value = _finite_float(kl_reg, "kl_reg")
    return CollapseMetrics(
        pred_cos_mean=_mean(
            _cosine(pred, tgt) for pred, tgt in zip(pred_rows, target_rows, strict=True)
        ),
        pred_l2_mean=_mean(
            _euclidean(pred, tgt) for pred, tgt in zip(pred_rows, target_rows, strict=True)
        ),
        target_var_per_dim=_mean_variance_per_dim(target_rows),
        pred_var_per_dim=_mean_variance_per_dim(pred_rows),
        pred_target_corr=_pearson_corr(_flatten(pred_rows), _flatten(target_rows)),
        pairwise_pred_dist_mean=_pairwise_dist_mean(pred_rows),
        kl_reg=kl_value,
    )


def detect_collapse(
    metrics: CollapseMetrics,
    *,
    thresholds: CollapseThresholds | None = None,
    initial_pairwise_pred_dist_mean: float | None = None,
) -> tuple[CollapseAlert, ...]:
    """Return the RFC-0005 §3.6 alert criteria tripped by ``metrics``."""
    active_thresholds = thresholds if thresholds is not None else CollapseThresholds()
    if initial_pairwise_pred_dist_mean is not None:
        _require_nonnegative_finite(
            "initial_pairwise_pred_dist_mean",
            initial_pairwise_pred_dist_mean,
        )

    alerts: list[CollapseAlert] = []
    pred_var_threshold = active_thresholds.pred_var_to_target_var * metrics.target_var_per_dim
    if metrics.pred_var_per_dim < pred_var_threshold:
        alerts.append(
            CollapseAlert(
                criterion="pred_var_per_dim",
                value=metrics.pred_var_per_dim,
                threshold=pred_var_threshold,
            )
        )

    if initial_pairwise_pred_dist_mean is not None:
        pairwise_threshold = active_thresholds.pairwise_to_initial * initial_pairwise_pred_dist_mean
        if metrics.pairwise_pred_dist_mean < pairwise_threshold:
            alerts.append(
                CollapseAlert(
                    criterion="pairwise_pred_dist_mean",
                    value=metrics.pairwise_pred_dist_mean,
                    threshold=pairwise_threshold,
                )
            )

    if metrics.kl_reg > active_thresholds.kl_reg_max:
        alerts.append(
            CollapseAlert(
                criterion="kl_reg",
                value=metrics.kl_reg,
                threshold=active_thresholds.kl_reg_max,
            )
        )

    return tuple(alerts)


def record_collapse_metrics(
    metrics: CollapseMetrics,
    *,
    alerts: Iterable[CollapseAlert] = (),
    logger: GenoLeWMLogger | None = None,
    step: int | None = None,
) -> None:
    """Write collapse metrics to the registry and optional structured logs."""
    if step is not None:
        _require_nonnegative_int("step", step)

    get_gauge("geno_lewm.training.collapse.pred_cos_mean").set(metrics.pred_cos_mean)
    get_gauge("geno_lewm.training.collapse.pred_l2_mean").set(metrics.pred_l2_mean)
    get_gauge("geno_lewm.training.collapse.target_var_per_dim").set(metrics.target_var_per_dim)
    get_gauge("geno_lewm.training.collapse.pred_var_per_dim").set(metrics.pred_var_per_dim)
    get_gauge("geno_lewm.training.collapse.pred_target_corr").set(metrics.pred_target_corr)
    get_gauge("geno_lewm.training.collapse.pairwise_pred_dist_mean").set(
        metrics.pairwise_pred_dist_mean
    )
    get_gauge("geno_lewm.training.collapse.kl_reg").set(metrics.kl_reg)

    if logger is not None:
        for name, value in _metric_items(metrics):
            _log_training_metric(logger, name=name, value=value, step=step)

    for alert in alerts:
        get_counter("geno_lewm.training.collapse.alert").inc()
        if logger is not None:
            _log_collapse_alert(logger, alert, step=step)


def _metric_items(metrics: CollapseMetrics) -> tuple[tuple[str, float], ...]:
    return (
        ("geno_lewm.training.collapse.pred_cos_mean", metrics.pred_cos_mean),
        ("geno_lewm.training.collapse.pred_l2_mean", metrics.pred_l2_mean),
        ("geno_lewm.training.collapse.target_var_per_dim", metrics.target_var_per_dim),
        ("geno_lewm.training.collapse.pred_var_per_dim", metrics.pred_var_per_dim),
        ("geno_lewm.training.collapse.pred_target_corr", metrics.pred_target_corr),
        (
            "geno_lewm.training.collapse.pairwise_pred_dist_mean",
            metrics.pairwise_pred_dist_mean,
        ),
        ("geno_lewm.training.collapse.kl_reg", metrics.kl_reg),
    )


def _log_training_metric(
    logger: GenoLeWMLogger,
    *,
    name: str,
    value: float,
    step: int | None,
) -> None:
    if step is None:
        logger.info("training.metric", name=name, value=value, unit="unitless", kind="gauge")
    else:
        logger.info(
            "training.metric",
            step=step,
            name=name,
            value=value,
            unit="unitless",
            kind="gauge",
        )


def _log_collapse_alert(
    logger: GenoLeWMLogger,
    alert: CollapseAlert,
    *,
    step: int | None,
) -> None:
    if step is None:
        logger.warn(
            "training.collapse.alert",
            criterion=alert.criterion,
            value=alert.value,
            threshold=alert.threshold,
        )
    else:
        logger.warn(
            "training.collapse.alert",
            step=step,
            criterion=alert.criterion,
            value=alert.value,
            threshold=alert.threshold,
        )


def _as_rows(value: object, name: str) -> Matrix:
    raw = _tensor_like_to_python(value)
    if isinstance(raw, str | bytes | bytearray) or not isinstance(raw, Sequence):
        raise InputError(
            f"{name} must be a 2D numeric sequence",
            details={"field": name, "type": type(raw).__name__},
        )
    if not raw:
        raise InputError(f"{name} must contain at least one row", details={"field": name})

    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for row_index, raw_row in enumerate(raw):
        if isinstance(raw_row, str | bytes | bytearray) or not isinstance(raw_row, Sequence):
            raise InputError(
                f"{name} rows must be numeric sequences",
                details={"field": name, "row": row_index, "type": type(raw_row).__name__},
            )
        if not raw_row:
            raise InputError(
                f"{name} rows must contain at least one value",
                details={"field": name, "row": row_index},
            )
        row = tuple(
            _finite_float(item, name, row_index=row_index, column_index=column_index)
            for column_index, item in enumerate(raw_row)
        )
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise InputError(
                f"{name} must be rectangular",
                details={"field": name, "expected_width": width, "row": row_index},
            )
        rows.append(row)
    return tuple(rows)


def _tensor_like_to_python(value: object) -> object:
    out = _call_zero_arg_method(value, "detach")
    out = _call_zero_arg_method(out, "cpu")
    return _call_zero_arg_method(out, "tolist")


def _call_zero_arg_method(value: object, method_name: str) -> object:
    method = getattr(value, method_name, None)
    if method is None or not callable(method):
        return value
    return cast(Callable[[], object], method)()


def _finite_float(
    value: object,
    name: str,
    *,
    row_index: int | None = None,
    column_index: int | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        details: dict[str, object] = {
            "field": name,
            "value": value,
            "type": type(value).__name__,
        }
        if row_index is not None:
            details["row"] = row_index
        if column_index is not None:
            details["column"] = column_index
        raise InputError(f"{name} values must be numeric", details=details)
    out = float(value)
    if not math.isfinite(out):
        details = {"field": name, "value": out}
        if row_index is not None:
            details["row"] = row_index
        if column_index is not None:
            details["column"] = column_index
        raise InputError(f"{name} values must be finite", details=details)
    return out


def _require_positive_finite(name: str, value: float) -> None:
    if _finite_float(value, name) <= 0.0:
        raise InputError(
            f"{name} must be positive",
            details={"field": name, "value": value},
        )


def _require_nonnegative_finite(name: str, value: float) -> None:
    if _finite_float(value, name) < 0.0:
        raise InputError(
            f"{name} must be non-negative",
            details={"field": name, "value": value},
        )


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(
            f"{name} must be a non-negative integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    if count == 0:
        raise InputError("mean requires at least one value")
    return total / count


def _flatten(rows: Matrix) -> tuple[float, ...]:
    return tuple(value for row in rows for value in row)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b, strict=True)))


def _mean_variance_per_dim(rows: Matrix) -> float:
    n_rows = len(rows)
    n_dims = len(rows[0])
    variances: list[float] = []
    for dim in range(n_dims):
        values = [row[dim] for row in rows]
        mean = sum(values) / n_rows
        variances.append(sum((value - mean) * (value - mean) for value in values) / n_rows)
    return sum(variances) / n_dims


def _pearson_corr(a: Sequence[float], b: Sequence[float]) -> float:
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    centered_a = [value - mean_a for value in a]
    centered_b = [value - mean_b for value in b]
    numerator = sum(x * y for x, y in zip(centered_a, centered_b, strict=True))
    denom = math.sqrt(sum(x * x for x in centered_a) * sum(y * y for y in centered_b))
    if denom == 0.0:
        return 0.0
    return numerator / denom


def _pairwise_dist_mean(rows: Matrix) -> float:
    if len(rows) < 2:
        return 0.0
    total = 0.0
    count = 0
    for left_index, left in enumerate(rows[:-1]):
        for right in rows[left_index + 1 :]:
            total += _euclidean(left, right)
            count += 1
    return total / count
