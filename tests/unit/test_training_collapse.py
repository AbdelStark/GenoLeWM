"""Unit tests for training contract collapse monitoring."""

from __future__ import annotations

import math
from typing import Any

import pytest

from geno_lewm import metrics as metric_registry
from geno_lewm.errors import InputError
from geno_lewm.training.collapse import (
    CollapseAlert,
    CollapseMonitor,
    CollapseThresholds,
    compute_collapse_metrics,
    detect_collapse,
    record_collapse_metrics,
)


@pytest.fixture(autouse=True)
def _reset_metrics() -> Any:
    metric_registry._reset_for_tests()
    yield
    metric_registry._reset_for_tests()


class _FakeLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[str, dict[str, object]]] = []
        self.warnings: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **fields: object) -> None:
        self.infos.append((event, fields))

    def warn(self, event: str, **fields: object) -> None:
        self.warnings.append((event, fields))


class _TensorLike:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def detach(self) -> _TensorLike:
        return self

    def cpu(self) -> _TensorLike:
        return self

    def tolist(self) -> list[list[float]]:
        return self.rows


def test_compute_collapse_metrics_for_known_batch() -> None:
    metrics = compute_collapse_metrics(
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 1.0]],
        kl_reg=0.25,
    )

    assert metrics.pred_cos_mean == pytest.approx((1.0 + (1.0 / math.sqrt(2.0))) / 2.0)
    assert metrics.pred_l2_mean == pytest.approx(0.5)
    assert metrics.target_var_per_dim == pytest.approx(0.125)
    assert metrics.pred_var_per_dim == pytest.approx(0.25)
    assert metrics.pred_target_corr == pytest.approx(1.0 / math.sqrt(3.0))
    assert metrics.pairwise_pred_dist_mean == pytest.approx(math.sqrt(2.0))
    assert metrics.kl_reg == pytest.approx(0.25)


def test_tensor_like_inputs_use_detach_cpu_tolist() -> None:
    metrics = compute_collapse_metrics(
        _TensorLike([[1.0, 0.0], [0.0, 1.0]]),
        _TensorLike([[1.0, 0.0], [0.0, 1.0]]),
        kl_reg=0.0,
    )

    assert metrics.pred_cos_mean == pytest.approx(1.0)
    assert metrics.pred_var_per_dim == pytest.approx(metrics.target_var_per_dim)


def test_collapsed_predictions_trigger_alert_and_logging() -> None:
    logger = _FakeLogger()
    monitor = CollapseMonitor()

    result = monitor.observe(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]],
        kl_reg=0.0,
        step=500,
        logger=logger,
    )

    assert result is not None
    assert result.tripped
    assert [alert.criterion for alert in result.alerts] == ["pred_var_per_dim"]
    assert metric_registry.get_counter("geno_lewm.training.collapse.alert").value() == 1.0
    assert metric_registry.get_gauge("geno_lewm.training.collapse.pred_var_per_dim").value() == 0.0
    assert len(logger.infos) == 7
    assert logger.warnings == [
        (
            "training.collapse.alert",
            {"step": 500, "criterion": "pred_var_per_dim", "value": 0.0, "threshold": 2.5},
        )
    ]


def test_healthy_targets_do_not_trigger_alerts_on_10k_synthetic_batches() -> None:
    monitor = CollapseMonitor()
    prediction = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    target = [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

    for i in range(10_000):
        result = monitor.observe(prediction, target, kl_reg=0.0, step=(i + 1) * 500)
        assert result is not None
        assert not result.alerts

    assert metric_registry.get_counter("geno_lewm.training.collapse.alert").value() == 0.0


def test_monitor_skips_unscheduled_steps_unless_forced() -> None:
    monitor = CollapseMonitor()
    prediction = [[0.0, 0.0], [1.0, 1.0]]
    target = [[0.0, 0.0], [1.0, 1.0]]

    assert monitor.observe(prediction, target, kl_reg=0.0, step=499) is None
    assert monitor.observe(prediction, target, kl_reg=0.0, step=499, force=True) is not None


def test_pairwise_and_kl_criteria_trip() -> None:
    baseline = compute_collapse_metrics(
        [[-1.0, -1.0], [1.0, 1.0]],
        [[-1.0, -1.0], [1.0, 1.0]],
        kl_reg=0.0,
    )
    collapsed = compute_collapse_metrics(
        [[0.0, 0.0], [0.0, 0.0]],
        [[-1.0, -1.0], [1.0, 1.0]],
        kl_reg=11.0,
    )

    alerts = detect_collapse(
        collapsed,
        initial_pairwise_pred_dist_mean=baseline.pairwise_pred_dist_mean,
    )

    assert [alert.criterion for alert in alerts] == [
        "pred_var_per_dim",
        "pairwise_pred_dist_mean",
        "kl_reg",
    ]


def test_record_collapse_metrics_updates_registry_without_logger() -> None:
    metrics = compute_collapse_metrics(
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        kl_reg=0.0,
    )

    record_collapse_metrics(
        metrics,
        alerts=(CollapseAlert("kl_reg", 11.0, 10.0),),
    )

    assert metric_registry.get_gauge("geno_lewm.training.collapse.pred_cos_mean").value() == 1.0
    assert metric_registry.get_counter("geno_lewm.training.collapse.alert").value() == 1.0


def test_record_collapse_metrics_logs_without_step() -> None:
    logger = _FakeLogger()
    metrics = compute_collapse_metrics([[0.0, 0.0]], [[0.0, 0.0]], kl_reg=0.0)

    record_collapse_metrics(
        metrics,
        alerts=(CollapseAlert("kl_reg", 11.0, 10.0),),
        logger=logger,
    )

    assert len(logger.infos) == 7
    assert logger.infos[0] == (
        "training.metric",
        {
            "name": "geno_lewm.training.collapse.pred_cos_mean",
            "value": 0.0,
            "unit": "unitless",
            "kind": "gauge",
        },
    )
    assert logger.warnings == [
        (
            "training.collapse.alert",
            {"criterion": "kl_reg", "value": 11.0, "threshold": 10.0},
        )
    ]


def test_single_zero_vector_batch_has_zero_similarity_corr_and_pairwise() -> None:
    metrics = compute_collapse_metrics([[0.0, 0.0]], [[0.0, 0.0]], kl_reg=0.0)

    assert metrics.pred_cos_mean == 0.0
    assert metrics.pred_target_corr == 0.0
    assert metrics.pairwise_pred_dist_mean == 0.0


def test_monitor_accepts_explicit_initial_pairwise_baseline() -> None:
    monitor = CollapseMonitor(initial_pairwise_pred_dist_mean=1.0)

    assert monitor.initial_pairwise_pred_dist_mean == 1.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CollapseMonitor(log_every_steps=0),
        lambda: CollapseMonitor(initial_pairwise_pred_dist_mean=-1.0),
        lambda: CollapseMonitor().should_log(-1),
        lambda: detect_collapse(
            compute_collapse_metrics([[1.0]], [[1.0]], kl_reg=0.0),
            initial_pairwise_pred_dist_mean=-1.0,
        ),
        lambda: record_collapse_metrics(
            compute_collapse_metrics([[1.0]], [[1.0]], kl_reg=0.0),
            step=-1,
        ),
    ],
)
def test_invalid_monitor_configuration_raises_input_error(factory: Any) -> None:
    with pytest.raises(InputError):
        factory()


@pytest.mark.parametrize(
    ("prediction", "target", "kl_reg"),
    [
        (1.0, [[1.0]], 0.0),
        ([], [], 0.0),
        ([1.0], [[1.0]], 0.0),
        ([[]], [[1.0]], 0.0),
        ([[1.0], [2.0]], [[1.0]], 0.0),
        ([[1.0]], [[1.0, 2.0]], 0.0),
        ([[1.0], [1.0, 2.0]], [[1.0], [2.0]], 0.0),
        ([[1.0], [math.nan]], [[1.0], [2.0]], 0.0),
        ([[True]], [[1.0]], 0.0),
        ([[1.0]], [[1.0]], math.inf),
    ],
)
def test_invalid_inputs_raise_input_error(
    prediction: object,
    target: object,
    kl_reg: float,
) -> None:
    with pytest.raises(InputError):
        compute_collapse_metrics(prediction, target, kl_reg=kl_reg)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CollapseThresholds(pred_var_to_target_var=0.0),
        lambda: CollapseThresholds(pairwise_to_initial=0.0),
        lambda: CollapseThresholds(kl_reg_max=0.0),
    ],
)
def test_invalid_thresholds_raise_input_error(factory: Any) -> None:
    with pytest.raises(InputError):
        factory()
