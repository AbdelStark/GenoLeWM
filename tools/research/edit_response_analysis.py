# SPDX-License-Identifier: Apache-2.0
"""Analyse the edit-response geometry of frozen Carbon states (study R1).

This is the analysis half of the edit-response-geometry probe. Its input is
the per-``(variant, pool_radius)`` embeddings Parquet emitted by
:mod:`tools.research.edit_response_spectroscopy` (columns include
``label_group``, ``continuous_score``, ``pool_radius``, ``cos_ref_alt``,
``rel_delta`` and the raw pooled state vectors ``s_ref`` / ``s_alt`` as
``list<float32>``). For every pooling radius it produces a JSON report with:

1. **Sensitivity ceiling** — mean / median / std of ``cos_ref_alt`` and
   ``rel_delta`` per ``label_group``: how much the edit moves the state at all.
2. **ClinVar path-vs-benign discrimination** — AUROC of (a) the raw
   displacement magnitude ``||s_alt - s_ref||``, (b) the magnitude of the
   unit-sphere displacement ``normalize(s_alt) - normalize(s_ref)``, and
   (c) a leakage-free *directional* score: stratified K-fold CV learns the
   axis ``mean(delta|path) - mean(delta|benign)`` on the training folds and
   scores held-out variants by projection, pooling the held-out scores into a
   single AUROC. Each AUROC carries a 95 % bootstrap confidence interval.
3. **BRCA2 Spearman** — the rank correlation of the unit-sphere displacement
   magnitude with the functional (SGE) score, and of the ClinVar-learned
   direction *transferred* to BRCA2 (BRCA2 displacement projected onto the
   global ClinVar path-benign axis) with the functional score.
4. **PCA of the displacement** — the explained-variance spectrum of the
   centred unit-sphere displacement matrix (the shared "edit-happened" axis
   versus edit-specific residual).

The point metrics reuse the paper's tested primitives
(:func:`geno_lewm.evaluation._auroc`, :func:`geno_lewm.evaluation._spearman_rho`
and the confidence-interval quantile helper) so the numbers match the rest of
the evaluation surface; no SciPy or scikit-learn dependency is introduced.
NumPy and PyArrow are imported lazily (inside the functions that need them),
mirroring :mod:`tools.research.edit_response_spectroscopy`, so the module
imports and type-checks without the ``train`` / ``eval`` extras installed.

Usage::

    python -m tools.research.edit_response_analysis \
        --embeddings emb.parquet --out-report r1.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.evaluation import (
    _auroc as _mann_whitney_auroc,
    _confidence_interval as _bootstrap_confidence_interval,
    _spearman_rho as _spearman_correlation,
)

__all__ = [
    "GENERATED_BY",
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "AnalysisConfig",
    "VariantGeometry",
    "analyze_embeddings",
    "analyze_radius",
    "load_embeddings",
    "main",
    "run_analysis",
]

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.edit_response_analysis"
_UNLABELED: Final = "__unlabeled__"
_CLINVAR_PATH: Final = "clinvar_path"
_CLINVAR_BENIGN: Final = "clinvar_benign"
_BRCA2: Final = "brca2_sge"

#: Columns the analysis reads from the embeddings Parquet. Extra columns are
#: ignored; any missing column fails closed.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "variant_id",
    "label_group",
    "continuous_score",
    "pool_radius",
    "cos_ref_alt",
    "rel_delta",
    "s_ref",
    "s_alt",
    "chrom",
    "ref",
    "alt",
)


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Deterministic knobs for the edit-response-geometry analysis.

    Every random draw (bootstrap resampling, stratified K-fold shuffling,
    Spearman permutation) is seeded so the report is reproducible.
    """

    kfold_splits: int = 5
    kfold_seed: int = 0
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0
    spearman_permutations: int = 1000
    spearman_seed: int = 0
    ci_level: float = 0.95
    min_class_count: int = 5
    min_clinvar_total: int = 20
    min_brca2: int = 30

    def __post_init__(self) -> None:
        _require_positive_int("kfold_splits", self.kfold_splits)
        _require_non_negative_int("bootstrap_resamples", self.bootstrap_resamples)
        _require_non_negative_int("spearman_permutations", self.spearman_permutations)
        _require_non_negative_int("min_class_count", self.min_class_count)
        _require_non_negative_int("min_clinvar_total", self.min_clinvar_total)
        _require_non_negative_int("min_brca2", self.min_brca2)
        if not 0.0 < self.ci_level < 1.0:
            raise InputError(
                "ci_level must lie strictly between 0 and 1",
                details={"ci_level": self.ci_level},
            )
        if self.kfold_splits < 2:
            raise InputError(
                "kfold_splits must be at least 2 for cross-validation",
                details={"kfold_splits": self.kfold_splits},
            )


@dataclass(frozen=True, slots=True)
class VariantGeometry:
    """One variant's pooled reference/edited states at a single radius."""

    variant_id: str
    label_group: str | None
    continuous_score: float | None
    cos_ref_alt: float
    rel_delta: float
    s_ref: tuple[float, ...]
    s_alt: tuple[float, ...]
    chrom: str | None = None
    ref: str | None = None
    alt: str | None = None

    @property
    def is_snv(self) -> bool:
        """True when this row is a single-base substitution.

        Multi-base alleles displace the pooled state far further than a point
        edit purely because they change sequence length, so they are excluded
        from the controlled statistics rather than allowed to inflate them.
        """
        return self.ref is not None and self.alt is not None and len(self.ref) == len(self.alt) == 1

    @property
    def substitution(self) -> str | None:
        """The ``REF>ALT`` substitution class, or ``None`` for non-SNV rows."""
        if not self.is_snv:
            return None
        return f"{self.ref}>{self.alt}"


# ---------------------------------------------------------------------------
# Lazy optional dependencies (kept out of module import so the tool loads and
# type-checks without the train / eval extras, mirroring the spectroscopy tool).


def _numpy() -> Any:
    return cast(Any, importlib.import_module("numpy"))


def _parquet() -> Any:
    return cast(Any, importlib.import_module("pyarrow.parquet"))


# ---------------------------------------------------------------------------
# Linear-algebra helpers (all operate on NumPy arrays typed as ``Any``).


def _stack_vectors(rows: Sequence[VariantGeometry], attr: str) -> Any:
    """Stack a per-variant vector attribute into a ``(n, d)`` float64 matrix."""
    np = _numpy()
    if not rows:
        raise InputError("cannot stack vectors from an empty variant set")
    width = len(getattr(rows[0], attr))
    if width == 0:
        raise InputError("pooled state vectors must be non-empty", details={"attr": attr})
    for row in rows:
        vector = getattr(row, attr)
        if len(vector) != width:
            raise InputError(
                "pooled state vectors must share a width within a radius",
                details={"attr": attr, "expected": width, "observed": len(vector)},
            )
    return np.asarray([list(getattr(row, attr)) for row in rows], dtype=np.float64)


def _l2_normalize_rows(matrix: Any) -> Any:
    np = _numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _learn_direction(delta: Any, labels: Any) -> Any:
    """Return the unit ``mean(delta|positive) - mean(delta|negative)`` axis."""
    np = _numpy()
    positive = delta[labels == 1].mean(axis=0)
    negative = delta[labels == 0].mean(axis=0)
    axis = positive - negative
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        return axis
    return axis / norm


def _row_magnitudes(matrix: Any) -> list[float]:
    np = _numpy()
    return [float(value) for value in np.linalg.norm(matrix, axis=1).tolist()]


# ---------------------------------------------------------------------------
# Metric primitives reusing the tested evaluation helpers.


def _auroc_with_ci(
    scores: Sequence[float], labels: Sequence[int], config: AnalysisConfig
) -> dict[str, Any]:
    """AUROC (Mann-Whitney) plus a stratified bootstrap 95 % interval."""
    bool_labels = [bool(label) for label in labels]
    float_scores = [float(score) for score in scores]
    auroc = _mann_whitney_auroc(bool_labels, float_scores)
    ci = _bootstrap_auroc_ci(float_scores, bool_labels, config=config)
    return {"auroc": auroc, "ci95": list(ci)}


def _bootstrap_auroc_ci(
    scores: Sequence[float],
    labels: Sequence[bool],
    config: AnalysisConfig,
) -> tuple[float, float]:
    """Stratified bootstrap CI, resampling positives and negatives separately."""
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    if config.bootstrap_resamples <= 0 or not positives or not negatives:
        auroc = _mann_whitney_auroc(list(labels), list(scores))
        return (auroc, auroc)
    rng = random.Random(config.bootstrap_seed)
    samples: list[float] = []
    for _ in range(config.bootstrap_resamples):
        sampled_pos = rng.choices(positives, k=len(positives))
        sampled_neg = rng.choices(negatives, k=len(negatives))
        sampled_labels = [True] * len(sampled_pos) + [False] * len(sampled_neg)
        samples.append(_mann_whitney_auroc(sampled_labels, sampled_pos + sampled_neg))
    return _bootstrap_confidence_interval(samples, ci_level=config.ci_level)


def _spearman_with_p(
    scores: Sequence[float],
    targets: Sequence[float],
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Spearman rho with a deterministic two-sided permutation p-value.

    The p-value is a permutation test (no SciPy): the target order is shuffled
    ``spearman_permutations`` times and the fraction of permutations whose
    ``|rho|`` reaches the observed ``|rho|`` gives ``(count + 1) / (perms + 1)``.
    """
    score_list = [float(value) for value in scores]
    target_list = [float(value) for value in targets]
    rho = _spearman_correlation(target_list, score_list)
    p_value = _permutation_p_value(score_list, target_list, rho, config)
    return {
        "rho": rho,
        "p": p_value,
        "n": len(score_list),
        "n_permutations": config.spearman_permutations,
    }


def _permutation_p_value(
    scores: list[float],
    targets: list[float],
    observed_rho: float,
    config: AnalysisConfig,
) -> float:
    if config.spearman_permutations <= 0:
        return float("nan")
    rng = random.Random(config.spearman_seed)
    shuffled = list(targets)
    threshold = abs(observed_rho)
    at_least = 0
    for _ in range(config.spearman_permutations):
        rng.shuffle(shuffled)
        try:
            permuted = _spearman_correlation(shuffled, scores)
        except InputError:
            continue
        if abs(permuted) >= threshold:
            at_least += 1
    return (at_least + 1) / (config.spearman_permutations + 1)


# ---------------------------------------------------------------------------
# Deterministic stratified K-fold (NumPy only; no scikit-learn).


def _stratified_kfold_indices(labels: Any, n_splits: int, seed: int) -> list[tuple[Any, Any]]:
    """Round-robin stratified folds after a seeded per-class permutation."""
    np = _numpy()
    fold_members: list[list[int]] = [[] for _ in range(n_splits)]
    rng = np.random.default_rng(seed)
    for cls in (0, 1):
        class_indices = np.nonzero(labels == cls)[0]
        for position, index in enumerate(rng.permutation(class_indices).tolist()):
            fold_members[position % n_splits].append(int(index))
    all_indices = set(range(int(labels.shape[0])))
    splits: list[tuple[Any, Any]] = []
    for members in fold_members:
        test = sorted(members)
        train = sorted(all_indices.difference(test))
        splits.append((np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)))
    return splits


def _directional_cv_auroc(
    delta: Any,
    labels: Any,
    config: AnalysisConfig,
) -> tuple[dict[str, Any], Any]:
    """Leakage-free directional AUROC: learn the axis on train, score held-out."""
    np = _numpy()
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    n_splits = min(config.kfold_splits, n_positive, n_negative)
    scores = np.full(int(labels.shape[0]), np.nan, dtype=np.float64)
    for train_idx, test_idx in _stratified_kfold_indices(labels, n_splits, config.kfold_seed):
        axis = _learn_direction(delta[train_idx], labels[train_idx])
        scores[test_idx] = delta[test_idx] @ axis
    score_list = [float(value) for value in scores.tolist()]
    label_list = [int(value) for value in labels.tolist()]
    result = _auroc_with_ci(score_list, label_list, config)
    result["n_splits"] = n_splits
    return result, scores


# ---------------------------------------------------------------------------
# Per-radius analysis blocks.


def _describe(values: Any) -> dict[str, float]:
    np = _numpy()
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def _sensitivity_ceiling(rows: Sequence[VariantGeometry]) -> dict[str, Any]:
    np = _numpy()
    groups = np.asarray([row.label_group or _UNLABELED for row in rows])
    cos = np.asarray([row.cos_ref_alt for row in rows], dtype=np.float64)
    rel = np.asarray([row.rel_delta for row in rows], dtype=np.float64)
    out: dict[str, Any] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        out[str(group)] = {
            "n": int(mask.sum()),
            "cos_ref_alt": _describe(cos[mask]),
            "rel_delta": _describe(rel[mask]),
        }
    return out


def _clinvar_block(
    rows: Sequence[VariantGeometry],
    delta_raw: Any,
    delta_norm: Any,
    config: AnalysisConfig,
) -> tuple[dict[str, Any], Any]:
    """ClinVar path-vs-benign AUROCs plus the global transfer axis (or ``None``)."""
    np = _numpy()
    groups = np.asarray([row.label_group or _UNLABELED for row in rows])
    is_path = groups == _CLINVAR_PATH
    is_benign = groups == _CLINVAR_BENIGN
    keep = is_path | is_benign
    n_path = int(is_path.sum())
    n_benign = int(is_benign.sum())
    if (
        int(keep.sum()) < config.min_clinvar_total
        or n_path < config.min_class_count
        or n_benign < config.min_class_count
    ):
        return (
            {
                "skipped": "insufficient ClinVar path/benign variants",
                "n_path": n_path,
                "n_benign": n_benign,
            },
            None,
        )
    labels = is_path[keep].astype(np.int64)
    raw = delta_raw[keep]
    norm = delta_norm[keep]
    label_list = [int(value) for value in labels.tolist()]

    result: dict[str, Any] = {"n_path": n_path, "n_benign": n_benign}
    result["magnitude_raw"] = _auroc_with_ci(_row_magnitudes(raw), label_list, config)
    result["magnitude_norm"] = _auroc_with_ci(_row_magnitudes(norm), label_list, config)
    result["directional_raw_cv"], _ = _directional_cv_auroc(raw, labels, config)
    result["directional_norm_cv"], _ = _directional_cv_auroc(norm, labels, config)

    transfer_axis = _learn_direction(norm, labels)
    return result, transfer_axis


def _brca2_block(
    rows: Sequence[VariantGeometry],
    delta_norm: Any,
    transfer_axis: Any,
    config: AnalysisConfig,
) -> dict[str, Any]:
    np = _numpy()
    groups = np.asarray([row.label_group or _UNLABELED for row in rows])
    is_brca2 = groups == _BRCA2
    n_brca2 = int(is_brca2.sum())
    if n_brca2 < config.min_brca2:
        return {"skipped": "insufficient BRCA2 SGE variants", "n": n_brca2}

    scored = [(index, row) for index, row in enumerate(rows) if is_brca2[index]]
    missing = [row.variant_id for _index, row in scored if row.continuous_score is None]
    if missing:
        raise InputError(
            "BRCA2 variants must carry a continuous_score",
            details={"variant_ids": missing[:10], "n_missing": len(missing)},
        )
    indices = np.asarray([index for index, _row in scored], dtype=np.int64)
    functional = [float(cast(float, row.continuous_score)) for _index, row in scored]
    magnitudes = _row_magnitudes(delta_norm[indices])

    result: dict[str, Any] = {
        "n": n_brca2,
        "spearman_mag_vs_func": _spearman_with_p(magnitudes, functional, config),
    }
    if transfer_axis is not None:
        projections = [float(value) for value in (delta_norm[indices] @ transfer_axis).tolist()]
        result["spearman_clinvar_dir_transfer_vs_func"] = _spearman_with_p(
            projections, functional, config
        )
    return result


def _confound_controls(
    rows: Sequence[VariantGeometry],
    delta_raw: Any,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Test whether the ClinVar displacement AUROC survives its rival explanations.

    A raw ``||delta||`` AUROC is not evidence that the encoder registers biology:
    three cheaper explanations predict the same number, and each is checked here
    against the identical variant set.

    ``substitution_only_auroc``
        Scores each variant by the pathogenic rate of its ``REF>ALT`` class and
        nothing else. ClinVar's mutation spectrum is skewed (C>T at CpG is a
        deamination hotspot), so a model that has learned only which base swaps
        are chemically common would reproduce that much AUROC with no geometry.
    ``within_substitution_auroc``
        Pools ``||delta||`` AUROC computed *inside* each substitution class, where
        the spectrum is constant by construction. Signal that survives here
        cannot be the spectrum.
    ``reference_only_*``
        Probes the unedited state ``s_ref`` alone. Pathogenic and benign variants
        occupy systematically different genomic neighbourhoods, so a probe with
        no access to the edit at all can appear to score variants. This is the
        sharpest threat: if it matches the displacement AUROC, the geometry is
        reading the region rather than the intervention. The chromosome-held-out
        fold assignment additionally denies the probe any same-locus leakage.

    The displacement claim is only supported when the controlled AUROCs stay far
    above these baselines.
    """
    np = _numpy()
    snv = np.asarray([row.is_snv for row in rows])
    groups = np.asarray([row.label_group or _UNLABELED for row in rows])
    keep = ((groups == _CLINVAR_PATH) | (groups == _CLINVAR_BENIGN)) & snv
    n_path = int((keep & (groups == _CLINVAR_PATH)).sum())
    n_benign = int((keep & (groups == _CLINVAR_BENIGN)).sum())
    if (
        int(keep.sum()) < config.min_clinvar_total
        or n_path < config.min_class_count
        or n_benign < config.min_class_count
    ):
        return {
            "skipped": "insufficient ClinVar path/benign SNVs",
            "n_path": n_path,
            "n_benign": n_benign,
            "n_non_snv_excluded": int((~snv).sum()),
        }

    labels = (groups[keep] == _CLINVAR_PATH).astype(np.int64)
    label_list = [int(value) for value in labels.tolist()]
    magnitudes = _row_magnitudes(delta_raw[keep])
    kept_rows = [row for row, flag in zip(rows, keep.tolist(), strict=True) if flag]
    subs = np.asarray([row.substitution for row in kept_rows])

    result: dict[str, Any] = {
        "n_path": n_path,
        "n_benign": n_benign,
        "n_non_snv_excluded": int((~snv).sum()),
        "magnitude_raw_snv_only": _auroc_with_ci(magnitudes, label_list, config),
    }

    # Baseline 1: substitution-class pathogenic rate, carrying zero geometry.
    rates = {
        str(name): float(labels[subs == name].mean()) for name in np.unique(subs)  # noqa: PD011
    }
    lookup = [rates[str(name)] for name in subs.tolist()]
    result["substitution_only_auroc"] = _auroc_with_ci(lookup, label_list, config)

    # Baseline 2: displacement AUROC within each substitution class, pooled by n.
    per_class: dict[str, Any] = {}
    weighted_sum = 0.0
    weight_total = 0
    magnitude_array = np.asarray(magnitudes, dtype=np.float64)
    for name in np.unique(subs):
        mask = subs == name
        class_labels = labels[mask]
        n_pos = int(class_labels.sum())
        n_neg = int((1 - class_labels).sum())
        if n_pos < config.min_class_count * 2 or n_neg < config.min_class_count * 2:
            continue
        auroc = _mann_whitney_auroc(
            [bool(value) for value in class_labels.tolist()],
            [float(value) for value in magnitude_array[mask].tolist()],
        )
        size = int(mask.sum())
        per_class[str(name)] = {"n": size, "n_path": n_pos, "auroc": auroc}
        weighted_sum += auroc * size
        weight_total += size
    result["within_substitution_per_class"] = per_class
    result["within_substitution_auroc"] = (
        weighted_sum / weight_total if weight_total > 0 else None
    )
    result["within_substitution_n"] = weight_total

    # Baseline 3: the unedited reference state, which knows nothing of the edit.
    s_ref = _stack_vectors(kept_rows, "s_ref")
    result["reference_only_norm_auroc"] = _auroc_with_ci(
        _row_magnitudes(s_ref), label_list, config
    )
    result["reference_only_probe_auroc"], _ = _directional_cv_auroc(s_ref, labels, config)
    chroms = [row.chrom for row in kept_rows]
    if all(value is not None for value in chroms) and len(set(chroms)) >= config.kfold_splits:
        result["reference_only_probe_chrom_holdout_auroc"] = _grouped_probe_auroc(
            s_ref, labels, np.asarray(chroms), config
        )
    return result


def _grouped_probe_auroc(
    features: Any,
    labels: Any,
    groups: Any,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Mean-difference probe AUROC with whole groups held out per fold.

    Random folds let a probe memorise a locus seen in training; assigning entire
    chromosomes to a fold forces it to generalise across genomic location.
    """
    np = _numpy()
    unique = sorted({str(value) for value in groups.tolist()})
    fold_of = {name: index % config.kfold_splits for index, name in enumerate(unique)}
    folds = np.asarray([fold_of[str(value)] for value in groups.tolist()])
    scores = np.zeros(len(labels), dtype=np.float64)
    for fold in range(config.kfold_splits):
        test = folds == fold
        train = ~test
        if not bool(test.any()) or len(np.unique(labels[train])) < 2:
            continue
        axis = _learn_direction(features[train], labels[train])
        scores[test] = features[test] @ axis
    return {
        "auroc": _mann_whitney_auroc(
            [bool(value) for value in labels.tolist()],
            [float(value) for value in scores.tolist()],
        ),
        "n_groups": len(unique),
    }


def _pca_delta(delta_norm: Any) -> dict[str, Any]:
    np = _numpy()
    if int(delta_norm.shape[0]) < 2:
        return {"skipped": "PCA requires at least two variants", "n": int(delta_norm.shape[0])}
    centred = delta_norm - delta_norm.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    variances = singular_values**2
    total = float(variances.sum())
    if total == 0.0:
        ratios = np.zeros_like(variances)
    else:
        ratios = variances / total
    cumulative = np.cumsum(ratios)
    return {
        "explained_variance_ratio_top10": [float(value) for value in ratios[:10].tolist()],
        "explained_variance_ratio_sum": float(ratios.sum()),
        "n_components_50pct": _components_for(cumulative, 0.5),
        "n_components_90pct": _components_for(cumulative, 0.9),
    }


def _components_for(cumulative: Any, fraction: float) -> int:
    np = _numpy()
    return int(np.searchsorted(cumulative, fraction) + 1)


def analyze_radius(
    rows: Sequence[VariantGeometry],
    *,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Run the full edit-response-geometry analysis for one pooling radius."""
    if not rows:
        raise InputError("analyze_radius requires at least one variant row")
    cfg = config if config is not None else AnalysisConfig()

    s_ref = _stack_vectors(rows, "s_ref")
    s_alt = _stack_vectors(rows, "s_alt")
    delta_raw = s_alt - s_ref
    delta_norm = _l2_normalize_rows(s_alt) - _l2_normalize_rows(s_ref)

    clinvar, transfer_axis = _clinvar_block(rows, delta_raw, delta_norm, cfg)
    return {
        "n": len(rows),
        "sensitivity": _sensitivity_ceiling(rows),
        "clinvar_path_vs_benign": clinvar,
        "confound_controls": _confound_controls(rows, delta_raw, cfg),
        "brca2": _brca2_block(rows, delta_norm, transfer_axis, cfg),
        "pca_delta": _pca_delta(delta_norm),
    }


def analyze_embeddings(
    rows_by_radius: Mapping[int, Sequence[VariantGeometry]],
    *,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Analyse each pooling radius independently and key the report by radius."""
    if not rows_by_radius:
        raise InputError("no variant rows to analyse")
    cfg = config if config is not None else AnalysisConfig()
    by_radius = {
        str(radius): analyze_radius(rows_by_radius[radius], config=cfg)
        for radius in sorted(rows_by_radius)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "config": _config_payload(cfg),
        "radii": sorted(rows_by_radius),
        "by_radius": by_radius,
    }


def _config_payload(config: AnalysisConfig) -> dict[str, Any]:
    return {
        "kfold_splits": config.kfold_splits,
        "kfold_seed": config.kfold_seed,
        "bootstrap_resamples": config.bootstrap_resamples,
        "bootstrap_seed": config.bootstrap_seed,
        "spearman_permutations": config.spearman_permutations,
        "spearman_seed": config.spearman_seed,
        "ci_level": config.ci_level,
        "min_class_count": config.min_class_count,
        "min_clinvar_total": config.min_clinvar_total,
        "min_brca2": config.min_brca2,
    }


# ---------------------------------------------------------------------------
# Parquet loading.


def load_embeddings(path: str | Path) -> dict[int, list[VariantGeometry]]:
    """Load the embeddings Parquet into per-radius :class:`VariantGeometry` lists."""
    pq = _parquet()
    source = Path(path)
    if not source.is_file():
        raise InputError("embeddings Parquet not found", details={"path": str(source)})
    try:
        table = pq.read_table(source)
    except (OSError, ValueError) as exc:
        raise InputError(
            "could not read embeddings Parquet",
            details={"path": str(source), "error": str(exc)},
        ) from exc

    missing = [name for name in REQUIRED_COLUMNS if name not in table.column_names]
    if missing:
        raise InputError(
            "embeddings Parquet is missing required columns",
            details={"path": str(source), "missing": missing},
        )
    records = cast("list[dict[str, Any]]", table.select(REQUIRED_COLUMNS).to_pylist())
    by_radius: dict[int, list[VariantGeometry]] = {}
    for line_no, record in enumerate(records, start=1):
        radius = _require_row_int(record, "pool_radius", source, line_no)
        by_radius.setdefault(radius, []).append(_row_to_geometry(record, source, line_no))
    if not by_radius:
        raise InputError("embeddings Parquet contains no rows", details={"path": str(source)})
    return by_radius


def _row_to_geometry(record: Mapping[str, Any], source: Path, line_no: int) -> VariantGeometry:
    return VariantGeometry(
        variant_id=str(record.get("variant_id") or f"row:{line_no}"),
        label_group=_optional_row_str(record.get("label_group")),
        continuous_score=_optional_row_float(record, "continuous_score", source, line_no),
        cos_ref_alt=_require_row_float(record, "cos_ref_alt", source, line_no),
        rel_delta=_require_row_float(record, "rel_delta", source, line_no),
        s_ref=_require_row_vector(record, "s_ref", source, line_no),
        s_alt=_require_row_vector(record, "s_alt", source, line_no),
        chrom=_optional_row_str(record.get("chrom")),
        ref=_optional_row_str(record.get("ref")),
        alt=_optional_row_str(record.get("alt")),
    )


def _optional_row_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _require_row_int(record: Mapping[str, Any], field: str, source: Path, line_no: int) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(
            "embeddings Parquet field must be an integer",
            details={"path": str(source), "row": line_no, "field": field},
        )
    return value


def _require_row_float(record: Mapping[str, Any], field: str, source: Path, line_no: int) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(
            "embeddings Parquet field must be a number",
            details={"path": str(source), "row": line_no, "field": field},
        )
    return float(value)


def _optional_row_float(
    record: Mapping[str, Any], field: str, source: Path, line_no: int
) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(
            "embeddings Parquet field must be a number when present",
            details={"path": str(source), "row": line_no, "field": field},
        )
    return float(value)


def _require_row_vector(
    record: Mapping[str, Any], field: str, source: Path, line_no: int
) -> tuple[float, ...]:
    value = record.get(field)
    if not isinstance(value, (list, tuple)) or not value:
        raise InputError(
            "embeddings Parquet field must be a non-empty numeric list",
            details={"path": str(source), "row": line_no, "field": field},
        )
    vector: list[float] = []
    for element in value:
        if isinstance(element, bool) or not isinstance(element, int | float):
            raise InputError(
                "embeddings Parquet vector must contain only numbers",
                details={"path": str(source), "row": line_no, "field": field},
            )
        vector.append(float(element))
    return tuple(vector)


def run_analysis(
    path: str | Path,
    *,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Load the embeddings Parquet and produce the full R1 report."""
    cfg = config if config is not None else AnalysisConfig()
    rows_by_radius = load_embeddings(path)
    report = analyze_embeddings(rows_by_radius, config=cfg)
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
    commit = result.stdout.strip()
    return commit or None


def _parser() -> argparse.ArgumentParser:
    defaults = AnalysisConfig()
    parser = argparse.ArgumentParser(description="Edit-response-geometry analysis (R1).")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--kfold-splits", type=int, default=defaults.kfold_splits)
    parser.add_argument("--kfold-seed", type=int, default=defaults.kfold_seed)
    parser.add_argument("--bootstrap-resamples", type=int, default=defaults.bootstrap_resamples)
    parser.add_argument("--bootstrap-seed", type=int, default=defaults.bootstrap_seed)
    parser.add_argument("--spearman-permutations", type=int, default=defaults.spearman_permutations)
    parser.add_argument("--spearman-seed", type=int, default=defaults.spearman_seed)
    parser.add_argument("--ci-level", type=float, default=defaults.ci_level)
    return parser


def _config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    return replace(
        AnalysisConfig(),
        kfold_splits=args.kfold_splits,
        kfold_seed=args.kfold_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        spearman_permutations=args.spearman_permutations,
        spearman_seed=args.spearman_seed,
        ci_level=args.ci_level,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: analyse an embeddings Parquet and write the JSON report."""
    args = _parser().parse_args(argv)
    try:
        config = _config_from_args(args)
        report = run_analysis(args.embeddings, config=config)
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
                "radii": report["radii"],
                "n_radii": len(report["radii"]),
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
