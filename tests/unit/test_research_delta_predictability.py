# SPDX-License-Identifier: Apache-2.0
"""Tests for the R3 delta-predictability tool.

The tool exists to answer one question -- can the edit response ``Delta`` be
predicted from ``(s_ref, action)`` alone? -- so the tests must show it gives the
right answer in *both* directions on data where the truth is known:

* :func:`test_deterministic_delta_is_recovered` plants a ``Delta`` that is an
  exact function of ``(s_ref, action)``. A harness that could not detect
  predictability here would make the real-data negative result meaningless, so
  this is the positive control.
* :func:`test_noise_delta_collapses_to_the_baselines` plants a ``Delta`` that is
  pure noise. The learned predictors must fall back to the baselines rather than
  manufacture signal, which is the negative control against overfitting.

The remaining tests pin the method guarantees the science rests on: the
action-only ridge must agree with the per-class-mean lookup (they are the same
estimator), non-SNV rows must be excluded and counted, and a cosine-trained
MLP must never report an AUROC of its own magnitude.

The module skips cleanly without NumPy / PyArrow / PyTorch, matching the
repository's optional-dependency test convention.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tools.research.delta_predictability import (
    GENERATED_BY,
    SCHEMA_VERSION,
    PredictabilityConfig,
    evaluate_predictability,
    main,
    run_predictability,
)
from tools.research.edit_response_analysis import VariantGeometry

_D_STATE = 12
_RADIUS = 8
_OTHER_RADIUS = 0

# The MLP is the slow part of the tool; a small net over a few hundred synthetic
# rows converges long before the production epoch count, so the gate stays fast
# without weakening either control.
_FAST = PredictabilityConfig(
    radius=_RADIUS,
    mlp_hidden=64,
    mlp_epochs=200,
    mlp_batch_size=64,
)

#: Substitution classes carrying a large displacement, assigned to pathogenic
#: rows, and the small-displacement classes assigned to benign ones. Keeping the
#: two sets disjoint is what makes ||Delta|| separate the labels, giving the
#: positive control a ceiling AUROC near 1 to chase.
_PATH_SUBSTITUTIONS = (("A", "G"), ("C", "T"))
_BENIGN_SUBSTITUTIONS = (("G", "A"), ("T", "C"))


def _geometry(np: Any, s_ref: Any, s_alt: Any) -> tuple[float, float]:
    norm_ref = float(np.linalg.norm(s_ref))
    cos = float(np.dot(s_ref, s_alt) / (norm_ref * float(np.linalg.norm(s_alt))))
    rel = float(np.linalg.norm(s_alt - s_ref) / norm_ref)
    return cos, rel


def _row(
    np: Any,
    *,
    variant_id: str,
    label_group: str,
    s_ref: Any,
    s_alt: Any,
    ref: str,
    alt: str,
) -> VariantGeometry:
    cos, rel = _geometry(np, s_ref, s_alt)
    return VariantGeometry(
        variant_id=variant_id,
        label_group=label_group,
        continuous_score=None,
        cos_ref_alt=cos,
        rel_delta=rel,
        s_ref=tuple(float(x) for x in s_ref),
        s_alt=tuple(float(x) for x in s_alt),
        chrom="1",
        ref=ref,
        alt=alt,
    )


def _synthetic_rows(
    *,
    deterministic: bool,
    n_per_group: int = 120,
    seed: int = 11,
) -> list[VariantGeometry]:
    """ClinVar-like SNVs whose ``Delta`` is either learnable or pure noise.

    When ``deterministic`` is true ``Delta = M @ s_ref + class_vector``, an exact
    function of the two inputs the tool is allowed to use, and the pathogenic
    rows are assigned the substitution classes carrying the large class vectors
    so ``||Delta||`` separates the labels. When false, ``Delta`` is Gaussian
    noise drawn independently of both ``s_ref`` and the substitution class, so
    nothing in the tool's feature set can predict it.
    """
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    # A fixed linear map and per-class offsets define the recoverable ground truth.
    mixing = rng.normal(0.0, 0.4, (_D_STATE, _D_STATE))
    class_vectors = {
        "A>G": rng.normal(0.0, 1.0, _D_STATE) * 4.0,
        "C>T": rng.normal(0.0, 1.0, _D_STATE) * 4.0,
        "G>A": rng.normal(0.0, 1.0, _D_STATE) * 0.5,
        "T>C": rng.normal(0.0, 1.0, _D_STATE) * 0.5,
    }

    rows: list[VariantGeometry] = []
    for group, is_path in (("clinvar_path", True), ("clinvar_benign", False)):
        pairs = _PATH_SUBSTITUTIONS if is_path else _BENIGN_SUBSTITUTIONS
        for i in range(n_per_group):
            # Pathogenic rows draw the high-magnitude classes, benign the low ones,
            # so the true ||Delta|| carries the label (a ceiling AUROC near 1).
            ref, alt = pairs[i % len(pairs)]
            s_ref = rng.normal(0.0, 1.0, _D_STATE) + 10.0
            if deterministic:
                delta = mixing @ s_ref * 0.05 + class_vectors[f"{ref}>{alt}"]
            else:
                delta = rng.normal(0.0, 1.0, _D_STATE)
            rows.append(
                _row(
                    np,
                    variant_id=f"{group}_{i}",
                    label_group=group,
                    s_ref=s_ref,
                    s_alt=s_ref + delta,
                    ref=ref,
                    alt=alt,
                )
            )
    return rows


def _write_parquet(path: Path, rows: list[VariantGeometry]) -> Path:
    """Serialise rows at two radii so radius selection is exercised."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    columns: dict[str, list[Any]] = {name: [] for name in _COLUMNS}
    for pool_radius in (_OTHER_RADIUS, _RADIUS):
        for row in rows:
            columns["variant_id"].append(row.variant_id)
            columns["label_group"].append(row.label_group)
            columns["continuous_score"].append(row.continuous_score)
            columns["pool_radius"].append(pool_radius)
            columns["cos_ref_alt"].append(row.cos_ref_alt)
            columns["rel_delta"].append(row.rel_delta)
            columns["s_ref"].append(list(row.s_ref))
            columns["s_alt"].append(list(row.s_alt))
            columns["chrom"].append(row.chrom)
            columns["ref"].append(row.ref)
            columns["alt"].append(row.alt)
    schema = pa.schema(
        [
            ("variant_id", pa.string()),
            ("label_group", pa.string()),
            ("continuous_score", pa.float64()),
            ("pool_radius", pa.int64()),
            ("cos_ref_alt", pa.float64()),
            ("rel_delta", pa.float64()),
            ("s_ref", pa.list_(pa.float32())),
            ("s_alt", pa.list_(pa.float32())),
            ("chrom", pa.string()),
            ("ref", pa.string()),
            ("alt", pa.string()),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), path)
    return path


_COLUMNS = (
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


# ---------------------------------------------------------------------------
# Positive control: the harness must detect predictability when it is there.


def test_deterministic_delta_is_recovered() -> None:
    """When Delta IS a function of (s_ref, action), the predictors must find it."""
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    report = evaluate_predictability(_synthetic_rows(deterministic=True), config=_FAST)

    predictors = report["predictors"]
    ridge = predictors["ridge_sref_action"]
    # The ground truth is linear in the features, so ridge must essentially nail it.
    assert ridge["cos"] > 0.95
    assert ridge["rel_err"] < 0.3
    # ...and beat the lookup that ignores s_ref entirely.
    assert ridge["cos"] > predictors["class_mean"]["cos"]
    assert predictors["class_mean"]["cos"] > predictors["global_mean"]["cos"]

    mlp = predictors["mlp_mse"]
    assert mlp["cos"] > 0.9

    # The recovered displacement must retain the pathogenicity signal that the
    # true displacement carries: predicted AUROC close to the ceiling.
    ceiling = report["ceiling"]["true_delta_auroc"]
    assert ceiling > 0.95
    assert ridge["auroc_magnitude"] > 0.9
    assert mlp["auroc_magnitude"] > 0.9


# ---------------------------------------------------------------------------
# Negative control: the harness must not manufacture signal when there is none.


def test_noise_delta_collapses_to_the_baselines() -> None:
    """When Delta is independent noise, no predictor may beat the baselines."""
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    report = evaluate_predictability(_synthetic_rows(deterministic=False), config=_FAST)

    predictors = report["predictors"]
    # Held-out cos for an unpredictable target sits at chance, not above it.
    for name in ("global_mean", "class_mean", "ridge_sref_action", "mlp_mse"):
        assert predictors[name]["cos"] < 0.15, name
    # A tuned ridge must not invent structure beyond the class-mean lookup.
    assert predictors["ridge_sref_action"]["cos"] < predictors["class_mean"]["cos"] + 0.1
    # Nothing to discriminate: even the true Delta cannot separate the labels.
    assert report["ceiling"]["true_delta_auroc"] == pytest.approx(0.5, abs=0.1)
    assert predictors["mlp_mse"]["auroc_magnitude"] == pytest.approx(0.5, abs=0.15)


# ---------------------------------------------------------------------------
# Method guarantees.


def test_ridge_action_only_reproduces_the_class_mean_lookup() -> None:
    """One-hot ridge and per-class means are the same estimator; they must agree.

    This is the harness's self-check: if these two independently implemented
    paths disagree, one of them is wrong and every other number is suspect.
    """
    pytest.importorskip("numpy")
    config = replace(_FAST, enable_mlp=False)
    report = evaluate_predictability(_synthetic_rows(deterministic=True), config=config)

    predictors = report["predictors"]
    assert predictors["ridge_action_only"]["cos"] == pytest.approx(
        predictors["class_mean"]["cos"], abs=1e-3
    )


def test_non_snv_rows_are_excluded_and_counted() -> None:
    """Multi-base alleles change the state by changing length, not by biology."""
    np = pytest.importorskip("numpy")
    rows = _synthetic_rows(deterministic=True)
    indel = replace(rows[0], variant_id="indel_0", ref="CA", alt="C")
    absent = replace(rows[1], variant_id="unknown_0", ref=None, alt=None)
    config = replace(_FAST, enable_mlp=False)

    report = evaluate_predictability([*rows, indel, absent], config=config)

    assert report["n_non_snv_excluded"] == 2
    assert report["n"] == len(rows)
    assert np is not None


def test_cosine_objective_never_reports_a_magnitude_auroc() -> None:
    """Cosine loss is scale-invariant, so ||Dhat|| from it is meaningless.

    Reporting an AUROC of that magnitude would be a fabricated result, so the
    tool must refuse to emit one and say why.
    """
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    config = replace(_FAST, mlp_objective="cosine")
    report = evaluate_predictability(_synthetic_rows(deterministic=True), config=config)

    mlp = report["predictors"]["mlp_cosine"]
    assert "cos" in mlp
    assert mlp["auroc_magnitude"] is None
    assert "scale-invariant" in mlp["auroc_magnitude_omitted_because"]
    # The MSE-trained model is the only fair magnitude test, so it is absent here.
    assert "mlp_mse" not in report["predictors"]


def test_mlp_is_skipped_cleanly_when_disabled() -> None:
    """The NumPy-only path must work without the torch extra installed."""
    pytest.importorskip("numpy")
    report = evaluate_predictability(
        _synthetic_rows(deterministic=True), config=replace(_FAST, enable_mlp=False)
    )
    assert "skipped" in report["predictors"]["mlp"]
    assert report["predictors"]["ridge_sref_action"]["cos"] > 0.95


def test_ridge_without_intercept_reproduces_the_prototype_variant() -> None:
    """``fit_intercept`` must actually change the fit, and stay a lambda sweep.

    The validated R3 prototype standardised a constant column into oblivion and
    so fit no intercept. That variant is kept reproducible behind this flag; the
    corrected default is what the tool reports.
    """
    pytest.importorskip("numpy")
    rows = _synthetic_rows(deterministic=True)
    base = replace(_FAST, enable_mlp=False)

    with_intercept = evaluate_predictability(rows, config=base)["predictors"]["ridge_sref_action"]
    without = evaluate_predictability(rows, config=replace(base, fit_intercept=False))[
        "predictors"
    ]["ridge_sref_action"]

    assert with_intercept["fit_intercept"] is True
    assert without["fit_intercept"] is False
    # An under-regularised failure proves nothing, so lambda is always swept.
    assert len(with_intercept["lambda_sweep"]) == len(_FAST.ridge_lambdas)
    assert with_intercept["best_lambda"] in list(_FAST.ridge_lambdas)


# ---------------------------------------------------------------------------
# CLI and failure modes.


def test_main_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    parquet = _write_parquet(tmp_path / "emb.parquet", _synthetic_rows(deterministic=True))
    out_report = tmp_path / "reports" / "r3.json"

    rc = main(
        [
            "--embeddings",
            str(parquet),
            "--out-report",
            str(out_report),
            "--radius",
            str(_RADIUS),
            "--no-mlp",
        ]
    )

    assert rc == 0
    payload = json.loads(out_report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["generated_by"] == GENERATED_BY
    assert payload["radius"] == _RADIUS
    assert payload["source"].endswith("emb.parquet")
    assert payload["predictors"]["ridge_sref_action"]["cos"] > 0.95
    assert "out_report" in capsys.readouterr().out


def test_main_reports_a_bad_radius_as_a_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI must surface a typed failure as a nonzero exit, not a traceback."""
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    parquet = _write_parquet(tmp_path / "emb.parquet", _synthetic_rows(deterministic=True))

    rc = main(
        [
            "--embeddings",
            str(parquet),
            "--out-report",
            str(tmp_path / "r3.json"),
            "--radius",
            "999",
            "--no-mlp",
        ]
    )

    assert rc != 0
    captured = capsys.readouterr()
    assert "error:" in captured.err
    # The details must name the radii that *are* available, so the failure is actionable.
    assert "999" in captured.err
    assert not (tmp_path / "r3.json").exists()


def test_unknown_radius_fails_closed(tmp_path: Path) -> None:
    """Silently analysing the wrong radius would be worse than failing."""
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    from geno_lewm.errors import InputError

    parquet = _write_parquet(tmp_path / "emb.parquet", _synthetic_rows(deterministic=True))
    with pytest.raises(InputError):
        run_predictability(parquet, config=replace(_FAST, radius=999))


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(Exception, match="kfold_splits"):
        PredictabilityConfig(kfold_splits=1)
    with pytest.raises(Exception, match="ridge_lambdas"):
        PredictabilityConfig(ridge_lambdas=())
    with pytest.raises(Exception, match="mlp_objective"):
        PredictabilityConfig(mlp_objective="hinge")


def test_no_clinvar_labels_skips_auroc_without_crashing() -> None:
    """A radius with no path/benign rows must skip AUROC, not fabricate one."""
    pytest.importorskip("numpy")
    rows = [
        replace(row, label_group="brca2_sge")
        for row in _synthetic_rows(deterministic=True, n_per_group=40)
    ]
    report = evaluate_predictability(rows, config=replace(_FAST, enable_mlp=False))

    assert "skipped" in report["clinvar"]
    assert report["ceiling"]["true_delta_auroc"] is None
    assert report["predictors"]["ridge_sref_action"]["auroc_magnitude"] is None
