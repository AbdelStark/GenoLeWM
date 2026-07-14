"""Tests for the R1 edit-response-geometry analysis tool.

These build a small synthetic embeddings Parquet with a *planted* signal:
ClinVar pathogenic/benign variants share the same displacement magnitude but
differ along a hidden direction, and BRCA2 variants have a displacement whose
magnitude grows monotonically with the functional score. The tool must then
recover the directional signal (directional-CV AUROC well above the magnitude
AUROC) and the monotone BRCA2 relation (positive, significant Spearman), while
skipping label groups that are absent instead of crashing.

The whole module is skipped cleanly when NumPy / PyArrow are unavailable,
matching the repository's optional-dependency test convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tools.research.edit_response_analysis import (
    GENERATED_BY,
    SCHEMA_VERSION,
    AnalysisConfig,
    VariantGeometry,
    analyze_radius,
    main,
    run_analysis,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

_D_STATE = 16
_REF_AXIS = 0
_DIR_AXIS = 1
_N_PER_GROUP = 60
_RADII = (0, 64)

# Reduced resampling keeps the gate fast; the planted signal is strong enough
# that the point estimates are unambiguous at these resample counts.
_FAST_CONFIG = AnalysisConfig(
    bootstrap_resamples=200,
    spearman_permutations=300,
    min_brca2=30,
)


def _reference_vector(np: Any) -> Any:
    r = np.zeros(_D_STATE, dtype=np.float64)
    r[_REF_AXIS] = 100.0
    return r


def _geometry(np: Any, s_ref: Any, s_alt: Any) -> tuple[float, float]:
    norm_ref = float(np.linalg.norm(s_ref))
    norm_alt = float(np.linalg.norm(s_alt))
    cos = float(np.dot(s_ref, s_alt) / (norm_ref * norm_alt))
    rel = float(np.linalg.norm(s_alt - s_ref) / norm_ref)
    return cos, rel


def _synth_variants(seed: int = 20240714) -> list[dict[str, Any]]:
    """Build per-variant (s_ref, s_alt) pairs with a planted signal."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed)
    r = _reference_vector(np)
    variants: list[dict[str, Any]] = []

    for group, sign in (("clinvar_path", 1.0), ("clinvar_benign", -1.0)):
        for i in range(_N_PER_GROUP):
            noise = rng.normal(0.0, 1.0, _D_STATE)
            noise[_REF_AXIS] = 0.0
            noise[_DIR_AXIS] = 0.0
            delta = noise.copy()
            # Hidden direction: path and benign are offset in opposite ways so
            # their magnitudes match but their projection separates them.
            delta[_DIR_AXIS] = sign * 1.5 + rng.normal(0.0, 0.3)
            s_alt = r + delta
            cos, rel = _geometry(np, r, s_alt)
            variants.append(
                {
                    "variant_id": f"{group}_{i}",
                    "label_group": group,
                    "continuous_score": None,
                    "cos_ref_alt": cos,
                    "rel_delta": rel,
                    "s_ref": [float(x) for x in r],
                    "s_alt": [float(x) for x in s_alt],
                }
            )

    for i in range(_N_PER_GROUP):
        t = (i + 1) / _N_PER_GROUP
        magnitude = 0.5 + 3.0 * t
        delta = rng.normal(0.0, 0.1, _D_STATE)
        delta[_REF_AXIS] = 0.0
        delta[_DIR_AXIS] = magnitude
        s_alt = r + delta
        cos, rel = _geometry(np, r, s_alt)
        variants.append(
            {
                "variant_id": f"brca2_{i}",
                "label_group": "brca2_sge",
                "continuous_score": t,
                "cos_ref_alt": cos,
                "rel_delta": rel,
                "s_ref": [float(x) for x in r],
                "s_alt": [float(x) for x in s_alt],
            }
        )

    return variants


def _write_parquet(path: Path, variants: list[dict[str, Any]]) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    columns: dict[str, list[Any]] = {
        "variant_id": [],
        "label_group": [],
        "continuous_score": [],
        "pool_radius": [],
        "cos_ref_alt": [],
        "rel_delta": [],
        "s_ref": [],
        "s_alt": [],
    }
    for pool_radius in _RADII:
        for variant in variants:
            columns["variant_id"].append(variant["variant_id"])
            columns["label_group"].append(variant["label_group"])
            columns["continuous_score"].append(variant["continuous_score"])
            columns["pool_radius"].append(pool_radius)
            columns["cos_ref_alt"].append(variant["cos_ref_alt"])
            columns["rel_delta"].append(variant["rel_delta"])
            columns["s_ref"].append(variant["s_ref"])
            columns["s_alt"].append(variant["s_alt"])
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
        ]
    )
    table = pa.table(columns, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _variant_geometry_rows() -> list[VariantGeometry]:
    np = pytest.importorskip("numpy")
    rows: list[VariantGeometry] = []
    for variant in _synth_variants():
        s_ref = tuple(float(x) for x in variant["s_ref"])
        s_alt = tuple(float(x) for x in variant["s_alt"])
        rows.append(
            VariantGeometry(
                variant_id=variant["variant_id"],
                label_group=variant["label_group"],
                continuous_score=variant["continuous_score"],
                cos_ref_alt=float(variant["cos_ref_alt"]),
                rel_delta=float(variant["rel_delta"]),
                s_ref=s_ref,
                s_alt=s_alt,
            )
        )
    assert np is not None
    return rows


def test_directional_cv_beats_magnitude_and_recovers_planted_signal(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    parquet = _write_parquet(tmp_path / "emb.parquet", _synth_variants())

    report = run_analysis(parquet, config=_FAST_CONFIG)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["generated_by"] == GENERATED_BY
    assert set(report["by_radius"]) == {"0", "64"}

    for radius in _RADII:
        radius_report = report["by_radius"][str(radius)]
        assert radius_report["n"] == 3 * _N_PER_GROUP

        # Per-label_group sensitivity aggregation.
        sensitivity = radius_report["sensitivity"]
        assert sensitivity["clinvar_path"]["n"] == _N_PER_GROUP
        assert sensitivity["clinvar_benign"]["n"] == _N_PER_GROUP
        assert sensitivity["brca2_sge"]["n"] == _N_PER_GROUP
        assert set(sensitivity["clinvar_path"]["cos_ref_alt"]) == {"mean", "median", "std"}

        clinvar = radius_report["clinvar_path_vs_benign"]
        mag = clinvar["magnitude_norm"]["auroc"]
        directional = clinvar["directional_norm_cv"]["auroc"]
        # The planted signal is directional: displacement magnitude barely
        # separates the classes, but the learned axis nearly perfectly does.
        assert directional > mag
        assert directional > 0.8
        assert mag < 0.75
        for key in ("magnitude_raw", "magnitude_norm", "directional_norm_cv"):
            ci = clinvar[key]["ci95"]
            assert len(ci) == 2
            assert ci[0] <= ci[1]

        brca2 = radius_report["brca2"]
        assert brca2["spearman_mag_vs_func"]["rho"] > 0.7
        assert brca2["spearman_mag_vs_func"]["p"] < 0.05
        # The ClinVar-learned direction transfers to BRCA2 monotonically too.
        assert brca2["spearman_clinvar_dir_transfer_vs_func"]["rho"] > 0.5

        pca = radius_report["pca_delta"]
        assert abs(pca["explained_variance_ratio_sum"] - 1.0) < 1e-6
        assert len(pca["explained_variance_ratio_top10"]) <= 10
        assert 1 <= pca["n_components_50pct"] <= pca["n_components_90pct"]


def test_main_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    parquet = _write_parquet(tmp_path / "emb.parquet", _synth_variants())
    out_report = tmp_path / "reports" / "r1.json"

    rc = main(
        [
            "--embeddings",
            str(parquet),
            "--out-report",
            str(out_report),
            "--bootstrap-resamples",
            "200",
            "--spearman-permutations",
            "300",
        ]
    )

    assert rc == 0
    assert out_report.is_file()
    payload = json.loads(out_report.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["source"].endswith("emb.parquet")
    assert set(payload["by_radius"]) == {"0", "64"}
    captured = capsys.readouterr()
    assert "out_report" in captured.out


def test_absent_label_group_skips_without_crashing() -> None:
    pytest.importorskip("numpy")
    rows = _variant_geometry_rows()

    brca2_only = [row for row in rows if row.label_group == "brca2_sge"]
    result = analyze_radius(brca2_only, config=_FAST_CONFIG)
    assert "skipped" in result["clinvar_path_vs_benign"]
    assert result["brca2"]["spearman_mag_vs_func"]["rho"] > 0.7

    clinvar_only = [row for row in rows if row.label_group != "brca2_sge"]
    result = analyze_radius(clinvar_only, config=_FAST_CONFIG)
    assert "skipped" in result["brca2"]
    assert "auroc" in result["clinvar_path_vs_benign"]["directional_norm_cv"]

    path_only = [row for row in rows if row.label_group == "clinvar_path"]
    result = analyze_radius(path_only, config=_FAST_CONFIG)
    assert "skipped" in result["clinvar_path_vs_benign"]
    assert "skipped" in result["brca2"]
    # PCA is still defined for a single populated group.
    assert abs(result["pca_delta"]["explained_variance_ratio_sum"] - 1.0) < 1e-6


def test_missing_required_column_fails_closed(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from geno_lewm.errors import InputError

    table = pa.table({"variant_id": ["v0"], "pool_radius": [0]})
    bad = tmp_path / "bad.parquet"
    pq.write_table(table, bad)

    with pytest.raises(InputError):
        run_analysis(bad, config=_FAST_CONFIG)
