"""Unit tests for surprise-scoring contract calibration table building."""

from __future__ import annotations

from pathlib import Path

import pytest

from geno_lewm.errors import InputError, SchemaCompatError
from geno_lewm.surprise import (
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_CDF_POINTS,
    CalibrationBucket,
    CalibrationExample,
    build_calibration_table,
    read_calibration_table,
    write_calibration_table,
)


def test_builder_is_deterministic_given_seed_and_input_order() -> None:
    examples = _examples("coding_missense|mid|none", 50)

    first = build_calibration_table(
        examples,
        seed=17,
        per_bucket_sample=10,
        grid_size=11,
        warn_sparse=False,
    )
    second = build_calibration_table(
        reversed(examples),
        seed=17,
        per_bucket_sample=10,
        grid_size=11,
        warn_sparse=False,
    )

    assert first.buckets == second.buckets
    assert first.schema_version == CALIBRATION_SCHEMA_VERSION


def test_builder_samples_buckets_and_records_parent_backoff() -> None:
    examples = (
        CalibrationExample("enhancer|high|simple", 0.0),
        CalibrationExample("enhancer|high|simple", 1.0),
        CalibrationExample("enhancer|high|none", 2.0),
        CalibrationExample("enhancer|high|none", 3.0),
    )

    table = build_calibration_table(
        examples,
        seed=1,
        grid_size=5,
        min_bucket_size=3,
        low_confidence_size=1,
        warn_sparse=False,
    )
    by_id = {bucket.bucket_id: bucket for bucket in table.buckets}

    assert by_id["enhancer|high|simple"].n_calibration == 2
    assert by_id["enhancer|high|simple"].back_off_to == "enhancer|high"
    assert by_id["enhancer|high"].n_calibration == 4
    assert by_id["enhancer|high"].back_off_to is None
    assert by_id["*"].n_calibration == 4
    assert table.warnings == ()


def test_builder_warns_when_backoff_remains_sparse() -> None:
    examples = (
        CalibrationExample("splice|low|segmental_dup", 0.25),
        CalibrationExample("splice|low|segmental_dup", 0.75),
    )

    with pytest.warns(RuntimeWarning, match="calibration bucket remains sparse after backoff"):
        table = build_calibration_table(
            examples,
            seed=0,
            grid_size=5,
            min_bucket_size=5,
            low_confidence_size=3,
        )

    assert len(table.warnings) == 1
    warning = table.warnings[0]
    assert warning.bucket_id == "splice|low|segmental_dup"
    assert warning.resolved_bucket_id == "*"
    assert warning.n_calibration == 2
    assert warning.low_confidence is True


def test_bucket_confidence_indicators_are_derived_from_row_count() -> None:
    low = CalibrationBucket(
        bucket_id="intergenic|low|none",
        n_calibration=50,
        cdf=(0.0, 1.0),
        sigma_grid=(0.0, 1.0),
    )
    high = CalibrationBucket(
        bucket_id="intergenic|low|none",
        n_calibration=2_000,
        cdf=(0.0, 1.0),
        sigma_grid=(0.0, 1.0),
    )

    assert low.confidence == 0.05
    assert low.low_confidence is True
    assert high.confidence == 1.0
    assert high.low_confidence is False


def test_parquet_round_trip_matches_documented_schema(tmp_path: Path) -> None:
    table = build_calibration_table(
        _examples("coding_synonymous|mid|none", 6),
        seed=2,
        grid_size=7,
        warn_sparse=False,
    )
    path = write_calibration_table(table, tmp_path / "calibration.parquet")

    restored = read_calibration_table(path)

    assert path.is_file()
    assert [bucket.bucket_id for bucket in restored.buckets] == [
        bucket.bucket_id for bucket in table.buckets
    ]
    assert [bucket.n_calibration for bucket in restored.buckets] == [
        bucket.n_calibration for bucket in table.buckets
    ]
    assert all(len(bucket.cdf) == 7 for bucket in restored.buckets)
    assert all(len(bucket.sigma_grid) == 7 for bucket in restored.buckets)

    import pyarrow as pa
    import pyarrow.parquet as pq

    arrow_table = pq.read_table(path)
    assert tuple(arrow_table.column_names) == (
        "bucket_id",
        "n_calibration",
        "cdf",
        "sigma_grid",
        "back_off_to",
        "schema_version",
    )
    assert arrow_table.schema.field("n_calibration").type == pa.int64()
    assert arrow_table.schema.field("cdf").type.value_type == pa.float32()
    assert arrow_table.schema.field("sigma_grid").type.value_type == pa.float32()


def test_read_rejects_parquet_with_wrong_schema(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pydict({"bucket_id": ["*"]}), path)

    with pytest.raises(SchemaCompatError):
        read_calibration_table(path)


def test_validation_rejects_invalid_examples_and_options() -> None:
    with pytest.raises(InputError):
        CalibrationExample("coding_missense|mid|none", -1.0)
    with pytest.raises(InputError):
        CalibrationExample("coding_missense|middle|none", 1.0)
    with pytest.raises(InputError):
        build_calibration_table([], seed=0)
    with pytest.raises(InputError):
        build_calibration_table(_examples("coding_missense|mid|none", 1), grid_size=1)
    with pytest.raises(InputError):
        build_calibration_table(
            _examples("coding_missense|mid|none", 1),
            min_bucket_size=5,
            low_confidence_size=6,
        )


def test_table_resolves_sparse_bucket_to_parent() -> None:
    table = build_calibration_table(
        (
            CalibrationExample("promoter|mid|none", 0.0),
            CalibrationExample("promoter|mid|none", 1.0),
            CalibrationExample("promoter|mid|simple", 2.0),
            CalibrationExample("promoter|mid|simple", 3.0),
        ),
        min_bucket_size=3,
        low_confidence_size=1,
        warn_sparse=False,
    )

    resolved = table.resolve("promoter|mid|none", min_bucket_size=3)

    assert resolved.bucket_id == "promoter|mid"
    assert resolved.n_calibration == 4


def test_default_grid_size_matches_rfc_contract() -> None:
    table = build_calibration_table(
        _examples("intergenic|mid|none", 2),
        seed=0,
        warn_sparse=False,
    )

    assert {len(bucket.cdf) for bucket in table.buckets} == {DEFAULT_CDF_POINTS}
    assert {len(bucket.sigma_grid) for bucket in table.buckets} == {DEFAULT_CDF_POINTS}


def _examples(bucket_id: str, count: int) -> tuple[CalibrationExample, ...]:
    return tuple(CalibrationExample(bucket_id, float(idx)) for idx in range(count))
