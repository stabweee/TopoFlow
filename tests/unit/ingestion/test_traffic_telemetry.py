"""Clean-room tests for 5-minute traffic telemetry validation, aggregation and storage.

Covers the pydantic ``TelemetryRecord`` contract (no fabricated defaults), row-level
partitioning into valid/rejected frames with ordered rejection reasons, strict and
lenient validation, volume-weighted harmonic speed aggregation, and the append-only
Hive-partitioned Parquet lakehouse (write, read with de-duplication, end-to-end pipeline).
All data is synthetic and built in-test or in ``tmp_path``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from topoflow.ingestion.schemas import (
    TELEMETRY_OPTIONAL_COLUMNS,
    TELEMETRY_POLARS_SCHEMA,
    TELEMETRY_REQUIRED_COLUMNS,
    TelemetryRecord,
)
from topoflow.ingestion.traffic_telemetry import (
    TelemetryValidationError,
    aggregate_telemetry_5min,
    ingest_telemetry_pipeline,
    partition_telemetry,
    read_partitioned_telemetry,
    read_raw_telemetry,
    validate_telemetry,
    write_partitioned_telemetry,
)

SEG = "alb_i495_inner_loop_s0001"
T0 = datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC)


def _record_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "segment_id": "alb_i495_inner_loop_s0038",
        "corridor": "alb_i495",
        "direction": "inner_loop",
        "timestamp_utc": T0,
        "speed_mph": 55.0,
    }
    base.update(overrides)
    return base


def _frame(n: int = 1, **overrides: list) -> pl.DataFrame:
    """Build a fully valid synthetic telemetry frame of ``n`` rows, with column overrides."""
    data: dict[str, list] = {
        "segment_id": [SEG] * n,
        "corridor": ["alb_i495"] * n,
        "direction": ["inner_loop"] * n,
        "timestamp_utc": [T0 + timedelta(minutes=5 * i) for i in range(n)],
        "speed_mph": [50.0] * n,
        "free_flow_speed_mph": [65.0] * n,
        "volume_vph": [100.0] * n,
        "occupancy_pct": [10.0] * n,
        "confidence_score": [0.9] * n,
    }
    data.update(overrides)
    return pl.DataFrame(data, strict=False)


def _parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet")) if root.exists() else []


# =============================================================================
# Schema constants and TelemetryRecord
# =============================================================================


class TestSchemaConstants:
    def test_no_fabricated_free_flow_default_constant(self) -> None:
        import topoflow.constants as constants

        assert not hasattr(constants, "DEFAULT_FREE_FLOW_SPEED_MPH")

    def test_required_and_optional_columns(self) -> None:
        assert TELEMETRY_REQUIRED_COLUMNS == (
            "segment_id",
            "corridor",
            "direction",
            "timestamp_utc",
            "speed_mph",
        )
        assert TELEMETRY_OPTIONAL_COLUMNS == (
            "free_flow_speed_mph",
            "volume_vph",
            "occupancy_pct",
            "confidence_score",
        )

    def test_polars_schema_unchanged(self) -> None:
        assert list(TELEMETRY_POLARS_SCHEMA.keys()) == [
            *TELEMETRY_REQUIRED_COLUMNS,
            *TELEMETRY_OPTIONAL_COLUMNS,
        ]
        assert TELEMETRY_POLARS_SCHEMA["timestamp_utc"] == pl.Datetime("ms", "UTC")
        assert isinstance(TELEMETRY_POLARS_SCHEMA["corridor"], pl.Enum)
        assert isinstance(TELEMETRY_POLARS_SCHEMA["direction"], pl.Enum)
        assert set(TELEMETRY_POLARS_SCHEMA["corridor"].categories.to_list()) == {
            "alb_i495",
            "i95_lorton_woodbridge",
            "i295_anacostia",
        }
        assert set(TELEMETRY_POLARS_SCHEMA["direction"].categories.to_list()) == {
            "inner_loop",
            "outer_loop",
            "northbound",
            "southbound",
        }


class TestTelemetrySchema:
    def test_optional_fields_default_to_none(self) -> None:
        """No fabricated defaults (65 mph / 0 vph / confidence 1) may be injected."""
        record = TelemetryRecord(**_record_kwargs())
        assert record.free_flow_speed_mph is None
        assert record.volume_vph is None
        assert record.occupancy_pct is None
        assert record.confidence_score is None

    def test_optional_fields_accept_explicit_none(self) -> None:
        record = TelemetryRecord(
            **_record_kwargs(
                free_flow_speed_mph=None,
                volume_vph=None,
                occupancy_pct=None,
                confidence_score=None,
            )
        )
        assert record.volume_vph is None

    def test_fields_accept_boundaries(self) -> None:
        low = TelemetryRecord(
            **_record_kwargs(
                speed_mph=0.0,
                free_flow_speed_mph=30.0,
                volume_vph=0.0,
                occupancy_pct=0.0,
                confidence_score=0.0,
            )
        )
        high = TelemetryRecord(
            **_record_kwargs(
                speed_mph=120.0,
                free_flow_speed_mph=90.0,
                volume_vph=20000.0,
                occupancy_pct=100.0,
                confidence_score=1.0,
            )
        )
        assert low.speed_mph == 0.0
        assert high.volume_vph == 20000.0

    def test_negative_speed(self) -> None:
        with pytest.raises(ValidationError):
            TelemetryRecord(**_record_kwargs(speed_mph=-10.0))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("speed_mph", 150.0),
            ("free_flow_speed_mph", 29.9),
            ("free_flow_speed_mph", 90.1),
            ("volume_vph", -1.0),
            ("volume_vph", 20000.1),
            ("occupancy_pct", 105.0),
            ("occupancy_pct", -0.1),
            ("confidence_score", 1.5),
            ("confidence_score", -0.01),
            ("segment_id", ""),
        ],
    )
    def test_out_of_bounds_fields(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            TelemetryRecord(**_record_kwargs(**{field: value}))

    @pytest.mark.parametrize(
        "missing", ["segment_id", "corridor", "direction", "timestamp_utc", "speed_mph"]
    )
    def test_required_fields(self, missing: str) -> None:
        kwargs = _record_kwargs()
        kwargs.pop(missing)
        with pytest.raises(ValidationError):
            TelemetryRecord(**kwargs)

    def test_invalid_corridor_and_direction(self) -> None:
        with pytest.raises(ValidationError):
            TelemetryRecord(**_record_kwargs(corridor="i66"))
        with pytest.raises(ValidationError):
            TelemetryRecord(**_record_kwargs(direction="eastbound"))

    def test_forbid_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TelemetryRecord(**_record_kwargs(unknown_sensor_field="noise"))

    def test_utc_timezone_correction(self) -> None:
        naive = TelemetryRecord(**_record_kwargs(timestamp_utc=datetime(2024, 10, 15, 8, 0, 0)))
        assert naive.timestamp_utc.utcoffset() == timedelta(0)
        assert naive.timestamp_utc.hour == 8

        aware = TelemetryRecord(
            **_record_kwargs(
                timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=timezone(timedelta(hours=-4)))
            )
        )
        assert aware.timestamp_utc.utcoffset() == timedelta(0)
        assert aware.timestamp_utc.hour == 12  # 8 AM EDT is 12 PM UTC


# =============================================================================
# partition_telemetry
# =============================================================================


class TestPartitionTelemetry:
    def test_all_valid_rows(self) -> None:
        valid, rejected = partition_telemetry(_frame(3))
        assert len(valid) == 3
        assert len(rejected) == 0
        assert valid.schema == TELEMETRY_POLARS_SCHEMA

    def test_empty_input(self) -> None:
        valid, rejected = partition_telemetry(_frame(0))
        assert len(valid) == 0
        assert len(rejected) == 0
        assert valid.schema == TELEMETRY_POLARS_SCHEMA

    @pytest.mark.parametrize("missing", list(TELEMETRY_REQUIRED_COLUMNS))
    def test_missing_required_column_raises(self, missing: str) -> None:
        with pytest.raises(ValueError):
            partition_telemetry(_frame(2).drop(missing))

    def test_missing_optional_columns_added_as_null(self) -> None:
        df = _frame(2).drop(list(TELEMETRY_OPTIONAL_COLUMNS))
        valid, rejected = partition_telemetry(df)
        assert len(valid) == 2
        assert len(rejected) == 0
        assert valid.columns == list(TELEMETRY_POLARS_SCHEMA.keys())
        assert valid.schema == TELEMETRY_POLARS_SCHEMA
        for col in TELEMETRY_OPTIONAL_COLUMNS:
            assert valid[col].null_count() == 2

    def test_null_optional_values_are_valid_and_stay_null(self) -> None:
        df = _frame(
            2,
            free_flow_speed_mph=[None, 65.0],
            volume_vph=[None, 100.0],
            occupancy_pct=[None, 10.0],
            confidence_score=[None, 0.5],
        )
        valid, rejected = partition_telemetry(df)
        assert len(valid) == 2
        assert len(rejected) == 0
        first = valid.sort("timestamp_utc").row(0, named=True)
        assert first["free_flow_speed_mph"] is None
        assert first["volume_vph"] is None
        assert first["occupancy_pct"] is None
        assert first["confidence_score"] is None

    def test_extra_columns_dropped_from_valid(self) -> None:
        df = _frame(1).with_columns(pl.lit("x").alias("vendor_note"))
        valid, _ = partition_telemetry(df)
        assert valid.columns == list(TELEMETRY_POLARS_SCHEMA.keys())

    def test_casts_schema_types(self) -> None:
        df = pl.DataFrame(
            {
                "segment_id": [SEG],
                "corridor": ["alb_i495"],
                "direction": ["inner_loop"],
                "timestamp_utc": [T0],
                "speed_mph": [50],
                "free_flow_speed_mph": [65],
                "volume_vph": [100],
                "occupancy_pct": [50],
                "confidence_score": [1],
            }
        )
        valid, _ = partition_telemetry(df)
        assert valid["speed_mph"].dtype == pl.Float64
        assert valid.schema == TELEMETRY_POLARS_SCHEMA

    @pytest.mark.parametrize(
        ("column", "bad_values", "reason"),
        [
            ("speed_mph", [50.0, -0.1], "speed_mph_out_of_range"),
            ("speed_mph", [50.0, 120.1], "speed_mph_out_of_range"),
            ("free_flow_speed_mph", [65.0, 25.0], "free_flow_speed_mph_out_of_range"),
            ("free_flow_speed_mph", [65.0, 95.0], "free_flow_speed_mph_out_of_range"),
            ("volume_vph", [100.0, 25000.0], "volume_vph_out_of_range"),
            ("volume_vph", [100.0, -5.0], "volume_vph_out_of_range"),
            ("occupancy_pct", [10.0, 120.0], "occupancy_pct_out_of_range"),
            ("confidence_score", [0.5, 1.2], "confidence_score_out_of_range"),
        ],
    )
    def test_out_of_range_reasons(self, column: str, bad_values: list, reason: str) -> None:
        valid, rejected = partition_telemetry(_frame(2, **{column: bad_values}))
        assert len(valid) == 1
        assert len(rejected) == 1
        assert "rejection_reason" in rejected.columns
        assert rejected["rejection_reason"].to_list() == [reason]

    @pytest.mark.parametrize("column", list(TELEMETRY_REQUIRED_COLUMNS))
    def test_missing_required_value_reasons(self, column: str) -> None:
        good = _frame(1)[column].to_list()
        valid, rejected = partition_telemetry(_frame(2, **{column: [good[0], None]}))
        assert len(valid) == 1
        assert rejected["rejection_reason"].to_list() == [f"missing_{column}"]

    def test_invalid_timestamp_string(self) -> None:
        df = _frame(2, timestamp_utc=["2024-10-15T08:00:00", "not-a-date"])
        valid, rejected = partition_telemetry(df)
        assert len(valid) == 1
        assert rejected["rejection_reason"].to_list() == ["invalid_timestamp_utc"]
        assert valid["timestamp_utc"].dtype == pl.Datetime("ms", "UTC")

    def test_invalid_corridor_and_direction(self) -> None:
        df = _frame(
            3,
            corridor=["alb_i495", "i66", "alb_i495"],
            direction=["inner_loop", "inner_loop", "eastbound"],
        )
        valid, rejected = partition_telemetry(df)
        assert len(valid) == 1
        assert sorted(rejected["rejection_reason"].to_list()) == [
            "invalid_corridor",
            "invalid_direction",
        ]

    def test_invalid_numeric_string(self) -> None:
        df = _frame(2, speed_mph=["50.0", "fast"])
        valid, rejected = partition_telemetry(df)
        assert len(valid) == 1
        assert rejected["rejection_reason"].to_list() == ["invalid_speed_mph"]

    def test_reason_precedence_invalid_before_missing_before_range(self) -> None:
        df = _frame(
            4,
            segment_id=[SEG, None, None, None],
            # row 1: missing segment + unparseable timestamp -> invalid wins
            timestamp_utc=[
                "2024-10-15T08:00:00",
                "garbage",
                "2024-10-15T08:10:00",
                "2024-10-15T08:15:00",
            ],
            # row 2: missing segment + out-of-range speed -> missing wins
            speed_mph=[50.0, 50.0, 500.0, 50.0],
            # row 3: missing segment + out-of-range volume -> missing wins
            volume_vph=[100.0, 100.0, 100.0, 99999.0],
        )
        valid, rejected = partition_telemetry(df)
        assert len(valid) == 1
        assert sorted(rejected["rejection_reason"].to_list()) == [
            "invalid_timestamp_utc",
            "missing_segment_id",
            "missing_segment_id",
        ]

    def test_invalid_columns_checked_in_schema_order(self) -> None:
        # corridor precedes timestamp_utc in TELEMETRY_POLARS_SCHEMA
        df = _frame(2, corridor=["alb_i495", "i66"], timestamp_utc=["2024-10-15T08:00", "nope"])
        _, rejected = partition_telemetry(df)
        assert rejected["rejection_reason"].to_list() == ["invalid_corridor"]

    def test_range_rules_checked_in_order(self) -> None:
        # speed_mph precedes volume_vph in the range rule order
        df = _frame(2, speed_mph=[50.0, 200.0], volume_vph=[100.0, 99999.0])
        _, rejected = partition_telemetry(df)
        assert rejected["rejection_reason"].to_list() == ["speed_mph_out_of_range"]

    def test_rejected_rows_keep_offending_values(self) -> None:
        df = _frame(2, speed_mph=[50.0, 130.0])
        _, rejected = partition_telemetry(df)
        assert rejected["speed_mph"].cast(pl.Float64).to_list() == [130.0]


# =============================================================================
# validate_telemetry
# =============================================================================


class TestValidateTelemetry:
    def test_missing_columns_raises_error(self) -> None:
        df = pl.DataFrame({"segment_id": ["seg_001"], "speed_mph": [50.0]})
        with pytest.raises(ValueError):
            validate_telemetry(df)

    def test_all_valid_returns_all_rows(self) -> None:
        out = validate_telemetry(_frame(4))
        assert len(out) == 4
        assert out.schema == TELEMETRY_POLARS_SCHEMA

    def test_strict_raises_with_violations_and_rejected(self) -> None:
        df = _frame(
            4,
            segment_id=[SEG, SEG, SEG, None],
            speed_mph=[50.0, -10.0, 130.0, 90.0],
            occupancy_pct=[50.0, 50.0, 120.0, 50.0],
        )
        with pytest.raises(TelemetryValidationError) as excinfo:
            validate_telemetry(df)

        err = excinfo.value
        assert isinstance(err, ValueError)
        assert err.violations == {"speed_mph_out_of_range": 2, "missing_segment_id": 1}
        assert isinstance(err.rejected, pl.DataFrame)
        assert len(err.rejected) == 3
        assert "rejection_reason" in err.rejected.columns

    def test_strict_is_default(self) -> None:
        with pytest.raises(TelemetryValidationError):
            validate_telemetry(_frame(1, speed_mph=[999.0]))

    def test_lenient_returns_only_valid_rows(self) -> None:
        df = _frame(
            4,
            segment_id=[
                "alb_i495_inner_loop_s0001",
                "alb_i495_inner_loop_s0002",
                "alb_i495_inner_loop_s0003",
                None,
            ],
            speed_mph=[50.0, -10.0, 130.0, 90.0],
            occupancy_pct=[50.0, 50.0, 120.0, 50.0],
        )
        clean = validate_telemetry(df, strict=False)
        assert clean.shape[0] == 1
        assert clean["segment_id"][0] == "alb_i495_inner_loop_s0001"
        assert clean.schema == TELEMETRY_POLARS_SCHEMA

    def test_lenient_all_rejected_returns_empty(self) -> None:
        clean = validate_telemetry(_frame(2, speed_mph=[-1.0, 500.0]), strict=False)
        assert len(clean) == 0
        assert clean.schema == TELEMETRY_POLARS_SCHEMA


# =============================================================================
# aggregate_telemetry_5min
# =============================================================================


class TestAggregateTelemetry5Min:
    def test_timestamp_truncation(self) -> None:
        df = _frame(
            3,
            timestamp_utc=[
                datetime(2024, 10, 15, 8, 0, 30, tzinfo=UTC),
                datetime(2024, 10, 15, 8, 2, 45, tzinfo=UTC),
                datetime(2024, 10, 15, 8, 4, 59, tzinfo=UTC),
            ],
            speed_mph=[50.0, 60.0, 70.0],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg.shape[0] == 1
        assert agg["timestamp_utc"][0] == datetime(2024, 10, 15, 8, 0, tzinfo=UTC)
        assert agg["n_raw_records"][0] == 3
        assert agg["n_raw_records"].dtype == pl.UInt32

    def test_epoch_boundary_splits(self) -> None:
        df = _frame(
            2,
            timestamp_utc=[
                datetime(2024, 10, 15, 8, 4, 59, tzinfo=UTC),
                datetime(2024, 10, 15, 8, 5, 0, tzinfo=UTC),
            ],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg["timestamp_utc"].to_list() == [
            datetime(2024, 10, 15, 8, 0, tzinfo=UTC),
            datetime(2024, 10, 15, 8, 5, tzinfo=UTC),
        ]
        assert agg["n_raw_records"].to_list() == [1, 1]

    def test_volume_weighted_harmonic_speed(self) -> None:
        df = _frame(
            2,
            timestamp_utc=[T0, T0 + timedelta(minutes=3)],
            speed_mph=[40.0, 60.0],
            volume_vph=[100.0, 300.0],
        )
        agg = aggregate_telemetry_5min(df)
        expected = 400.0 / (100.0 / 40.0 + 300.0 / 60.0)
        assert agg["speed_mph"][0] == pytest.approx(expected)
        assert agg["speed_mph"][0] == pytest.approx(53.333333, rel=1e-6)
        assert agg["speed_aggregation"][0] == "harmonic_volume_weighted"

    def test_harmonic_ignores_zero_and_null_volume_readings(self) -> None:
        df = _frame(
            3,
            timestamp_utc=[T0, T0 + timedelta(minutes=1), T0 + timedelta(minutes=2)],
            speed_mph=[40.0, 60.0, 80.0],
            volume_vph=[100.0, 0.0, None],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg["speed_mph"][0] == pytest.approx(40.0)
        # A null volume in the epoch marks the harmonic mean as partial.
        assert agg["speed_aggregation"][0] == "harmonic_partial_volume"
        assert agg["n_raw_records"][0] == 3
        assert agg["n_volume_weighted"][0] == 1

    def test_zero_volume_is_not_null_volume(self) -> None:
        """A non-null 0 vph reading is unweighted but does not make the epoch partial."""
        df = _frame(
            2,
            timestamp_utc=[T0, T0 + timedelta(minutes=1)],
            speed_mph=[40.0, 60.0],
            volume_vph=[100.0, 0.0],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg["speed_mph"][0] == pytest.approx(40.0)
        assert agg["speed_aggregation"][0] == "harmonic_volume_weighted"
        assert agg["n_volume_weighted"][0] == 1

    def test_stopped_flow_weighted_reading_gives_zero(self) -> None:
        """Any weighted reading at 0 mph (stopped flow) forces the epoch speed to 0."""
        df = _frame(
            2,
            timestamp_utc=[T0, T0 + timedelta(minutes=1)],
            speed_mph=[0.0, 50.0],
            volume_vph=[200.0, 100.0],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg["speed_mph"][0] == 0.0
        assert agg["speed_aggregation"][0] == "harmonic_volume_weighted"
        assert agg["n_volume_weighted"][0] == 2

    def test_stopped_flow_partial_volume_example(self) -> None:
        df = _frame(
            4,
            timestamp_utc=[T0 + timedelta(minutes=i) for i in range(4)],
            speed_mph=[0.0, 0.0, 60.0, 60.0],
            volume_vph=[1000.0, 1000.0, 100.0, None],
        )
        row = aggregate_telemetry_5min(df).row(0, named=True)
        assert row["speed_mph"] == 0.0
        assert row["speed_aggregation"] == "harmonic_partial_volume"
        assert row["n_volume_weighted"] == 3
        assert row["n_raw_records"] == 4

    def test_zero_speed_with_zero_volume_not_stopped_flow(self) -> None:
        """A 0 mph reading with 0 vph is not weighted, so it cannot force 0 mph."""
        df = _frame(
            2,
            timestamp_utc=[T0, T0 + timedelta(minutes=1)],
            speed_mph=[0.0, 50.0],
            volume_vph=[0.0, 100.0],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg["speed_mph"][0] == pytest.approx(50.0)
        assert agg["speed_aggregation"][0] == "harmonic_volume_weighted"

    def test_n_volume_weighted_dtype_and_counts(self) -> None:
        df = _frame(
            5,
            timestamp_utc=[T0 + timedelta(minutes=i) for i in range(4)]
            + [T0 + timedelta(minutes=5)],
            volume_vph=[100.0, 0.0, None, 50.0, 0.0],
        )
        agg = aggregate_telemetry_5min(df).sort("timestamp_utc")
        assert agg["n_volume_weighted"].dtype == pl.UInt32
        assert agg["n_volume_weighted"].to_list() == [2, 0]
        assert agg["speed_aggregation"].to_list() == [
            "harmonic_partial_volume",
            "arithmetic_unweighted",
        ]

    def test_zero_volume_arithmetic_mean_fallback(self) -> None:
        df = _frame(
            2,
            timestamp_utc=[T0, T0 + timedelta(minutes=3)],
            speed_mph=[40.0, 60.0],
            volume_vph=[0.0, 0.0],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg["speed_mph"][0] == pytest.approx(50.0)
        assert agg["speed_aggregation"][0] == "arithmetic_unweighted"

    def test_null_volume_arithmetic_mean_fallback(self) -> None:
        df = _frame(
            2,
            timestamp_utc=[T0, T0 + timedelta(minutes=3)],
            speed_mph=[40.0, 70.0],
            volume_vph=[None, None],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg["speed_mph"][0] == pytest.approx(55.0)
        assert agg["speed_aggregation"][0] == "arithmetic_unweighted"
        assert agg["volume_vph"][0] is None

    def test_all_zero_speed_with_volume_is_stopped_flow(self) -> None:
        df = _frame(
            2,
            timestamp_utc=[T0, T0 + timedelta(minutes=1)],
            speed_mph=[0.0, 0.0],
            volume_vph=[100.0, 100.0],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg["speed_mph"][0] == 0.0
        assert agg["speed_aggregation"][0] == "harmonic_volume_weighted"
        assert agg["n_volume_weighted"][0] == 2

    def test_optional_column_aggregations(self) -> None:
        df = _frame(
            3,
            timestamp_utc=[T0, T0 + timedelta(minutes=1), T0 + timedelta(minutes=2)],
            free_flow_speed_mph=[60.0, None, 70.0],
            volume_vph=[100.0, None, 300.0],
            occupancy_pct=[None, 20.0, 40.0],
            confidence_score=[0.9, 0.4, None],
        )
        row = aggregate_telemetry_5min(df).row(0, named=True)
        assert row["free_flow_speed_mph"] == pytest.approx(65.0)
        assert row["volume_vph"] == pytest.approx(200.0)
        assert row["occupancy_pct"] == pytest.approx(30.0)
        assert row["confidence_score"] == pytest.approx(0.4)

    def test_all_null_optionals_stay_null(self) -> None:
        df = _frame(
            2,
            timestamp_utc=[T0, T0 + timedelta(minutes=1)],
            free_flow_speed_mph=[None, None],
            volume_vph=[None, None],
            occupancy_pct=[None, None],
            confidence_score=[None, None],
        )
        row = aggregate_telemetry_5min(df).row(0, named=True)
        assert row["free_flow_speed_mph"] is None
        assert row["volume_vph"] is None
        assert row["occupancy_pct"] is None
        assert row["confidence_score"] is None

    def test_groups_by_segment_corridor_direction(self) -> None:
        df = _frame(
            3,
            segment_id=[SEG, SEG, "alb_i495_outer_loop_s0001"],
            direction=["inner_loop", "inner_loop", "outer_loop"],
            timestamp_utc=[T0, T0 + timedelta(minutes=1), T0],
            speed_mph=[40.0, 60.0, 30.0],
            volume_vph=[100.0, 100.0, 50.0],
        )
        agg = aggregate_telemetry_5min(df)
        assert len(agg) == 2
        inner = agg.filter(pl.col("segment_id") == SEG).row(0, named=True)
        assert inner["n_raw_records"] == 2
        assert inner["speed_mph"] == pytest.approx(200.0 / (100.0 / 40.0 + 100.0 / 60.0))

    def test_empty_input(self) -> None:
        agg = aggregate_telemetry_5min(validate_telemetry(_frame(0)))
        assert len(agg) == 0
        assert "n_raw_records" in agg.columns
        assert "speed_aggregation" in agg.columns

    def test_sorting_order(self) -> None:
        df = _frame(
            3,
            segment_id=[
                "alb_i495_inner_loop_s0002",
                "alb_i495_inner_loop_s0001",
                "alb_i495_inner_loop_s0001",
            ],
            timestamp_utc=[
                datetime(2024, 10, 15, 8, 5, 0, tzinfo=UTC),
                datetime(2024, 10, 15, 8, 10, 0, tzinfo=UTC),
                datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
            ],
            speed_mph=[60.0, 40.0, 40.0],
        )
        agg = aggregate_telemetry_5min(df)
        assert agg["segment_id"].to_list() == [
            "alb_i495_inner_loop_s0001",
            "alb_i495_inner_loop_s0001",
            "alb_i495_inner_loop_s0002",
        ]
        assert agg["timestamp_utc"].to_list() == [
            datetime(2024, 10, 15, 8, 0, tzinfo=UTC),
            datetime(2024, 10, 15, 8, 10, tzinfo=UTC),
            datetime(2024, 10, 15, 8, 5, tzinfo=UTC),
        ]


# =============================================================================
# Raw readers, partitioned lakehouse, pipeline
# =============================================================================


class TestReadRawTelemetry:
    def test_read_raw_telemetry_formats(self, tmp_path: Path) -> None:
        df = _frame(1)
        csv_file = tmp_path / "sample_telemetry.csv"
        json_file = tmp_path / "sample_telemetry.json"
        parquet_file = tmp_path / "sample_telemetry.parquet"
        df.write_csv(csv_file)
        df.write_json(json_file)
        df.write_parquet(parquet_file)

        csv_df = validate_telemetry(read_raw_telemetry(csv_file))
        json_df = validate_telemetry(read_raw_telemetry(json_file))
        parquet_df = validate_telemetry(read_raw_telemetry(parquet_file))

        assert csv_df.equals(parquet_df)
        assert json_df.equals(parquet_df)

    def test_read_unsupported_format(self, tmp_path: Path) -> None:
        unsupported_file = tmp_path / "sample_telemetry.txt"
        unsupported_file.write_text("This is not a supported format.")
        with pytest.raises(ValueError):
            read_raw_telemetry(unsupported_file)


@pytest.fixture
def two_corridor_frame() -> pl.DataFrame:
    return validate_telemetry(
        _frame(
            2,
            segment_id=[SEG, "i95_lorton_woodbridge_northbound_s0001"],
            corridor=["alb_i495", "i95_lorton_woodbridge"],
            direction=["inner_loop", "northbound"],
            timestamp_utc=[T0, datetime(2024, 3, 2, 7, 0, tzinfo=UTC)],
            speed_mph=[50.0, 60.0],
        )
    )


class TestWritePartitionedTelemetry:
    def test_layout_and_return_value(
        self, tmp_path: Path, two_corridor_frame: pl.DataFrame
    ) -> None:
        out = write_partitioned_telemetry(two_corridor_frame, tmp_path, run_id="run_a")
        assert Path(out) == tmp_path
        alb = tmp_path / "corridor=alb_i495" / "year=2024" / "month=10" / "run_a.parquet"
        i95 = tmp_path / "corridor=i95_lorton_woodbridge" / "year=2024" / "month=3"
        assert alb.exists()
        assert (i95 / "run_a.parquet").exists()
        # month is not zero padded
        assert not (tmp_path / "corridor=i95_lorton_woodbridge" / "year=2024" / "month=03").exists()
        assert len(_parquet_files(tmp_path)) == 2

    def test_files_exclude_partition_keys_and_include_ingested_at(
        self, tmp_path: Path, two_corridor_frame: pl.DataFrame
    ) -> None:
        write_partitioned_telemetry(two_corridor_frame, tmp_path, run_id="run_a")
        files = _parquet_files(tmp_path)
        assert files
        for file in files:
            cols = pl.read_parquet(file, hive_partitioning=False).columns
            assert "corridor" not in cols
            assert "year" not in cols
            assert "month" not in cols
            assert "ingested_at_utc" in cols
            assert "segment_id" in cols

    def test_existing_run_id_raises(self, tmp_path: Path, two_corridor_frame: pl.DataFrame) -> None:
        write_partitioned_telemetry(two_corridor_frame, tmp_path, run_id="run_a")
        with pytest.raises(FileExistsError):
            write_partitioned_telemetry(two_corridor_frame, tmp_path, run_id="run_a")

    def test_append_only_distinct_run_ids(
        self, tmp_path: Path, two_corridor_frame: pl.DataFrame
    ) -> None:
        write_partitioned_telemetry(two_corridor_frame, tmp_path, run_id="run_a")
        write_partitioned_telemetry(two_corridor_frame, tmp_path, run_id="run_b")
        names = sorted(p.name for p in _parquet_files(tmp_path / "corridor=alb_i495"))
        assert names == ["run_a.parquet", "run_b.parquet"]
        assert len(_parquet_files(tmp_path)) == 4

    def test_default_run_id_generates_unique_files(
        self, tmp_path: Path, two_corridor_frame: pl.DataFrame
    ) -> None:
        write_partitioned_telemetry(two_corridor_frame, tmp_path)
        write_partitioned_telemetry(two_corridor_frame, tmp_path)
        assert len(_parquet_files(tmp_path)) == 4

    def test_empty_frame_writes_nothing(self, tmp_path: Path) -> None:
        empty = validate_telemetry(_frame(0))
        write_partitioned_telemetry(empty, tmp_path / "lake", run_id="run_a")
        assert _parquet_files(tmp_path) == []


class TestReadPartitionedTelemetry:
    def test_round_trip(self, tmp_path: Path) -> None:
        write_partitioned_telemetry(validate_telemetry(_frame(3)), tmp_path, run_id="r1")
        back = read_partitioned_telemetry(tmp_path)
        assert len(back) == 3
        assert back["corridor"].dtype == TELEMETRY_POLARS_SCHEMA["corridor"]
        for col in TELEMETRY_POLARS_SCHEMA:
            assert col in back.columns
        assert back["speed_mph"].to_list() == [50.0, 50.0, 50.0]

    def test_year_month_from_directory_as_int32(
        self, tmp_path: Path, two_corridor_frame: pl.DataFrame
    ) -> None:
        write_partitioned_telemetry(two_corridor_frame, tmp_path, run_id="r1")
        back = read_partitioned_telemetry(tmp_path)
        assert back["year"].dtype == pl.Int32
        assert back["month"].dtype == pl.Int32
        by_corridor = {
            row["corridor"]: (row["year"], row["month"]) for row in back.iter_rows(named=True)
        }
        assert by_corridor == {"alb_i495": (2024, 10), "i95_lorton_woodbridge": (2024, 3)}

    @staticmethod
    def _write_legacy_file(root: Path, df: pl.DataFrame) -> Path:
        """Hand-write a pre-provenance partition file: no ingested_at/n_raw/speed_aggregation."""
        part = root / "corridor=alb_i495" / "year=2024" / "month=10"
        part.mkdir(parents=True, exist_ok=True)
        path = part / "legacy.parquet"
        df.drop("corridor").write_parquet(path)
        return path

    def test_reads_legacy_file_alongside_new_file(self, tmp_path: Path) -> None:
        legacy = validate_telemetry(
            _frame(1, segment_id=["alb_i495_inner_loop_s0099"], speed_mph=[44.0])
        )
        legacy_path = self._write_legacy_file(tmp_path, legacy)
        assert "ingested_at_utc" not in pl.read_parquet(legacy_path).columns

        new = aggregate_telemetry_5min(validate_telemetry(_frame(1, speed_mph=[52.0])))
        write_partitioned_telemetry(new, tmp_path, run_id="r_new")

        back = read_partitioned_telemetry(tmp_path)
        assert len(back) == 2
        assert back["corridor"].dtype == TELEMETRY_POLARS_SCHEMA["corridor"]
        rows = {row["segment_id"]: row for row in back.iter_rows(named=True)}
        old_row = rows["alb_i495_inner_loop_s0099"]
        assert old_row["speed_mph"] == pytest.approx(44.0)
        assert old_row["ingested_at_utc"] is None
        assert old_row["n_raw_records"] is None
        assert old_row["speed_aggregation"] is None
        new_row = rows[SEG]
        assert new_row["speed_mph"] == pytest.approx(52.0)
        assert new_row["ingested_at_utc"] is not None
        assert new_row["n_raw_records"] == 1

    def test_dedup_prefers_non_null_ingested_at(self, tmp_path: Path) -> None:
        # Same (segment_id, timestamp) key in a legacy file and a newly ingested one.
        self._write_legacy_file(tmp_path, validate_telemetry(_frame(1, speed_mph=[44.0])))
        write_partitioned_telemetry(
            validate_telemetry(_frame(1, speed_mph=[52.0])), tmp_path, run_id="r_new"
        )
        back = read_partitioned_telemetry(tmp_path)
        assert len(back) == 1
        assert back["speed_mph"][0] == pytest.approx(52.0)
        assert back["ingested_at_utc"][0] is not None

    def test_multi_corridor_round_trip(
        self, tmp_path: Path, two_corridor_frame: pl.DataFrame
    ) -> None:
        write_partitioned_telemetry(two_corridor_frame, tmp_path, run_id="r1")
        back = read_partitioned_telemetry(tmp_path)
        assert sorted(back["corridor"].cast(pl.String).to_list()) == [
            "alb_i495",
            "i95_lorton_woodbridge",
        ]

    def test_dedup_keeps_most_recent_ingestion(self, tmp_path: Path) -> None:
        write_partitioned_telemetry(validate_telemetry(_frame(1)), tmp_path, run_id="r1")
        time.sleep(0.05)
        revised = validate_telemetry(_frame(1, speed_mph=[31.0]))
        write_partitioned_telemetry(revised, tmp_path, run_id="r2")

        back = read_partitioned_telemetry(tmp_path)
        assert len(back) == 1
        assert back["speed_mph"][0] == pytest.approx(31.0)

    def test_dedup_distinct_keys_preserved(self, tmp_path: Path) -> None:
        write_partitioned_telemetry(validate_telemetry(_frame(2)), tmp_path, run_id="r1")
        time.sleep(0.05)
        write_partitioned_telemetry(validate_telemetry(_frame(3)), tmp_path, run_id="r2")
        back = read_partitioned_telemetry(tmp_path)
        assert len(back) == 3
        assert back.select(["segment_id", "timestamp_utc"]).is_unique().all()

    def test_empty_directory_returns_empty_schema(self, tmp_path: Path) -> None:
        back = read_partitioned_telemetry(tmp_path)
        assert len(back) == 0
        assert back.schema == TELEMETRY_POLARS_SCHEMA


class TestIngestTelemetryPipeline:
    def test_csv_end_to_end(self, tmp_path: Path) -> None:
        raw = _frame(
            3,
            timestamp_utc=[T0, T0 + timedelta(minutes=2), T0 + timedelta(minutes=7)],
            speed_mph=[40.0, 60.0, 55.0],
            volume_vph=[100.0, 300.0, 50.0],
        )
        src = tmp_path / "raw.csv"
        raw.write_csv(src)
        out_dir = tmp_path / "lake"

        result = ingest_telemetry_pipeline(src, out_dir)
        assert Path(result) == out_dir
        back = read_partitioned_telemetry(out_dir).sort("timestamp_utc")
        assert len(back) == 2
        assert back["speed_mph"][0] == pytest.approx(53.333333, rel=1e-6)
        assert back["speed_mph"][1] == pytest.approx(55.0)

    def test_strict_pipeline_raises_on_bad_rows(self, tmp_path: Path) -> None:
        src = tmp_path / "raw.parquet"
        _frame(2, speed_mph=[50.0, 500.0]).write_parquet(src)
        out_dir = tmp_path / "lake"
        with pytest.raises(TelemetryValidationError):
            ingest_telemetry_pipeline(src, out_dir)
        assert _parquet_files(out_dir) == []

    def test_lenient_pipeline_writes_valid_rows(self, tmp_path: Path) -> None:
        src = tmp_path / "raw.parquet"
        _frame(2, speed_mph=[50.0, 500.0]).write_parquet(src)
        out_dir = tmp_path / "lake"
        ingest_telemetry_pipeline(src, out_dir, strict=False)
        back = read_partitioned_telemetry(out_dir)
        assert len(back) == 1
        assert back["speed_mph"][0] == pytest.approx(50.0)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        src = tmp_path / "raw.xml"
        src.write_text("<x/>")
        with pytest.raises(ValueError):
            ingest_telemetry_pipeline(src, tmp_path / "lake")


class TestMockFixture:
    def test_mock_fixture_integrity(self, fixtures_dir: Path) -> None:
        df = pl.read_parquet(fixtures_dir / "mock_telemetry_5min.parquet")
        assert len(df) == 3168
        assert df.null_count().sum_horizontal()[0] == 0
        assert df["speed_mph"].is_between(10, 85).all()
