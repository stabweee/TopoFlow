from datetime import UTC, datetime, timedelta, timezone

import polars as pl
import pytest
from pydantic import ValidationError

from topoflow.ingestion.schemas import TELEMETRY_POLARS_SCHEMA, TelemetryRecord
from topoflow.ingestion.traffic_telemetry import (
    aggregate_telemetry_5min,
    read_raw_telemetry,
    validate_telemetry,
    write_partitioned_telemetry,
)


class TestTelemetrySchema:
    def test_valid_defaults(self):
        record = TelemetryRecord(
            segment_id="alb_i495_inner_loop_s0038",
            corridor="alb_i495",
            direction="inner_loop",
            timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
            speed_mph=55.0,
        )

        assert record.free_flow_speed_mph == 65.0
        assert record.volume_vph == 0.0
        assert record.occupancy_pct == 0.0
        assert record.confidence_score == 1.0

    def test_negative_speed(self):
        with pytest.raises(ValidationError):
            TelemetryRecord(
                segment_id="alb_i495_inner_loop_s0038",
                corridor="alb_i495",
                direction="inner_loop",
                timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                speed_mph=-10.0,
            )

    def test_out_of_bounds_fields(self):
        # speed_mph > 120
        with pytest.raises(ValidationError):
            TelemetryRecord(
                segment_id="alb_i495_inner_loop_s0038",
                corridor="alb_i495",
                direction="inner_loop",
                timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                speed_mph=150.0,
            )

        # occupancy_pct > 100
        with pytest.raises(ValidationError):
            TelemetryRecord(
                segment_id="alb_i495_inner_loop_s0038",
                corridor="alb_i495",
                direction="inner_loop",
                timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                speed_mph=50.0,
                occupancy_pct=105,
            )

        # confidence_score > 1
        with pytest.raises(ValidationError):
            TelemetryRecord(
                segment_id="alb_i495_inner_loop_s0038",
                corridor="alb_i495",
                direction="inner_loop",
                timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                speed_mph=50.0,
                confidence_score=1.5,
            )

        # schema_id is empty (min_length < 1)
        with pytest.raises(ValidationError):
            TelemetryRecord(
                segment_id="",
                corridor="alb_i495",
                direction="inner_loop",
                timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                speed_mph=50.0,
            )

    def test_invalid_corridor_and_direction(self):
        with pytest.raises(ValidationError):
            TelemetryRecord(
                segment_id="alb_i495_inner_loop_s0038",
                corridor="i66",
                direction="inner_loop",
                timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                speed_mph=50.0,
            )

        with pytest.raises(ValidationError):
            TelemetryRecord(
                segment_id="alb_i495_inner_loop_s0038",
                corridor="alb_i495",
                direction="eastbound",
                timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                speed_mph=50.0,
            )

    def test_forbid_extra_fields(self):
        with pytest.raises(ValidationError):
            TelemetryRecord(
                segment_id="alb_i495_inner_loop_s0038",
                corridor="alb_i495",
                direction="inner_loop",
                timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                speed_mph=50.0,
                unknown_sensor_field="noise",
            )

    def test_utc_timezone_correction(self):
        record = TelemetryRecord(
            segment_id="alb_i495_inner_loop_s0038",
            corridor="alb_i495",
            direction="inner_loop",
            timestamp_utc=datetime(2024, 10, 15, 8, 0, 0),
            speed_mph=50.0,
        )
        assert record.timestamp_utc.tzinfo == UTC

        record = TelemetryRecord(
            segment_id="alb_i495_inner_loop_s0038",
            corridor="alb_i495",
            direction="inner_loop",
            timestamp_utc=datetime(2024, 10, 15, 8, 0, 0, tzinfo=timezone(timedelta(hours=-4))),
            speed_mph=50.0,
        )
        assert record.timestamp_utc.tzinfo == UTC
        assert record.timestamp_utc.hour == 12  # 8 AM EDT is 12 PM UTC


class TestValidateTelemetry:
    def test_missing_columns_raises_error(self):
        df = pl.DataFrame({"segment_id": ["seg_001"], "speed_mph": [50.0]})

        with pytest.raises(ValueError):
            validate_telemetry(df)

    def test_filters_out_of_bounds_rows(self):
        df = pl.DataFrame(
            {
                "segment_id": [
                    "alb_i495_inner_loop_s0001",
                    "alb_i495_inner_loop_s0002",
                    "alb_i495_inner_loop_s0003",
                    None,
                ],
                "corridor": ["alb_i495", "alb_i495", "alb_i495", "alb_i495"],
                "direction": ["inner_loop", "inner_loop", "inner_loop", "inner_loop"],
                "timestamp_utc": [
                    datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 5, 0, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 10, 0, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 15, 0, tzinfo=UTC),
                ],
                "speed_mph": [50.0, -10.0, 130.0, 90.0],
                "free_flow_speed_mph": [65.0, 65.0, 65.0, 65.0],
                "volume_vph": [100.0, 100.0, 100.0, 100.0],
                "occupancy_pct": [50.0, 50.0, 120.0, 50.0],
                "confidence_score": [1.0, 1.0, 1.0, 1.0],
            }
        )

        clean_df = validate_telemetry(df)
        assert clean_df.shape[0] == 1
        assert clean_df["segment_id"][0] == "alb_i495_inner_loop_s0001"

    def test_casts_schema_types(self):
        df = pl.DataFrame(
            {
                "segment_id": ["alb_i495_inner_loop_s0001"],
                "corridor": ["alb_i495"],
                "direction": ["inner_loop"],
                "timestamp_utc": [datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC)],
                "speed_mph": [50],
                "free_flow_speed_mph": [65],
                "volume_vph": [100],
                "occupancy_pct": [50],
                "confidence_score": [1],
            }
        )

        clean_df = validate_telemetry(df)
        assert clean_df["speed_mph"].dtype == pl.Float64
        assert clean_df.schema == TELEMETRY_POLARS_SCHEMA


class TestAggregateTelemetry5Min:
    def test_timestamp_truncation(self):
        df = pl.DataFrame(
            {
                "segment_id": ["alb_i495_inner_loop_s0001"] * 3,
                "corridor": ["alb_i495"] * 3,
                "direction": ["inner_loop"] * 3,
                "timestamp_utc": [
                    datetime(2024, 10, 15, 8, 0, 30, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 2, 45, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 4, 59, tzinfo=UTC),
                ],
                "speed_mph": [50.0, 60.0, 70.0],
                "free_flow_speed_mph": [65.0] * 3,
                "volume_vph": [100.0] * 3,
                "occupancy_pct": [50.0] * 3,
                "confidence_score": [1.0] * 3,
            }
        )

        aggregated_df = aggregate_telemetry_5min(df)
        assert aggregated_df.shape[0] == 1
        assert aggregated_df["timestamp_utc"][0] == datetime(2024, 10, 15, 8, 0, tzinfo=UTC)

    def test_weighted_average_speed(self):
        df = pl.DataFrame(
            {
                "segment_id": ["alb_i495_inner_loop_s0001"] * 2,
                "corridor": ["alb_i495"] * 2,
                "direction": ["inner_loop"] * 2,
                "timestamp_utc": [
                    datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 3, 0, tzinfo=UTC),
                ],
                "speed_mph": [40.0, 60.0],
                "free_flow_speed_mph": [65.0] * 2,
                "volume_vph": [100.0, 300.0],
                "occupancy_pct": [50.0] * 2,
                "confidence_score": [1.0] * 2,
            }
        )

        aggregated_df = aggregate_telemetry_5min(df)
        expected_weighted_speed = (40.0 * 100 + 60.0 * 300) / (100 + 300)
        assert aggregated_df["speed_mph"][0] == pytest.approx(expected_weighted_speed)

    def test_zero_volume_arithmetic_mean_fallback(self):
        df = pl.DataFrame(
            {
                "segment_id": ["alb_i495_inner_loop_s0001"] * 2,
                "corridor": ["alb_i495"] * 2,
                "direction": ["inner_loop"] * 2,
                "timestamp_utc": [
                    datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 3, 0, tzinfo=UTC),
                ],
                "speed_mph": [40.0, 60.0],
                "free_flow_speed_mph": [65.0] * 2,
                "volume_vph": [0.0, 0.0],
                "occupancy_pct": [50.0] * 2,
                "confidence_score": [1.0] * 2,
            }
        )

        aggregated_df = aggregate_telemetry_5min(df)
        expected_mean_speed = (40.0 + 60.0) / 2
        assert aggregated_df["speed_mph"][0] == pytest.approx(expected_mean_speed)

    def test_sorting_order(self):
        df = pl.DataFrame(
            {
                "segment_id": [
                    "alb_i495_inner_loop_s0002",
                    "alb_i495_inner_loop_s0001",
                    "alb_i495_inner_loop_s0001",
                ],
                "corridor": ["alb_i495", "alb_i495", "alb_i495"],
                "direction": ["inner_loop", "inner_loop", "inner_loop"],
                "timestamp_utc": [
                    datetime(2024, 10, 15, 8, 5, 0, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 10, 0, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                ],
                "speed_mph": [60.0, 40.0, 40.0],
                "free_flow_speed_mph": [65.0] * 3,
                "volume_vph": [100.0] * 3,
                "occupancy_pct": [50.0] * 3,
                "confidence_score": [1.0] * 3,
            }
        )

        aggregated_df = aggregate_telemetry_5min(df)
        assert aggregated_df["segment_id"].to_list() == [
            "alb_i495_inner_loop_s0001",
            "alb_i495_inner_loop_s0001",
            "alb_i495_inner_loop_s0002",
        ]
        assert aggregated_df["timestamp_utc"].to_list() == [
            datetime(2024, 10, 15, 8, 0, tzinfo=UTC),
            datetime(2024, 10, 15, 8, 10, tzinfo=UTC),
            datetime(2024, 10, 15, 8, 5, tzinfo=UTC),
        ]


class TestTelemetryIOAndPartitions:
    def test_read_raw_telemetry_formats(self, tmp_path):
        csv_file = tmp_path / "sample_telemetry.csv"
        json_file = tmp_path / "sample_telemetry.json"
        parquet_file = tmp_path / "sample_telemetry.parquet"

        df = pl.DataFrame(
            {
                "segment_id": ["alb_i495_inner_loop_s0001"],
                "corridor": ["alb_i495"],
                "direction": ["inner_loop"],
                "timestamp_utc": [datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC)],
                "speed_mph": [50.0],
                "free_flow_speed_mph": [65.0],
                "volume_vph": [100.0],
                "occupancy_pct": [50.0],
                "confidence_score": [1.0],
            }
        )

        df.write_csv(csv_file)
        df.write_json(json_file)
        df.write_parquet(parquet_file)

        csv_df = validate_telemetry(read_raw_telemetry(csv_file))
        json_df = validate_telemetry(read_raw_telemetry(json_file))
        parquet_df = validate_telemetry(read_raw_telemetry(parquet_file))

        assert csv_df.equals(parquet_df)
        assert json_df.equals(parquet_df)

    def test_read_unsupported_format(self, tmp_path):
        unsupported_file = tmp_path / "sample_telemetry.txt"
        unsupported_file.write_text("This is not a supported format.")

        with pytest.raises(ValueError):
            read_raw_telemetry(unsupported_file)

    def test_write_partitioned_telemetry(self, tmp_path):
        df = pl.DataFrame(
            {
                "segment_id": [
                    "alb_i495_inner_loop_s0001",
                    "i95_lorton_woodbridge_northbound_s0001",
                ],
                "corridor": ["alb_i495", "i95_lorton_woodbridge"],
                "direction": ["inner_loop", "northbound"],
                "timestamp_utc": [
                    datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                    datetime(2024, 10, 15, 8, 0, 0, tzinfo=UTC),
                ],
                "speed_mph": [50.0, 60.0],
                "free_flow_speed_mph": [65.0] * 2,
                "volume_vph": [100.0] * 2,
                "occupancy_pct": [50.0] * 2,
                "confidence_score": [1.0] * 2,
            }
        )

        write_partitioned_telemetry(df, tmp_path)

        # Check if partitioned directories are created
        corridor_dir = tmp_path / "corridor=alb_i495"
        assert corridor_dir.exists()

        read_df = pl.read_parquet(tmp_path)
        assert len(read_df) == len(df)

    def test_mock_fixture_integrity(self):
        df = pl.read_parquet("tests/fixtures/mock_telemetry_5min.parquet")

        assert len(df) == 3168
        assert df.null_count().sum_horizontal()[0] == 0
        assert df["speed_mph"].is_between(10, 85).all()
