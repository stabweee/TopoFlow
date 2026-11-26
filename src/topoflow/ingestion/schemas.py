"""Telemetry data contracts (Pydantic record model and columnar Polars schema).

Only ``speed_mph`` is a required measurement. Free-flow speed, volume, occupancy and
confidence are optional and nullable: when a source does not report them they stay
null. Nothing is defaulted to a plausible-looking value (e.g. 65 mph or 0 vph), because
fabricated defaults are indistinguishable from real observations downstream.
"""

from datetime import UTC, datetime
from typing import Literal, get_args

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

CORRIDORS = Literal["alb_i495", "i95_lorton_woodbridge", "i295_anacostia"]
DIRECTIONS = Literal["inner_loop", "outer_loop", "northbound", "southbound"]

# Physical bounds shared by the record model and the vectorized Polars validator.
SPEED_MPH_BOUNDS: tuple[float, float] = (0.0, 120.0)
FREE_FLOW_SPEED_MPH_BOUNDS: tuple[float, float] = (30.0, 90.0)
VOLUME_VPH_BOUNDS: tuple[float, float] = (0.0, 20000.0)
OCCUPANCY_PCT_BOUNDS: tuple[float, float] = (0.0, 100.0)
CONFIDENCE_SCORE_BOUNDS: tuple[float, float] = (0.0, 1.0)


class TelemetryRecord(BaseModel):
    """A single observed telemetry reading for one 100 m station."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    segment_id: str = Field(min_length=1)
    corridor: CORRIDORS
    direction: DIRECTIONS
    timestamp_utc: datetime
    speed_mph: float = Field(ge=SPEED_MPH_BOUNDS[0], le=SPEED_MPH_BOUNDS[1])
    free_flow_speed_mph: float | None = Field(
        default=None, ge=FREE_FLOW_SPEED_MPH_BOUNDS[0], le=FREE_FLOW_SPEED_MPH_BOUNDS[1]
    )
    volume_vph: float | None = Field(default=None, ge=VOLUME_VPH_BOUNDS[0], le=VOLUME_VPH_BOUNDS[1])
    occupancy_pct: float | None = Field(
        default=None, ge=OCCUPANCY_PCT_BOUNDS[0], le=OCCUPANCY_PCT_BOUNDS[1]
    )
    confidence_score: float | None = Field(
        default=None, ge=CONFIDENCE_SCORE_BOUNDS[0], le=CONFIDENCE_SCORE_BOUNDS[1]
    )

    @field_validator("timestamp_utc", mode="after")
    @classmethod
    def ensure_utc_timezone(cls, v: datetime) -> datetime:
        if not v.tzinfo:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


TELEMETRY_POLARS_SCHEMA = pl.Schema(
    {
        "segment_id": pl.Utf8,
        "corridor": pl.Enum(list(get_args(CORRIDORS))),
        "direction": pl.Enum(list(get_args(DIRECTIONS))),
        "timestamp_utc": pl.Datetime(time_unit="ms", time_zone="UTC"),
        "speed_mph": pl.Float64,
        "free_flow_speed_mph": pl.Float64,
        "volume_vph": pl.Float64,
        "occupancy_pct": pl.Float64,
        "confidence_score": pl.Float64,
    }
)

# Columns every telemetry source must supply (non-null) for a row to be valid.
TELEMETRY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "segment_id",
    "corridor",
    "direction",
    "timestamp_utc",
    "speed_mph",
)

# Optional measurements: missing columns are added as all-null, never defaulted.
TELEMETRY_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "free_flow_speed_mph",
    "volume_vph",
    "occupancy_pct",
    "confidence_score",
)
