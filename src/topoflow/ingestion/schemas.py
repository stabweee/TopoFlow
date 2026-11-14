from datetime import UTC, datetime
from typing import Literal, get_args

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from topoflow.constants import DEFAULT_FREE_FLOW_SPEED_MPH

CORRIDORS = Literal["alb_i495", "i95_lorton_woodbridge", "i295_anacostia"]
DIRECTIONS = Literal["inner_loop", "outer_loop", "northbound", "southbound"]


class TelemetryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True)

    segment_id: str = Field(min_length=1)
    corridor: CORRIDORS
    direction: DIRECTIONS
    timestamp_utc: datetime
    speed_mph: float = Field(ge=0, le=120)
    free_flow_speed_mph: float = Field(default=DEFAULT_FREE_FLOW_SPEED_MPH, ge=30, le=90)
    volume_vph: float = Field(default=0, ge=0, le=20000)
    occupancy_pct: float = Field(default=0, ge=0, le=100)
    confidence_score: float = Field(default=1, ge=0, le=1)

    @field_validator("timestamp_utc", mode="after")
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
