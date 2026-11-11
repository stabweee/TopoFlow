"""Configuration management and schema validation for TopoFlow.

Defines Pydantic settings for paths, database credentials, external APIs,
and geometric bounding schemas for study and control corridors.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from topoflow.constants import (
    CRS_METRIC,
    CRS_WGS84,
    SPATIAL_RESOLUTION_METERS,
)


def _find_project_root() -> Path:
    """Find the project root directory by inspecting parent directories for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


# =============================================================================
# Spatial & Corridor Bounding Models
# =============================================================================


class BoundingBox(BaseModel):
    """Geographic bounding box defined by latitude and longitude bounds (EPSG:4326)."""

    min_lat: float = Field(..., description="Minimum latitude (degrees North)")
    max_lat: float = Field(..., description="Maximum latitude (degrees North)")
    min_lon: float = Field(..., description="Minimum longitude (degrees East/West)")
    max_lon: float = Field(..., description="Maximum longitude (degrees East/West)")

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        """Verify that bounds are within valid geodetic ranges and non-inverted."""
        if not (-90.0 <= self.min_lat <= 90.0 and -90.0 <= self.max_lat <= 90.0):
            raise ValueError(
                f"Latitudes must be in [-90, 90]. Received min_lat={self.min_lat}, max_lat={self.max_lat}"
            )
        if not (-180.0 <= self.min_lon <= 180.0 and -180.0 <= self.max_lon <= 180.0):
            raise ValueError(
                f"Longitudes must be in [-180, 180]. Received min_lon={self.min_lon}, max_lon={self.max_lon}"
            )
        if self.min_lat >= self.max_lat:
            raise ValueError(
                f"min_lat ({self.min_lat}) must be strictly less than max_lat ({self.max_lat})"
            )
        if self.min_lon >= self.max_lon:
            raise ValueError(
                f"min_lon ({self.min_lon}) must be strictly less than max_lon ({self.max_lon})"
            )
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return bounds as standard GIS bounding box tuple: (min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def as_lat_lon_dict(self) -> dict[str, float]:
        """Return bounds as a latitude/longitude dictionary."""
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
        }

    def contains(self, lat: float, lon: float) -> bool:
        """Check whether a (latitude, longitude) coordinate falls within the bounding box."""
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon


class Centroid(BaseModel):
    """Geographic centroid coordinate."""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude (degrees North)")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude (degrees East/West)")


class Mileposts(BaseModel):
    """Corridor start and end milepost markers."""

    start: str
    end: str


class CorridorConfig(BaseModel):
    """Schema for individual highway study or control corridor."""

    corridor_id: str
    name: str
    corridor_type: Literal["study", "control"]
    route: str
    state: str
    mileposts: Mileposts | None = None
    bounding_box: BoundingBox
    centroid: Centroid
    directions: list[str]

    @model_validator(mode="after")
    def validate_centroid_in_bounds(self) -> "CorridorConfig":
        """Verify that the corridor centroid is positioned within its bounding box."""
        if not self.bounding_box.contains(self.centroid.lat, self.centroid.lon):
            raise ValueError(
                f"Centroid ({self.centroid.lat}, {self.centroid.lon}) for corridor '{self.corridor_id}' "
                f"falls outside its bounding box: {self.bounding_box.as_lat_lon_dict()}"
            )
        return self


class ElevationSmoothingConfig(BaseModel):
    """Savitzky-Golay filtering parameters for elevation profile smoothing."""

    savitzky_golay_window: int = 5
    polynomial_order: int = 2


class CorridorDefaults(BaseModel):
    """Default processing settings across corridors."""

    elevation_smoothing: ElevationSmoothingConfig = Field(default_factory=ElevationSmoothingConfig)
    spatial_resolution_meters: float = SPATIAL_RESOLUTION_METERS


class CorridorsConfig(BaseModel):
    """Root configuration holding all study and control corridors."""

    corridors: dict[str, CorridorConfig]
    defaults: CorridorDefaults = Field(default_factory=CorridorDefaults)

    def get_study_corridor(self) -> CorridorConfig:
        """Return the primary study corridor."""
        for corridor in self.corridors.values():
            if corridor.corridor_type == "study":
                return corridor
        raise ValueError("No study corridor defined in configuration.")

    def get_control_corridors(self) -> dict[str, CorridorConfig]:
        """Return all synthetic control corridors."""
        return {k: v for k, v in self.corridors.items() if v.corridor_type == "control"}

    def get_corridor(self, corridor_id: str) -> CorridorConfig:
        """Retrieve a corridor by its identifier."""
        if corridor_id not in self.corridors:
            raise KeyError(
                f"Corridor '{corridor_id}' not found. Available: {list(self.corridors.keys())}"
            )
        return self.corridors[corridor_id]


# =============================================================================
# Global Application Settings
# =============================================================================


class Settings(BaseSettings):
    """Centralized TopoFlow runtime settings and filesystem paths."""

    model_config = SettingsConfigDict(
        env_prefix="TOPOFLOW_",
        env_file=".env",
        extra="ignore",
    )

    # Base Project Paths
    project_root: Path = Field(default_factory=_find_project_root)

    # Database & Storage
    motherduck_token: str | None = None
    motherduck_database: str = "topoflow"

    # API Endpoints & Timeouts
    open_meteo_base_url: str = "https://archive-api.open-meteo.com/v1/archive"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    request_timeout_seconds: int = 30

    # Geospatial Constants
    crs_metric: str = CRS_METRIC
    crs_wgs84: str = CRS_WGS84
    spatial_resolution_meters: float = SPATIAL_RESOLUTION_METERS

    @property
    def data_dir(self) -> Path:
        """Path to data root directory."""
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Path to raw immutable input data."""
        return self.data_dir / "raw"

    @property
    def interim_data_dir(self) -> Path:
        """Path to intermediate cached data."""
        return self.data_dir / "interim"

    @property
    def processed_data_dir(self) -> Path:
        """Path to processed analytical data marts."""
        return self.data_dir / "processed"

    @property
    def configs_dir(self) -> Path:
        """Path to configuration directory."""
        return self.project_root / "configs"

    @property
    def corridors_config_path(self) -> Path:
        """Path to corridors.yaml."""
        return self.configs_dir / "corridors.yaml"

    @property
    def outputs_dir(self) -> Path:
        """Path to output artifacts directory."""
        return self.project_root / "outputs"

    def load_corridors(self, config_path: Path | None = None) -> CorridorsConfig:
        """Load and validate the corridors configuration YAML file."""
        target_path = config_path or self.corridors_config_path
        if not target_path.exists():
            raise FileNotFoundError(f"Corridors configuration not found at {target_path}")

        with open(target_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        return CorridorsConfig.model_validate(raw_data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of global Settings."""
    return Settings()
