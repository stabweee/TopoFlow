"""Tests for configuration management, domain constants, and corridor bounding schemas."""

from pathlib import Path

import pyproj
import pytest
from pydantic import ValidationError

from topoflow.config import (
    BoundingBox,
    Centroid,
    CorridorConfig,
    ElevationSmoothingConfig,
    Settings,
    get_settings,
)
from topoflow.constants import (
    CRS_METRIC,
    CRS_WGS84,
    JAM_DENSITY_VPMPL,
    MAX_STRAIGHT_RADIUS_METERS,
    RAIN_SHOCK_THRESHOLD_MM,
    SPATIAL_RESOLUTION_METERS,
    VOT_COMMERCIAL_PER_HOUR,
    VOT_PERSONAL_PER_HOUR,
    VOT_REGIONAL_WAGE_PER_HOUR,
)


class TestConstants:
    """Validate domain constants and CRS strings."""

    def test_crs_validity(self) -> None:
        """Verify that CRS definition strings parse into valid pyproj CoordinateReferenceSystems."""
        crs_metric = pyproj.CRS.from_user_input(CRS_METRIC)
        crs_wgs84 = pyproj.CRS.from_user_input(CRS_WGS84)

        assert crs_metric.is_projected
        assert crs_wgs84.is_geographic
        assert "18N" in crs_metric.name
        assert "NAD83" in crs_metric.name

    def test_domain_constants(self) -> None:
        """Verify domain constants match analytical specifications."""
        assert SPATIAL_RESOLUTION_METERS == 100.0
        assert MAX_STRAIGHT_RADIUS_METERS == 10000.0
        assert JAM_DENSITY_VPMPL == 220.0
        assert RAIN_SHOCK_THRESHOLD_MM == 1.25
        assert VOT_PERSONAL_PER_HOUR == 24.60
        assert VOT_COMMERCIAL_PER_HOUR == 42.00
        assert VOT_REGIONAL_WAGE_PER_HOUR == 38.00


class TestBoundingBox:
    """Validate BoundingBox schema, bounding checks, and edge cases."""

    def test_bounding_box_valid(self) -> None:
        """Verify valid bounding box creation and helper methods."""
        bbox = BoundingBox(
            min_lat=38.95,
            max_lat=39.00,
            min_lon=-77.23,
            max_lon=-77.18,
        )
        assert bbox.as_tuple() == (-77.23, 38.95, -77.18, 39.00)
        assert bbox.as_lat_lon_dict() == {
            "min_lat": 38.95,
            "max_lat": 39.00,
            "min_lon": -77.23,
            "max_lon": -77.18,
        }
        # Point inside
        assert bbox.contains(38.97, -77.20)
        # Point outside latitude
        assert not bbox.contains(39.05, -77.20)
        # Point outside longitude
        assert not bbox.contains(38.97, -77.15)

    def test_bounding_box_inverted_lat(self) -> None:
        """Reject bounding box where min_lat >= max_lat."""
        with pytest.raises(ValidationError, match="min_lat.*must be strictly less than max_lat"):
            BoundingBox(min_lat=39.00, max_lat=38.95, min_lon=-77.23, max_lon=-77.18)

    def test_bounding_box_inverted_lon(self) -> None:
        """Reject bounding box where min_lon >= max_lon."""
        with pytest.raises(ValidationError, match="min_lon.*must be strictly less than max_lon"):
            BoundingBox(min_lat=38.95, max_lat=39.00, min_lon=-77.18, max_lon=-77.23)

    def test_bounding_box_out_of_range_coords(self) -> None:
        """Reject latitudes or longitudes exceeding geodetic limits."""
        with pytest.raises(ValidationError, match="Latitudes must be in"):
            BoundingBox(min_lat=-95.0, max_lat=39.00, min_lon=-77.23, max_lon=-77.18)

        with pytest.raises(ValidationError, match="Longitudes must be in"):
            BoundingBox(min_lat=38.95, max_lat=39.00, min_lon=-185.0, max_lon=-77.18)


class TestCorridorConfig:
    """Validate CorridorConfig models and centroid spatial containment."""

    def test_corridor_centroid_outside_box(self) -> None:
        """Reject corridor where centroid lies outside its bounding box."""
        with pytest.raises(ValidationError, match="falls outside its bounding box"):
            CorridorConfig(
                corridor_id="test_corridor",
                name="Test Corridor",
                corridor_type="study",
                route="I-495",
                state="VA/MD",
                bounding_box=BoundingBox(
                    min_lat=38.95, max_lat=39.00, min_lon=-77.23, max_lon=-77.18
                ),
                centroid=Centroid(lat=38.90, lon=-77.20),  # Lat is below min_lat
                directions=["inner_loop", "outer_loop"],
            )


class TestCorridorsYamlLoading:
    """Validate reading and parsing configs/corridors.yaml."""

    def test_load_corridors_defaults(self, configs_dir: Path) -> None:
        """Load default corridors.yaml and assert presence of study and control corridors."""
        settings = Settings()
        config = settings.load_corridors(configs_dir / "corridors.yaml")

        assert "alb_i495" in config.corridors
        assert "i95_lorton_woodbridge" in config.corridors
        assert "i295_anacostia" in config.corridors

        study = config.get_study_corridor()
        assert study.corridor_id == "alb_i495"
        assert study.corridor_type == "study"
        assert study.route == "I-495"
        assert study.bounding_box.contains(study.centroid.lat, study.centroid.lon)

        controls = config.get_control_corridors()
        assert len(controls) == 2
        assert "i95_lorton_woodbridge" in controls
        assert "i295_anacostia" in controls

    def test_corridor_lookup_and_defaults(self, configs_dir: Path) -> None:
        """Verify corridor lookup and default smoothing parameters."""
        settings = Settings()
        config = settings.load_corridors(configs_dir / "corridors.yaml")

        alb = config.get_corridor("alb_i495")
        assert alb.name.startswith("I-495 American Legion")

        with pytest.raises(KeyError, match="Corridor 'nonexistent' not found"):
            config.get_corridor("nonexistent")

        assert isinstance(config.defaults.elevation_smoothing, ElevationSmoothingConfig)
        assert config.defaults.elevation_smoothing.savitzky_golay_window == 5
        assert config.defaults.elevation_smoothing.polynomial_order == 2

    def test_load_corridors_missing_file(self, tmp_path: Path) -> None:
        """Raise FileNotFoundError when target yaml does not exist."""
        settings = Settings()
        with pytest.raises(FileNotFoundError, match="Corridors configuration not found"):
            settings.load_corridors(tmp_path / "nonexistent.yaml")


class TestSettings:
    """Validate runtime Settings, path discovery, and environment overrides."""

    def test_settings_paths(self) -> None:
        """Verify default path properties resolve correctly."""
        settings = get_settings()
        assert settings.project_root.exists()
        assert settings.data_dir == settings.project_root / "data"
        assert settings.raw_data_dir == settings.data_dir / "raw"
        assert settings.interim_data_dir == settings.data_dir / "interim"
        assert settings.processed_data_dir == settings.data_dir / "processed"
        assert settings.configs_dir == settings.project_root / "configs"
        assert settings.outputs_dir == settings.project_root / "outputs"
        assert settings.corridors_config_path.exists()

    def test_settings_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify environment variables with prefix TOPOFLOW_ override defaults."""
        monkeypatch.setenv("TOPOFLOW_MOTHERDUCK_DATABASE", "test_lakehouse")
        monkeypatch.setenv("TOPOFLOW_REQUEST_TIMEOUT_SECONDS", "45")

        fresh_settings = Settings()
        assert fresh_settings.motherduck_database == "test_lakehouse"
        assert fresh_settings.request_timeout_seconds == 45
