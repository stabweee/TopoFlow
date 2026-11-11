"""Unit tests for horizontal curvature visualization utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest

from topoflow.config import Settings
from topoflow.geo.discretization import SpatialDiscretizer
from topoflow.geo.osm_extractor import OSMExtractor
from topoflow.visualization.curvature_plot import plot_horizontal_curvature


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to tests/fixtures directory."""
    return Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def sample_stations_gdf(fixtures_dir: Path) -> gpd.GeoDataFrame:
    """Generate sample stations GeoDataFrame from offline fixture."""
    fixture_path = fixtures_dir / "osm_corridors_sample.geojson"
    corridors_gdf = OSMExtractor.load_from_fixture(fixture_path)
    discretizer = SpatialDiscretizer()
    return discretizer.discretize_corridors_dataframe(corridors_gdf)


class TestCurvaturePlotting:
    """Test horizontal curvature plotting generation and saving."""

    def test_plot_curvature_saves_file(
        self,
        sample_stations_gdf: gpd.GeoDataFrame,
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "test_curvature.png"
        saved_path = plot_horizontal_curvature(sample_stations_gdf, output_path=out_file, dpi=100)

        assert saved_path == out_file
        assert out_file.exists()
        assert out_file.stat().st_size > 10000  # Non-trivial figure size

    def test_plot_curvature_default_path(
        self,
        sample_stations_gdf: gpd.GeoDataFrame,
        tmp_path: Path,
    ) -> None:
        custom_settings = Settings(project_root=tmp_path)
        saved_path = plot_horizontal_curvature(
            sample_stations_gdf, settings=custom_settings, dpi=80
        )

        expected = tmp_path / "outputs" / "figures" / "p1_horizontal_curvature.png"
        assert saved_path == expected
        assert expected.exists()

    def test_plot_curvature_empty_gdf_raises(self, tmp_path: Path) -> None:
        empty_gdf = gpd.GeoDataFrame(columns=["segment_id", "geometry"])
        with pytest.raises(ValueError, match="no station records"):
            plot_horizontal_curvature(empty_gdf, output_path=tmp_path / "empty.png")
