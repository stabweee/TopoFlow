"""Unit tests for corridor centerline visualization utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest

from topoflow.config import Settings
from topoflow.geo.osm_extractor import OSMExtractor
from topoflow.visualization.corridor_map import plot_corridors


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to tests/fixtures directory."""
    return Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def sample_gdf(fixtures_dir: Path) -> gpd.GeoDataFrame:
    """Load sample corridor geometries from offline fixture."""
    fixture_path = fixtures_dir / "osm_corridors_sample.geojson"
    return OSMExtractor.load_from_fixture(fixture_path)


class TestCorridorMapPlotting:
    """Test corridor visualization generation and saving."""

    def test_plot_corridors_saves_file(
        self,
        sample_gdf: gpd.GeoDataFrame,
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "test_corridors.png"
        saved_path = plot_corridors(sample_gdf, output_path=out_file, dpi=100)

        assert saved_path == out_file
        assert out_file.exists()
        assert out_file.stat().st_size > 5000  # Non-trivial PNG file size

    def test_plot_corridors_default_path(
        self,
        sample_gdf: gpd.GeoDataFrame,
        tmp_path: Path,
    ) -> None:
        custom_settings = Settings(project_root=tmp_path)
        saved_path = plot_corridors(sample_gdf, settings=custom_settings, dpi=80)

        expected = tmp_path / "outputs" / "figures" / "p1_corridor_centerlines.png"
        assert saved_path == expected
        assert expected.exists()

    def test_plot_corridors_empty_gdf_raises(self, tmp_path: Path) -> None:
        empty_gdf = gpd.GeoDataFrame(columns=["corridor_id", "geometry"])
        with pytest.raises(ValueError, match="no corridor records"):
            plot_corridors(empty_gdf, output_path=tmp_path / "empty.png")
