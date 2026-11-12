"""Unit tests for elevation and Savitzky-Golay visualization utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from topoflow.visualization.elevation_plot import (
    plot_corridor_elevation_profile,
    plot_savgol_vs_raw_differences,
)


@pytest.fixture
def sample_elevation_df() -> pl.DataFrame:
    """Generate realistic synthetic elevation profile DataFrame for testing."""
    rng = np.random.default_rng(42)
    chainage = np.arange(0.0, 4000.0, 100.0)
    z_raw = 80.0 - 0.03 * chainage + rng.normal(0, 0.05, len(chainage))
    z_smooth = 80.0 - 0.03 * chainage
    grade_pct = np.full(len(chainage), -3.0)
    grade_raw_pct = np.full(len(chainage), -3.0) + rng.normal(0, 0.2, len(chainage))
    curvature = np.zeros(len(chainage))

    is_bridge = (chainage >= 1800.0) & (chainage <= 2600.0)
    z_raw[is_bridge] = 43.0
    z_smooth[is_bridge] = 43.0

    return pl.DataFrame(
        {
            "segment_id": [f"alb_s{i:04d}" for i in range(len(chainage))],
            "corridor_id": ["alb_i495"] * len(chainage),
            "direction": ["inner_loop"] * len(chainage),
            "station_index": list(range(len(chainage))),
            "chainage_m": chainage,
            "easting": [311000.0 + i * 10 for i in range(len(chainage))],
            "northing": [4315000.0 + i * 100 for i in range(len(chainage))],
            "elevation_raw_m": z_raw,
            "elevation_smooth_m": z_smooth,
            "grade_pct": grade_pct,
            "grade_raw_pct": grade_raw_pct,
            "vertical_curvature": curvature,
            "is_sag": [False] * len(chainage),
            "is_crest": [False] * len(chainage),
            "is_bridge_deck": is_bridge,
        }
    )


class TestElevationPlotting:
    """Tests for elevation profile figure generation."""

    def test_plot_corridor_elevation_profile_saves_file(
        self,
        sample_elevation_df: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "test_elevation.png"
        saved_path = plot_corridor_elevation_profile(
            sample_elevation_df, output_path=out_file, dpi=80
        )
        assert saved_path == out_file
        assert out_file.exists()
        assert out_file.stat().st_size > 10000

    def test_plot_savgol_vs_raw_differences_saves_file(
        self,
        sample_elevation_df: pl.DataFrame,
        tmp_path: Path,
    ) -> None:
        out_file = tmp_path / "test_savgol.png"
        saved_path = plot_savgol_vs_raw_differences(
            sample_elevation_df, output_path=out_file, dpi=80
        )
        assert saved_path == out_file
        assert out_file.exists()
        assert out_file.stat().st_size > 10000

    def test_plot_elevation_empty_df_raises(self, tmp_path: Path) -> None:
        empty_df = pl.DataFrame()
        with pytest.raises(ValueError):
            plot_corridor_elevation_profile(empty_df, output_path=tmp_path / "err.png")

    def test_plot_savgol_empty_df_raises(self, tmp_path: Path) -> None:
        empty_df = pl.DataFrame()
        with pytest.raises(ValueError):
            plot_savgol_vs_raw_differences(empty_df, output_path=tmp_path / "err.png")
