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

    def test_plot_corridor_elevation_profile_control_corridors_steep_grades(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify dynamic headroom accommodates control corridors with grades exceeding +/-2.0%."""
        chainage = np.arange(0.0, 3000.0, 100.0)
        n = len(chainage)

        # Control corridor I-95 with steep rolling grades from -3.0% to +3.5%
        g95 = np.linspace(-3.0, 3.5, n)
        df_i95 = pl.DataFrame(
            {
                "segment_id": [f"i95_s{i:04d}" for i in range(n)],
                "corridor_id": ["i95_lorton_woodbridge"] * n,
                "direction": ["northbound"] * n,
                "station_index": list(range(n)),
                "chainage_m": chainage,
                "easting": [310000.0 + i * 10 for i in range(n)],
                "northing": [4315000.0 + i * 100 for i in range(n)],
                "elevation_raw_m": [50.0 + i * 0.5 for i in range(n)],
                "elevation_smooth_m": [50.0 + i * 0.5 for i in range(n)],
                "grade_pct": g95,
                "grade_raw_pct": g95,
                "vertical_curvature": [0.0] * n,
                "is_sag": [False] * n,
                "is_crest": [False] * n,
                "is_bridge_deck": [False] * n,
            }
        )

        # Control corridor I-295 with rolling grades exceeding +/-2.0%
        g295 = np.linspace(-2.5, 3.2, n)
        df_i295 = pl.DataFrame(
            {
                "segment_id": [f"i295_s{i:04d}" for i in range(n)],
                "corridor_id": ["i295_anacostia"] * n,
                "direction": ["northbound"] * n,
                "station_index": list(range(n)),
                "chainage_m": chainage,
                "easting": [315000.0 + i * 10 for i in range(n)],
                "northing": [4320000.0 + i * 100 for i in range(n)],
                "elevation_raw_m": [30.0 + i * 0.3 for i in range(n)],
                "elevation_smooth_m": [30.0 + i * 0.3 for i in range(n)],
                "grade_pct": g295,
                "grade_raw_pct": g295,
                "vertical_curvature": [0.0] * n,
                "is_sag": [False] * n,
                "is_crest": [False] * n,
                "is_bridge_deck": [False] * n,
            }
        )

        df_multi = pl.concat([df_i95, df_i295])
        out_file = tmp_path / "test_steep_control_corridors.png"

        saved_path = plot_corridor_elevation_profile(df_multi, output_path=out_file, dpi=80)
        assert saved_path == out_file
        assert out_file.exists()
        assert out_file.stat().st_size > 10000
