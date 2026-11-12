"""Unit tests for Spatial Feature Mart Visualization (Ticket TF-013).

Strict clean-room specification tests verifying signatures and outputs for
src/topoflow/visualization/feature_mart_plot.py:
1. plot_spatial_feature_mart
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from topoflow.visualization.feature_mart_plot import plot_spatial_feature_mart


@pytest.fixture
def synthetic_mart_parquet(tmp_path: Path) -> Path:
    """Create a minimal synthetic spatial feature mart parquet file for plotting tests."""
    corridors = ["alb_i495", "i95_lorton_woodbridge", "i295_anacostia"]
    rows = []
    seg_idx = 0

    for corr in corridors:
        chainage = np.arange(0.0, 1500.0, 100.0)
        for i, c in enumerate(chainage):
            grade = -3.2 if (corr == "alb_i495" and c < 700) else 3.8
            kappa = 0.001 if c < 800 else 0.003
            tfi = 0.7 * (abs(grade) / 3.0) + 0.3 * (kappa / 0.002)

            rows.append(
                {
                    "segment_id": f"seg_{seg_idx:04d}",
                    "corridor_id": corr,
                    "corridor": corr,
                    "direction": "inner_loop",
                    "station_index": int(i),
                    "chainage_m": float(c),
                    "easting": 311000.0 + c,
                    "northing": 4315000.0 + c * 0.5,
                    "elevation_smooth_m": 80.0 - 0.02 * c,
                    "grade_pct": float(grade),
                    "grade_climb_pct": max(0.0, float(grade)),
                    "grade_descent_pct": min(0.0, float(grade)),
                    "grade_abs_pct": abs(float(grade)),
                    "grade_class": "steep_downgrade" if grade <= -2.5 else "steep_upgrade",
                    "curvature_kappa": float(kappa),
                    "topographical_friction_index": float(tfi),
                    "sin_slope_angle": float(grade / 100.0),
                    "is_sag": bool(c == 700.0),
                    "is_crest": False,
                    "is_bridge_deck": bool((c >= 600.0) and (c <= 900.0)),
                }
            )
            seg_idx += 1

    df = pl.DataFrame(
        rows,
        schema_overrides={
            "is_sag": pl.Boolean,
            "is_crest": pl.Boolean,
            "is_bridge_deck": pl.Boolean,
        },
    )
    mart_file = tmp_path / "synthetic_feature_mart.parquet"
    df.write_parquet(mart_file)
    return mart_file


class TestPlotSpatialFeatureMart:
    """Tests for plot_spatial_feature_mart."""

    def test_plot_spatial_feature_mart_explicit_output(
        self, synthetic_mart_parquet: Path, tmp_path: Path
    ) -> None:
        """Plotting function must generate a non-empty PNG at the designated output path."""
        out_png = tmp_path / "feature_mart_summary.png"

        result_path = plot_spatial_feature_mart(
            synthetic_mart_parquet,
            output_path=out_png,
        )

        assert isinstance(result_path, Path)
        assert result_path == out_png
        assert out_png.exists()
        assert out_png.suffix == ".png"
        assert out_png.stat().st_size > 1000  # Non-trivial image content

    def test_plot_with_dataframe_input(self, synthetic_mart_parquet: Path, tmp_path: Path) -> None:
        """Plotting function should accept in-memory DataFrame as input."""
        df = pl.read_parquet(synthetic_mart_parquet)
        out_png = tmp_path / "feature_mart_from_df.png"

        result_path = plot_spatial_feature_mart(
            df,
            output_path=out_png,
        )

        assert isinstance(result_path, Path)
        assert out_png.exists()
        assert out_png.stat().st_size > 1000

    def test_plot_nonexistent_mart_raises(self, tmp_path: Path) -> None:
        """Calling plot function with non-existent mart path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            plot_spatial_feature_mart(
                tmp_path / "missing_mart.parquet",
                output_path=tmp_path / "output.png",
            )

    def test_plot_default_output_path(self, synthetic_mart_parquet: Path, tmp_path: Path) -> None:
        """When output_path is omitted, it should default to a valid figure path."""
        result_path = plot_spatial_feature_mart(
            synthetic_mart_parquet,
            output_path=None,
        )
        assert isinstance(result_path, Path)
        assert result_path.exists()
        assert result_path.suffix == ".png"
        assert result_path.stat().st_size > 1000
