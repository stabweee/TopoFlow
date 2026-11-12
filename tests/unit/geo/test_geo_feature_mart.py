"""Unit tests for Geospatial Feature Mart Integration (Ticket TF-013).

Strict clean-room specification tests verifying signatures, domain invariants,
and data contracts for src/topoflow/geo/feature_mart.py:
1. compute_asymmetric_grades
2. classify_grade_regimes
3. compute_topographical_friction_index
4. build_spatial_feature_mart
5. register_spatial_feature_mart
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from topoflow.geo.feature_mart import (
    build_spatial_feature_mart,
    classify_grade_regimes,
    compute_asymmetric_grades,
    compute_topographical_friction_index,
    register_spatial_feature_mart,
)

CANONICAL_STATIONS_PATH = Path("data/interim/geo/stations_100m.parquet")
CANONICAL_ELEVATION_PATH = Path("data/interim/geo/elevation_profile.parquet")


@pytest.fixture
def synthetic_grades_df() -> pl.DataFrame:
    """Synthetic DataFrame with diverse grades for testing regime and asymmetric grade computations."""
    return pl.DataFrame(
        {
            "segment_id": [f"seg_{i:03d}" for i in range(11)],
            "grade_pct": [
                -4.0,  # steep downgrade
                -2.5,  # boundary: steep downgrade
                -2.49,  # moderate downgrade
                -1.5,  # moderate downgrade
                -0.5,  # boundary: flat tangent
                0.0,  # flat tangent
                0.5,  # boundary: flat tangent
                0.51,  # moderate upgrade
                2.0,  # moderate upgrade
                2.5,  # boundary: steep upgrade
                3.8,  # steep upgrade (e.g. ALB climb)
            ],
            "curvature_kappa": [0.001 * i for i in range(11)],
        }
    )


@pytest.fixture
def synthetic_mart_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal valid synthetic stations and elevation parquet files."""
    chainage = np.arange(0.0, 1000.0, 100.0)
    n = len(chainage)

    stations_df = pl.DataFrame(
        {
            "segment_id": [f"seg_{i:03d}" for i in range(n)],
            "corridor_id": ["alb_i495"] * n,
            "direction": ["inner_loop"] * n,
            "station_index": list(range(n)),
            "chainage_m": chainage,
            "easting": [311000.0 + i * 10.0 for i in range(n)],
            "northing": [4315000.0 + i * 100.0 for i in range(n)],
            "longitude": [-77.1 + i * 0.001 for i in range(n)],
            "latitude": [38.9 + i * 0.001 for i in range(n)],
            "heading_deg": [45.0] * n,
            "curvature_kappa": [0.0005 * i for i in range(n)],
            "radius_m": [2000.0] * n,
            "route": ["I-495"] * n,
            "name": ["Capital Beltway"] * n,
            "lanes": [4] * n,
            "maxspeed": ["55 mph"] * n,
            "has_bridge": [False] * n,
        }
    )

    elevation_df = pl.DataFrame(
        {
            "segment_id": [f"seg_{i:03d}" for i in range(n)],
            "corridor_id": ["alb_i495"] * n,
            "direction": ["inner_loop"] * n,
            "station_index": list(range(n)),
            "chainage_m": chainage,
            "easting": [311000.0 + i * 10.0 for i in range(n)],
            "northing": [4315000.0 + i * 100.0 for i in range(n)],
            "elevation_raw_m": [80.0 - 0.03 * c for c in chainage],
            "elevation_smooth_m": [80.0 - 0.03 * c for c in chainage],
            "grade_pct": [-3.0 + 0.6 * i for i in range(n)],
            "grade_raw_pct": [-3.0 + 0.6 * i for i in range(n)],
            "vertical_curvature": [0.0] * n,
            "is_sag": [False] * n,
            "is_crest": [False] * n,
            "is_bridge_deck": [False] * n,
        }
    )

    stations_file = tmp_path / "synthetic_stations.parquet"
    elevation_file = tmp_path / "synthetic_elevation.parquet"
    stations_df.write_parquet(stations_file)
    elevation_df.write_parquet(elevation_file)

    return stations_file, elevation_file


class TestComputeAsymmetricGrades:
    """Tests for compute_asymmetric_grades."""

    def test_positive_grade_decomposition(self) -> None:
        """Positive grades must yield positive climb, zero descent, and matching absolute grade."""
        df = pl.DataFrame({"grade_pct": [3.8]})
        res = compute_asymmetric_grades(df)

        assert "grade_climb_pct" in res.columns
        assert "grade_descent_pct" in res.columns
        assert "grade_abs_pct" in res.columns

        row = res.to_dicts()[0]
        assert row["grade_climb_pct"] == pytest.approx(3.8)
        assert row["grade_descent_pct"] == pytest.approx(0.0)
        assert row["grade_abs_pct"] == pytest.approx(3.8)

    def test_negative_grade_decomposition(self) -> None:
        """Negative grades must yield zero climb, negative descent, and positive absolute grade."""
        df = pl.DataFrame({"grade_pct": [-3.2]})
        res = compute_asymmetric_grades(df)

        row = res.to_dicts()[0]
        assert row["grade_climb_pct"] == pytest.approx(0.0)
        assert row["grade_descent_pct"] == pytest.approx(-3.2)
        assert row["grade_abs_pct"] == pytest.approx(3.2)

    def test_zero_grade_decomposition(self) -> None:
        """Zero grade must yield zero across climb, descent, and absolute grade."""
        df = pl.DataFrame({"grade_pct": [0.0]})
        res = compute_asymmetric_grades(df)

        row = res.to_dicts()[0]
        assert row["grade_climb_pct"] == pytest.approx(0.0)
        assert row["grade_descent_pct"] == pytest.approx(0.0)
        assert row["grade_abs_pct"] == pytest.approx(0.0)

    def test_grade_decomposition_invariants(self, synthetic_grades_df: pl.DataFrame) -> None:
        """Mathematical invariants: climb >= 0, descent <= 0, abs == climb - descent, climb + descent == grade_pct."""
        res = compute_asymmetric_grades(synthetic_grades_df)

        climb = res["grade_climb_pct"].to_numpy()
        descent = res["grade_descent_pct"].to_numpy()
        grade_abs = res["grade_abs_pct"].to_numpy()
        grade = res["grade_pct"].to_numpy()

        assert np.all(climb >= 0.0)
        assert np.all(descent <= 0.0)
        np.testing.assert_allclose(climb + descent, grade, atol=1e-7)
        np.testing.assert_allclose(grade_abs, climb - descent, atol=1e-7)

    def test_missing_grade_column_raises(self) -> None:
        """Input without grade_pct must raise an informative error."""
        df = pl.DataFrame({"other_col": [1.0, 2.0]})
        with pytest.raises((ValueError, KeyError, pl.exceptions.ColumnNotFoundError)):
            compute_asymmetric_grades(df)


class TestClassifyGradeRegimes:
    """Tests for classify_grade_regimes."""

    def test_grade_regime_classification(self, synthetic_grades_df: pl.DataFrame) -> None:
        """Verify regime classification matches specification across thresholds."""
        res = classify_grade_regimes(synthetic_grades_df)
        assert "grade_class" in res.columns

        expected = [
            "steep_downgrade",  # -4.0 <= -2.5
            "steep_downgrade",  # -2.5 <= -2.5
            "moderate_downgrade",  # -2.5 < -2.49 < -0.5
            "moderate_downgrade",  # -2.5 < -1.5 < -0.5
            "flat_tangent",  # -0.5 <= -0.5 <= 0.5
            "flat_tangent",  # -0.5 <= 0.0 <= 0.5
            "flat_tangent",  # -0.5 <= 0.5 <= 0.5
            "moderate_upgrade",  # 0.5 < 0.51 < 2.5
            "moderate_upgrade",  # 0.5 < 2.0 < 2.5
            "steep_upgrade",  # 2.5 >= 2.5
            "steep_upgrade",  # 3.8 >= 2.5
        ]
        actual = res["grade_class"].to_list()
        assert actual == expected

    @pytest.mark.parametrize(
        ("grade", "expected_class"),
        [
            (-10.0, "steep_downgrade"),
            (-2.50001, "steep_downgrade"),
            (-2.5, "steep_downgrade"),
            (-2.49999, "moderate_downgrade"),
            (-0.50001, "moderate_downgrade"),
            (-0.5, "flat_tangent"),
            (0.0, "flat_tangent"),
            (0.5, "flat_tangent"),
            (0.50001, "moderate_upgrade"),
            (2.49999, "moderate_upgrade"),
            (2.5, "steep_upgrade"),
            (7.5, "steep_upgrade"),
        ],
    )
    def test_grade_regime_exact_boundaries(self, grade: float, expected_class: str) -> None:
        """Check exact boundary values for classification."""
        df = pl.DataFrame({"grade_pct": [grade]})
        res = classify_grade_regimes(df)
        assert res["grade_class"][0] == expected_class

    def test_missing_column_raises(self) -> None:
        """Input without grade_pct must raise error."""
        df = pl.DataFrame({"elevation": [10.0]})
        with pytest.raises((ValueError, KeyError, pl.exceptions.ColumnNotFoundError)):
            classify_grade_regimes(df)


class TestComputeTopographicalFrictionIndex:
    """Tests for compute_topographical_friction_index."""

    def test_default_weights_and_normalization(self) -> None:
        """TFI and sin_slope_angle calculation with default parameters."""
        df = pl.DataFrame(
            {
                "grade_pct": [3.0, -3.0, 0.0],
                "curvature_kappa": [0.002, 0.0, 0.004],
            }
        )
        res = compute_topographical_friction_index(df)

        assert "topographical_friction_index" in res.columns
        assert "sin_slope_angle" in res.columns

        # Row 0: 0.7 * (3.0/3.0) + 0.3 * (0.002/0.002) = 0.7 + 0.3 = 1.0
        # sin_slope_angle = 3.0 / 100.0 = 0.03
        assert res["topographical_friction_index"][0] == pytest.approx(1.0)
        assert res["sin_slope_angle"][0] == pytest.approx(0.03)

        # Row 1: 0.7 * (|-3.0|/3.0) + 0.3 * (0.0/0.002) = 0.7 + 0.0 = 0.7
        # sin_slope_angle = -3.0 / 100.0 = -0.03
        assert res["topographical_friction_index"][1] == pytest.approx(0.7)
        assert res["sin_slope_angle"][1] == pytest.approx(-0.03)

        # Row 2: 0.7 * (0.0/3.0) + 0.3 * (0.004/0.002) = 0.0 + 0.6 = 0.6
        # sin_slope_angle = 0.0 / 100.0 = 0.0
        assert res["topographical_friction_index"][2] == pytest.approx(0.6)
        assert res["sin_slope_angle"][2] == pytest.approx(0.0)

    def test_custom_weights_and_norms(self) -> None:
        """Custom weights and norms are applied correctly."""
        df = pl.DataFrame(
            {
                "grade_pct": [2.0],
                "curvature_kappa": [0.001],
            }
        )
        res = compute_topographical_friction_index(
            df,
            weight_grade=0.5,
            weight_curvature=0.5,
            grade_norm=2.0,
            curvature_norm=0.001,
        )
        # 0.5 * (2.0/2.0) + 0.5 * (0.001/0.001) = 0.5 + 0.5 = 1.0
        assert res["topographical_friction_index"][0] == pytest.approx(1.0)

    def test_non_negative_invariants(self, synthetic_grades_df: pl.DataFrame) -> None:
        """TFI must be strictly non-negative given non-negative kappa."""
        res = compute_topographical_friction_index(synthetic_grades_df)
        tfi = res["topographical_friction_index"].to_numpy()
        assert np.all(tfi >= 0.0)

    def test_zero_division_guard(self) -> None:
        """Non-positive norms must raise ValueError."""
        df = pl.DataFrame({"grade_pct": [1.0], "curvature_kappa": [0.001]})

        with pytest.raises(ValueError):
            compute_topographical_friction_index(df, grade_norm=0.0)

        with pytest.raises(ValueError):
            compute_topographical_friction_index(df, grade_norm=-1.0)

        with pytest.raises(ValueError):
            compute_topographical_friction_index(df, curvature_norm=0.0)

        with pytest.raises(ValueError):
            compute_topographical_friction_index(df, curvature_norm=-0.002)

    def test_missing_columns_raises(self) -> None:
        """Missing required input columns must raise an error."""
        with pytest.raises((ValueError, KeyError, pl.exceptions.ColumnNotFoundError)):
            compute_topographical_friction_index(pl.DataFrame({"grade_pct": [1.0]}))

        with pytest.raises((ValueError, KeyError, pl.exceptions.ColumnNotFoundError)):
            compute_topographical_friction_index(pl.DataFrame({"curvature_kappa": [0.001]}))


class TestBuildSpatialFeatureMart:
    """Tests for build_spatial_feature_mart."""

    def test_build_mart_with_synthetic_inputs(
        self, synthetic_mart_inputs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Build feature mart from synthetic inputs and verify joined schema and content."""
        stations_path, elevation_path = synthetic_mart_inputs
        output_file = tmp_path / "mart_output" / "mart.parquet"

        res = build_spatial_feature_mart(
            stations_path=stations_path,
            elevation_path=elevation_path,
            output_path=output_file,
            partition_by_corridor=False,
        )

        assert isinstance(res, pl.DataFrame)
        assert len(res) == 10

        # Required columns check
        expected_cols = {
            "segment_id",
            "corridor_id",
            "corridor",
            "chainage_m",
            "easting",
            "northing",
            "grade_pct",
            "grade_climb_pct",
            "grade_descent_pct",
            "grade_abs_pct",
            "grade_class",
            "curvature_kappa",
            "topographical_friction_index",
            "sin_slope_angle",
        }
        assert expected_cols.issubset(set(res.columns))

        # Output parquet file exists and is readable
        assert output_file.exists()
        reloaded = pl.read_parquet(output_file)
        assert len(reloaded) == len(res)

        # Invariant: corridor alias equals corridor_id
        assert (res["corridor"] == res["corridor_id"]).all()

        # Invariant: zero nulls across critical features
        for col in expected_cols:
            assert res[col].null_count() == 0, f"Nulls found in critical column {col}"
            if res[col].dtype.is_numeric():
                assert not res[col].is_nan().any(), f"NaNs found in numeric column {col}"

    def test_build_mart_partitioned(
        self, synthetic_mart_inputs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Build partitioned feature mart and verify directory structure and readability."""
        stations_path, elevation_path = synthetic_mart_inputs
        output_dir = tmp_path / "partitioned_mart"

        res = build_spatial_feature_mart(
            stations_path=stations_path,
            elevation_path=elevation_path,
            output_path=output_dir,
            partition_by_corridor=True,
        )

        assert isinstance(res, pl.DataFrame)
        assert len(res) == 10
        assert output_dir.exists()

        # Reading back from partitioned path should yield full dataset
        reloaded = pl.read_parquet(output_dir)
        assert len(reloaded) == len(res)

    def test_nonexistent_inputs_raise(self, tmp_path: Path) -> None:
        """Non-existent input paths must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            build_spatial_feature_mart(
                stations_path=tmp_path / "missing_stations.parquet",
                elevation_path=tmp_path / "missing_elevation.parquet",
            )

    @pytest.mark.skipif(
        not (CANONICAL_STATIONS_PATH.exists() and CANONICAL_ELEVATION_PATH.exists()),
        reason="Canonical interim parquet files not found",
    )
    def test_build_canonical_feature_mart(self, tmp_path: Path) -> None:
        """Test on actual canonical datasets: exactly 406 rows, 3 corridors, zero nulls/NaNs."""
        out_path = tmp_path / "canonical_mart.parquet"
        res = build_spatial_feature_mart(
            stations_path=CANONICAL_STATIONS_PATH,
            elevation_path=CANONICAL_ELEVATION_PATH,
            output_path=out_path,
            partition_by_corridor=False,
        )

        # Domain Invariant: Row conservation - exactly 406 rows across 3 corridors
        assert len(res) == 406

        corridors = set(res["corridor_id"].unique().to_list())
        expected_corridors = {"alb_i495", "i95_lorton_woodbridge", "i295_anacostia"}
        assert corridors == expected_corridors

        # Domain Invariant: UTM 18N (EPSG:26918) coordinates preserved
        assert res["easting"].min() > 200000.0  # reasonable UTM 18N easting
        assert res["northing"].min() > 4000000.0  # reasonable UTM 18N northing

        # Zero nulls / NaNs across all columns
        assert sum(res[col].null_count() for col in res.columns) == 0


class TestRegisterSpatialFeatureMart:
    """Tests for register_spatial_feature_mart."""

    def test_register_duckdb_view(
        self, synthetic_mart_inputs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Register mart with DuckDB and verify SQL analytical query execution."""
        stations_path, elevation_path = synthetic_mart_inputs
        output_file = tmp_path / "mart_for_duckdb.parquet"

        build_spatial_feature_mart(
            stations_path=stations_path,
            elevation_path=elevation_path,
            output_path=output_file,
            partition_by_corridor=False,
        )

        # Register with new connection
        con = register_spatial_feature_mart(
            con=None,
            mart_path=output_file,
            view_name="spatial_feature_mart",
        )
        assert isinstance(con, duckdb.DuckDBPyConnection)

        query = """
            SELECT
                corridor,
                COUNT(*) as segment_count,
                AVG(grade_pct) as avg_grade,
                AVG(topographical_friction_index) as avg_tfi
            FROM spatial_feature_mart
            GROUP BY corridor
        """
        result = con.execute(query).fetchall()
        assert len(result) == 1
        assert result[0][0] == "alb_i495"
        assert result[0][1] == 10

    def test_register_with_existing_connection(
        self, synthetic_mart_inputs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Register mart with custom view name in an existing connection."""
        stations_path, elevation_path = synthetic_mart_inputs
        output_file = tmp_path / "mart_custom.parquet"

        build_spatial_feature_mart(
            stations_path=stations_path,
            elevation_path=elevation_path,
            output_path=output_file,
            partition_by_corridor=False,
        )

        existing_con = duckdb.connect(":memory:")
        ret_con = register_spatial_feature_mart(
            con=existing_con,
            mart_path=output_file,
            view_name="custom_mart_view",
        )
        assert ret_con is existing_con

        count = ret_con.execute("SELECT COUNT(*) FROM custom_mart_view").fetchone()[0]
        assert count == 10

    def test_register_nonexistent_mart_raises(self, tmp_path: Path) -> None:
        """Registering non-existent parquet path must raise FileNotFoundError or duckdb error."""
        with pytest.raises((FileNotFoundError, duckdb.Error)):
            register_spatial_feature_mart(
                con=None,
                mart_path=tmp_path / "non_existent_mart.parquet",
            )
