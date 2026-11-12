"""Unit tests for bare-earth DEM and bridge deck LiDAR elevation profiling.

Validates 1m NAVD88 bare-earth DEM sampling, ASPRS Class 17 bridge deck LiDAR
extraction in UTM Zone 18N (EPSG:26918), VDOT/MDOT as-built benchmark validation,
Savitzky-Golay profile smoothing, raw discrete finite difference gradients,
noise filtering evaluation (Total Variation and Jerk reduction), vertical sag/crest
alignment classification, and the vectorized corridor processing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pytest
from shapely.geometry import Point

from topoflow.config import Settings
from topoflow.constants import CRS_METRIC
from topoflow.geo.elevation import (
    ElevationExtractor,
    classify_vertical_alignment,
    compute_raw_discrete_differences,
    evaluate_filtering_metrics,
    extract_bare_earth_dem,
    extract_bridge_deck_laz,
    process_corridor_stations,
    smooth_elevation_profile,
    validate_bridge_as_built,
)

# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to tests/fixtures directory."""
    return Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def configs_dir() -> Path:
    """Return path to configs directory in project root."""
    return Path(__file__).resolve().parents[3] / "configs"


@pytest.fixture
def mock_dem_path(fixtures_dir: Path) -> Path:
    """Return verified path to synthetic bare-earth DEM GeoTIFF."""
    path = fixtures_dir / "mock_dem.tif"
    assert path.exists(), f"Missing mock DEM fixture at {path}"
    return path


@pytest.fixture
def mock_laz_path(fixtures_dir: Path) -> Path:
    """Return verified path to synthetic bridge deck LAZ point cloud."""
    path = fixtures_dir / "mock_alb_laz.laz"
    assert path.exists(), f"Missing mock LAZ fixture at {path}"
    return path


@pytest.fixture
def as_built_csv_path(configs_dir: Path) -> Path:
    """Return verified path to official as-built bridge profile CSV."""
    path = configs_dir / "bridge_profile.csv"
    assert path.exists(), f"Missing as-built bridge profile fixture at {path}"
    return path


@pytest.fixture
def extractor() -> ElevationExtractor:
    """Return standard ElevationExtractor instance with default settings."""
    return ElevationExtractor()


@pytest.fixture
def dem_sample_points_gdf() -> gpd.GeoDataFrame:
    """Generate sample test points within mock DEM extent in EPSG:26918."""
    # Coordinates located squarely inside mock_dem.tif bounds:
    # [left=308000, bottom=4312000, right=314000, top=4319000]
    eastings = [310000.0, 310100.0, 310200.0, 310300.0, 310400.0]
    northings = [4315000.0, 4315100.0, 4315200.0, 4315300.0, 4315400.0]
    geometries = [Point(e, n) for e, n in zip(eastings, northings, strict=True)]
    return gpd.GeoDataFrame(
        {
            "station_index": list(range(len(geometries))),
            "chainage_m": [i * 100.0 for i in range(len(geometries))],
            "easting": eastings,
            "northing": northings,
        },
        geometry=geometries,
        crs=CRS_METRIC,
    )


@pytest.fixture
def bridge_deck_sample_points_gdf() -> gpd.GeoDataFrame:
    """Generate bridge centerline points hitting ASPRS Class 17 returns in mock LAZ."""
    # Points along American Legion Bridge centerline in UTM 18N where C17 returns exist:
    # LAZ C17 extent: X in [311152, 311338], Y in [4315400, 4316400]
    northings = [4315600.0, 4315800.0, 4316000.0, 4316200.0]
    eastings = [0.1706 * y - 425050.888 for y in northings]
    geometries = [Point(e, n) for e, n in zip(eastings, northings, strict=True)]
    return gpd.GeoDataFrame(
        {
            "segment_id": [f"alb_bridge_s{i:04d}" for i in range(len(geometries))],
            "corridor_id": ["alb_i495"] * len(geometries),
            "direction": ["inner_loop"] * len(geometries),
            "station_index": list(range(len(geometries))),
            "chainage_m": [i * 100.0 for i in range(len(geometries))],
            "easting": eastings,
            "northing": northings,
            "has_bridge": [True] * len(geometries),
        },
        geometry=geometries,
        crs=CRS_METRIC,
    )


# =============================================================================
# 1. Initialization Tests
# =============================================================================


class TestElevationExtractorInit:
    """Tests for ElevationExtractor lifecycle and configuration."""

    def test_default_initialization(self, extractor: ElevationExtractor) -> None:
        """Verify default instantiation uses standard project settings."""
        assert extractor is not None
        assert hasattr(extractor, "settings")
        assert extractor.settings.crs_metric == CRS_METRIC

    def test_custom_settings_initialization(self) -> None:
        """Verify initialization with explicitly provided Settings instance."""
        custom_settings = Settings()
        extractor = ElevationExtractor(settings=custom_settings)
        assert extractor.settings is custom_settings


# =============================================================================
# 2. Bare-Earth DEM Extraction Tests
# =============================================================================


class TestExtractBareEarthDem:
    """Tests for sampling 1m bare-earth DEM elevations in NAVD88 meters."""

    def test_extract_bare_earth_dem_valid_points(
        self,
        extractor: ElevationExtractor,
        dem_sample_points_gdf: gpd.GeoDataFrame,
        mock_dem_path: Path,
    ) -> None:
        """Verify sampling valid points inside DEM returns correct 1D array."""
        elevations = extractor.extract_bare_earth_dem(dem_sample_points_gdf, mock_dem_path)
        assert isinstance(elevations, np.ndarray)
        assert elevations.ndim == 1
        assert len(elevations) == len(dem_sample_points_gdf)
        assert np.issubdtype(elevations.dtype, np.floating)
        # All sampled elevations in this fixture region should be realistic Potomac elevations (~30-70m)
        assert np.all(np.isfinite(elevations))
        assert np.all((elevations >= 30.0) & (elevations <= 80.0))

    def test_extract_bare_earth_dem_empty_gdf(
        self,
        extractor: ElevationExtractor,
        mock_dem_path: Path,
    ) -> None:
        """Verify extracting from empty GeoDataFrame returns empty 1D float array."""
        empty_gdf = gpd.GeoDataFrame(
            {"station_index": [], "chainage_m": []},
            geometry=[],
            crs=CRS_METRIC,
        )
        elevations = extractor.extract_bare_earth_dem(empty_gdf, mock_dem_path)
        assert isinstance(elevations, np.ndarray)
        assert len(elevations) == 0

    def test_extract_bare_earth_dem_missing_raster_raises(
        self,
        extractor: ElevationExtractor,
        dem_sample_points_gdf: gpd.GeoDataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify non-existent DEM raster path raises FileNotFoundError."""
        missing_path = tmp_path / "non_existent_dem.tif"
        with pytest.raises(FileNotFoundError):
            extractor.extract_bare_earth_dem(dem_sample_points_gdf, missing_path)

    def test_extract_bare_earth_dem_standalone_function(
        self,
        dem_sample_points_gdf: gpd.GeoDataFrame,
        mock_dem_path: Path,
    ) -> None:
        """Verify standalone extract_bare_earth_dem function works equivalently."""
        elevations = extract_bare_earth_dem(dem_sample_points_gdf, mock_dem_path)
        assert isinstance(elevations, np.ndarray)
        assert len(elevations) == len(dem_sample_points_gdf)
        assert np.all(np.isfinite(elevations))


# =============================================================================
# 3. ASPRS Class 17 Bridge Deck LiDAR Extraction Tests
# =============================================================================


class TestExtractBridgeDeckLaz:
    """Tests for asphalt deck extraction strictly filtering ASPRS Class 17 returns."""

    def test_extract_bridge_deck_laz_asprs_class_17_filtering(
        self,
        extractor: ElevationExtractor,
        bridge_deck_sample_points_gdf: gpd.GeoDataFrame,
        mock_laz_path: Path,
    ) -> None:
        """Verify extracted elevations reflect Class 17 deck (~42-45m) and reject Class 2 ground (~6m)."""
        deck_elevations = extractor.extract_bridge_deck_laz(
            bridge_deck_sample_points_gdf,
            mock_laz_path,
            buffer_radius_m=15.0,
        )
        assert isinstance(deck_elevations, np.ndarray)
        assert len(deck_elevations) == len(bridge_deck_sample_points_gdf)
        # Class 17 bridge deck returns in fixture are strictly between 42.0m and 46.0m
        # Ground (Class 2) returns under bridge are ~4.7 - 8.2m; assert deck is strictly separated
        assert np.all((deck_elevations >= 42.0) & (deck_elevations <= 46.0))
        assert not np.any(deck_elevations < 20.0), (
            "Ground returns leaked into bridge deck sampling!"
        )

    def test_extract_bridge_deck_laz_standalone_function(
        self,
        bridge_deck_sample_points_gdf: gpd.GeoDataFrame,
        mock_laz_path: Path,
    ) -> None:
        """Verify standalone extract_bridge_deck_laz function works equivalently."""
        deck_elevations = extract_bridge_deck_laz(
            bridge_deck_sample_points_gdf,
            mock_laz_path,
            buffer_radius_m=15.0,
        )
        assert len(deck_elevations) == len(bridge_deck_sample_points_gdf)
        assert np.all((deck_elevations >= 42.0) & (deck_elevations <= 46.0))

    def test_extract_bridge_deck_laz_no_returns_in_buffer(
        self,
        extractor: ElevationExtractor,
        mock_laz_path: Path,
    ) -> None:
        """Verify points far outside the bridge footprint return NaN for missing returns."""
        far_gdf = gpd.GeoDataFrame(
            {"station_index": [0], "chainage_m": [0.0]},
            geometry=[Point(300000.0, 4300000.0)],
            crs=CRS_METRIC,
        )
        deck_elevations = extractor.extract_bridge_deck_laz(
            far_gdf,
            mock_laz_path,
            buffer_radius_m=15.0,
        )
        assert len(deck_elevations) == 1
        assert np.isnan(deck_elevations[0])

    def test_extract_bridge_deck_laz_missing_file_raises(
        self,
        extractor: ElevationExtractor,
        bridge_deck_sample_points_gdf: gpd.GeoDataFrame,
        tmp_path: Path,
    ) -> None:
        """Verify non-existent LAZ file raises FileNotFoundError."""
        missing_laz = tmp_path / "non_existent.laz"
        with pytest.raises(FileNotFoundError):
            extractor.extract_bridge_deck_laz(bridge_deck_sample_points_gdf, missing_laz)


# =============================================================================
# 4. As-Built Benchmark Validation Tests
# =============================================================================


class TestValidateBridgeAsBuilt:
    """Tests for cross-referencing extracted deck heights against VDOT/MDOT as-built reference."""

    def test_validate_bridge_as_built_exact_match(
        self,
        extractor: ElevationExtractor,
        as_built_csv_path: Path,
    ) -> None:
        """Verify extracted heights matching benchmark within tolerance return True."""
        df_as_built = pl.read_csv(as_built_csv_path, comment_prefix="#")
        chainage = df_as_built["station_offset_m"].to_numpy()
        reference_z = df_as_built["elevation_navd88_m"].to_numpy()

        is_valid = extractor.validate_bridge_as_built(
            extracted_deck_z=reference_z,
            bridge_stations_chainage_m=chainage,
            as_built_csv_path=as_built_csv_path,
            tolerance_m=0.2,
        )
        assert is_valid is True

    def test_validate_bridge_as_built_standalone_function(
        self,
        as_built_csv_path: Path,
    ) -> None:
        """Verify standalone validate_bridge_as_built function works equivalently."""
        df_as_built = pl.read_csv(as_built_csv_path, comment_prefix="#")
        chainage = df_as_built["station_offset_m"].to_numpy()
        reference_z = df_as_built["elevation_navd88_m"].to_numpy()

        is_valid = validate_bridge_as_built(
            extracted_deck_z=reference_z,
            bridge_stations_chainage_m=chainage,
            as_built_csv_path=as_built_csv_path,
            tolerance_m=0.2,
        )
        assert is_valid is True

    def test_validate_bridge_as_built_exceeds_tolerance_raises(
        self,
        extractor: ElevationExtractor,
        as_built_csv_path: Path,
    ) -> None:
        """Verify error exceeding tolerance_m raises ValueError or AssertionError."""
        df_as_built = pl.read_csv(as_built_csv_path, comment_prefix="#")
        chainage = df_as_built["station_offset_m"].to_numpy()
        reference_z = df_as_built["elevation_navd88_m"].to_numpy()
        bad_z = reference_z.copy()
        bad_z[4] += 0.55  # 55cm discrepancy exceeds 20cm tolerance

        with pytest.raises((ValueError, AssertionError)):
            extractor.validate_bridge_as_built(
                extracted_deck_z=bad_z,
                bridge_stations_chainage_m=chainage,
                as_built_csv_path=as_built_csv_path,
                tolerance_m=0.2,
            )

    def test_validate_bridge_as_built_missing_csv_raises(
        self,
        extractor: ElevationExtractor,
        tmp_path: Path,
    ) -> None:
        """Verify missing as-built CSV raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extractor.validate_bridge_as_built(
                extracted_deck_z=np.array([42.5]),
                bridge_stations_chainage_m=np.array([0.0]),
                as_built_csv_path=tmp_path / "missing.csv",
                tolerance_m=0.2,
            )


# =============================================================================
# 5. Savitzky-Golay Smoothing & Synthetic AASHTO Sag Curve Verification
# =============================================================================


class TestSmoothElevationProfile:
    """Tests for Savitzky-Golay polynomial profile smoothing and mathematical gradient recovery."""

    @pytest.fixture
    def synthetic_sag_curve(self) -> dict[str, np.ndarray]:
        """Generate known AASHTO vertical sag curve z(x) = z0 + g1*x + ((g2 - g1)/(2L))*x^2.

        Parameters model I-495 American Legion Bridge sag curve:
        - g1 = -3.2% (-0.032)
        - g2 = +3.8% (+0.038)
        - Length L = 1000m
        - Station spacing dx = 10m (101 stations)
        - Gaussian LiDAR/DEM noise sigma = 0.05m (5cm)
        """
        dx = 10.0
        chainage_m = np.arange(0.0, 1000.0 + dx, dx)
        length_m = 1000.0
        g1 = -0.032
        g2 = 0.038
        z0 = 50.0

        analytical_z = z0 + g1 * chainage_m + ((g2 - g1) / (2.0 * length_m)) * (chainage_m**2)
        analytical_grade_pct = 100.0 * (g1 + ((g2 - g1) / length_m) * chainage_m)
        analytical_curvature = np.full_like(chainage_m, (g2 - g1) / length_m)

        # Deterministic pseudo-random seed for repeatable clean-room validation
        rng = np.random.default_rng(42)
        noisy_z = analytical_z + rng.normal(loc=0.0, scale=0.05, size=len(chainage_m))

        return {
            "chainage_m": chainage_m,
            "analytical_z": analytical_z,
            "noisy_z": noisy_z,
            "analytical_grade_pct": analytical_grade_pct,
            "analytical_curvature": analytical_curvature,
        }

    def test_smooth_elevation_profile_standalone_function(
        self,
        synthetic_sag_curve: dict[str, np.ndarray],
    ) -> None:
        """Verify standalone smooth_elevation_profile function works equivalently."""
        chainage = synthetic_sag_curve["chainage_m"]
        clean_z = synthetic_sag_curve["analytical_z"]

        z_smooth, grade_pct, curvature = smooth_elevation_profile(
            chainage,
            clean_z,
            window_length=5,
            polyorder=2,
        )
        assert len(z_smooth) == len(chainage)
        assert len(grade_pct) == len(chainage)
        assert len(curvature) == len(chainage)

    def test_smooth_elevation_profile_noise_reduction_rmse(
        self,
        extractor: ElevationExtractor,
        synthetic_sag_curve: dict[str, np.ndarray],
    ) -> None:
        """Assert Savitzky-Golay grade RMSE is strictly < 0.6 * raw finite difference RMSE."""
        chainage = synthetic_sag_curve["chainage_m"]
        noisy_z = synthetic_sag_curve["noisy_z"]
        true_grade = synthetic_sag_curve["analytical_grade_pct"]

        # 1. Raw central differences
        raw_grade = extractor.compute_raw_discrete_differences(chainage, noisy_z)

        # 2. Savitzky-Golay smoothing
        _, sg_grade, _ = extractor.smooth_elevation_profile(
            chainage,
            noisy_z,
            window_length=5,
            polyorder=2,
        )

        rmse_raw = float(np.sqrt(np.mean((raw_grade - true_grade) ** 2)))
        rmse_sg = float(np.sqrt(np.mean((sg_grade - true_grade) ** 2)))

        assert rmse_raw > 0.0
        assert rmse_sg < 0.6 * rmse_raw, (
            f"Expected RMSE_sg < 0.6 * RMSE_raw, but got RMSE_sg={rmse_sg:.4f}, RMSE_raw={rmse_raw:.4f}"
        )


# =============================================================================
# 6. Raw Discrete Finite Differences Tests
# =============================================================================


class TestComputeRawDiscreteDifferences:
    """Tests for finite difference central gradient computation."""

    def test_raw_discrete_differences_constant_linear_grade(
        self,
        extractor: ElevationExtractor,
    ) -> None:
        """Verify constant 3.0% slope returns 3.0% grade across all stations."""
        x = np.arange(0.0, 1000.0, 100.0)
        # z(x) = 10.0 + 0.03 * x => dz/dx = 0.03 => grade = 3.0%
        z = 10.0 + 0.03 * x
        grade_raw_pct = extractor.compute_raw_discrete_differences(x, z)

        assert len(grade_raw_pct) == len(x)
        np.testing.assert_allclose(grade_raw_pct, 3.0, atol=1e-5)

    def test_raw_discrete_differences_standalone_function(self) -> None:
        """Verify standalone compute_raw_discrete_differences function works equivalently."""
        x = np.arange(0.0, 500.0, 100.0)
        z = 10.0 + 0.03 * x
        grade = compute_raw_discrete_differences(x, z)
        assert len(grade) == len(x)
        np.testing.assert_allclose(grade, 3.0, atol=1e-5)


# =============================================================================
# 7. Filtering Metrics & Total Variation Reduction Tests
# =============================================================================


class TestEvaluateFilteringMetrics:
    """Tests for Total Variation (TV) and Jerk noise filtering metrics."""

    def test_evaluate_filtering_metrics_standalone_function(self) -> None:
        """Verify standalone evaluate_filtering_metrics function works equivalently."""
        chainage = np.arange(0.0, 500.0, 100.0)
        raw_z = np.array([50.0, 48.0, 47.0, 48.0, 51.0])
        raw_grade = np.array([-2.0, -1.5, 0.0, 1.5, 3.0])
        sg_grade = np.array([-2.0, -1.4, 0.1, 1.4, 2.9])

        metrics = evaluate_filtering_metrics(chainage, raw_z, raw_grade, sg_grade)
        assert "tv_reduction_pct" in metrics

    def test_evaluate_filtering_metrics_tv_reduction_greater_than_50_pct(
        self,
        extractor: ElevationExtractor,
    ) -> None:
        """Assert tv_reduction_pct > 50.0 on noisy AASHTO sag curve."""
        dx = 10.0
        x = np.arange(0.0, 1000.0 + dx, dx)
        L = 1000.0
        g1, g2 = -0.032, 0.038
        z_true = 50.0 + g1 * x + ((g2 - g1) / (2.0 * L)) * (x**2)

        rng = np.random.default_rng(42)
        z_noisy = z_true + rng.normal(0.0, 0.05, size=len(x))

        raw_grade = extractor.compute_raw_discrete_differences(x, z_noisy)
        _, sg_grade, _ = extractor.smooth_elevation_profile(
            x,
            z_noisy,
            window_length=5,
            polyorder=2,
        )

        metrics = extractor.evaluate_filtering_metrics(x, z_noisy, raw_grade, sg_grade)

        assert metrics["total_variation_savgol"] < metrics["total_variation_raw"]
        assert metrics["jerk_savgol"] < metrics["jerk_raw"]
        assert metrics["tv_reduction_pct"] > 50.0, (
            f"Expected tv_reduction_pct > 50.0, got {metrics['tv_reduction_pct']:.2f}%"
        )


# =============================================================================
# 8. Vertical Alignment Classification Tests
# =============================================================================


class TestClassifyVerticalAlignment:
    """Tests for classifying sag curves, crest curves, and inflection zones."""

    def test_classify_vertical_alignment_sag_curve(
        self,
        extractor: ElevationExtractor,
    ) -> None:
        """Verify sag curve has is_sag=True at inflection bottom and is_crest=False."""
        chainage = np.arange(0.0, 1000.0, 50.0)
        # Sag curve: grade transitions from -3.0% to +3.0%, vertical curvature > 0
        grade_pct = np.linspace(-3.0, 3.0, len(chainage))
        curvature = np.full_like(chainage, 0.006)  # Positive curvature = sag

        is_sag, is_crest = extractor.classify_vertical_alignment(chainage, grade_pct, curvature)

        assert isinstance(is_sag, np.ndarray)
        assert isinstance(is_crest, np.ndarray)
        assert is_sag.dtype == bool
        assert is_crest.dtype == bool
        assert np.any(is_sag)
        assert not np.any(is_crest)

    def test_classify_vertical_alignment_crest_curve(
        self,
        extractor: ElevationExtractor,
    ) -> None:
        """Verify crest curve has is_crest=True at summit and is_sag=False."""
        chainage = np.arange(0.0, 1000.0, 50.0)
        # Crest curve: grade transitions from +3.0% to -3.0%, vertical curvature < 0
        grade_pct = np.linspace(3.0, -3.0, len(chainage))
        curvature = np.full_like(chainage, -0.006)  # Negative curvature = crest

        is_sag, is_crest = extractor.classify_vertical_alignment(chainage, grade_pct, curvature)

        assert np.any(is_crest)
        assert not np.any(is_sag)

    def test_classify_vertical_alignment_standalone_function(self) -> None:
        """Verify standalone classify_vertical_alignment function works equivalently."""
        chainage = np.arange(0.0, 500.0, 100.0)
        grade_pct = np.zeros_like(chainage)
        curvature = np.zeros_like(chainage)

        is_sag, is_crest = classify_vertical_alignment(chainage, grade_pct, curvature)
        assert not np.any(is_sag)
        assert not np.any(is_crest)


# =============================================================================
# 9. End-to-End Vectorized Corridor Station Pipeline Tests
# =============================================================================


class TestProcessCorridorStationsPipeline:
    """Tests for process_corridor_stations vectorized end-to-end integration."""

    @pytest.fixture
    def sample_corridor_stations_gdf(
        self,
        dem_sample_points_gdf: gpd.GeoDataFrame,
        bridge_deck_sample_points_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Combine bare-earth and bridge points into a multi-station corridor GDF."""
        gdf_non_bridge = dem_sample_points_gdf.copy()
        gdf_non_bridge["segment_id"] = [f"alb_road_s{i:04d}" for i in range(len(gdf_non_bridge))]
        gdf_non_bridge["corridor_id"] = "alb_i495"
        gdf_non_bridge["direction"] = "inner_loop"
        gdf_non_bridge["has_bridge"] = False

        combined = gpd.GeoDataFrame(
            pd.concat([gdf_non_bridge, bridge_deck_sample_points_gdf], ignore_index=True),
            geometry="geometry",
            crs=CRS_METRIC,
        )
        # Update station indices and continuous chainage
        combined["station_index"] = list(range(len(combined)))
        combined["chainage_m"] = [i * 100.0 for i in range(len(combined))]
        return combined

    def test_process_corridor_stations_schema_and_columns(
        self,
        extractor: ElevationExtractor,
        sample_corridor_stations_gdf: gpd.GeoDataFrame,
        mock_dem_path: Path,
        mock_laz_path: Path,
    ) -> None:
        """Verify output Polars DataFrame strictly matches required 15-column schema."""
        expected_columns = [
            "segment_id",
            "corridor_id",
            "direction",
            "station_index",
            "chainage_m",
            "easting",
            "northing",
            "elevation_raw_m",
            "elevation_smooth_m",
            "grade_pct",
            "grade_raw_pct",
            "vertical_curvature",
            "is_sag",
            "is_crest",
            "is_bridge_deck",
        ]

        df_result = extractor.process_corridor_stations(
            sample_corridor_stations_gdf,
            dem_path=mock_dem_path,
            laz_path=mock_laz_path,
        )

        assert isinstance(df_result, pl.DataFrame)
        assert len(df_result) == len(sample_corridor_stations_gdf)
        assert df_result.columns == expected_columns

    def test_process_corridor_stations_bridge_flagging_and_deck_elevations(
        self,
        extractor: ElevationExtractor,
        sample_corridor_stations_gdf: gpd.GeoDataFrame,
        mock_dem_path: Path,
        mock_laz_path: Path,
    ) -> None:
        """Verify bridge stations receive is_bridge_deck=True and deck heights (~42-45m)."""
        df_result = extractor.process_corridor_stations(
            sample_corridor_stations_gdf,
            dem_path=mock_dem_path,
            laz_path=mock_laz_path,
        )

        bridge_rows = df_result.filter(pl.col("is_bridge_deck"))
        road_rows = df_result.filter(~pl.col("is_bridge_deck"))

        assert len(bridge_rows) > 0
        assert len(road_rows) > 0

        # Bridge deck elevations should be asphalt returns ~42-45m
        bridge_elevations = bridge_rows["elevation_smooth_m"].to_numpy()
        assert np.all((bridge_elevations >= 42.0) & (bridge_elevations <= 46.0))

    def test_process_corridor_stations_standalone_function(
        self,
        sample_corridor_stations_gdf: gpd.GeoDataFrame,
        mock_dem_path: Path,
        mock_laz_path: Path,
    ) -> None:
        """Verify module-level process_corridor_stations function works identically."""
        df_result = process_corridor_stations(
            sample_corridor_stations_gdf,
            dem_path=mock_dem_path,
            laz_path=mock_laz_path,
        )
        assert isinstance(df_result, pl.DataFrame)
        assert "elevation_smooth_m" in df_result.columns
