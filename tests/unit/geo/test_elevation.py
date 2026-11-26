"""Unit tests for bare-earth DEM and bridge deck LiDAR elevation profiling.

Validates 1m NAVD88 bare-earth DEM sampling, ASPRS Class 17 bridge deck LiDAR
extraction in UTM Zone 18N (EPSG:26918), VDOT/MDOT as-built benchmark validation,
Savitzky-Golay profile smoothing, raw discrete finite difference gradients,
noise filtering evaluation (Total Variation and Jerk reduction), vertical sag/crest
alignment classification, and the vectorized corridor processing pipeline.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pyproj
import pytest
from shapely.geometry import Point

from topoflow.config import Settings
from topoflow.constants import CRS_METRIC
from topoflow.geo.elevation import (
    ELEVATION_SOURCE_UNDER_BRIDGE,
    EPQS_MAX_STATION_OFFSET_M,
    ElevationExtractor,
    ElevationExtractorError,
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

    def test_extract_bridge_deck_laz_default_buffer_radius_35m(
        self,
        extractor: ElevationExtractor,
        mock_laz_path: Path,
    ) -> None:
        """Verify extract_bridge_deck_laz defaults to buffer_radius_m=35.0 and queries within 35m."""
        sig = inspect.signature(extractor.extract_bridge_deck_laz)
        assert sig.parameters["buffer_radius_m"].default == 35.0

        # Construct a station offset ~22.0 meters outside the C17 return boundary (X_max ~311252.0)
        # at northing 4315800.0:
        northing = 4315800.0
        offset_easting = 311252.0 + 22.0

        offset_gdf = gpd.GeoDataFrame(
            {
                "segment_id": ["alb_offset_s0001"],
                "corridor_id": ["alb_i495"],
                "direction": ["inner_loop"],
                "station_index": [0],
                "chainage_m": [3900.0],
                "easting": [offset_easting],
                "northing": [northing],
            },
            geometry=[Point(offset_easting, northing)],
            crs=CRS_METRIC,
        )

        # With tight 15m radius, a 22m offset point finds no returns (returns NaN)
        tight_elevations = extractor.extract_bridge_deck_laz(
            offset_gdf,
            mock_laz_path,
            buffer_radius_m=15.0,
        )
        assert len(tight_elevations) == 1
        assert np.isnan(tight_elevations[0])

        # With default buffer_radius_m=35.0 (omitted argument), the point captures C17 returns
        default_elevations = extractor.extract_bridge_deck_laz(
            offset_gdf,
            mock_laz_path,
        )
        assert len(default_elevations) == 1
        assert np.isfinite(default_elevations[0])
        assert 42.0 <= default_elevations[0] <= 46.0


# =============================================================================
# 4. As-Built Benchmark Validation Tests
# =============================================================================


def _write_as_built_csv(
    path: Path,
    offsets: list[float],
    elevations: list[float],
    sources: list[str] | None = None,
    *,
    include_source: bool = True,
) -> Path:
    """Write a synthetic as-built profile CSV with a leading comment block."""
    header = ["station_offset_m", "elevation_navd88_m"]
    if include_source:
        header.append("source")
    rows = ["# Synthetic as-built profile for unit tests", "# units: metres NAVD88"]
    rows.append(",".join(header))
    for i, (off, z) in enumerate(zip(offsets, elevations, strict=True)):
        cells = [f"{off}", f"{z}"]
        if include_source:
            cells.append(sources[i] if sources is not None else "synthetic plan sheet")
        rows.append(",".join(cells))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


AS_BUILT_OFFSETS = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0]
AS_BUILT_Z = [42.10, 42.35, 42.60, 42.80, 42.95, 43.05]


class TestValidateBridgeAsBuilt:
    """Tests for cross-referencing extracted deck heights against an as-built reference CSV."""

    @pytest.fixture
    def as_built_csv_path(self, tmp_path: Path) -> Path:
        return _write_as_built_csv(tmp_path / "as_built.csv", AS_BUILT_OFFSETS, AS_BUILT_Z)

    def test_exact_match(self, extractor: ElevationExtractor, as_built_csv_path: Path) -> None:
        is_valid = extractor.validate_bridge_as_built(
            extracted_deck_z=np.array(AS_BUILT_Z),
            bridge_stations_chainage_m=np.array(AS_BUILT_OFFSETS),
            as_built_csv_path=as_built_csv_path,
            tolerance_m=0.2,
        )
        assert is_valid is True

    def test_within_tolerance(self, extractor: ElevationExtractor, as_built_csv_path: Path) -> None:
        z = np.array(AS_BUILT_Z) + np.array([0.1, -0.15, 0.0, 0.19, -0.05, 0.0])
        assert (
            extractor.validate_bridge_as_built(
                extracted_deck_z=z,
                bridge_stations_chainage_m=np.array(AS_BUILT_OFFSETS),
                as_built_csv_path=as_built_csv_path,
                tolerance_m=0.2,
            )
            is True
        )

    def test_standalone_function(self, as_built_csv_path: Path) -> None:
        is_valid = validate_bridge_as_built(
            extracted_deck_z=np.array(AS_BUILT_Z),
            bridge_stations_chainage_m=np.array(AS_BUILT_OFFSETS),
            as_built_csv_path=as_built_csv_path,
            tolerance_m=0.2,
        )
        assert is_valid is True

    def test_exceeds_tolerance_raises(
        self, extractor: ElevationExtractor, as_built_csv_path: Path
    ) -> None:
        bad_z = np.array(AS_BUILT_Z)
        bad_z[4] += 0.55  # 55 cm discrepancy exceeds 20 cm tolerance
        with pytest.raises(ValueError):
            extractor.validate_bridge_as_built(
                extracted_deck_z=bad_z,
                bridge_stations_chainage_m=np.array(AS_BUILT_OFFSETS),
                as_built_csv_path=as_built_csv_path,
                tolerance_m=0.2,
            )

    def test_missing_csv_raises(self, extractor: ElevationExtractor, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extractor.validate_bridge_as_built(
                extracted_deck_z=np.array([42.5]),
                bridge_stations_chainage_m=np.array([0.0]),
                as_built_csv_path=tmp_path / "missing.csv",
                tolerance_m=0.2,
            )

    def test_missing_source_column_raises(
        self, extractor: ElevationExtractor, tmp_path: Path
    ) -> None:
        path = _write_as_built_csv(
            tmp_path / "no_source.csv", AS_BUILT_OFFSETS, AS_BUILT_Z, include_source=False
        )
        with pytest.raises(ValueError):
            extractor.validate_bridge_as_built(
                extracted_deck_z=np.array(AS_BUILT_Z),
                bridge_stations_chainage_m=np.array(AS_BUILT_OFFSETS),
                as_built_csv_path=path,
                tolerance_m=0.2,
            )

    @pytest.mark.parametrize("bad_source", ["", "   "])
    def test_blank_or_null_source_raises(
        self, extractor: ElevationExtractor, tmp_path: Path, bad_source: str
    ) -> None:
        sources = ["plan sheet"] * len(AS_BUILT_OFFSETS)
        sources[2] = bad_source
        path = _write_as_built_csv(tmp_path / "blank.csv", AS_BUILT_OFFSETS, AS_BUILT_Z, sources)
        with pytest.raises(ValueError):
            validate_bridge_as_built(
                extracted_deck_z=np.array(AS_BUILT_Z),
                bridge_stations_chainage_m=np.array(AS_BUILT_OFFSETS),
                as_built_csv_path=path,
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

PIPELINE_COLUMNS = [
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
    "elevation_source",
    "grade_edge_effect",
]

# Synthetic station layout in UTM 18N (well away from any real fixture coverage).
SYN_E0 = 500000.0
SYN_N0 = 4200000.0


def _stations(
    n: int,
    *,
    corridor_id: str = "i95_lorton_woodbridge",
    direction: str = "northbound",
    has_bridge: list[bool] | None = None,
    spacing_m: float = 100.0,
    e0: float = SYN_E0,
    n0: float = SYN_N0,
    chainage: list[float] | None = None,
    include_has_bridge: bool = True,
) -> gpd.GeoDataFrame:
    """Build a due-north synthetic carriageway of ``n`` stations in EPSG:26918."""
    ch = np.asarray(chainage if chainage is not None else [i * spacing_m for i in range(n)])
    northings = n0 + ch
    eastings = np.full(n, e0)
    data: dict[str, object] = {
        "segment_id": [f"{corridor_id}_{direction}_s{i:04d}" for i in range(n)],
        "corridor_id": [corridor_id] * n,
        "direction": [direction] * n,
        "station_index": list(range(n)),
        "chainage_m": ch.astype(float),
        "easting": eastings,
        "northing": northings,
    }
    if include_has_bridge:
        data["has_bridge"] = has_bridge if has_bridge is not None else [False] * n
    return gpd.GeoDataFrame(
        data,
        geometry=[Point(e, y) for e, y in zip(eastings, northings, strict=True)],
        crs=CRS_METRIC,
    )


def _write_epqs(path: Path, segment_ids: list[str], elevations: list[float | None]) -> Path:
    pl.DataFrame(
        {"segment_id": segment_ids, "elevation_usgs_m": elevations},
        schema={"segment_id": pl.String, "elevation_usgs_m": pl.Float64},
    ).write_parquet(path)
    return path


def _write_epqs_with_coords(
    path: Path,
    gdf: gpd.GeoDataFrame,
    elevations: list[float],
    *,
    shift_m: float = 0.0,
    shift_index: int | None = None,
    shift_index_m: float = 0.0,
) -> Path:
    """Write an EPQS cache with longitude/latitude of each station, shifted east by shift_m."""
    to_wgs84 = pyproj.Transformer.from_crs(CRS_METRIC, "EPSG:4326", always_xy=True)
    eastings = gdf["easting"].to_numpy(dtype=float) + shift_m
    if shift_index is not None:
        eastings[shift_index] += shift_index_m
    lon, lat = to_wgs84.transform(eastings, gdf["northing"].to_numpy(dtype=float))
    pl.DataFrame(
        {
            "segment_id": gdf["segment_id"].tolist(),
            "elevation_usgs_m": elevations,
            "longitude": np.asarray(lon, dtype=float),
            "latitude": np.asarray(lat, dtype=float),
        }
    ).write_parquet(path)
    return path


def _write_plane_dem(
    path: Path,
    *,
    base_z: float = 20.0,
    slope: float = 0.01,
    e0: float = SYN_E0,
    n0: float = SYN_N0,
    extent_m: float = 3000.0,
    res_m: float = 1.0,
) -> Path:
    """Write a GeoTIFF whose value is base_z + slope * (northing - n0) in EPSG:26918."""
    import rasterio
    from rasterio.transform import from_origin

    width = int(200 / res_m)
    height = int(extent_m / res_m)
    west = e0 - 100.0
    north = n0 + extent_m - 100.0
    row_centres_n = north - (np.arange(height) + 0.5) * res_m
    col = base_z + slope * (row_centres_n - n0)
    data = np.repeat(col[:, None], width, axis=1).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS_METRIC,
        transform=from_origin(west, north, res_m, res_m),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def _write_c17_laz(
    path: Path, centres: list[tuple[float, float]], deck_z: list[float], *, ground_z: float = 2.0
) -> Path:
    """Write a LAZ with Class 17 deck returns around each centre plus Class 2 ground below."""
    import laspy

    rng = np.random.default_rng(7)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    cls: list[np.ndarray] = []
    for (cx, cy), z in zip(centres, deck_z, strict=True):
        k = 40
        xs.append(cx + rng.uniform(-5.0, 5.0, k))
        ys.append(cy + rng.uniform(-5.0, 5.0, k))
        zs.append(np.full(k, z))
        cls.append(np.full(k, 17, dtype=np.uint8))
        xs.append(cx + rng.uniform(-5.0, 5.0, k))
        ys.append(cy + rng.uniform(-5.0, 5.0, k))
        zs.append(np.full(k, ground_z))
        cls.append(np.full(k, 2, dtype=np.uint8))
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([SYN_E0, SYN_N0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    header.add_crs(pyproj.CRS.from_user_input(CRS_METRIC))
    las = laspy.LasData(header)
    las.x = np.concatenate(xs)
    las.y = np.concatenate(ys)
    las.z = np.concatenate(zs)
    las.classification = np.concatenate(cls)
    las.write(str(path))
    return path


def _by_index(df: pl.DataFrame) -> dict[int, dict]:
    return {row["station_index"]: row for row in df.iter_rows(named=True)}


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
        combined["station_index"] = list(range(len(combined)))
        combined["chainage_m"] = [i * 100.0 for i in range(len(combined))]
        return combined

    def test_schema_17_columns_epqs_only(self, tmp_path: Path) -> None:
        gdf = _stations(8)
        epqs = _write_epqs(
            tmp_path / "epqs.parquet", gdf["segment_id"].tolist(), [30.0 + i for i in range(8)]
        )
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs)
        assert isinstance(df, pl.DataFrame)
        assert df.columns == PIPELINE_COLUMNS
        assert len(df) == len(gdf)
        assert df["is_bridge_deck"].dtype == pl.Boolean
        assert df["grade_edge_effect"].dtype == pl.Boolean

    def test_mock_fixture_bridge_deck_elevations(
        self,
        extractor: ElevationExtractor,
        sample_corridor_stations_gdf: gpd.GeoDataFrame,
        mock_dem_path: Path,
        mock_laz_path: Path,
    ) -> None:
        """Bridge stations over Class 17 returns take LiDAR deck heights (~42-46 m)."""
        df = extractor.process_corridor_stations(
            sample_corridor_stations_gdf, dem_path=mock_dem_path, laz_path=mock_laz_path
        )
        assert df.columns == PIPELINE_COLUMNS
        bridge = df.filter(pl.col("is_bridge_deck"))
        road = df.filter(~pl.col("is_bridge_deck"))
        assert len(bridge) == 4
        assert len(road) == 5
        z = bridge["elevation_raw_m"].to_numpy()
        assert np.all((z >= 42.0) & (z <= 46.0))
        assert set(bridge["elevation_source"].to_list()) == {"lidar_c17"}
        # LiDAR never applies to non-bridge stations.
        assert set(road["elevation_source"].to_list()) == {"usgs_dem"}

    def test_standalone_matches_method(self, tmp_path: Path) -> None:
        gdf = _stations(6)
        epqs = _write_epqs(
            tmp_path / "epqs.parquet", gdf["segment_id"].tolist(), [10.0, 11, 12, 13, 14, 15]
        )
        a = process_corridor_stations(gdf, usgs_epqs_path=epqs)
        b = ElevationExtractor().process_corridor_stations(gdf, usgs_epqs_path=epqs)
        assert a.equals(b)

    # --- Source priority -------------------------------------------------------------

    def test_epqs_values_used_verbatim(self, tmp_path: Path) -> None:
        gdf = _stations(7)
        z = [31.2, 30.8, 29.5, 28.0, 29.1, 30.3, 31.9]
        epqs = _write_epqs(tmp_path / "epqs.parquet", gdf["segment_id"].tolist(), z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("chainage_m")
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), z)
        assert set(df["elevation_source"].to_list()) == {"usgs_epqs"}

    def test_epqs_takes_priority_over_dem(self, tmp_path: Path) -> None:
        gdf = _stations(6)
        dem = _write_plane_dem(tmp_path / "dem.tif", base_z=20.0)
        epqs = _write_epqs(tmp_path / "epqs.parquet", gdf["segment_id"].tolist(), [50.0] * 6)
        df = process_corridor_stations(gdf, dem_path=dem, usgs_epqs_path=epqs)
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), 50.0)
        assert set(df["elevation_source"].to_list()) == {"usgs_epqs"}

    def test_dem_fills_missing_epqs(self, tmp_path: Path) -> None:
        gdf = _stations(6)
        dem = _write_plane_dem(tmp_path / "dem.tif", base_z=20.0, slope=0.01)
        ids = gdf["segment_id"].tolist()
        # station 2 NaN, station 4 null, station 5 absent from the EPQS file
        epqs = _write_epqs(
            tmp_path / "epqs.parquet", ids[:5], [50.0, 51.0, float("nan"), 53.0, None]
        )
        df = process_corridor_stations(gdf, dem_path=dem, usgs_epqs_path=epqs)
        rows = _by_index(df)
        for i in (0, 1, 3):
            assert rows[i]["elevation_source"] == "usgs_epqs"
        for i in (2, 4, 5):
            assert rows[i]["elevation_source"] == "usgs_dem"
            assert rows[i]["elevation_raw_m"] == pytest.approx(20.0 + 0.01 * 100.0 * i, abs=0.05)

    def test_dem_only(self, tmp_path: Path) -> None:
        gdf = _stations(6)
        dem = _write_plane_dem(tmp_path / "dem.tif", base_z=20.0, slope=0.01)
        df = process_corridor_stations(gdf, dem_path=dem).sort("chainage_m")
        expected = 20.0 + 0.01 * df["chainage_m"].to_numpy()
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), expected, atol=0.05)
        assert set(df["elevation_source"].to_list()) == {"usgs_dem"}

    def test_lidar_overrides_on_bridge_station(self, tmp_path: Path) -> None:
        gdf = _stations(6, has_bridge=[False, True, False, False, False, False])
        epqs = _write_epqs(tmp_path / "epqs.parquet", gdf["segment_id"].tolist(), [10.0] * 6)
        # Class 17 returns only around station 1 (bridge) at z=40; ground at 2 m ignored.
        laz = _write_c17_laz(tmp_path / "deck.laz", [(SYN_E0, SYN_N0 + 100.0)], [40.0])
        df = process_corridor_stations(gdf, laz_path=laz, usgs_epqs_path=epqs)
        rows = _by_index(df)
        assert rows[1]["elevation_source"] == "lidar_c17"
        assert rows[1]["elevation_raw_m"] == pytest.approx(40.0, abs=1e-3)
        for i in (0, 2, 3, 4, 5):
            assert rows[i]["elevation_source"] == "usgs_epqs"
            assert rows[i]["elevation_raw_m"] == pytest.approx(10.0)

    def test_lidar_deck_over_non_bridge_station_ignored(self, tmp_path: Path) -> None:
        """A Class 17 deck crossing above a roadway station must not override it."""
        gdf = _stations(6)
        epqs = _write_epqs(tmp_path / "epqs.parquet", gdf["segment_id"].tolist(), [10.0] * 6)
        laz = _write_c17_laz(tmp_path / "deck.laz", [(SYN_E0, SYN_N0 + 200.0)], [40.0])
        df = process_corridor_stations(gdf, laz_path=laz, usgs_epqs_path=epqs)
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), 10.0)
        assert set(df["elevation_source"].to_list()) == {"usgs_epqs"}

    def test_lidar_deck_over_non_bridge_station_ignored_with_dem(self, tmp_path: Path) -> None:
        gdf = _stations(6)
        dem = _write_plane_dem(tmp_path / "dem.tif", base_z=20.0, slope=0.0)
        laz = _write_c17_laz(tmp_path / "deck.laz", [(SYN_E0, SYN_N0 + 300.0)], [40.0])
        df = process_corridor_stations(gdf, dem_path=dem, laz_path=laz)
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), 20.0, atol=0.05)
        assert set(df["elevation_source"].to_list()) == {"usgs_dem"}

    def test_lidar_never_applied_without_has_bridge_column(self, tmp_path: Path) -> None:
        gdf = _stations(5, include_has_bridge=False)
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), [10.0] * 5)
        laz = _write_c17_laz(tmp_path / "deck.laz", [(SYN_E0, SYN_N0 + 200.0)], [40.0])
        df = process_corridor_stations(gdf, laz_path=laz, usgs_epqs_path=epqs)
        assert set(df["elevation_source"].to_list()) == {"usgs_epqs"}
        assert not df["is_bridge_deck"].any()

    @pytest.mark.parametrize("null_value", [None, np.nan])
    def test_null_has_bridge_treated_as_false(self, tmp_path: Path, null_value: object) -> None:
        gdf = _stations(5)
        gdf["has_bridge"] = pd.Series([False, False, null_value, False, False], dtype=object)
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), [10.0] * 5)
        laz = _write_c17_laz(tmp_path / "deck.laz", [(SYN_E0, SYN_N0 + 200.0)], [40.0])
        df = process_corridor_stations(gdf, laz_path=laz, usgs_epqs_path=epqs)
        rows = _by_index(df)
        assert rows[2]["is_bridge_deck"] is False
        assert rows[2]["elevation_source"] == "usgs_epqs"
        assert rows[2]["elevation_raw_m"] == pytest.approx(10.0)

    def test_lidar_search_radius_35m(self, tmp_path: Path) -> None:
        gdf = _stations(6, has_bridge=[False, False, True, False, True, False])
        epqs = _write_epqs(tmp_path / "epqs.parquet", gdf["segment_id"].tolist(), [10.0] * 6)
        # Returns 25 m east of station 2 (within 35 m) and 60 m east of station 4 (outside).
        laz = _write_c17_laz(
            tmp_path / "deck.laz",
            [(SYN_E0 + 25.0, SYN_N0 + 200.0), (SYN_E0 + 60.0, SYN_N0 + 400.0)],
            [33.0, 44.0],
        )
        df = process_corridor_stations(gdf, laz_path=laz, usgs_epqs_path=epqs)
        rows = _by_index(df)
        assert rows[2]["elevation_source"] == "lidar_c17"
        assert rows[2]["elevation_raw_m"] == pytest.approx(33.0, abs=1e-3)
        # Station 4 sees no returns within 35 m -> interpolated between anchors 3 and 5.
        assert rows[4]["elevation_source"] == "interpolated"
        assert rows[4]["elevation_raw_m"] == pytest.approx(10.0)

    def test_no_source_raises(self, tmp_path: Path) -> None:
        gdf = _stations(5)
        ids = gdf["segment_id"].tolist()
        epqs = _write_epqs(tmp_path / "epqs.parquet", ids, [10.0, 11.0, None, 13.0, 14.0])
        with pytest.raises(ElevationExtractorError):
            process_corridor_stations(gdf, usgs_epqs_path=epqs)

    def test_no_paths_at_all_raises(self) -> None:
        with pytest.raises(ElevationExtractorError):
            process_corridor_stations(_stations(5))

    def test_dem_outside_extent_raises(self, tmp_path: Path) -> None:
        """Stations outside the DEM with no other source must not get a synthetic value."""
        gdf = _stations(5, n0=SYN_N0 + 50000.0)
        dem = _write_plane_dem(tmp_path / "dem.tif")
        with pytest.raises(ElevationExtractorError):
            process_corridor_stations(gdf, dem_path=dem)

    # --- Bridge flags & deck interpolation ------------------------------------------

    def test_is_bridge_deck_mirrors_has_bridge(self, tmp_path: Path) -> None:
        flags = [False, False, True, True, False, False, True, False]
        gdf = _stations(8, has_bridge=flags)
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), [20.0] * 8)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("station_index")
        assert df["is_bridge_deck"].to_list() == flags

    def test_missing_has_bridge_column_all_false(self, tmp_path: Path) -> None:
        gdf = _stations(6, include_has_bridge=False)
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), [20.0] * 6)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs)
        assert not df["is_bridge_deck"].any()

    def test_no_hard_coded_alb_bridge_span(self, tmp_path: Path) -> None:
        """Former fabricated ALB chainage/northing window must not imply a bridge."""
        chainage = np.arange(3500.0, 4350.0, 50.0)
        n = len(chainage)
        gdf = _stations(
            n,
            corridor_id="alb_i495",
            direction="inner_loop",
            e0=311200.0,
            n0=4315000.0 - 3500.0,
            chainage=list(chainage),
        )
        z = [40.0 + 0.1 * i for i in range(n)]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("chainage_m")
        assert not df["is_bridge_deck"].any()
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), z)
        assert set(df["elevation_source"].to_list()) == {"usgs_epqs"}

    def test_bridge_stations_interpolated_between_anchors(self, tmp_path: Path) -> None:
        flags = [False, False, False, False, True, True, False, False, False, False]
        gdf = _stations(10, has_bridge=flags)
        # Ground under the bridge is the river (~ -0.5 m); deck must be bridged by anchors.
        z = [15.0, 15.0, 15.0, 15.0, -0.5, -0.5, 18.0, 18.0, 18.0, 18.0]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs)
        rows = _by_index(df)
        assert rows[4]["elevation_raw_m"] == pytest.approx(16.0)
        assert rows[5]["elevation_raw_m"] == pytest.approx(17.0)
        assert rows[4]["elevation_source"] == "interpolated"
        assert rows[5]["elevation_source"] == "interpolated"
        for i in (0, 1, 2, 3, 6, 7, 8, 9):
            assert rows[i]["elevation_raw_m"] == pytest.approx(z[i])
            assert rows[i]["elevation_source"] == "usgs_epqs"

    def test_interpolation_uses_chainage_not_index(self, tmp_path: Path) -> None:
        chainage = [0.0, 100.0, 130.0, 400.0, 500.0]
        flags = [False, False, True, False, False]
        gdf = _stations(5, has_bridge=flags, chainage=chainage)
        z = [10.0, 10.0, 0.0, 40.0, 40.0]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        rows = _by_index(process_corridor_stations(gdf, usgs_epqs_path=epqs))
        # Linear between (100, 10) and (400, 40) at 130 m -> 13.0
        assert rows[2]["elevation_raw_m"] == pytest.approx(13.0)

    def test_lidar_bridge_station_kept_and_used_as_anchor(self, tmp_path: Path) -> None:
        flags = [False, False, True, True, True, False, False]
        gdf = _stations(7, has_bridge=flags)
        z = [20.0, 20.0, 1.0, 1.0, 1.0, 26.0, 26.0]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        # Deck LiDAR only over station 3 (middle of the span) at 30 m.
        laz = _write_c17_laz(tmp_path / "deck.laz", [(SYN_E0, SYN_N0 + 300.0)], [30.0])
        rows = _by_index(process_corridor_stations(gdf, laz_path=laz, usgs_epqs_path=epqs))
        assert rows[3]["elevation_source"] == "lidar_c17"
        assert rows[3]["elevation_raw_m"] == pytest.approx(30.0, abs=1e-3)
        # Station 2 between anchors (100 m, 20) and (300 m, 30) -> 25
        assert rows[2]["elevation_source"] == "interpolated"
        assert rows[2]["elevation_raw_m"] == pytest.approx(25.0, abs=1e-3)
        # Station 4 between anchors (300 m, 30) and (500 m, 26) -> 28
        assert rows[4]["elevation_source"] == "interpolated"
        assert rows[4]["elevation_raw_m"] == pytest.approx(28.0, abs=1e-3)

    def test_under_bridge_constant(self) -> None:
        assert ELEVATION_SOURCE_UNDER_BRIDGE == "bare_earth_under_bridge"
        assert EPQS_MAX_STATION_OFFSET_M == 0.5

    def test_fewer_than_two_anchors_keeps_measured(self, tmp_path: Path) -> None:
        flags = [True, True, True, True, False]
        gdf = _stations(5, has_bridge=flags)
        z = [5.0, 6.0, 7.0, 8.0, 9.0]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("station_index")
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), z)
        assert df["elevation_source"].to_list() == ["bare_earth_under_bridge"] * 4 + ["usgs_epqs"]
        assert df["is_bridge_deck"].to_list() == flags

    def test_all_bridge_no_anchors_keeps_measured(self, tmp_path: Path) -> None:
        gdf = _stations(5, has_bridge=[True] * 5)
        z = [5.0, 6.0, 7.0, 8.0, 9.0]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("station_index")
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), z)
        assert set(df["elevation_source"].to_list()) == {ELEVATION_SOURCE_UNDER_BRIDGE}

    def test_bridge_at_carriageway_start_not_extrapolated(self, tmp_path: Path) -> None:
        flags = [True, True, False, False, False, False, False]
        gdf = _stations(7, has_bridge=flags)
        z = [1.0, 2.0, 30.0, 31.0, 32.0, 33.0, 34.0]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("station_index")
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), z)
        assert (
            df["elevation_source"].to_list()
            == [ELEVATION_SOURCE_UNDER_BRIDGE] * 2 + ["usgs_epqs"] * 5
        )

    def test_bridge_at_carriageway_end_not_held_flat(self, tmp_path: Path) -> None:
        flags = [False, False, False, False, False, True, True]
        gdf = _stations(7, has_bridge=flags)
        z = [30.0, 31.0, 32.0, 33.0, 34.0, -0.4, 0.7]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("station_index")
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), z)
        assert (
            df["elevation_source"].to_list()
            == ["usgs_epqs"] * 5 + [ELEVATION_SOURCE_UNDER_BRIDGE] * 2
        )

    def test_bridge_end_and_interior_spans_mixed(self, tmp_path: Path) -> None:
        """Interior span is interpolated while the dangling start span keeps bare earth."""
        flags = [True, False, False, True, False, False]
        gdf = _stations(6, has_bridge=flags)
        z = [3.0, 10.0, 10.0, 0.0, 20.0, 20.0]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        rows = _by_index(process_corridor_stations(gdf, usgs_epqs_path=epqs))
        assert rows[0]["elevation_source"] == ELEVATION_SOURCE_UNDER_BRIDGE
        assert rows[0]["elevation_raw_m"] == pytest.approx(3.0)
        assert rows[3]["elevation_source"] == "interpolated"
        assert rows[3]["elevation_raw_m"] == pytest.approx(15.0)

    def test_lidar_anchor_enables_interpolation_at_end(self, tmp_path: Path) -> None:
        """A LiDAR bridge station at the end is an anchor 'after' the preceding bridge station."""
        flags = [False, False, True, True]
        gdf = _stations(4, has_bridge=flags)
        z = [10.0, 10.0, 0.0, 0.0]
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        laz = _write_c17_laz(tmp_path / "deck.laz", [(SYN_E0, SYN_N0 + 300.0)], [30.0])
        rows = _by_index(process_corridor_stations(gdf, laz_path=laz, usgs_epqs_path=epqs))
        assert rows[3]["elevation_source"] == "lidar_c17"
        assert rows[2]["elevation_source"] == "interpolated"
        assert rows[2]["elevation_raw_m"] == pytest.approx(20.0, abs=1e-3)

    # --- EPQS cache alignment -------------------------------------------------------

    def test_epqs_coordinates_aligned_ok(self, tmp_path: Path) -> None:
        gdf = _stations(5)
        z = [10.0, 11.0, 12.0, 13.0, 14.0]
        epqs = _write_epqs_with_coords(tmp_path / "e.parquet", gdf, z, shift_m=0.2)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("station_index")
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), z)
        assert set(df["elevation_source"].to_list()) == {"usgs_epqs"}

    def test_stale_epqs_cache_raises(self, tmp_path: Path) -> None:
        gdf = _stations(5)
        epqs = _write_epqs_with_coords(tmp_path / "e.parquet", gdf, [10.0] * 5, shift_m=10.0)
        with pytest.raises(ElevationExtractorError, match="stale"):
            process_corridor_stations(gdf, usgs_epqs_path=epqs)

    def test_single_stale_row_raises(self, tmp_path: Path) -> None:
        gdf = _stations(5)
        epqs = _write_epqs_with_coords(
            tmp_path / "e.parquet", gdf, [10.0] * 5, shift_m=0.0, shift_index=3, shift_index_m=2.0
        )
        with pytest.raises(ElevationExtractorError, match="stale"):
            process_corridor_stations(gdf, usgs_epqs_path=epqs)

    def test_conflicting_duplicate_epqs_raises(self, tmp_path: Path) -> None:
        gdf = _stations(4)
        ids = gdf["segment_id"].tolist()
        epqs = _write_epqs(tmp_path / "e.parquet", [*ids, ids[1]], [10.0, 11.0, 12.0, 13.0, 99.0])
        with pytest.raises(ElevationExtractorError, match="conflicting"):
            process_corridor_stations(gdf, usgs_epqs_path=epqs)

    def test_identical_duplicate_epqs_ok(self, tmp_path: Path) -> None:
        gdf = _stations(4)
        ids = gdf["segment_id"].tolist()
        epqs = _write_epqs(tmp_path / "e.parquet", [*ids, ids[1]], [10.0, 11.0, 12.0, 13.0, 11.0])
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("station_index")
        assert len(df) == 4
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), [10.0, 11.0, 12.0, 13.0])

    def test_epqs_without_coordinates_joins_by_id(self, tmp_path: Path) -> None:
        gdf = _stations(4)
        ids = list(reversed(gdf["segment_id"].tolist()))
        epqs = _write_epqs(tmp_path / "e.parquet", ids, [13.0, 12.0, 11.0, 10.0])
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("station_index")
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), [10.0, 11.0, 12.0, 13.0])

    # --- Short carriageways -----------------------------------------------------------

    def test_two_station_carriageway_nan_grade(self, tmp_path: Path) -> None:
        gdf = _stations(2)
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), [10.0, 12.0])
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("station_index")
        assert len(df) == 2
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), [10.0, 12.0])
        assert np.all(np.isnan(df["grade_pct"].to_numpy()))
        assert np.all(np.isnan(df["vertical_curvature"].to_numpy()))
        assert df["grade_edge_effect"].all()

    def test_interpolation_scoped_per_carriageway(self, tmp_path: Path) -> None:
        nb = _stations(5, direction="northbound", has_bridge=[False, False, True, False, False])
        sb = _stations(5, direction="southbound", has_bridge=[False] * 5, e0=SYN_E0 + 40.0)
        gdf = gpd.GeoDataFrame(pd.concat([nb, sb], ignore_index=True), crs=CRS_METRIC)
        ids = gdf["segment_id"].tolist()
        z = [10.0, 10.0, 0.0, 20.0, 20.0] + [100.0] * 5
        epqs = _write_epqs(tmp_path / "e.parquet", ids, z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs)
        nb_rows = _by_index(df.filter(pl.col("direction") == "northbound"))
        sb_df = df.filter(pl.col("direction") == "southbound")
        assert nb_rows[2]["elevation_raw_m"] == pytest.approx(15.0)
        np.testing.assert_allclose(sb_df["elevation_raw_m"].to_numpy(), 100.0)

    def test_unsorted_input_ordered_by_chainage(self, tmp_path: Path) -> None:
        gdf = _stations(6, has_bridge=[False, False, True, False, False, False])
        shuffled = gdf.iloc[[3, 0, 5, 2, 1, 4]].reset_index(drop=True)
        z_by_id = dict(zip(gdf["segment_id"], [10.0, 10.0, 0.0, 30.0, 30.0, 30.0], strict=True))
        epqs = _write_epqs(tmp_path / "e.parquet", list(z_by_id.keys()), list(z_by_id.values()))
        rows = _by_index(process_corridor_stations(shuffled, usgs_epqs_path=epqs))
        assert rows[2]["elevation_raw_m"] == pytest.approx(20.0)
        assert rows[2]["elevation_source"] == "interpolated"

    # --- No fabricated repairs on roadway stations ----------------------------------

    def test_non_bridge_dips_and_low_values_kept_exactly(self, tmp_path: Path) -> None:
        """No dip filling, no <2 m replacement, no endpoint extrapolation on roadway."""
        z = [0.5, 30.0, 30.0, 30.0, 30.0, 24.0, 30.0, 30.0, -0.53, 30.0, 1.5]
        gdf = _stations(len(z))
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("chainage_m")
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), z)
        assert not df["is_bridge_deck"].any()
        assert set(df["elevation_source"].to_list()) == {"usgs_epqs"}

    def test_monkeypatched_dem_dip_not_repaired(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even when sampled from a DEM, roadway dips survive untouched."""
        z = np.array([42.0, 32.0, 44.0, 46.0, 36.0, 49.0, 51.0])
        gdf = _stations(len(z))
        dem = _write_plane_dem(tmp_path / "dem.tif")
        monkeypatch.setattr(
            ElevationExtractor, "extract_bare_earth_dem", lambda *args, **kwargs: z.copy()
        )
        df = process_corridor_stations(gdf, dem_path=dem).sort("chainage_m")
        np.testing.assert_allclose(df["elevation_raw_m"].to_numpy(), z)

    # --- Grade edge effect ------------------------------------------------------------

    def test_grade_edge_effect_flags(self, tmp_path: Path) -> None:
        gdf = _stations(10)
        epqs = _write_epqs(
            tmp_path / "e.parquet", gdf["segment_id"].tolist(), [20.0 + i for i in range(10)]
        )
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs).sort("chainage_m")
        assert df["grade_edge_effect"].to_list() == [True, True] + [False] * 6 + [True, True]

    def test_grade_edge_effect_per_carriageway_unsorted(self, tmp_path: Path) -> None:
        nb = _stations(6, direction="northbound")
        sb = _stations(7, direction="southbound", e0=SYN_E0 + 40.0)
        gdf = gpd.GeoDataFrame(pd.concat([sb, nb], ignore_index=True), crs=CRS_METRIC)
        gdf = gdf.sample(frac=1.0, random_state=3).reset_index(drop=True)
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), [20.0] * len(gdf))
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs)
        for direction, n in (("northbound", 6), ("southbound", 7)):
            sub = df.filter(pl.col("direction") == direction).sort("chainage_m")
            assert sub["grade_edge_effect"].to_list() == [True, True] + [False] * (n - 4) + [
                True,
                True,
            ]

    def test_grade_edge_effect_short_carriageway(self, tmp_path: Path) -> None:
        gdf = _stations(4)
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), [1.0, 2, 3, 4])
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs)
        assert df["grade_edge_effect"].all()

    def test_smoothed_outputs_finite(self, tmp_path: Path) -> None:
        z = [50.0 - 0.032 * 100 * i for i in range(6)] + [
            50.0 - 0.032 * 500 + 0.038 * 100 * i for i in range(1, 7)
        ]
        gdf = _stations(len(z))
        epqs = _write_epqs(tmp_path / "e.parquet", gdf["segment_id"].tolist(), z)
        df = process_corridor_stations(gdf, usgs_epqs_path=epqs)
        for col in ("elevation_smooth_m", "grade_pct", "grade_raw_pct", "vertical_curvature"):
            assert np.all(np.isfinite(df[col].to_numpy())), col
        assert df["is_sag"].any()


class TestSmoothElevationProfileEdgeCases:
    """Guard rails for NaN handling in Savitzky-Golay smoothing."""

    def test_all_nan_raises(self) -> None:
        x = np.arange(0.0, 500.0, 100.0)
        with pytest.raises(ElevationExtractorError):
            smooth_elevation_profile(x, np.full(5, np.nan))

    def test_single_valid_value_raises(self) -> None:
        x = np.arange(0.0, 500.0, 100.0)
        z = np.array([np.nan, np.nan, 12.0, np.nan, np.nan])
        with pytest.raises(ElevationExtractorError):
            smooth_elevation_profile(x, z)

    def test_method_also_raises(self, extractor: ElevationExtractor) -> None:
        x = np.arange(0.0, 300.0, 100.0)
        with pytest.raises(ElevationExtractorError):
            extractor.smooth_elevation_profile(x, np.array([np.nan, 1.0, np.nan]))

    def test_interior_nan_interpolated(self) -> None:
        x = np.arange(0.0, 1000.0, 100.0)
        z = 10.0 + 0.02 * x
        z_gap = z.copy()
        z_gap[4] = np.nan
        z_smooth, grade, curvature = smooth_elevation_profile(x, z_gap)
        assert np.all(np.isfinite(z_smooth))
        assert np.all(np.isfinite(grade))
        assert np.all(np.isfinite(curvature))
        np.testing.assert_allclose(z_smooth, z, atol=1e-6)
        np.testing.assert_allclose(grade, 2.0, atol=1e-6)

    def test_two_valid_values_ok(self) -> None:
        x = np.arange(0.0, 500.0, 100.0)
        z = np.array([10.0, np.nan, np.nan, np.nan, 14.0])
        z_smooth, grade, _ = smooth_elevation_profile(x, z)
        assert np.all(np.isfinite(z_smooth))
        np.testing.assert_allclose(grade, 1.0, atol=1e-6)

    @pytest.mark.parametrize("n", [1, 2])
    def test_fewer_than_three_returns_nan_derivatives(self, n: int) -> None:
        """Short profiles: elevation unchanged, grade and curvature all-NaN (never zeros)."""
        x = np.arange(0.0, 100.0 * n, 100.0)
        z = np.array([10.0, 12.5][:n])
        z_out, grade, curvature = smooth_elevation_profile(x, z)
        np.testing.assert_array_equal(np.asarray(z_out), z)
        assert len(grade) == n
        assert len(curvature) == n
        assert np.all(np.isnan(grade))
        assert np.all(np.isnan(curvature))
