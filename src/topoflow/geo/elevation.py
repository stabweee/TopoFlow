"""Hybrid elevation extraction, LiDAR ASPRS Class 17 bridge profiling, and Savitzky-Golay grade estimation.

Uses measured USGS 3DEP bare-earth elevations (EPQS / DEM rasters) for land stations,
measured ASPRS Class 17 (Bridge Deck) LiDAR returns where available, and linear
interpolation between measured neighbours only for OSM-tagged bridge stations that lack
deck returns. Every station records its ``elevation_source``; no elevation is invented.
Optionally validates deck heights against an independent, cited reference profile, and
computes analytical vertical grade (% slope) and curvature via Savitzky-Golay filtering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import polars as pl
import pyproj
import rasterio
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

from topoflow.config import Settings, get_settings
from topoflow.constants import CRS_METRIC, CRS_WGS84

logger = logging.getLogger(__name__)

# Provenance labels for the ``elevation_source`` column of the elevation mart.
ELEVATION_SOURCE_EPQS: str = "usgs_epqs"
ELEVATION_SOURCE_DEM: str = "usgs_dem"
ELEVATION_SOURCE_LIDAR: str = "lidar_c17"
ELEVATION_SOURCE_INTERPOLATED: str = "interpolated"
# Bridge station with no deck measurement and no anchors on both sides: the value is the
# bare-earth surface beneath the bridge, NOT the deck. Exclude from deck-sensitive analysis.
ELEVATION_SOURCE_UNDER_BRIDGE: str = "bare_earth_under_bridge"

# Maximum horizontal offset (EPSG:26918 meters) between a cached EPQS sample and the
# station it is joined to before the cache is considered stale.
EPQS_MAX_STATION_OFFSET_M: float = 0.5


class ElevationExtractorError(Exception):
    """Base exception for elevation extraction and vertical profiling errors."""


class ElevationExtractor:
    """Extracts, cross-validates, and filters roadway elevation and vertical grade profiles."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize ElevationExtractor with project settings.

        Args:
            settings: Optional TopoFlow application Settings instance.
        """
        self.settings: Settings = settings or get_settings()

    def extract_bare_earth_dem(
        self,
        points_gdf: gpd.GeoDataFrame,
        dem_raster_path: Path,
    ) -> np.ndarray:
        """Sample 1m bare-earth DEM elevations (NAVD88 meters) at station coordinates.

        Args:
            points_gdf: GeoDataFrame containing station Point geometries.
            dem_raster_path: Path to bare-earth GeoTIFF raster file.

        Returns:
            1D float64 array of sampled bare-earth elevations in meters.

        Raises:
            FileNotFoundError: If dem_raster_path does not exist.
        """
        dem_raster_path = Path(dem_raster_path)
        if not dem_raster_path.exists():
            raise FileNotFoundError(f"DEM raster file not found at: {dem_raster_path}")

        if len(points_gdf) == 0:
            return np.empty(0, dtype=np.float64)

        with rasterio.open(dem_raster_path) as src:
            gdf_projected = points_gdf.to_crs(src.crs) if points_gdf.crs != src.crs else points_gdf

            coords = [(p.x, p.y) for p in gdf_projected.geometry]
            sampled_values = list(src.sample(coords))
            elevations = np.array([val[0] for val in sampled_values], dtype=np.float64)

            # Mask nodata values with NaN if present
            if src.nodata is not None:
                elevations[elevations == src.nodata] = np.nan

        return elevations

    def extract_bridge_deck_laz(
        self,
        bridge_points_gdf: gpd.GeoDataFrame,
        laz_path: Path,
        buffer_radius_m: float = 35.0,
    ) -> np.ndarray:
        """Extract asphalt deck elevations from ASPRS Class 17 returns in LiDAR point cloud.

        Filters strictly for ASPRS Class 17 (Bridge Deck) laser returns, rejecting ground,
        water, and noise returns. Computes median deck elevation within buffer_radius_m
        of each station Point in UTM Zone 18N (EPSG:26918).

        Args:
            bridge_points_gdf: GeoDataFrame of bridge centerline stations in EPSG:26918.
            laz_path: Path to raw USGS 3DEP .laz point cloud file.
            buffer_radius_m: Search radius in meters around station coordinates (default 15m).

        Returns:
            1D float64 array of extracted bridge deck elevations (NaN for stations with no returns).

        Raises:
            FileNotFoundError: If laz_path does not exist.
        """
        import laspy

        laz_path = Path(laz_path)
        if not laz_path.exists():
            raise FileNotFoundError(f"LiDAR point cloud file not found at: {laz_path}")

        if len(bridge_points_gdf) == 0:
            return np.empty(0, dtype=np.float64)

        gdf_metric = (
            bridge_points_gdf.to_crs(CRS_METRIC)
            if bridge_points_gdf.crs != CRS_METRIC
            else bridge_points_gdf
        )

        las = laspy.read(laz_path)
        mask_class17 = las.classification == 17

        if not np.any(mask_class17):
            logger.warning("No ASPRS Class 17 (Bridge Deck) points found in %s", laz_path)
            return np.full(len(bridge_points_gdf), np.nan, dtype=np.float64)

        deck_x = np.asarray(las.x[mask_class17], dtype=np.float64)
        deck_y = np.asarray(las.y[mask_class17], dtype=np.float64)
        deck_z = np.asarray(las.z[mask_class17], dtype=np.float64)

        tree = cKDTree(np.column_stack([deck_x, deck_y]))

        station_coords = np.column_stack(
            [
                gdf_metric.geometry.x.to_numpy(),
                gdf_metric.geometry.y.to_numpy(),
            ]
        )

        extracted_z = np.full(len(station_coords), np.nan, dtype=np.float64)
        for i, coord in enumerate(station_coords):
            neighbor_indices = tree.query_ball_point(coord, r=buffer_radius_m)
            if len(neighbor_indices) > 0:
                extracted_z[i] = float(np.median(deck_z[neighbor_indices]))

        return extracted_z

    def validate_bridge_as_built(
        self,
        extracted_deck_z: np.ndarray,
        bridge_stations_chainage_m: np.ndarray,
        as_built_csv_path: Path,
        tolerance_m: float = 0.2,
    ) -> bool:
        """Validate extracted bridge deck elevations against an independent reference profile.

        The reference must be independent of this pipeline (e.g. VDOT/MDOT SHA as-built
        plan sheets or a separate survey). Every reference row must cite its origin in a
        non-empty ``source`` column; references derived from TopoFlow's own output are
        circular and must not be used.

        Args:
            extracted_deck_z: 1D array of extracted bridge deck heights (meters).
            bridge_stations_chainage_m: 1D array of bridge chainage offsets (meters).
            as_built_csv_path: CSV with ``station_offset_m``, ``elevation_navd88_m``, ``source``.
            tolerance_m: Maximum acceptable elevation error in meters (default 0.2m / 20cm).

        Returns:
            True if all benchmark checkpoints pass tolerance.

        Raises:
            FileNotFoundError: If as_built_csv_path does not exist.
            ValueError: If rows lack a cited ``source`` or discrepancy exceeds tolerance_m.
        """
        as_built_csv_path = Path(as_built_csv_path)
        if not as_built_csv_path.exists():
            raise FileNotFoundError(f"As-built reference CSV not found at: {as_built_csv_path}")

        df_as_built = pl.read_csv(as_built_csv_path, comment_prefix="#")
        if "source" not in df_as_built.columns:
            raise ValueError("Reference profile must include a per-row 'source' citation column.")
        uncited = df_as_built.filter(
            pl.col("source").is_null() | (pl.col("source").cast(pl.Utf8).str.strip_chars() == "")
        )
        if uncited.height > 0:
            raise ValueError(f"{uncited.height} reference row(s) have no cited 'source'.")
        ref_offsets = df_as_built["station_offset_m"].to_numpy().astype(np.float64)
        ref_z = df_as_built["elevation_navd88_m"].to_numpy().astype(np.float64)

        if len(extracted_deck_z) == len(ref_z) and np.allclose(
            bridge_stations_chainage_m - bridge_stations_chainage_m[0],
            ref_offsets - ref_offsets[0],
            atol=1.0,
        ):
            errors = np.abs(extracted_deck_z - ref_z)
        else:
            # Interpolate benchmark elevations across extracted bridge chainage
            min_chainage = bridge_stations_chainage_m[0]
            norm_chainage = bridge_stations_chainage_m - min_chainage
            interp_ref_z = np.interp(norm_chainage, ref_offsets, ref_z)
            errors = np.abs(extracted_deck_z - interp_ref_z)

        max_error = float(np.max(errors))
        if max_error > tolerance_m:
            raise ValueError(
                f"Extracted bridge deck elevation error ({max_error:.4f}m) exceeds "
                f"VDOT/MDOT As-Built tolerance of {tolerance_m:.4f}m."
            )

        logger.info(
            "Bridge deck As-Built verification passed: max discrepancy = %.4fm <= %.4fm",
            max_error,
            tolerance_m,
        )
        return True

    @staticmethod
    def _align_epqs_to_stations(
        gdf_metric: gpd.GeoDataFrame,
        segment_ids: list[str],
        df_epqs: pl.DataFrame,
        max_offset_m: float = EPQS_MAX_STATION_OFFSET_M,
    ) -> np.ndarray:
        """Join cached EPQS elevations to stations, refusing stale or ambiguous cache rows.

        The EPQS cache is keyed by ``segment_id``; if station geometry changes, the same id
        can refer to a different location. When the cache carries ``longitude``/``latitude``
        the sampled point must lie within ``max_offset_m`` (EPSG:26918) of the station.

        Returns:
            Elevations aligned to ``segment_ids`` (NaN where the cache has no value).

        Raises:
            ElevationExtractorError: On conflicting duplicate ids or stale coordinates.
        """
        conflicts = (
            df_epqs.group_by("segment_id")
            .agg(pl.col("elevation_usgs_m").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
        )
        if conflicts.height:
            raise ElevationExtractorError(
                f"EPQS cache has conflicting elevations for {conflicts.height} segment_id(s): "
                f"{conflicts['segment_id'].to_list()[:10]}"
            )

        has_coords = {"longitude", "latitude"}.issubset(df_epqs.columns)
        cols = ["segment_id", "elevation_usgs_m"] + (
            ["longitude", "latitude"] if has_coords else []
        )
        joined = pl.DataFrame({"segment_id": segment_ids}).join(
            df_epqs.select(cols).unique("segment_id", keep="first"),
            on="segment_id",
            how="left",
            maintain_order="left",
        )
        z = joined["elevation_usgs_m"].cast(pl.Float64).fill_nan(None).to_numpy()

        if not has_coords:
            logger.warning("EPQS cache has no coordinates; station alignment cannot be verified.")
            return z

        lon = joined["longitude"].cast(pl.Float64).to_numpy()
        lat = joined["latitude"].cast(pl.Float64).to_numpy()
        cached = np.isfinite(lon) & np.isfinite(lat)
        ex, ny = pyproj.Transformer.from_crs(CRS_WGS84, CRS_METRIC, always_xy=True).transform(
            lon, lat
        )
        offset = np.hypot(
            np.asarray(ex) - gdf_metric.geometry.x.to_numpy(),
            np.asarray(ny) - gdf_metric.geometry.y.to_numpy(),
        )
        stale = cached & (offset > max_offset_m)
        if np.any(stale):
            stale_ids = [segment_ids[i] for i in np.where(stale)[0]]
            raise ElevationExtractorError(
                f"EPQS cache is stale: {len(stale_ids)} station(s) moved more than "
                f"{max_offset_m} m since sampling (max {float(np.nanmax(offset)):.1f} m), "
                f"e.g. {stale_ids[:5]}. Re-run EPQS ingestion with force_download=True."
            )
        return z

    def smooth_elevation_profile(
        self,
        chainage_m: np.ndarray,
        elevations: np.ndarray,
        window_length: int = 5,
        polyorder: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply Savitzky-Golay polynomial filter to compute smoothed elevation and derivatives.

        Calculates continuous smoothed elevation, analytical first derivative (% grade = dz/dx * 100),
        and analytical second derivative (vertical curvature = d^2z/dx^2).

        Args:
            chainage_m: 1D array of metric chainage along centerline (meters).
            elevations: 1D array of raw elevations in meters.
            window_length: Savitzky-Golay window size in stations (default 5 stations = 500m).
            polyorder: Savitzky-Golay polynomial degree (default 2 for AASHTO parabolic vertical curves).

        Returns:
            Tuple of (elevation_smooth_m, grade_pct, vertical_curvature).
        """
        elevations = np.asarray(elevations, dtype=np.float64)
        chainage_m = np.asarray(chainage_m, dtype=np.float64)

        if len(elevations) < 3:
            # Too few stations to estimate derivatives: report them as unknown, not 0%.
            nan_arr = np.full_like(elevations, np.nan)
            return elevations.copy(), nan_arr, nan_arr.copy()

        if len(elevations) < window_length:
            window_length = max(
                3, len(elevations) if len(elevations) % 2 != 0 else len(elevations) - 1
            )
            polyorder = min(polyorder, window_length - 1)

        # Handle any NaN entries via linear interpolation prior to filtering
        nans = np.isnan(elevations)
        if np.any(nans):
            valid_x = chainage_m[~nans]
            valid_y = elevations[~nans]
            if len(valid_x) < 2:
                raise ElevationExtractorError(
                    "Cannot smooth an elevation profile with fewer than 2 measured values."
                )
            elevations_clean = np.interp(chainage_m, valid_x, valid_y)
        else:
            elevations_clean = elevations.copy()

        delta_x = float(np.mean(np.diff(chainage_m))) if len(chainage_m) > 1 else 100.0
        delta_x = max(1.0, delta_x)

        elevation_smooth = savgol_filter(
            elevations_clean,
            window_length=window_length,
            polyorder=polyorder,
            deriv=0,
        )
        dz_dx = savgol_filter(
            elevations_clean,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=delta_x,
        )
        d2z_dx2 = savgol_filter(
            elevations_clean,
            window_length=window_length,
            polyorder=polyorder,
            deriv=2,
            delta=delta_x,
        )

        grade_pct = dz_dx * 100.0
        return elevation_smooth, grade_pct, d2z_dx2

    def compute_raw_discrete_differences(
        self,
        chainage_m: np.ndarray,
        elevations: np.ndarray,
    ) -> np.ndarray:
        """Compute standard discrete finite central difference vertical grade (% slope).

        Args:
            chainage_m: 1D array of metric chainage along centerline (meters).
            elevations: 1D array of raw elevations in meters.

        Returns:
            1D array of raw discrete grades in percent (100 * dz/dx).
        """
        chainage_m = np.asarray(chainage_m, dtype=np.float64)
        elevations = np.asarray(elevations, dtype=np.float64)
        dz_dx_raw = np.gradient(elevations, chainage_m)
        return dz_dx_raw * 100.0

    def evaluate_filtering_metrics(
        self,
        chainage_m: np.ndarray,
        raw_z: np.ndarray,
        raw_grade: np.ndarray,
        savgol_grade: np.ndarray,
    ) -> dict[str, float]:
        """Compute Total Variation, Jerk, and noise reduction metrics between raw and Savitzky-Golay.

        Args:
            chainage_m: 1D array of chainage (meters).
            raw_z: 1D array of raw elevations (meters).
            raw_grade: 1D array of raw discrete difference grades (%).
            savgol_grade: 1D array of Savitzky-Golay analytical grades (%).

        Returns:
            Dictionary with keys:
            - total_variation_raw
            - total_variation_savgol
            - jerk_raw
            - jerk_savgol
            - tv_reduction_pct
        """
        chainage_m = np.asarray(chainage_m, dtype=np.float64)
        raw_grade = np.asarray(raw_grade, dtype=np.float64)
        savgol_grade = np.asarray(savgol_grade, dtype=np.float64)

        delta_x = float(np.mean(np.diff(chainage_m))) if len(chainage_m) > 1 else 100.0
        delta_x = max(1.0, delta_x)

        tv_raw = float(np.sum(np.abs(np.diff(raw_grade))))
        tv_sg = float(np.sum(np.abs(np.diff(savgol_grade))))

        if len(raw_grade) > 2:
            jerk_raw = float(
                np.mean(
                    ((raw_grade[2:] - 2.0 * raw_grade[1:-1] + raw_grade[:-2]) / (delta_x**2)) ** 2
                )
            )
            jerk_sg = float(
                np.mean(
                    (
                        (savgol_grade[2:] - 2.0 * savgol_grade[1:-1] + savgol_grade[:-2])
                        / (delta_x**2)
                    )
                    ** 2
                )
            )
        else:
            jerk_raw = 0.0
            jerk_sg = 0.0

        tv_reduction = float((tv_raw - tv_sg) / tv_raw * 100.0) if tv_raw > 0 else 0.0

        return {
            "total_variation_raw": tv_raw,
            "total_variation_savgol": tv_sg,
            "jerk_raw": jerk_raw,
            "jerk_savgol": jerk_sg,
            "tv_reduction_pct": tv_reduction,
        }

    def classify_vertical_alignment(
        self,
        chainage_m: np.ndarray,
        grade_pct: np.ndarray,
        vertical_curvature: np.ndarray,
        curvature_threshold: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify vertical geometry into sag curves and crest curves.

        A sag curve has positive vertical curvature (d^2z/dx^2 > 0) with a bottom-of-grade
        inflection (trough). A crest curve has negative vertical curvature (d^2z/dx^2 < 0)
        with a top-of-grade summit.

        Args:
            chainage_m: 1D array of chainage (meters).
            grade_pct: 1D array of vertical grades (%).
            vertical_curvature: 1D array of second derivatives (d^2z/dx^2).
            curvature_threshold: Minimum curvature magnitude to distinguish from flat tangent.

        Returns:
            Tuple of boolean arrays (is_sag, is_crest).
        """
        vertical_curvature = np.asarray(vertical_curvature, dtype=np.float64)
        is_sag = vertical_curvature > curvature_threshold
        is_crest = vertical_curvature < -curvature_threshold
        return is_sag, is_crest

    def process_corridor_stations(
        self,
        stations_gdf: gpd.GeoDataFrame,
        dem_path: Path | None = None,
        laz_path: Path | None = None,
        usgs_epqs_path: Path | None = None,
    ) -> pl.DataFrame:
        """Vectorized end-to-end pipeline generating the complete elevation profile mart.

        Args:
            stations_gdf: GeoDataFrame containing 100m discretized stations.
            dem_path: Optional path to bare-earth 1m DEM GeoTIFF raster.
            laz_path: Optional path to USGS 3DEP LiDAR .laz point cloud file.
            usgs_epqs_path: Optional path to authentic USGS EPQS bare-earth Parquet dataset.

        Returns:
            Polars DataFrame with the 17-column canonical schema (incl. ``elevation_source``, ``grade_edge_effect``).

        Raises:
            ElevationExtractorError: If any station has no measured elevation.
        """
        if len(stations_gdf) == 0:
            return pl.DataFrame()

        gdf = stations_gdf.copy()
        if gdf.crs != CRS_METRIC:
            gdf = gdf.to_crs(CRS_METRIC)

        has_bridge_series = (
            gdf["has_bridge"].fillna(False).astype(bool).to_numpy()
            if "has_bridge" in gdf.columns
            else np.zeros(len(gdf), dtype=bool)
        )

        n_total = len(gdf)
        raw_elevations = np.full(n_total, np.nan, dtype=np.float64)
        elevation_source = np.full(n_total, "", dtype=object)
        # Bridge deck flag comes from the per-station OSM bridge tag (TF-010/TF-011).
        is_bridge_deck_arr = has_bridge_series.copy()
        segment_ids = (
            gdf["segment_id"].astype(str).tolist()
            if "segment_id" in gdf.columns
            else [f"seg_{i:04d}" for i in range(n_total)]
        )

        # 1. Authentic USGS 3DEP EPQS bare-earth elevations (vectorized join on segment_id)
        if usgs_epqs_path is not None and Path(usgs_epqs_path).exists():
            df_epqs = pl.read_parquet(usgs_epqs_path)
            if {"segment_id", "elevation_usgs_m"}.issubset(df_epqs.columns):
                epqs_z = self._align_epqs_to_stations(gdf, segment_ids, df_epqs)
                hit = np.isfinite(epqs_z)
                raw_elevations[hit] = epqs_z[hit]
                elevation_source[hit] = ELEVATION_SOURCE_EPQS

        # 2. Bare-earth DEM raster for stations not yet measured
        if dem_path is not None and Path(dem_path).exists():
            dem_z = self.extract_bare_earth_dem(gdf, Path(dem_path))
            fill_mask = np.isnan(raw_elevations) & np.isfinite(dem_z)
            raw_elevations[fill_mask] = dem_z[fill_mask]
            elevation_source[fill_mask] = ELEVATION_SOURCE_DEM

        # 3. Measured ASPRS Class 17 deck returns override bare-earth on bridge stations only.
        # Deck returns near at-grade stations belong to overpasses, ramps or the opposite
        # carriageway and must not replace the roadway's own surface.
        if laz_path is not None and Path(laz_path).exists():
            deck_z = self.extract_bridge_deck_laz(gdf, Path(laz_path), buffer_radius_m=35.0)
            has_deck = np.isfinite(deck_z)
            valid_deck = has_deck & has_bridge_series
            ignored = int((has_deck & ~has_bridge_series).sum())
            if ignored:
                logger.info(
                    "Ignored Class 17 returns near %d non-bridge station(s) "
                    "(overpass/ramp/adjacent deck)",
                    ignored,
                )
            raw_elevations[valid_deck] = deck_z[valid_deck]
            elevation_source[valid_deck] = ELEVATION_SOURCE_LIDAR

        # 4. No station may be left without a measurement: never invent elevations.
        missing = np.isnan(raw_elevations)
        if np.any(missing):
            missing_ids = [segment_ids[i] for i in np.where(missing)[0]]
            raise ElevationExtractorError(
                f"{len(missing_ids)} station(s) have no measured elevation "
                f"(EPQS/DEM/LiDAR): {missing_ids[:10]}"
            )

        smooth_elevations = np.empty(n_total, dtype=np.float64)
        grade_pct_arr = np.empty(n_total, dtype=np.float64)
        grade_raw_pct_arr = np.empty(n_total, dtype=np.float64)
        curvature_arr = np.empty(n_total, dtype=np.float64)
        is_sag_arr = np.zeros(n_total, dtype=bool)
        is_crest_arr = np.zeros(n_total, dtype=bool)
        # Stations whose 5-station S-G window is truncated by the carriageway end; their
        # grades are less reliable and should be excluded from grade-sensitive analysis.
        grade_edge_effect_arr = np.zeros(n_total, dtype=bool)
        sg_window = 5
        half_window = sg_window // 2

        for (corridor_key, direction_key), group_indices in gdf.groupby(
            ["corridor_id", "direction"]
        ).groups.items():
            idx = np.asarray(group_indices)
            idx = idx[np.argsort(gdf.iloc[idx]["chainage_m"].to_numpy())]
            sub_chainage = gdf.iloc[idx]["chainage_m"].to_numpy().astype(np.float64)
            sub_z = raw_elevations[idx].copy()
            sub_src = elevation_source[idx].copy()

            # Bare-earth DEMs remove bridge decks and sample the ground/water beneath.
            # On OSM bridge stations without a LiDAR deck return, interpolate the deck
            # linearly between measured anchors on BOTH sides and flag it. Bridge stations
            # without an anchor on each side (e.g. a bridge at a carriageway end) are never
            # extrapolated: they keep the bare-earth value, labelled as such.
            on_bridge = is_bridge_deck_arr[idx]
            has_lidar = sub_src == ELEVATION_SOURCE_LIDAR
            needs_deck = on_bridge & ~has_lidar
            anchors = ~on_bridge | has_lidar
            if np.any(needs_deck):
                anchor_c = sub_chainage[anchors]
                if anchor_c.size >= 2:
                    bracketed = (sub_chainage > anchor_c.min()) & (sub_chainage < anchor_c.max())
                else:
                    bracketed = np.zeros(len(idx), dtype=bool)
                interp_mask = needs_deck & bracketed
                if np.any(interp_mask):
                    sub_z[interp_mask] = np.interp(
                        sub_chainage[interp_mask], anchor_c, sub_z[anchors]
                    )
                    sub_src[interp_mask] = ELEVATION_SOURCE_INTERPOLATED
                unresolved = needs_deck & ~bracketed
                if np.any(unresolved):
                    sub_src[unresolved] = ELEVATION_SOURCE_UNDER_BRIDGE
                    logger.warning(
                        "%s/%s: %d bridge station(s) lack anchors on both sides; kept "
                        "bare-earth elevation labelled '%s' (deck height unknown)",
                        corridor_key,
                        direction_key,
                        int(unresolved.sum()),
                        ELEVATION_SOURCE_UNDER_BRIDGE,
                    )

            low_off_bridge = ~on_bridge & (sub_z < 2.0)
            if np.any(low_off_bridge):
                logger.warning(
                    "%s/%s: %d off-bridge station(s) below 2 m NAVD88 (possible untagged "
                    "water crossing): %s",
                    corridor_key,
                    direction_key,
                    int(low_off_bridge.sum()),
                    [segment_ids[i] for i in idx[low_off_bridge]],
                )

            labels, counts = np.unique(sub_src.astype(str), return_counts=True)
            logger.info(
                "%s/%s elevation sources: %s",
                corridor_key,
                direction_key,
                dict(zip(labels.tolist(), counts.tolist(), strict=True)),
            )

            raw_elevations[idx] = sub_z
            elevation_source[idx] = sub_src

            sub_z_smooth, sub_grade, sub_curv = self.smooth_elevation_profile(
                sub_chainage, sub_z, window_length=sg_window, polyorder=2
            )
            edge = np.zeros(len(idx), dtype=bool)
            edge[:half_window] = True
            edge[len(idx) - half_window :] = True
            grade_edge_effect_arr[idx] = edge
            sub_grade_raw = self.compute_raw_discrete_differences(sub_chainage, sub_z)
            sub_sag, sub_crest = self.classify_vertical_alignment(sub_chainage, sub_grade, sub_curv)

            smooth_elevations[idx] = sub_z_smooth
            grade_pct_arr[idx] = sub_grade
            grade_raw_pct_arr[idx] = sub_grade_raw
            curvature_arr[idx] = sub_curv
            is_sag_arr[idx] = sub_sag
            is_crest_arr[idx] = sub_crest

        result_dict: dict[str, Any] = {
            "segment_id": segment_ids,
            "corridor_id": gdf["corridor_id"].to_list()
            if "corridor_id" in gdf.columns
            else ["unknown"] * len(gdf),
            "direction": gdf["direction"].to_list()
            if "direction" in gdf.columns
            else ["unknown"] * len(gdf),
            "station_index": gdf["station_index"].to_list()
            if "station_index" in gdf.columns
            else list(range(len(gdf))),
            "chainage_m": gdf["chainage_m"].to_numpy().astype(np.float64),
            "easting": gdf.geometry.x.to_numpy().astype(np.float64),
            "northing": gdf.geometry.y.to_numpy().astype(np.float64),
            "elevation_raw_m": raw_elevations,
            "elevation_smooth_m": smooth_elevations,
            "grade_pct": grade_pct_arr,
            "grade_raw_pct": grade_raw_pct_arr,
            "vertical_curvature": curvature_arr,
            "is_sag": is_sag_arr,
            "is_crest": is_crest_arr,
            "is_bridge_deck": is_bridge_deck_arr,
            "elevation_source": elevation_source.astype(str).tolist(),
            "grade_edge_effect": grade_edge_effect_arr,
        }

        return pl.DataFrame(result_dict)


# =============================================================================
# Standalone Module-Level Functions
# =============================================================================


def extract_bare_earth_dem(
    points_gdf: gpd.GeoDataFrame,
    dem_raster_path: Path,
) -> np.ndarray:
    """Sample 1m bare-earth DEM elevations at station coordinates."""
    return ElevationExtractor().extract_bare_earth_dem(points_gdf, dem_raster_path)


def extract_bridge_deck_laz(
    bridge_points_gdf: gpd.GeoDataFrame,
    laz_path: Path,
    buffer_radius_m: float = 15.0,
) -> np.ndarray:
    """Extract ASPRS Class 17 bridge deck elevations from raw LiDAR point cloud."""
    return ElevationExtractor().extract_bridge_deck_laz(
        bridge_points_gdf, laz_path, buffer_radius_m=buffer_radius_m
    )


def validate_bridge_as_built(
    extracted_deck_z: np.ndarray,
    bridge_stations_chainage_m: np.ndarray,
    as_built_csv_path: Path,
    tolerance_m: float = 0.2,
) -> bool:
    """Validate extracted bridge deck elevations against official As-Built blueprints."""
    return ElevationExtractor().validate_bridge_as_built(
        extracted_deck_z,
        bridge_stations_chainage_m,
        as_built_csv_path,
        tolerance_m=tolerance_m,
    )


def smooth_elevation_profile(
    chainage_m: np.ndarray,
    elevations: np.ndarray,
    window_length: int = 5,
    polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Savitzky-Golay polynomial filter to compute smoothed elevation and derivatives."""
    return ElevationExtractor().smooth_elevation_profile(
        chainage_m,
        elevations,
        window_length=window_length,
        polyorder=polyorder,
    )


def compute_raw_discrete_differences(
    chainage_m: np.ndarray,
    elevations: np.ndarray,
) -> np.ndarray:
    """Compute standard discrete finite central difference vertical grade (% slope)."""
    return ElevationExtractor().compute_raw_discrete_differences(chainage_m, elevations)


def evaluate_filtering_metrics(
    chainage_m: np.ndarray,
    raw_z: np.ndarray,
    raw_grade: np.ndarray,
    savgol_grade: np.ndarray,
) -> dict[str, float]:
    """Compute Total Variation, Jerk, and noise reduction metrics between raw and Savitzky-Golay."""
    return ElevationExtractor().evaluate_filtering_metrics(
        chainage_m, raw_z, raw_grade, savgol_grade
    )


def classify_vertical_alignment(
    chainage_m: np.ndarray,
    grade_pct: np.ndarray,
    vertical_curvature: np.ndarray,
    curvature_threshold: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify vertical geometry into sag curves, crest curves, and inflection zones."""
    return ElevationExtractor().classify_vertical_alignment(
        chainage_m, grade_pct, vertical_curvature, curvature_threshold=curvature_threshold
    )


def process_corridor_stations(
    stations_gdf: gpd.GeoDataFrame,
    dem_path: Path | None = None,
    laz_path: Path | None = None,
    usgs_epqs_path: Path | None = None,
) -> pl.DataFrame:
    """Vectorized pipeline generating the canonical elevation profile mart."""
    return ElevationExtractor().process_corridor_stations(
        stations_gdf, dem_path=dem_path, laz_path=laz_path, usgs_epqs_path=usgs_epqs_path
    )
