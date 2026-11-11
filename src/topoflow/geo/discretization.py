"""Spatial discretization and differential geometry profiling for highway centerlines.

Resamples continuous directional highway carriageway geometries into strict 100-meter
equidistant stations (chainage), computes differential geometric properties (tangent vectors,
azimuth headings, Three-Point Menger curvature, and horizontal radius of curvature),
and exports the canonical station matrix.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import pyproj
from shapely.geometry import LineString, Point
from shapely.ops import transform

from topoflow.constants import (
    CRS_METRIC,
    CRS_WGS84,
    MAX_STRAIGHT_RADIUS_METERS,
    SPATIAL_RESOLUTION_METERS,
)

logger = logging.getLogger(__name__)


class DiscretizationError(Exception):
    """Base exception for spatial discretization and geometric profiling errors."""


def project_linestring(
    line: LineString,
    source_crs: str = CRS_WGS84,
    target_crs: str = CRS_METRIC,
) -> LineString:
    """Project a Shapely LineString from source CRS to target metric CRS.

    Args:
        line: Input LineString geometry.
        source_crs: Coordinate Reference System of input (defaults to EPSG:4326).
        target_crs: Target metric Coordinate Reference System (defaults to EPSG:26918).

    Returns:
        Projected LineString in target metric coordinates.

    Raises:
        DiscretizationError: If line geometry is empty or invalid.
    """
    if line.is_empty:
        raise DiscretizationError("Cannot project empty LineString.")

    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected = transform(transformer.transform, line)
    return projected


def compute_heading(
    p_from: tuple[float, float],
    p_to: tuple[float, float],
) -> float:
    """Compute forward geographic azimuth heading in degrees clockwise from True North.

    Args:
        p_from: Origin point (x, y) in metric coordinates.
        p_to: Destination point (x, y) in metric coordinates.

    Returns:
        Azimuth in degrees in range [0.0, 360.0).
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    azimuth = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    return round(azimuth, 3)


def menger_curvature(
    p_prev: tuple[float, float],
    p_curr: tuple[float, float],
    p_next: tuple[float, float],
    max_radius: float = MAX_STRAIGHT_RADIUS_METERS,
) -> tuple[float, float]:
    """Calculate Three-Point Menger curvature and horizontal radius of curvature.

    Uses the circumradius formulation across metric station triplets:
        κ = 2 * |(x_B - x_A)(y_C - y_A) - (y_B - y_A)(x_C - x_A)| / (d_AB * d_BC * d_CA)
        R = 1 / κ

    Straight tangents with κ < 1 / max_radius (i.e. κ < 1e-4 m^-1) are cleanly
    capped at κ = 0.0 and R = max_radius.

    Args:
        p_prev: Previous station coordinate (x_A, y_A) in metric meters.
        p_curr: Current station coordinate (x_B, y_B) in metric meters.
        p_next: Next station coordinate (x_C, y_C) in metric meters.
        max_radius: Maximum radius of curvature cap for straightaways in meters.

    Returns:
        Tuple of (curvature_kappa in 1/m, radius_m in meters).
    """
    xa, ya = p_prev
    xb, yb = p_curr
    xc, yc = p_next

    d_ab = math.hypot(xb - xa, yb - ya)
    d_bc = math.hypot(xc - xb, yc - yb)
    d_ca = math.hypot(xa - xc, ya - yc)

    denom = d_ab * d_bc * d_ca
    cross = 2.0 * abs((xb - xa) * (yc - ya) - (yb - ya) * (xc - xa))

    if denom <= 1e-9 or cross <= 1e-9:
        return 0.0, float(max_radius)

    kappa = cross / denom
    min_kappa_threshold = 1.0 / max_radius

    if kappa < min_kappa_threshold:
        return 0.0, float(max_radius)

    radius = min(1.0 / kappa, float(max_radius))
    return float(kappa), float(radius)


def generate_segment_id(
    corridor_id: str,
    direction: str,
    station_index: int,
) -> str:
    """Generate canonical composite spatial identifier for a 100m station.

    Args:
        corridor_id: Corridor identifier (e.g., 'alb_i495').
        direction: Carriageway direction (e.g., 'inner_loop', 'outer_loop').
        station_index: 0-indexed station sequence number.

    Returns:
        Standardized identifier string (e.g., 'alb_i495_inner_loop_s0012').
    """
    return f"{corridor_id}_{direction}_s{station_index:04d}"


def resample_carriageway_stations(
    line_metric: LineString,
    corridor_id: str,
    direction: str,
    interval_m: float = SPATIAL_RESOLUTION_METERS,
    max_radius: float = MAX_STRAIGHT_RADIUS_METERS,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resample a single metric highway LineString into equidistant chainage stations.

    Args:
        line_metric: LineString in metric CRS (EPSG:26918).
        corridor_id: Identifier of the parent corridor.
        direction: Direction of travel along carriageway.
        interval_m: Equidistant chainage distance step (defaults to 100.0m).
        max_radius: Tangent cap threshold for radius of curvature (defaults to 10,000m).
        extra_metadata: Optional corridor metadata (e.g., route, lanes, maxspeed) to propagate.

    Returns:
        List of station dictionaries with complete spatial and differential geometry attributes.

    Raises:
        DiscretizationError: If line length is zero or invalid.
    """
    total_length = line_metric.length
    if total_length <= 0.0:
        raise DiscretizationError(
            f"Cannot discretize line with non-positive length ({total_length}m) for {corridor_id} {direction}"
        )

    # Determine chainage distances: exact multiples of interval_m up to total_length
    if total_length < interval_m:
        # Edge case: short segment
        chainages = [0.0, total_length]
    else:
        num_intervals = int(math.floor(total_length / interval_m))
        chainages = [float(k * interval_m) for k in range(num_intervals + 1)]

    coords_metric: list[tuple[float, float]] = []
    for dist in chainages:
        pt = line_metric.interpolate(dist)
        coords_metric.append((pt.x, pt.y))

    n_points = len(coords_metric)
    if n_points < 2:
        raise DiscretizationError(
            f"Insufficient stations ({n_points}) along {corridor_id} {direction}"
        )

    # Project coordinates back to WGS84 for storage / mapping
    transformer_to_wgs84 = pyproj.Transformer.from_crs(CRS_METRIC, CRS_WGS84, always_xy=True)
    xs_metric = [c[0] for c in coords_metric]
    ys_metric = [c[1] for c in coords_metric]
    lons, lats = transformer_to_wgs84.transform(xs_metric, ys_metric)

    # Compute differential geometry (heading and Menger curvature)
    stations: list[dict[str, Any]] = []
    meta = extra_metadata or {}

    for i in range(n_points):
        p_curr = coords_metric[i]

        # Heading: forward tangent vector
        if i < n_points - 1:
            heading = compute_heading(p_curr, coords_metric[i + 1])
        else:
            # End boundary adopts heading from penultimate station
            heading = compute_heading(coords_metric[i - 1], p_curr)

        # Menger curvature: three-point osculating circle
        if n_points == 2:
            kappa, radius = 0.0, float(max_radius)
        elif i == 0:
            # Start boundary adopts curvature of first interior triplet (0, 1, 2)
            kappa, radius = menger_curvature(
                coords_metric[0], coords_metric[1], coords_metric[2], max_radius
            )
        elif i == n_points - 1:
            # End boundary adopts curvature of last interior triplet (N-3, N-2, N-1)
            kappa, radius = menger_curvature(
                coords_metric[n_points - 3],
                coords_metric[n_points - 2],
                coords_metric[n_points - 1],
                max_radius,
            )
        else:
            kappa, radius = menger_curvature(
                coords_metric[i - 1], p_curr, coords_metric[i + 1], max_radius
            )

        segment_id = generate_segment_id(corridor_id, direction, i)
        station_record: dict[str, Any] = {
            "segment_id": segment_id,
            "corridor_id": corridor_id,
            "direction": direction,
            "station_index": i,
            "chainage_m": round(chainages[i], 2),
            "easting": round(p_curr[0], 3),
            "northing": round(p_curr[1], 3),
            "longitude": round(lons[i], 7),
            "latitude": round(lats[i], 7),
            "heading_deg": heading,
            "curvature_kappa": round(kappa, 8),
            "radius_m": round(radius, 2),
            "geometry": Point(lons[i], lats[i]),
        }

        # Propagate metadata if available
        for key, val in meta.items():
            if key not in station_record and key != "geometry":
                station_record[key] = val

        stations.append(station_record)

    return stations


class SpatialDiscretizer:
    """Discretizes highway vector corridors into canonical 100m station matrices."""

    def __init__(
        self,
        interval_m: float = SPATIAL_RESOLUTION_METERS,
        max_radius: float = MAX_STRAIGHT_RADIUS_METERS,
        source_crs: str = CRS_WGS84,
        metric_crs: str = CRS_METRIC,
    ) -> None:
        """Initialize SpatialDiscretizer with spatial parameters and CRS definitions.

        Args:
            interval_m: Station interval step in meters (defaults to 100.0m).
            max_radius: Maximum radius threshold in meters (defaults to 10,000m).
            source_crs: Input geometry CRS (defaults to EPSG:4326).
            metric_crs: Analytical metric CRS (defaults to EPSG:26918).
        """
        self.interval_m = interval_m
        self.max_radius = max_radius
        self.source_crs = source_crs
        self.metric_crs = metric_crs

    def discretize_corridors_dataframe(
        self,
        corridors_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Discretize all corridor carriageways in a GeoDataFrame into 100m stations.

        Args:
            corridors_gdf: GeoDataFrame containing corridor geometries (from TF-010).

        Returns:
            GeoDataFrame containing all 100m station points with differential geometry.

        Raises:
            DiscretizationError: If corridors_gdf is empty or missing required fields.
        """
        if corridors_gdf.empty:
            raise DiscretizationError("Cannot discretize empty GeoDataFrame.")

        required_cols = {"corridor_id", "direction", "geometry"}
        missing = required_cols - set(corridors_gdf.columns)
        if missing:
            raise DiscretizationError(f"Input GeoDataFrame missing required columns: {missing}")

        all_stations: list[dict[str, Any]] = []

        for _, row in corridors_gdf.iterrows():
            geom = row["geometry"]
            if not isinstance(geom, LineString):
                logger.warning(
                    "Skipping non-LineString geometry for %s %s",
                    row.get("corridor_id"),
                    row.get("direction"),
                )
                continue

            # Project line to metric UTM Zone 18N
            line_metric = project_linestring(
                geom,
                source_crs=self.source_crs,
                target_crs=self.metric_crs,
            )

            # Metadata extraction
            metadata = {
                k: v
                for k, v in row.items()
                if k not in ("geometry", "length_meters", "is_continuous")
            }

            stations = resample_carriageway_stations(
                line_metric=line_metric,
                corridor_id=str(row["corridor_id"]),
                direction=str(row["direction"]),
                interval_m=self.interval_m,
                max_radius=self.max_radius,
                extra_metadata=metadata,
            )
            all_stations.extend(stations)

        if not all_stations:
            raise DiscretizationError(
                "No stations were successfully generated from input geometries."
            )

        gdf = gpd.GeoDataFrame(all_stations, crs=CRS_WGS84)
        return gdf

    def export(
        self,
        stations_gdf: gpd.GeoDataFrame,
        output_path: Path,
    ) -> Path:
        """Export discretized stations GeoDataFrame to Parquet format.

        Args:
            stations_gdf: GeoDataFrame of stations to serialize.
            output_path: Destination Parquet file path.

        Returns:
            Absolute Path to written Parquet file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stations_gdf.to_parquet(output_path)
        logger.info(
            "Exported %d stations across %d unique corridors to %s",
            len(stations_gdf),
            stations_gdf["corridor_id"].nunique(),
            output_path,
        )
        return output_path
