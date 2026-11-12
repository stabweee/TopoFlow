"""USGS 3DEP LiDAR point cloud ingestion module.

Queries the official USGS 3DEP public archive on AWS Open Data
(USGS_LPC_VA_Fairfax_County_2018), downloads intersecting octree LAZ chunks
covering the American Legion Bridge Potomac River crossing, reprojects
laser returns into canonical NAD83 / UTM Zone 18N (EPSG:26918), and persists
standard compressed .laz point clouds.
"""

from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pyproj
import requests

from topoflow.config import get_settings
from topoflow.constants import CRS_METRIC

logger = logging.getLogger(__name__)

DEFAULT_EPT_BASE_URL: str = (
    "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/USGS_LPC_VA_Fairfax_County_2018"
)
CRS_EPT: str = "EPSG:3857"


def compute_node_bounds(
    node_key: str,
    root_bounds: list[float],
) -> tuple[float, float, float, float]:
    """Compute the 2D bounding box (min_x, min_y, max_x, max_y) for an EPT octree node.

    Args:
        node_key: Octree identifier formatted as '{depth}-{x}-{y}-{z}'.
        root_bounds: Root bounding box [min_x, min_y, min_z, max_x, max_y, max_z] in EPSG:3857.

    Returns:
        Tuple of (min_x, min_y, max_x, max_y).
    """
    depth, nx, ny, _ = [int(p) for p in node_key.split("-")]
    span = root_bounds[3] - root_bounds[0]
    node_span = span / (2**depth)
    min_x = root_bounds[0] + nx * node_span
    max_x = min_x + node_span
    min_y = root_bounds[1] + ny * node_span
    max_y = min_y + node_span
    return min_x, min_y, max_x, max_y


def node_intersects_bbox(
    node_bounds: tuple[float, float, float, float],
    target_bbox: tuple[float, float, float, float],
) -> bool:
    """Check if node bounding box intersects target bounding box.

    Args:
        node_bounds: (min_x, min_y, max_x, max_y)
        target_bbox: (min_x, min_y, max_x, max_y)

    Returns:
        True if bounding boxes intersect.
    """
    return (
        node_bounds[0] <= target_bbox[2]
        and node_bounds[2] >= target_bbox[0]
        and node_bounds[1] <= target_bbox[3]
        and node_bounds[3] >= target_bbox[1]
    )


def find_intersecting_ept_nodes(
    ept_base_url: str,
    target_bbox_3857: tuple[float, float, float, float],
    max_depth: int = 8,
    session: requests.Session | None = None,
) -> list[tuple[str, int]]:
    """Traverse EPT hierarchy to find all node keys intersecting target bbox up to max_depth.

    Args:
        ept_base_url: Base HTTPS URL to EPT dataset root.
        target_bbox_3857: (min_x, min_y, max_x, max_y) in EPSG:3857.
        max_depth: Maximum octree recursion depth.
        session: Optional HTTP session.

    Returns:
        List of tuples (node_key, point_count).
    """
    s = session or requests.Session()
    ept_info_resp = s.get(f"{ept_base_url}/ept.json", timeout=15)
    ept_info_resp.raise_for_status()
    ept_info = ept_info_resp.json()
    root_bounds = ept_info["bounds"]

    root_hier_resp = s.get(f"{ept_base_url}/ept-hierarchy/0-0-0-0.json", timeout=15)
    root_hier_resp.raise_for_status()
    root_hier = root_hier_resp.json()

    intersecting_nodes: list[tuple[str, int]] = []
    subhier_queue: list[str] = []

    for key, count in root_hier.items():
        depth = int(key.split("-")[0])
        if depth > max_depth:
            continue
        nb = compute_node_bounds(key, root_bounds)
        if node_intersects_bbox(nb, target_bbox_3857):
            if count == -1:
                subhier_queue.append(key)
            elif count > 0:
                intersecting_nodes.append((key, count))

    # Process sub-hierarchies
    for sh_key in subhier_queue:
        sh_url = f"{ept_base_url}/ept-hierarchy/{sh_key}.json"
        try:
            resp = s.get(sh_url, timeout=15)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            sh_data = resp.json()
            for key, count in sh_data.items():
                depth = int(key.split("-")[0])
                if depth > max_depth or count <= 0:
                    continue
                nb = compute_node_bounds(key, root_bounds)
                if node_intersects_bbox(nb, target_bbox_3857):
                    intersecting_nodes.append((key, count))
        except requests.RequestException as e:
            logger.warning("Failed to fetch sub-hierarchy %s: %s", sh_key, e)

    return intersecting_nodes


def fetch_ept_node_laz(
    node_key: str,
    ept_base_url: str,
    session: requests.Session | None = None,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> laspy.LasData | None:
    """Download and parse a single EPT node LAZ chunk with exponential backoff.

    Args:
        node_key: EPT node identifier (e.g. '6-42-50-31').
        ept_base_url: Base EPT archive URL.
        session: Optional HTTP session.
        max_retries: Retry attempts on transient errors.
        backoff_factor: Multiplier for exponential backoff sleep.

    Returns:
        laspy.LasData if successfully fetched, None if 404 or exhausted retries.
    """
    s = session or requests.Session()
    url = f"{ept_base_url}/ept-data/{node_key}.laz"

    for attempt in range(max_retries):
        try:
            resp = s.get(url, timeout=20)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return laspy.read(io.BytesIO(resp.content))
        except (requests.RequestException, TimeoutError) as e:
            if attempt == max_retries - 1:
                logger.warning(
                    "Failed to fetch node %s after %d retries: %s", node_key, max_retries, e
                )
                return None
            sleep_time = backoff_factor * (2**attempt)
            logger.debug(
                "Transient error fetching %s, retrying in %.1fs: %s", node_key, sleep_time, e
            )
            time.sleep(sleep_time)

    return None


def reproject_points(
    x: np.ndarray,
    y: np.ndarray,
    source_crs: str = CRS_EPT,
    target_crs: str = CRS_METRIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject 2D coordinate arrays from source CRS to target CRS (default UTM 18N).

    Args:
        x: 1D array of X coordinates.
        y: 1D array of Y coordinates.
        source_crs: Source coordinate system (default EPSG:3857).
        target_crs: Target coordinate system (default EPSG:26918).

    Returns:
        Tuple of (target_x, target_y) as float64 numpy arrays.
    """
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    tx, ty = transformer.transform(x, y)
    return np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)


def ingest_usgs_alb_lidar(
    output_path: Path,
    stations_gdf: gpd.GeoDataFrame | None = None,
    buffer_m: float = 120.0,
    ept_base_url: str = DEFAULT_EPT_BASE_URL,
    max_depth: int = 7,
    max_workers: int = 4,
    force_download: bool = False,
) -> Path:
    """Ingest authentic USGS 3DEP LiDAR returns for American Legion Bridge and persist to LAZ.

    Args:
        output_path: Target destination path for .laz file.
        stations_gdf: Optional GeoDataFrame of corridor stations (loads stations_100m.parquet if None).
        buffer_m: Lateral buffer distance around bridge centerline in meters.
        ept_base_url: Base EPT repository URL.
        max_depth: Maximum octree depth to query (depth 7 provides dense bridge deck coverage).
        max_workers: Concurrent thread workers for chunk downloading.
        force_download: If True, bypass cached file and overwrite.

    Returns:
        Path to persisted .laz file.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force_download:
        logger.info("Using cached authentic LiDAR point cloud at %s", output_path)
        return output_path

    settings = get_settings()
    if stations_gdf is None:
        stations_path = settings.interim_data_dir / "geo" / "stations_100m.parquet"
        if not stations_path.exists():
            raise FileNotFoundError(f"Stations dataset not found at {stations_path}")
        stations_gdf = gpd.read_parquet(stations_path)

    gdf_metric = (
        stations_gdf.to_crs(CRS_METRIC) if stations_gdf.crs != CRS_METRIC else stations_gdf.copy()
    )

    # Filter to ALB Potomac river crossing span
    alb_bridge = gdf_metric[
        (gdf_metric["corridor_id"] == "alb_i495")
        & (gdf_metric["has_bridge"].fillna(False))
        & (gdf_metric.geometry.y >= 4315400.0)
        & (gdf_metric.geometry.y <= 4316500.0)
    ]
    if len(alb_bridge) == 0:
        raise ValueError("No bridge stations found for alb_i495 within northing [4315400, 4316500]")

    bridge_union = alb_bridge.geometry.union_all()
    buffered_polygon = bridge_union.buffer(buffer_m)
    bounds_utm = buffered_polygon.bounds  # minx, miny, maxx, maxy

    # Transform target bounds to EPSG:3857 for octree traversal
    t_to_3857 = pyproj.Transformer.from_crs(CRS_METRIC, CRS_EPT, always_xy=True)
    p1_x, p1_y = t_to_3857.transform(bounds_utm[0], bounds_utm[1])
    p2_x, p2_y = t_to_3857.transform(bounds_utm[2], bounds_utm[3])
    target_bbox_3857 = (
        min(p1_x, p2_x),
        min(p1_y, p2_y),
        max(p1_x, p2_x),
        max(p1_y, p2_y),
    )

    logger.info("Traversing USGS 3DEP EPT hierarchy up to depth %d for ALB bridge...", max_depth)
    session = requests.Session()
    intersecting_nodes = find_intersecting_ept_nodes(
        ept_base_url=ept_base_url,
        target_bbox_3857=target_bbox_3857,
        max_depth=max_depth,
        session=session,
    )
    logger.info(
        "Discovered %d intersecting EPT octree nodes (%d estimated points).",
        len(intersecting_nodes),
        sum(count for _, count in intersecting_nodes),
    )

    # Download nodes concurrently
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    all_cls: list[np.ndarray] = []
    all_intensity: list[np.ndarray] = []
    all_returns: list[np.ndarray] = []

    logger.info(
        "Downloading %d octree chunks with %d workers...", len(intersecting_nodes), max_workers
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_ept_node_laz, node_key, ept_base_url, session): node_key
            for node_key, _ in intersecting_nodes
        }
        for future in as_completed(futures):
            las_chunk = future.result()
            if las_chunk is not None and len(las_chunk.points) > 0:
                all_x.append(np.asarray(las_chunk.x, dtype=np.float64))
                all_y.append(np.asarray(las_chunk.y, dtype=np.float64))
                all_z.append(np.asarray(las_chunk.z, dtype=np.float64))
                all_cls.append(np.asarray(las_chunk.classification, dtype=np.uint8))
                all_intensity.append(np.asarray(las_chunk.intensity, dtype=np.uint16))
                all_returns.append(np.asarray(las_chunk.return_number, dtype=np.uint8))

    if not all_x:
        raise RuntimeError("No LiDAR returns were retrieved from EPT service.")

    raw_x = np.concatenate(all_x)
    raw_y = np.concatenate(all_y)
    raw_z = np.concatenate(all_z)
    raw_cls = np.concatenate(all_cls)
    raw_intensity = np.concatenate(all_intensity)
    raw_returns = np.concatenate(all_returns)

    logger.info("Reprojecting %d downloaded points to %s (UTM 18N)...", len(raw_x), CRS_METRIC)
    utm_x, utm_y = reproject_points(raw_x, raw_y, source_crs=CRS_EPT, target_crs=CRS_METRIC)

    # Spatial filter: keep only points inside the buffered bridge polygon
    spatial_mask = (
        (utm_x >= bounds_utm[0])
        & (utm_x <= bounds_utm[2])
        & (utm_y >= bounds_utm[1])
        & (utm_y <= bounds_utm[3])
    )
    final_x = utm_x[spatial_mask]
    final_y = utm_y[spatial_mask]
    final_z = raw_z[spatial_mask]
    final_cls = raw_cls[spatial_mask]
    final_intensity = raw_intensity[spatial_mask]
    final_returns = raw_returns[spatial_mask]

    logger.info(
        "Retained %d points within bridge bounding box (Z: [%.2fm, %.2fm]).",
        len(final_x),
        float(np.min(final_z)),
        float(np.max(final_z)),
    )
    unique_cls, cls_counts = np.unique(final_cls, return_counts=True)
    logger.info(
        "Classification breakdown: %s",
        dict(zip(unique_cls.tolist(), cls_counts.tolist(), strict=False)),
    )

    # Persist as canonical compressed LAZ file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = [
        float(np.floor(np.min(final_x))),
        float(np.floor(np.min(final_y))),
        float(np.floor(np.min(final_z))),
    ]
    header.scales = [0.01, 0.01, 0.01]
    header.add_crs(pyproj.CRS.from_user_input(CRS_METRIC))

    las_out = laspy.LasData(header)
    las_out.x = final_x
    las_out.y = final_y
    las_out.z = final_z
    las_out.classification = final_cls
    las_out.intensity = final_intensity
    las_out.return_number = final_returns
    las_out.write(output_path)

    logger.info(
        "Successfully persisted authentic bridge LiDAR point cloud to %s (%d points)",
        output_path,
        len(final_x),
    )
    return output_path
